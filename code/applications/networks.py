"""Shared variance accounting and grouped-MGP inference for network applications."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from mgp_core import (
    Z95,
    moving_block_indices,
    regularity_report,
    sample_cov,
    solve_grouped_mgp_dual,
)
from near_optimal import delta_rate, near_optimal_derivative


def network_variance_matrix(mean_weights: np.ndarray, pii_a: np.ndarray, xi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return V such that aggregate variance from shock covariance S is <V,S>.

    With X_t = rho X_{t-1} + Pi_a eps_t + Xi eps_{t-1}, H_0=Pi_a,
    H_1=rho Pi_a + Xi, and H_h=rho H_{h-1} for h>=2. Thus
    Var(w'X) = <sum_h H_h' w w' H_h, Sigma_eps>.
    """
    w = np.asarray(mean_weights, dtype=float)
    w = w / w.sum()
    ww = np.outer(w, w)
    h1 = rho @ pii_a + xi
    u = solve_discrete_lyapunov(rho.T, ww)
    v = pii_a.T @ ww @ pii_a + h1.T @ u @ h1
    return 0.5 * (v + v.T)


def matrix_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def network_denominator(sigma_eps: np.ndarray, network_v: np.ndarray) -> float:
    return matrix_inner(network_v, sigma_eps)


def network_direct_diagonal_share(
    sigma_eps: np.ndarray,
    network_v: np.ndarray,
) -> float:
    """Naive network share that assigns every marginal variance to primitives.

    Unlike a homoskedastic mean-variance counterfactual, this object is
    invariant to reciprocal unit-specific rescalings of primitive shocks and
    their propagation exposures.
    """
    denominator = network_denominator(sigma_eps, network_v)
    return float(np.diag(network_v) @ np.diag(sigma_eps) / denominator)


def reported_dynamic_share(value: float) -> float:
    """Cap a dynamic grouped ratio at the logical upper bound of one.

    The uncapped ratio remains the object used for the quotient derivative and
    bootstrap centering.  Capping is only a reporting transformation.
    """
    value = float(value)
    return min(1.0, value) if np.isfinite(value) else value


_DYNAMIC_SHARE_REPORT_FIELDS = (
    "mgp_share",
    "normal_ucl95",
    "gradient_bootstrap_p05",
    "gradient_bootstrap_p95",
    "moving_block_gradient_bootstrap_p05",
    "moving_block_gradient_bootstrap_p95",
)


def cap_dynamic_share_report(summary: dict[str, object]) -> dict[str, object]:
    """Apply the logical cap to share-valued fields exposed for reporting."""
    for key in _DYNAMIC_SHARE_REPORT_FIELDS:
        if key in summary and not pd.isna(summary[key]):
            summary[key] = reported_dynamic_share(float(summary[key]))
    return summary


def network_share_directional_derivative(
    numerator_derivative: np.ndarray | float,
    denominator_derivative: np.ndarray | float,
    share: float,
    denominator: float,
) -> np.ndarray:
    """Quotient derivative for a structural network granularity share."""
    return (
        np.asarray(numerator_derivative) / denominator
        - share * np.asarray(denominator_derivative) / denominator
    )


def network_share_from_covariances(
    sigma_grouped: np.ndarray,
    sigma_eps: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    solver: str,
) -> dict[str, object]:
    mgp = solve_grouped_mgp_dual(sigma_grouped, solver)
    gamma_norm = float(mgp["value"])
    gamma_level = weight_scale * gamma_norm
    denom = network_denominator(sigma_eps, network_v)
    share_uncapped = gamma_level / denom
    return {
        "mgp_share": reported_dynamic_share(share_uncapped),
        "mgp_share_uncapped": share_uncapped,
        "mgp_level": gamma_level,
        "mgp_level_normalized": gamma_norm,
        "aggregate_variance": denom,
        "solver": mgp["solver"],
        "status": mgp["status"],
        "dual_min_eig": mgp["dual_min_eig"],
        "dual_min_diag": mgp["dual_min_diag"],
        "Y": mgp["Y"],
    }


def network_observed_stats(
    x_grouped: np.ndarray,
    eps_obs: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    solver: str,
) -> dict[str, object]:
    sigma_g = sample_cov(x_grouped)
    sigma_eps = sample_cov(eps_obs)
    observed = network_share_from_covariances(sigma_g, sigma_eps, network_v, weight_scale, solver)

    x_centered = x_grouped - x_grouped.mean(axis=0, keepdims=True)
    eps_centered = eps_obs - eps_obs.mean(axis=0, keepdims=True)
    y_dual = np.asarray(observed["Y"], dtype=float)
    q_level = weight_scale * np.einsum("ti,ij,tj->t", x_centered, y_dual, x_centered)
    denom_obs = np.einsum("ti,ij,tj->t", eps_centered, network_v, eps_centered)
    share = float(observed["mgp_share_uncapped"])
    denom = float(observed["aggregate_variance"])
    r_if = (q_level - share * denom_obs) / denom
    se = float(np.std(r_if, ddof=1) / np.sqrt(x_grouped.shape[0]))
    observed["se_share"] = se
    normal_ucl95_uncapped = share + Z95 * se
    observed["normal_ucl95"] = reported_dynamic_share(normal_ucl95_uncapped)
    observed["normal_ucl95_uncapped"] = normal_ucl95_uncapped
    observed.pop("Y", None)
    return observed


def network_gradient_family(
    x_grouped: np.ndarray,
    eps_obs: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    b: int,
    block_length: int,
    rng: np.random.Generator,
    solver: str,
) -> dict[str, object]:
    if b <= 0:
        return {}
    t, m = x_grouped.shape
    sigma_g = sample_cov(x_grouped)
    sigma_eps = sample_cov(eps_obs)
    mgp = solve_grouped_mgp_dual(sigma_g, solver)
    gamma_level = weight_scale * float(mgp["value"])
    denom = network_denominator(sigma_eps, network_v)
    g_hat = gamma_level / denom

    regularity = regularity_report(sigma_g, solver)
    # The numerator derivative is the SDP directional derivative Gamma_bar'(Sigma_g; H),
    # a minimum over the dual optimizer set -- NOT <Y*, H>, which is valid only in the
    # regular case. We estimate that set by near-optimality, as in
    # gradient_bootstrap.ShareDirectionalDerivative.
    delta = delta_rate(t)
    cap = float(mgp["value"]) * (1.0 + delta)
    print(
        f"[network_gradient_bootstrap] B={b}, block_length={block_length}, M={m}, "
        f"min over near-optimal set delta_T={delta:.4f} (T={t}); "
        f"diagnostics: common rank r={regularity['common_rank']} (shared draws)",
        flush=True,
    )

    def pass_draws(idx_list: list[np.ndarray]) -> np.ndarray:
        n = len(idx_list)
        h_g = np.empty((n, m, m))
        den_deriv = np.empty(n)
        for k, idx in enumerate(idx_list):
            sig_g_b = sample_cov(x_grouped[idx])
            sig_eps_b = sample_cov(eps_obs[idx])
            h_g[k] = np.sqrt(t) * (sig_g_b - sigma_g)
            den_deriv[k] = matrix_inner(network_v, np.sqrt(t) * (sig_eps_b - sigma_eps))
        level, ok = near_optimal_derivative(h_g, sigma_g[None], np.array([cap]))
        if not ok.all():
            raise RuntimeError(
                f"near-optimal derivative failed on {int((~ok).sum())}/{n} directions"
            )
        d_vals = network_share_directional_derivative(
            weight_scale * level, den_deriv, g_hat, denom
        )
        return d_vals

    idx_iid = [rng.choice(t, size=t, replace=True) for _ in range(b)]
    idx_blk = [moving_block_indices(t, block_length, rng) for _ in range(b)]
    d_iid = pass_draws(idx_iid)
    d_blk = pass_draws(idx_blk)

    rt = np.sqrt(t)
    grad_low = float(g_hat - np.quantile(d_iid, 0.95) / rt)
    grad_high = float(g_hat - np.quantile(d_iid, 0.05) / rt)
    block_grad_low = float(g_hat - np.quantile(d_blk, 0.95) / rt)
    block_grad_high = float(g_hat - np.quantile(d_blk, 0.05) / rt)
    return {
        "gradient_bootstrap_p05": reported_dynamic_share(grad_low),
        "gradient_bootstrap_p95": reported_dynamic_share(grad_high),
        "moving_block_gradient_bootstrap_p05": reported_dynamic_share(block_grad_low),
        "moving_block_gradient_bootstrap_p95": reported_dynamic_share(block_grad_high),
        "gradient_bootstrap_p05_uncapped": grad_low,
        "gradient_bootstrap_p95_uncapped": grad_high,
        "moving_block_gradient_bootstrap_p05_uncapped": block_grad_low,
        "moving_block_gradient_bootstrap_p95_uncapped": block_grad_high,
        "gradient_bootstrap_common_rank": regularity["common_rank"],
        "gradient_bootstrap_regular": regularity["regular"],
        "gradient_bootstrap_delta_T": delta,
    }
