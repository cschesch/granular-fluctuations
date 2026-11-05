#!/usr/bin/env python3
"""Grouped-MGP estimate on Foerster-Sarte-Watson benchmark epsilon shocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

import bias as B
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    rule_m,
    ROOT,
    OUTPUT_DIR,
    RUN_OUTPUT_CACHE,
    Z95,
    align_sizes_to_panel,
    build_lpt_grouped_data,
    choose_solver,
    default_subsample_length,
    make_panel_rng,
    moving_block_indices,
    plot_lpt_squared_bucket_sizes,
    regularity_report,
    sample_cov,
    solve_grouped_mgp_dual,
    subsample_indices,
    subsampling_summary,
)
from factor_rank import factor_rank_tests
from near_optimal import delta_rate, near_optimal_derivative
from run_multi_source import SourcePanel, lpt_diagnostics, output_prefix


from .prepare import EXPORTS as FOERSTER_OUT

DEFAULT_EPSILON_CSV = FOERSTER_OUT / "eps_q_98_0_1972_2007benchmark_wide.csv"
DEFAULT_WEIGHTS_CSV = FOERSTER_OUT / "l4wq_q_98_0_1972_2007.csv"
DEFAULT_COLUMNS_CSV = FOERSTER_OUT / "eps_q_98_0_1972_2007benchmark_columns.csv"
DEFAULT_PII_A_CSV = FOERSTER_OUT / "pii_a_98_benchmark.csv"
DEFAULT_XI_CSV = FOERSTER_OUT / "xi_98_benchmark.csv"
DEFAULT_CAP_RHO_CSV = FOERSTER_OUT / "cap_rho_98_benchmark.csv"
DEFAULT_RESULTS_DIR = OUTPUT_DIR


def cal_to_quarter(value: float) -> str:
    year = int(np.floor(float(value) + 1e-8))
    quarter = int(round((float(value) - year) * 4.0)) + 1
    if quarter < 1 or quarter > 4:
        raise ValueError(f"Invalid Foerster quarterly date: {value}")
    return f"{year}Q{quarter}"


def load_foerster_shocks_and_weights(
    epsilon_csv: Path, weights_csv: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    export = pd.read_csv(epsilon_csv)
    eps = export.filter(regex=r"^eps_")
    if eps.empty or not (eps.iloc[0] == 0).all():
        raise ValueError("Expected a Foerster epsilon export starting at initialization.")
    weights = pd.read_csv(weights_csv, header=None)
    if weights.shape != (len(eps), eps.shape[1] + 1) or not np.allclose(
        weights.iloc[:, 0], export["cal"]
    ):
        raise ValueError("Foerster weights do not align with the epsilon export.")
    eps.index = pd.PeriodIndex(export["cal"].map(cal_to_quarter), freq="Q")
    # factor_io.gss drops the first quarter with missing output growth via
    # packr(e~dy). Its second quarter retains the zero initial innovation.
    sizes = weights.iloc[:, 1:].mean(axis=0)
    sizes.index = eps.columns
    if not sizes.gt(0).all():
        raise ValueError("Foerster mean IP weights must be positive.")
    return eps.iloc[1:], sizes


def load_foerster_epsilon_panel(
    epsilon_csv: Path,
    weights_csv: Path,
    columns_csv: Path,
    m: int,
) -> SourcePanel:
    if not epsilon_csv.exists():
        raise FileNotFoundError(f"Missing epsilon export: {epsilon_csv}")
    if not weights_csv.exists():
        raise FileNotFoundError(
            f"Missing Foerster L4 weights export: {weights_csv}. "
            "Run python -m applications.foerster.prepare first."
        )
    if not columns_csv.exists():
        raise FileNotFoundError(f"Missing epsilon column map: {columns_csv}")

    eps, sizes = load_foerster_shocks_and_weights(epsilon_csv, weights_csv)
    eps_cols = eps.columns
    periods = eps.index.astype(str)
    y = eps.T.set_axis(periods.rename("quarter"), axis=1)
    y.index.name = "epsilon_column"

    meta = pd.read_csv(columns_csv)
    labels = (
        meta.set_index("epsilon_column")["description"]
        .astype(str)
        .str.strip()
        .reindex(eps_cols)
    )
    labels = labels.fillna(pd.Series(eps_cols, index=eps_cols))

    return SourcePanel(
        name=f"foerster_epsilon_M{m}",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=periods[0],
        last_date=periods[-1],
        frequency="quarterly",
        size_name="Mean IP relative-importance weight over 1972–2007",
        value_name="Foerster-Sarte-Watson benchmark epsilon innovation",
        coverage=1.0,
        notes=(
            "Benchmark 1998-NAICS L4 epsilon export from the localized Foerster et al. "
            "replication package; retains the 1972Q2 zero initial innovation "
            f"and the authors' mean IP weights; grouped with M={m}."
        ),
        weight_timing="sample_mean",
        weight_date="1972Q1–2007Q4",
    )


def read_network_matrix(path: Path, n: int, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Foerster network matrix export: {path}. "
            "Run python -m applications.foerster.prepare first."
        )
    mat = pd.read_csv(path, header=None).to_numpy(float)
    if mat.shape != (n, n):
        raise ValueError(f"{name} matrix has shape {mat.shape}, expected {(n, n)}.")
    return mat


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
    "bootstrap_mean",
    "bootstrap_p05",
    "bootstrap_p50",
    "bootstrap_p95",
    "moving_block_bootstrap_mean",
    "moving_block_bootstrap_p05",
    "moving_block_bootstrap_p50",
    "moving_block_bootstrap_p95",
    "gradient_bootstrap_p05",
    "gradient_bootstrap_p95",
    "moving_block_gradient_bootstrap_p05",
    "moving_block_gradient_bootstrap_p95",
    "subsample_p05",
    "subsample_p95",
    "subsample_ucl95",
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


def network_share_for_indices(
    x_grouped: np.ndarray,
    eps_obs: np.ndarray,
    idx: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    solver: str,
) -> dict[str, object]:
    sigma_g = sample_cov(x_grouped[idx])
    sigma_eps = sample_cov(eps_obs[idx])
    out = network_share_from_covariances(sigma_g, sigma_eps, network_v, weight_scale, solver)
    out.pop("Y", None)
    return out


def resample_network_mgp(
    x_grouped: np.ndarray,
    eps_obs: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    b: int,
    solver: str,
    rng: np.random.Generator,
    mode: str,
    progress_every: int,
    block_length: int | None = None,
) -> pd.DataFrame:
    if b <= 0:
        return pd.DataFrame()
    t = x_grouped.shape[0]
    rows: list[dict[str, object]] = []
    print(
        f"[{mode}] starting {b} network-exposure resamples"
        + (f", block_length={block_length}" if mode == "moving_block_bootstrap" else ""),
        flush=True,
    )
    for j in range(1, b + 1):
        if mode == "bootstrap":
            idx = rng.choice(t, size=t, replace=True)
        elif mode == "moving_block_bootstrap":
            idx = moving_block_indices(t, int(block_length), rng)
        else:
            raise ValueError(f"Unknown network resampling mode {mode!r}")
        row = network_share_for_indices(x_grouped, eps_obs, idx, network_v, weight_scale, solver)
        row.update({"mode": mode, "draw": j, "block_length": block_length if mode == "moving_block_bootstrap" else np.nan})
        rows.append(row)
        if j == 1 or j % progress_every == 0 or j == b:
            print(
                f"[{mode}] draw {j:>4}/{b}: "
                f"mean={np.mean([r['mgp_share'] for r in rows]):.4f}, "
                f"last={row['mgp_share']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


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
    y_dual = np.asarray(mgp["Y"], dtype=float)
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

    def pass_draws(idx_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(idx_list)
        g_vals = np.empty(n)
        h_g = np.empty((n, m, m))
        den_deriv = np.empty(n)
        for k, idx in enumerate(idx_list):
            sig_g_b = sample_cov(x_grouped[idx])
            sig_eps_b = sample_cov(eps_obs[idx])
            draw = network_share_from_covariances(sig_g_b, sig_eps_b, network_v, weight_scale, solver)
            g_vals[k] = float(draw["mgp_share_uncapped"])
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
        return g_vals, d_vals

    idx_iid = [rng.choice(t, size=t, replace=True) for _ in range(b)]
    idx_blk = [moving_block_indices(t, block_length, rng) for _ in range(b)]
    g_iid, d_iid = pass_draws(idx_iid)
    g_blk, d_blk = pass_draws(idx_blk)

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


def subsample_network_mgp(
    x_grouped: np.ndarray,
    eps_obs: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    ell: int,
    n_sub: int,
    solver: str,
    rng: np.random.Generator,
    progress_every: int,
    g_hat: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    t = x_grouped.shape[0]
    idx_list = subsample_indices(t, ell, n_sub, rng, overlapping=True)
    rows = []
    print(f"[network_subsample] ell={ell}, n={len(idx_list)} (overlapping blocks)", flush=True)
    for j, idx in enumerate(idx_list, 1):
        g_b = float(
            network_share_for_indices(
                x_grouped, eps_obs, idx, network_v, weight_scale, solver
            )["mgp_share_uncapped"]
        )
        rows.append({"subsample": j, "ell": ell, "g_sub": g_b, "L": float(np.sqrt(ell) * (g_b - g_hat))})
        if j == 1 or j % progress_every == 0 or j == len(idx_list):
            print(f"[network_subsample] {j:>4}/{len(idx_list)}", flush=True)
    draws = pd.DataFrame(rows)
    return draws, subsampling_summary(draws, g_hat, t)


def network_bias_terms(
    sigma_grouped: np.ndarray,
    sigma_eps: np.ndarray,
    grouping_matrix: np.ndarray,
    network_v: np.ndarray,
    weight_scale: float,
    t: int,
    solver: str,
    fragile_gap_ratio: float = 5.0,
    closed_disagreement_rel: float = 0.10,
) -> dict[str, object]:
    reg = regularity_report(sigma_grouped, solver)
    diag = B.closed_form_diagnostics(sigma_grouped, solver)
    rank_gap = float(reg["rank_gap_ratio"])
    gap_is_fragile = (not np.isfinite(rank_gap) and not np.isinf(rank_gap)) or (
        np.isfinite(rank_gap) and rank_gap < fragile_gap_ratio
    )
    structural_fragile = bool(
        (not reg["regular"])
        or gap_is_fragile
        or (not diag["strict_complementarity"])
        or (diag["sigma_rank"] < sigma_grouped.shape[0])
    )

    basis = B.sym_basis(sigma_grouped.shape[0])
    h = 1e-4 * np.linalg.norm(sigma_grouped)
    hg_cf = B.closed_value_hessian(sigma_grouped, basis, solver)
    hg_fd = B.operator_to_value_hessian([B.dYstar_fd(sigma_grouped, e, solver, h) for e in basis], basis)
    try:
        hg_ad = B.operator_to_value_hessian(B.dYstar_ad_all(sigma_grouped, basis), basis)
    except Exception:
        hg_ad = np.full_like(hg_cf, np.nan)

    y_dual = B.dual_optimizer(sigma_grouped, solver)
    gamma_norm = float(np.sum(y_dual * sigma_grouped))
    gamma_level = weight_scale * gamma_norm
    denom = network_denominator(sigma_eps, network_v)
    grad_num_full = weight_scale * grouping_matrix.T @ y_dual @ grouping_matrix
    omega_num_den = 2.0 * float(np.trace(grad_num_full @ sigma_eps @ network_v @ sigma_eps))
    omega_den_den = 2.0 * float(np.trace(network_v @ sigma_eps @ network_v @ sigma_eps))

    def quotient_bias(h_gamma: np.ndarray) -> float:
        if np.isnan(h_gamma).any():
            return np.nan
        gmat = B.gamma_matrix(sigma_grouped, basis)
        contr_num = weight_scale * float(np.sum(gmat * h_gamma))
        contr_share = (
            contr_num / denom
            - 2.0 * omega_num_den / denom**2
            + 2.0 * gamma_level * omega_den_den / denom**3
        )
        return contr_share / (2.0 * t)

    b_cf = quotient_bias(hg_cf)
    b_fd = quotient_bias(hg_fd)
    b_ad = quotient_bias(hg_ad)
    closed_fd_rel = B.relative_gap(b_cf, b_fd)
    closed_ad_rel = B.relative_gap(b_cf, b_ad)
    hessian_disagreement = (
        (np.isfinite(closed_ad_rel) and closed_ad_rel > closed_disagreement_rel)
        or (not np.isfinite(closed_ad_rel) and np.isfinite(closed_fd_rel) and closed_fd_rel > closed_disagreement_rel)
    )
    return {
        "bias_closed": b_cf,
        "bias_fd": b_fd,
        "bias_ad": b_ad,
        "bias_closed_fragile": bool(structural_fragile or hessian_disagreement),
        "bias_rank_gap_ratio": round(rank_gap, 1),
        "bias_strict_complementarity": diag["strict_complementarity"],
    }


def run_network_exposure_panel(
    panel: SourcePanel,
    original_weight_sizes: pd.Series,
    network_sizes: pd.Series,
    network_v: np.ndarray,
    m: int,
    block_length: int,
    bootstrap_b: int,
    block_bootstrap_b: int,
    subsample_n: int,
    solver: str,
    rng: np.random.Generator,
    gradient_rng: np.random.Generator,
    out_root: Path,
    progress_every: int,
    run_profile: str,
) -> dict[str, object]:
    y, sizes = align_sizes_to_panel(panel.y, network_sizes)
    order_pos = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    network_v_sorted = network_v[np.ix_(order_pos, order_pos)]

    shares = (sizes / sizes.sum()).to_numpy(dtype=float)
    weight_scale = float(sizes.sum() ** 2)
    y_np = y.to_numpy(dtype=float)
    eps_obs = y_np.T
    x_grouped, buckets, loads, grouping_matrix = build_lpt_grouped_data(y_np, shares, m)
    observed = network_observed_stats(x_grouped, eps_obs, network_v_sorted, weight_scale, solver)
    sigma_grouped = sample_cov(x_grouped)
    sigma_eps = sample_cov(eps_obs)
    regularity = regularity_report(sigma_grouped, solver)
    hhi = float(np.sum(shares**2))

    # Distribution-free J-test factor rank on the grouped panel, exactly as for the
    # M=10 columns of Table 1 (smallest non-rejected rank; ">=" if all rejected).
    try:
        _, rank_summary = factor_rank_tests(x_grouped, alpha=0.05, progress_label=f"foerster_network_M{m}")
        factor_lb_j = rank_summary.get("factor_lb_J", np.nan)
        regular_certified_j = bool(rank_summary.get("regular_certified_J", False))
    except Exception as exc:
        print(f"[foerster_epsilon_network_M{m}] J-test skipped: {type(exc).__name__}: {exc}", flush=True)
        factor_lb_j, regular_certified_j = np.nan, False

    print(
        f"\n[foerster_epsilon_network_M{m}] N={y.shape[0]:,}, T={y.shape[1]:,}, M={m}, "
        f"start={panel.start_date}, last={panel.last_date}, block={block_length}",
        flush=True,
    )
    print(
        f"[foerster_epsilon_network_M{m}] observed share={observed['mgp_share']:.4f}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}, exposure HHI={hhi:.6g}, "
        f"denom={observed['aggregate_variance']:.6g}",
        flush=True,
    )

    bootstrap = resample_network_mgp(
        x_grouped, eps_obs, network_v_sorted, weight_scale,
        b=bootstrap_b, solver=solver, rng=rng, mode="bootstrap",
        progress_every=progress_every,
    )
    block_bootstrap = resample_network_mgp(
        x_grouped, eps_obs, network_v_sorted, weight_scale,
        b=block_bootstrap_b, solver=solver, rng=rng, mode="moving_block_bootstrap",
        progress_every=progress_every, block_length=block_length,
    )
    gradient = network_gradient_family(
        x_grouped, eps_obs, network_v_sorted, weight_scale,
        b=min(bootstrap_b, block_bootstrap_b), block_length=block_length,
        rng=gradient_rng, solver=solver,
    )

    ell = default_subsample_length(y.shape[1], m)
    if subsample_n > 0 and ell < y.shape[1]:
        subsamples, sub_ci = subsample_network_mgp(
            x_grouped, eps_obs, network_v_sorted, weight_scale,
            ell=ell, n_sub=subsample_n, solver=solver, rng=rng,
            progress_every=progress_every, g_hat=observed["mgp_share_uncapped"],
        )
    else:
        subsamples = pd.DataFrame()
        sub_ci = {}

    out_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(f"foerster_epsilon_network_M{m}")
    firm_diag, bucket_diag = lpt_diagnostics(y.index, shares, panel.labels, buckets, loads)
    mean_weights = original_weight_sizes.reindex(y.index).astype(float)
    mean_weights = mean_weights / mean_weights.sum()
    firm_diag["ip_weight_share"] = mean_weights.to_numpy()
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat([bootstrap, block_bootstrap], ignore_index=True).to_pickle(
        RUN_OUTPUT_CACHE / f"{prefix}_draws_resampling.pkl"
    )
    if not subsamples.empty:
        subsamples.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_draws_subsampling.pkl")
    firm_diag.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_units.pkl")
    bucket_diag.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_buckets.pkl")
    try:
        plot_lpt_squared_bucket_sizes(
            firm_diag.rename(columns={"unit_id": "gvkey"}),
            bucket_diag,
            out_root / f"lpt_{prefix}",
            ylabel=r"Bucket squared-influence load: $\sum_{i\in B_m} V_i^2$",
        )
    except Exception as exc:
        print(f"[foerster_epsilon_network_M{m}] LPT plot skipped: {type(exc).__name__}: {exc}", flush=True)

    boot_vals = bootstrap["mgp_share_uncapped"] if not bootstrap.empty else pd.Series(dtype=float)
    block_vals = block_bootstrap["mgp_share_uncapped"] if not block_bootstrap.empty else pd.Series(dtype=float)
    bias_summary = network_bias_terms(
        sigma_grouped, sigma_eps, grouping_matrix, network_v_sorted, weight_scale,
        t=y.shape[1], solver=solver,
    )
    summary = {
        "dataset": f"foerster_epsilon_network_M{m}",
        "table_column": f"Network exposure M={m}",
        "table_outcome": "network-propagated epsilon bound",
        "run_profile": run_profile,
        "frequency": panel.frequency,
        "value_name": "Foerster epsilon innovations with FSW network exposure denominator",
        "size_name": "FSW network exposure L_i",
        "weight_timing": panel.weight_timing,
        "weight_date": str(panel.weight_date),
        "start_date": str(panel.start_date),
        "last_date": str(panel.last_date),
        "N": y.shape[0],
        "T": y.shape[1],
        "M": m,
        "bootstrap_b": bootstrap_b,
        "block_bootstrap_b": block_bootstrap_b,
        "requested_subsample_n": subsample_n,
        "block_length": block_length,
        "coverage": panel.coverage,
        "HHI": hhi,
        "top1_share": float(shares[0]),
        "top10_share": float(shares[: min(10, len(shares))].sum()),
        "top50_share": float(shares[: min(50, len(shares))].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        **{k: v for k, v in observed.items() if k not in {"solver", "status", "dual_min_eig", "dual_min_diag"}},
        "ledermann_kbar": regularity["ledermann_kbar"],
        "common_rank": regularity["common_rank"],
        "factor_lb_J": factor_lb_j,
        "regular_certified_J": regular_certified_j,
        "regular": regularity["regular"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
        "bootstrap_mean": float(boot_vals.mean()) if not boot_vals.empty else np.nan,
        "bootstrap_p05": reported_dynamic_share(boot_vals.quantile(0.05)) if not boot_vals.empty else np.nan,
        "bootstrap_p50": reported_dynamic_share(boot_vals.quantile(0.50)) if not boot_vals.empty else np.nan,
        "bootstrap_p95": reported_dynamic_share(boot_vals.quantile(0.95)) if not boot_vals.empty else np.nan,
        "moving_block_bootstrap_mean": float(block_vals.mean()) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p05": reported_dynamic_share(block_vals.quantile(0.05)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p50": reported_dynamic_share(block_vals.quantile(0.50)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p95": reported_dynamic_share(block_vals.quantile(0.95)) if not block_vals.empty else np.nan,
        **gradient,
        **sub_ci,
        **bias_summary,
        "network_weight_sum": float(sizes.sum()),
        "network_denominator_variance": float(observed["aggregate_variance"]),
        "notes": (
            "FSW observable-space upper bound: numerator uses network exposure "
            "L_i=sqrt(sum_h (w'H_h e_i)^2), LPT grouped on L_i^2; denominator is "
            "model-implied aggregate IP variance <V,Sigma_epsilon>."
        ),
    }
    cap_dynamic_share_report(summary)
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-csv", type=Path, default=DEFAULT_EPSILON_CSV)
    parser.add_argument("--weights-csv", type=Path, default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--columns-csv", type=Path, default=DEFAULT_COLUMNS_CSV)
    parser.add_argument("--pii-a-csv", type=Path, default=DEFAULT_PII_A_CSV)
    parser.add_argument("--xi-csv", type=Path, default=DEFAULT_XI_CSV)
    parser.add_argument("--cap-rho-csv", type=Path, default=DEFAULT_CAP_RHO_CSV)
    parser.add_argument("--M", type=int, default=10,
                        help="LPT groups; default 10.")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--block-bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--subsample-n", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--solver", default="auto")
    parser.add_argument(
        "--run-profile",
        choices=["medium", "full"],
        default="full",
        help="medium skips subsampling; full runs all standard diagnostics.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if args.M is not None and args.M <= 0:
        raise ValueError("M must be positive.")
    if args.run_profile == "medium":
        args.subsample_n = 0

    solver = choose_solver(args.solver)
    panel = load_foerster_epsilon_panel(args.epsilon_csv, args.weights_csv, args.columns_csv, args.M)
    block_length = analysis_block_length(panel.y.shape[1])
    n = panel.y.shape[0]
    pii_a = read_network_matrix(args.pii_a_csv, n, "Pi_a")
    xi = read_network_matrix(args.xi_csv, n, "Xi")
    cap_rho = read_network_matrix(args.cap_rho_csv, n, "cap_rho")
    network_v = network_variance_matrix(panel.sizes.to_numpy(float), pii_a, xi, cap_rho)
    network_sizes = pd.Series(np.sqrt(np.maximum(np.diag(network_v), 0.0)), index=panel.y.index)
    network_summary = run_network_exposure_panel(
        panel=panel,
        original_weight_sizes=panel.sizes,
        network_sizes=network_sizes,
        network_v=network_v,
        m=args.M,
        block_length=block_length,
        bootstrap_b=args.bootstrap_b,
        block_bootstrap_b=args.block_bootstrap_b,
        subsample_n=args.subsample_n,
        solver=solver,
        rng=make_panel_rng(args.seed, f"{panel.name}_network"),
        gradient_rng=make_panel_rng(args.seed, f"{panel.name}_network_gradient"),
        out_root=args.out_dir,
        progress_every=args.progress_every,
        run_profile=args.run_profile,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A single granular share: the network-exposure column (primitives regrouped by
    # total influence, denominator the model-implied aggregate variance).
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([network_summary]).to_pickle(RUN_OUTPUT_CACHE / "foerster_summary_all.pkl")

    display = pd.DataFrame([network_summary])
    cols = [
        "dataset",
        "N",
        "T",
        "M",
        "HHI",
        "mgp_share",
        "bootstrap_p95",
        "moving_block_bootstrap_p95",
        "subsample_p95",
    ]
    cols = [col for col in cols if col in display.columns]
    print("\nFoerster epsilon summary:", flush=True)
    print(display[cols].round(6).to_string(index=False), flush=True)
    print(f"Wrote {RUN_OUTPUT_CACHE / 'foerster_summary_all.pkl'}", flush=True)


if __name__ == "__main__":
    main()
