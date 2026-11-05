#!/usr/bin/env python3
"""Shapiro--Ten Berge (2000) O(1/T) bias of the grouped-MGP share, three ways.

The grouped share is gamma_bar(Sigma) = Gamma_bar(Sigma) / (1' Sigma 1), where
Gamma_bar(Sigma) = max_{D>=0 diag} tr(D) s.t. Sigma - D >= 0 is the grouped GLB
numerator (= -v in Shapiro's notation; our share is 1 - glb). In the regular case
(unique dual, reduced rank r >= Ledermann bound, interior D, strict
complementarity) the plug-in bias is

    E[gamma_hat] - gamma_bar = (1/2T) <Gamma, Hess gamma_bar(Sigma)> + o(1/T),     (ST00 eq 4.4)

with Gamma the asymptotic covariance of sqrt(T) vec(Sigma_hat). Under normality
Gamma[V] = 2 Sigma V Sigma.

Everything reduces to one primitive: the directional derivative of the dual
optimizer, dYstar[E] = d/dt Ystar(Sigma + tE).  The gradient of the numerator is
grad Gamma_bar = Ystar (the dual matrix), so Hess Gamma_bar[E] = dYstar[E]. Three
estimators of dYstar:

  closed : Shapiro's second-order sensitivity QP (ST00 eq 4.7) solved as an
           equality-constrained linear (KKT) system on the active manifold.
  fd     : central finite difference of the dual solve.
  ad     : cvxpy/diffcp implicit differentiation of the KKT system.

The quotient rule to the share and the Gaussian contraction are shared, so the
three differ only in dYstar and should coincide in the regular case (and are
meaningless in the non-regular case, where dYstar does not exist).
"""

from __future__ import annotations

import numpy as np

# Import mgp_core before cvxpy: it sets MOSEKLM_LICENSE_FILE, which the MOSEK C
# library reads once when cvxpy/mosek first initializes at import.
from mgp_core import numerical_rank, regularity_report, solve_grouped_mgp_dual, solve_primal_mtfa

import cvxpy as cp


def sym_basis(m: int) -> list[np.ndarray]:
    """Orthonormal (Frobenius) basis of the m x m symmetric matrices."""
    basis = []
    for i in range(m):
        e = np.zeros((m, m))
        e[i, i] = 1.0
        basis.append(e)
    for i in range(m):
        for j in range(i + 1, m):
            e = np.zeros((m, m))
            e[i, j] = e[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(e)
    return basis


def dual_optimizer(sigma: np.ndarray, solver: str) -> np.ndarray:
    """Ystar = grad Gamma_bar(Sigma), the optimizer of min <Sigma,Y> s.t. Y>=0, diag Y>=1."""
    return np.asarray(solve_grouped_mgp_dual(sigma, solver)["Y"], dtype=float)


def normalize_sigma(sigma: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetrize and scale a covariance to unit Frobenius norm.

    The optimizer Ystar and the share bias are homogeneous of degree zero in
    Sigma, while the level Hessian is homogeneous of degree -1. Normalizing
    before differentiating keeps the tiny empirical grouped covariances away
    from solver tolerances without changing the reported bias.
    """
    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    scale = float(np.linalg.norm(sigma))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Sigma must have positive finite Frobenius norm.")
    return sigma / scale, scale


# ---------------------------------------------------------------------------
# dYstar[E] : three estimators
# ---------------------------------------------------------------------------
def dYstar_fd(sigma: np.ndarray, e: np.ndarray, solver: str, h: float) -> np.ndarray:
    yp = dual_optimizer(sigma + h * e, solver)
    ym = dual_optimizer(sigma - h * e, solver)
    return (yp - ym) / (2.0 * h)


def make_ad_problem(m: int):
    sig = cp.Parameter((m, m), symmetric=True)
    y = cp.Variable((m, m), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(sig, y))), [y >> 0, cp.diag(y) >= 1])
    return prob, sig, y


def dYstar_ad_all(sigma: np.ndarray, basis: list[np.ndarray], ad_solver=cp.SCS) -> list[np.ndarray]:
    """Forward-mode derivative of Ystar for every basis direction (one solve)."""
    sigma_unit, scale = normalize_sigma(sigma)
    m = sigma_unit.shape[0]
    prob, sig, y = make_ad_problem(m)
    sig.value = sigma_unit
    prob.solve(requires_grad=True, solver=ad_solver, eps=1e-10, max_iters=200000)
    out = []
    for e in basis:
        sig.delta = e / scale
        prob.derivative()
        out.append(np.asarray(y.delta, dtype=float))
    return out


def _active_manifold(sigma: np.ndarray, solver: str, rel_tol: float = 1e-6):
    """F (m x (m-r) with Ystar=FF') and Phi=(Sigma-D)^+ at the optimum."""
    ystar = dual_optimizer(sigma, solver)
    c, _ = solve_primal_mtfa(sigma, solver)  # common part C = Sigma - D, rank r
    w_eig, w_vec = np.linalg.eigh(0.5 * (ystar + ystar.T))
    keep = w_eig > rel_tol * max(w_eig.max(), 1e-300)
    f = w_vec[:, keep] * np.sqrt(w_eig[keep])          # F F' = Ystar
    phi = np.linalg.pinv(0.5 * (c + c.T), rcond=rel_tol)
    return ystar, f, phi


def _vech_indices(q: int):
    return [(a, b) for a in range(q) for b in range(a, q)]


def _closed_value_hessian_unit(sigma: np.ndarray, basis: list[np.ndarray], solver: str) -> np.ndarray:
    """Level value-Hessian H^Gamma_ab = d^2 Gamma_bar / d(coord a) d(coord b) (orthonormal basis).

    From Shapiro--Ten Berge's second-order QP (eq 4.7): for direction E,
        theta(E) = min_h tr[Ystar (diag h + E) Phi (diag h + E)]  s.t. F'(diag h + E)F = 0,
    and Hess v = 2 theta (a quadratic form), so Hess Gamma_bar = -2 theta (since Gamma_bar = -v).
    h*(E) is linear in E (equality-constrained convex QP), solved here by its KKT system;
    G(E) = diag(h*(E)) + E, and H^Gamma_ab = -[tr(Ystar G_a Phi G_b) + tr(Ystar G_b Phi G_a)].
    """
    m = sigma.shape[0]
    ystar, f, phi = _active_manifold(sigma, solver)
    q = f.shape[1]
    P = ystar * phi                                   # Hadamard, PSD m x m
    # constraint operator A h = vech(F' diag(h) F): rows over a<=b, cols over i
    vidx = _vech_indices(q)
    A = np.array([[f[i, a] * f[i, b] for i in range(m)] for (a, b) in vidx])  # (q(q+1)/2) x m
    nc = A.shape[0]
    kkt = np.block([[2.0 * P, A.T], [A, np.zeros((nc, nc))]])
    kkt_pinv = np.linalg.pinv(kkt)

    g_mats = []
    for e in basis:
        b_lin = np.diag(phi @ e @ ystar)              # 2*this is the linear-term gradient
        rhs_top = -2.0 * b_lin
        fef = f.T @ e @ f
        rhs_bot = -np.array([fef[a, b] for (a, b) in vidx])
        sol = kkt_pinv @ np.concatenate([rhs_top, rhs_bot])
        h = sol[:m]
        g_mats.append(np.diag(h) + e)

    g_ten = np.stack(g_mats)                          # (d, m, m)
    m_ten = np.einsum("ij,ajk,kl->ail", ystar, g_ten, phi)   # Ystar G_a Phi
    term1 = np.einsum("aij,bji->ab", m_ten, g_ten)    # tr(Ystar G_a Phi G_b)
    h_gamma = -(term1 + term1.T)
    return h_gamma


def closed_value_hessian(sigma: np.ndarray, basis: list[np.ndarray], solver: str) -> np.ndarray:
    """Closed-form level Hessian, computed at unit scale for conditioning."""
    sigma_unit, scale = normalize_sigma(sigma)
    return _closed_value_hessian_unit(sigma_unit, basis, solver) / scale


# ---------------------------------------------------------------------------
# Contraction: level value-Hessian -> share bias under Gaussian Gamma
# ---------------------------------------------------------------------------
def operator_to_value_hessian(dYstar_list: list[np.ndarray], basis: list[np.ndarray]) -> np.ndarray:
    """H^Gamma_ab = <E_a, dYstar[E_b]>  (grad Gamma_bar = Ystar, so this is the level Hessian)."""
    e_ten = np.stack(basis)
    dy_ten = np.stack(dYstar_list)
    h = np.einsum("aij,bij->ab", e_ten, dy_ten)
    return 0.5 * (h + h.T)


def gamma_matrix(sigma: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Gaussian Gamma in the orthonormal basis: Gamma_ab = 2 tr(E_a Sigma E_b Sigma)."""
    e_ten = np.stack(basis)
    se = np.einsum("ij,ajk,kl->ail", sigma, e_ten, sigma)   # Sigma E_a Sigma
    return 2.0 * np.einsum("aij,bij->ab", e_ten, se)


def share_bias_from_value_hessian(h_gamma: np.ndarray, sigma: np.ndarray,
                                  basis: list[np.ndarray], t: int, solver: str) -> float:
    """(1/2T) <Gamma, Hess gamma_bar>, with the quotient rule gamma_bar = Gamma_bar / (1'Sigma1)."""
    one = np.ones(sigma.shape[0])
    ystar = dual_optimizer(sigma, solver)
    a = float(one @ sigma @ one)
    gamma_bar = float(np.sum(ystar * sigma))
    gmat = gamma_matrix(sigma, basis)
    contr_level = float(np.sum(gmat * h_gamma))            # <Gamma, Hess Gamma_bar>
    cross = 4.0 * float(one @ sigma @ ystar @ sigma @ one)  # <Gamma, Ystar (x) 11'> = 2*(1'S Y S 1); times 2
    contr_share = contr_level / a - cross / a**2 + 4.0 * gamma_bar / a
    return contr_share / (2.0 * t)


def mgp_share_value(sigma: np.ndarray, solver: str) -> float:
    ystar = dual_optimizer(sigma, solver)
    one = np.ones(sigma.shape[0])
    return float(np.sum(ystar * sigma) / (one @ sigma @ one))


def parametric_gaussian_bias(sigma: np.ndarray, t: int, b: int, solver: str,
                             rng: np.random.Generator) -> tuple[float, float]:
    """Cross-check: mean of share over Gaussian resamples minus the plug-in (+ MC se)."""
    sigma = 0.5 * (sigma + sigma.T)
    m = sigma.shape[0]
    plug = mgp_share_value(sigma, solver)
    vals = [mgp_share_value(np.cov(rng.multivariate_normal(np.zeros(m), sigma, size=t),
                                   rowvar=False, ddof=1), solver) for _ in range(b)]
    return float(np.mean(vals) - plug), float(np.std(vals, ddof=1) / np.sqrt(b))


def bias_all_methods(sigma: np.ndarray, t: int, solver: str, fd_h: float = 1e-4) -> dict:
    sigma = 0.5 * (sigma + sigma.T)
    basis = sym_basis(sigma.shape[0])
    h = fd_h * np.linalg.norm(sigma)

    hg_fd = operator_to_value_hessian([dYstar_fd(sigma, e, solver, h) for e in basis], basis)
    hg_ad = operator_to_value_hessian(dYstar_ad_all(sigma, basis), basis)
    hg_cf = closed_value_hessian(sigma, basis, solver)

    return {
        "bias_closed": share_bias_from_value_hessian(hg_cf, sigma, basis, t, solver),
        "bias_fd": share_bias_from_value_hessian(hg_fd, sigma, basis, t, solver),
        "bias_ad": share_bias_from_value_hessian(hg_ad, sigma, basis, t, solver),
    }


def closed_form_diagnostics(sigma: np.ndarray, solver: str, rel_tol: float = 1e-3) -> dict[str, object]:
    m = sigma.shape[0]
    sigma_eig = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))[::-1]
    sigma_rank = numerical_rank(sigma_eig, rel_tol)
    c, _ = solve_primal_mtfa(sigma, solver)
    common_eig = np.linalg.eigvalsh(0.5 * (c + c.T))[::-1]
    common_rank = numerical_rank(common_eig, rel_tol)
    y = np.asarray(solve_grouped_mgp_dual(sigma, solver)["Y"], dtype=float)
    dual_eig = np.linalg.eigvalsh(0.5 * (y + y.T))[::-1]
    dual_rank = numerical_rank(dual_eig, rel_tol)
    strict_complementarity = (common_rank + dual_rank) == m
    return {
        "sigma_rank": int(sigma_rank),
        "dual_rank": int(dual_rank),
        "strict_complementarity": bool(strict_complementarity),
    }


def relative_gap(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(abs(a - b) / max(abs(a), abs(b), 1e-12))


def grouped_bias_terms(
    sigma_grouped: np.ndarray,
    t: int,
    solver: str,
    fragile_gap_ratio: float = 5.0,
    closed_disagreement_rel: float = 0.10,
) -> dict[str, object]:
    """Regular O(1/T) curvature bias of the plain grouped share, three ways.

    The single-panel counterpart of ``run_foerster_epsilon.network_bias_terms``,
    for columns whose denominator is the grouped aggregate variance itself.
    Returns NaN biases on a non-regular column, where the leading term is
    O(1/sqrt T) instead.
    """
    reg = regularity_report(sigma_grouped, solver)
    diag = closed_form_diagnostics(sigma_grouped, solver)
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
    if not reg["regular"]:
        return {
            "bias_closed": np.nan,
            "bias_fd": np.nan,
            "bias_ad": np.nan,
            "bias_closed_fragile": True,
            "bias_rank_gap_ratio": round(rank_gap, 1),
            "bias_strict_complementarity": diag["strict_complementarity"],
        }

    basis = sym_basis(sigma_grouped.shape[0])
    h = 1e-4 * np.linalg.norm(sigma_grouped)
    hg_cf = closed_value_hessian(sigma_grouped, basis, solver)
    hg_fd = operator_to_value_hessian(
        [dYstar_fd(sigma_grouped, e, solver, h) for e in basis], basis
    )
    try:
        hg_ad = operator_to_value_hessian(dYstar_ad_all(sigma_grouped, basis), basis)
    except Exception:
        hg_ad = np.full_like(hg_cf, np.nan)
    methods = {
        name: share_bias_from_value_hessian(hessian, sigma_grouped, basis, t, solver)
        for name, hessian in (("bias_closed", hg_cf), ("bias_fd", hg_fd), ("bias_ad", hg_ad))
    }
    closed_fd_rel = relative_gap(methods["bias_closed"], methods["bias_fd"])
    closed_ad_rel = relative_gap(methods["bias_closed"], methods["bias_ad"])
    hessian_disagreement = (
        (np.isfinite(closed_ad_rel) and closed_ad_rel > closed_disagreement_rel)
        or (
            not np.isfinite(closed_ad_rel)
            and np.isfinite(closed_fd_rel)
            and closed_fd_rel > closed_disagreement_rel
        )
    )
    return {
        **methods,
        "bias_closed_fragile": bool(structural_fragile or hessian_disagreement),
        "bias_rank_gap_ratio": round(rank_gap, 1),
        "bias_strict_complementarity": diag["strict_complementarity"],
    }
