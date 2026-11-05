#!/usr/bin/env python3
"""Estimate directional derivatives over near-optimal dual solutions.

For each covariance perturbation H, minimize <Y, H> over feasible duals
whose objective is at most (1 + delta_T) times the estimated optimum.

Provides compiled and NumPy implementations with shared general-solver
retries. Unresolved directions return NaN and a failure flag.
See NUMERICAL_METHODS.md.
"""

from __future__ import annotations

import numpy as np

import mgp_kernels
from fast_mgp import (
    CENTER_TOL,
    DIVERGENCE_BOUND,
    FINAL_TOL,
    HESSIAN_JITTER,
    MAX_ITER,
    MU_SHRINK,
    _symmetrize,
)


SOLVER_POLICY_VERSION = "validated-nearopt-v1"


def delta_rate(t: int, scale: float = 1.0) -> float:
    """The tolerance sequence sqrt(log T / T), optionally rescaled by a constant."""
    return float(scale) * float(np.sqrt(np.log(max(t, 2)) / t))


def require_solved(values, solved, context="Near-optimal derivative"):
    """Reject an incomplete batch before it can enter inference or a checkpoint."""
    values = np.asarray(values, dtype=float)
    failed = ~np.asarray(solved, dtype=bool) | ~np.isfinite(values)
    if failed.any():
        rows = np.flatnonzero(failed)
        raise RuntimeError(
            f"{context}: {len(rows)}/{values.size} solves unsuccessful or nonfinite "
            f"(indices {rows[:10].tolist()}); inference is unavailable."
        )
    return values


def _positive_scale(sigma):
    """Check the original covariance before using its eigenvalue as a divisor."""
    if not np.isfinite(sigma).all():
        raise ValueError("Sigma must be finite and positive definite for the near-optimal set.")
    minimum = np.linalg.eigvalsh(sigma)[..., 0]
    if np.any(minimum <= 0):
        raise ValueError("Sigma must be positive definite for the near-optimal set.")
    return minimum


def _feasible_start(direction: np.ndarray, sigma: np.ndarray, lam_min):
    """(nu, t) with H + t Sigma - Diag(nu) > 0, nu > 0, t > 0.

    Taking t past ||H||_2 / lambda_min(Sigma) makes the pencil positive definite with
    room to spare, and nu is then put a little below the smallest eigenvalue.
    """
    norm = np.linalg.norm(direction, ord=2, axis=(1, 2))
    t0 = 2.0 * norm / np.atleast_1d(lam_min) + 1.0
    g = direction + t0[:, None, None] * sigma
    floor = np.linalg.eigvalsh(g)[:, 0]
    nu0 = np.maximum(0.25 * floor, 1e-8)
    return np.repeat(nu0[:, None], direction.shape[1], axis=1), t0


def near_optimal_derivative(
    directions: np.ndarray,
    sigma: np.ndarray,
    cap,
    owner: np.ndarray | None = None,
    *,
    mu_end: float = 1e-8,
    shrink: float = MU_SHRINK,
    center_tol: float = CENTER_TOL,
    final_tol: float = FINAL_TOL,
    max_iter: int = MAX_ITER,
    backstop: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate directional derivatives, using the NumPy solver without Numba.

    ``sigma`` is either one (M,M) covariance shared by every direction, or an (R,M,M)
    stack with ``owner`` saying which one each direction belongs to; ``cap`` matches.
    With ``backstop=True``, unsuccessful solves from either implementation are
    retried with general SDP solvers on the same directions. Unresolved values
    are NaN; callers must check the returned flags before computing inference.
    """
    directions = _symmetrize(np.asarray(directions, dtype=float))
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2:
        sigma = sigma[None]
    sigma = np.ascontiguousarray(_symmetrize(sigma))
    caps = np.ascontiguousarray(np.atleast_1d(np.asarray(cap, dtype=float)))
    k, m, _ = directions.shape
    if owner is None:
        owner = np.zeros(k, dtype=np.int64)
    owner = np.ascontiguousarray(owner, dtype=np.int64)
    if (sigma.shape[1:] != (m, m) or caps.shape != (len(sigma),)
            or owner.shape != (k,) or np.any((owner < 0) | (owner >= len(sigma)))):
        raise ValueError("Covariances, caps and direction owners have incompatible dimensions.")
    if not np.isfinite(directions).all() or not np.isfinite(caps).all():
        raise ValueError("Directions and objective caps must be finite.")
    sig_scale = _positive_scale(sigma)
    values = np.full(k, np.nan)
    ok = np.zeros(k, dtype=np.bool_)
    if k == 0:
        return values, ok
    if not mgp_kernels.NUMBA_AVAILABLE:
        for j in np.unique(owner):
            sel = owner == j
            try:
                values[sel], ok[sel] = near_optimal_derivative_numpy(
                    directions[sel], sigma[j], caps[j], mu_end=mu_end, shrink=shrink,
                    center_tol=center_tol, final_tol=final_tol, max_iter=max_iter,
                )
            except np.linalg.LinAlgError:
                # A failed factorization leaves this covariance group for the backstop.
                pass
    else:
        # Normalize directions by their Frobenius norms, and covariances/caps
        # together by their smallest eigenvalues. Restore direction scales below.
        scale = np.linalg.norm(directions, axis=(1, 2))
        scale = np.where(scale > 0, scale, 1.0)
        h = np.ascontiguousarray(directions / scale[:, None, None])
        scaled_sigma = np.ascontiguousarray(sigma / sig_scale[:, None, None])
        scaled_caps = np.ascontiguousarray(caps / sig_scale)
        lam_min = np.linalg.eigvalsh(scaled_sigma)[:, 0]
        nu, t = _feasible_start(h, scaled_sigma[owner], lam_min[owner])
        nu = np.ascontiguousarray(nu)
        t = np.ascontiguousarray(t)
        # Start where the barrier and objective are comparable in scale.
        mu = np.maximum((np.abs(nu.sum(axis=1)) + scaled_caps[owner] * t) / (m + 1), 1e-3)
        mgp_kernels.barrier_nearopt(
            h, scaled_sigma, scaled_caps, owner, mu.copy(), float(mu_end), float(shrink),
            float(center_tol), float(final_tol), int(max_iter), float(DIVERGENCE_BOUND),
            float(HESSIAN_JITTER), nu, t, mu, ok,
        )
        values = (nu.sum(axis=1) - scaled_caps[owner] * t) * scale
    ok &= np.isfinite(values)
    if backstop and not ok.all():
        _retry_directions(directions, sigma, caps, owner, values, ok, sig_scale)
    ok &= np.isfinite(values)
    return np.where(ok, values, np.nan), ok


def _retry_directions(directions, sigma, caps, owner, values, ok, sig_scale):
    """Retry only unresolved directions, without changing draws or the SDP."""
    from mosek_face import MosekError, NearOptimalSolver, licensed_mosek

    stuck = np.flatnonzero(~ok)
    use_mosek = licensed_mosek()
    for j in np.unique(owner[stuck]):
        rows = stuck[owner[stuck] == j]
        if use_mosek:
            try:
                with NearOptimalSolver(sigma[j], caps[j]) as solver:
                    got, bad = solver.values(directions[rows])
            except MosekError:
                pass
            else:
                accepted = ~bad & np.isfinite(got)
                values[rows[accepted]] = got[accepted]
                ok[rows[accepted]] = True
        rows = rows[~ok[rows]]
        if len(rows):
            got, bad = _open_solver(
                directions[rows], sigma[j] / sig_scale[j], caps[j] / sig_scale[j]
            )
            accepted = ~bad & np.isfinite(got)
            values[rows[accepted]] = got[accepted]
            ok[rows[accepted]] = True


def _open_solver(directions: np.ndarray, sigma: np.ndarray, cap: float):
    """Solve the same SDP with Clarabel if the MOSEK backstop is unavailable."""
    import cvxpy as cp

    y = cp.Variable(sigma.shape, symmetric=True)
    direction = cp.Parameter(sigma.shape, symmetric=True)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(direction, y))), [
        y >> 0, cp.diag(y) >= 1, cp.sum(cp.multiply(sigma, y)) <= cap,
    ])
    values = np.full(len(directions), np.nan)
    for k, matrix in enumerate(directions):
        scale = np.linalg.norm(matrix) or 1.0
        direction.value = matrix / scale
        try:
            problem.solve(solver="CLARABEL", warm_start=True)
        except cp.error.SolverError:
            continue
        if (problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                and _conic_solution_valid(y.value, direction.value, sigma, cap,
                                          problem.constraints[1].dual_value,
                                          problem.constraints[2].dual_value)):
            values[k] = problem.value * scale
    return values, ~np.isfinite(values)


def _conic_solution_valid(y, direction, sigma, cap, nu, t, tol=1e-6):
    """Check primal/dual feasibility and the gap in the scaled conic problem."""
    if y is None or nu is None or t is None:
        return False
    if not all(np.isfinite(value).all() for value in (y, nu, t)):
        return False
    y = _symmetrize(y)
    residual = _symmetrize(direction + t * sigma - np.diag(nu))
    upper, lower = float(np.sum(direction * y)), float(np.sum(nu) - cap * t)
    return (np.linalg.eigvalsh(y)[0] >= -tol * max(1.0, np.linalg.norm(y))
            and np.min(np.diag(y)) >= 1 - tol
            and float(np.sum(sigma * y)) <= cap + tol * max(1.0, abs(cap))
            and np.min(nu) >= -tol and t >= -tol
            and np.linalg.eigvalsh(residual)[0] >= -tol * max(1.0, np.linalg.norm(residual))
            and abs(upper - lower) <= tol * max(1.0, abs(upper), abs(lower)))


def near_optimal_derivative_numpy(
    directions: np.ndarray,
    sigma: np.ndarray,
    cap: float,
    *,
    mu_end: float = 1e-8,
    shrink: float = MU_SHRINK,
    center_tol: float = CENTER_TOL,
    final_tol: float = FINAL_TOL,
    max_iter: int = MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """min <Y,H> over the near-optimal set, by a damped-Newton barrier on the dual.

    Solves ``max 1'nu - cap * t`` subject to ``H + t Sigma - Diag(nu) >= 0``, ``nu >= 0``
    and ``t >= 0``, for a stack of directions H sharing one Sigma and one cap.
    """
    directions = _symmetrize(np.asarray(directions, dtype=float))
    sigma = _symmetrize(np.asarray(sigma, dtype=float))
    if not np.isfinite(directions).all() or not np.isfinite(cap):
        raise ValueError("Directions and objective caps must be finite.")
    k, m, _ = directions.shape
    scale = np.linalg.norm(directions, axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    # Normalize the direction only; the minimum is homogeneous in H.
    h = directions / scale[:, None, None]
    # Scale Sigma and its cap together to preserve the near-optimal set.
    sig_scale = float(_positive_scale(sigma))
    sigma = sigma / sig_scale
    caps = np.full(k, float(cap) / sig_scale)
    lam_min = float(np.linalg.eigvalsh(sigma)[0])
    if lam_min <= 0:
        raise ValueError("Sigma must be positive definite for the near-optimal set.")

    nu, t = _feasible_start(h, sigma, lam_min)
    mu = np.maximum((np.abs(nu.sum(axis=1)) + caps * t) / (m + 1), 1e-3)
    eye = np.eye(m)
    ok = np.zeros(k, dtype=bool)
    live = np.ones(k, dtype=bool)

    for _ in range(max_iter):
        if not live.any():
            break
        g = h + t[:, None, None] * sigma - nu[:, :, None] * eye
        a = _symmetrize(np.linalg.inv(g))
        asa = a @ sigma @ a
        inv_mu = 1.0 / mu

        grad = np.empty((k, m + 1))
        grad[:, :m] = np.einsum("kii->ki", a) - inv_mu[:, None] - 1.0 / nu
        grad[:, m] = caps * inv_mu - np.einsum("kij,ij->k", a, sigma) - 1.0 / t

        hess = np.zeros((k, m + 1, m + 1))
        hess[:, :m, :m] = a * a
        hess[:, np.arange(m), np.arange(m)] += 1.0 / nu**2
        cross = -np.einsum("kii->ki", asa)
        hess[:, :m, m] = cross
        hess[:, m, :m] = cross
        hess[:, m, m] = np.einsum("kij,ij->k", asa, sigma) + 1.0 / t**2
        hess += HESSIAN_JITTER * np.einsum("kii->k", hess)[:, None, None] * np.eye(m + 1)

        step = np.linalg.solve(hess, -grad[:, :, None])[:, :, 0]
        dec2 = np.maximum(-np.einsum("ki,ki->k", grad, step), 0.0)
        alpha = np.where(np.sqrt(dec2) > 0.25, 1.0 / (1.0 + np.sqrt(dec2)), 1.0)
        # fraction to the nu >= 0 and t >= 0 boundaries
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_nu = np.where(step[:, :m] < 0, -nu / step[:, :m], np.inf).min(axis=1)
            ratio_t = np.where(step[:, m] < 0, -t / step[:, m], np.inf)
        alpha = np.minimum(alpha, 0.99 * np.minimum(ratio_nu, ratio_t))

        for _ in range(60):
            nu_try = nu + alpha[:, None] * step[:, :m]
            t_try = t + alpha * step[:, m]
            g_try = h + t_try[:, None, None] * sigma - nu_try[:, :, None] * eye
            bad = (np.linalg.eigvalsh(g_try)[:, 0] <= 0) | (nu_try.min(axis=1) <= 0) | (t_try <= 0)
            if not (bad & live).any():
                break
            alpha = np.where(bad, alpha * 0.5, alpha)
        nu, t = nu_try, t_try

        gone = ~np.isfinite(t) | (np.abs(nu).max(axis=1) > DIVERGENCE_BOUND)
        live &= ~gone
        at_end = mu <= mu_end
        centred = dec2 < np.where(at_end, final_tol, center_tol)
        ok |= centred & at_end & live
        live &= ~(centred & at_end)
        mu = np.where(centred & ~at_end, np.maximum(mu * shrink, mu_end), mu)

    value = (nu.sum(axis=1) - caps * t) * scale
    ok &= np.isfinite(value)
    return np.where(ok, value, np.nan), ok
