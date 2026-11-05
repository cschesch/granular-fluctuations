#!/usr/bin/env python3
"""Fang--Santos directional derivative over the delta_T-near-optimal dual set.

The derivative of the grouped ceiling is a minimum over the dual optimizer set,

    Gamma_bar'(Sigma; H) = min{ <Y,H> : Y in Y*(Sigma) },

and the practical question is how to estimate that set. Reading it off the geometry of
the sample optimum -- the rank of C* and the activity of D* -- needs two tolerances, and
the face those tolerances cut out can be unbounded, at which point the derivative has no
finite value and the estimator has to fall back.

This module estimates it instead by near-optimality, which needs no rank and no null
space:

    Y_hat_T = { Y >= 0 : diag(Y) >= 1,  <Y, Sigma_hat> <= (1 + delta_T) Gamma_hat },

with the tolerance measured in units of the aggregate variance so that it is
dimensionless, and

    delta_T = sqrt(log T / T),    so    delta_T -> 0    and    sqrt(T) delta_T -> infinity.

The two rates do different jobs: sampling displaces a population optimizer from sample
optimality by O_p(T^{-1/2}), so ``sqrt(T) delta_T -> infinity`` keeps every population
optimizer inside the set, while ``delta_T -> 0`` eventually expels every genuinely
suboptimal dual. This is the value-function analogue of the shrinking-neighbourhood
derivative estimator of Fang and Santos (2019, Section 4.2).

The set is automatically bounded whenever ``Sigma_hat > 0``: for ``Y >= 0``,
``<Y,Sigma_hat> >= lambda_min(Sigma_hat) tr(Y)``, so near-optimality caps the trace.
There is therefore no unbounded face and no fallback.

Computationally this is one small SDP per direction, with constraints that do not depend
on the direction. Dualizing it collapses to the barrier family this package already
solves: with a multiplier ``nu >= 0`` on the diagonal constraints and ``t >= 0`` on the
near-optimality constraint,

    min{<Y,H> : Y in Y_hat_T} = max{ 1'nu - kappa t : H + t Sigma_hat - Diag(nu) >= 0 },

where ``kappa = (1 + delta_T) Gamma_hat``. That is M+1 variables against the
M(M+1)/2 of the primal, and it is exactly ``max w'nu s.t. Sigma - Diag(nu) >= 0`` with
one extra direction added to the linear matrix inequality. Slater holds on the primal --
``Y* + eps I`` is strictly feasible for small eps -- so the values agree.

The numpy implementation here is the reference; ``mgp_kernels.barrier_nearopt`` is the
compiled one, and ``test_inference_mc`` checks both against cvxpy.
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


def delta_rate(t: int, scale: float = 1.0) -> float:
    """The tolerance sequence sqrt(log T / T), optionally rescaled by a constant."""
    return float(scale) * float(np.sqrt(np.log(max(t, 2)) / t))


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
    """The compiled path; falls back to the numpy reference without numba.

    ``sigma`` is either one (M,M) covariance shared by every direction, or an (R,M,M)
    stack with ``owner`` saying which one each direction belongs to; ``cap`` matches.
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
    if not mgp_kernels.NUMBA_AVAILABLE:
        values = np.empty(k)
        flags = np.zeros(k, dtype=bool)
        for j in range(len(sigma)):
            sel = owner == j
            values[sel], flags[sel] = near_optimal_derivative_numpy(
                directions[sel], sigma[j], caps[j], mu_end=mu_end, shrink=shrink,
                center_tol=center_tol, final_tol=final_tol, max_iter=max_iter,
            )
        return values, flags
    # Both the direction and the covariance are normalised. The grouped covariance
    # carries the size weights and so has entries orders of magnitude below one, which
    # would otherwise force the multiplier on the near-optimality constraint to be huge.
    # Rescaling Sigma by its norm and the cap with it leaves the value unchanged.
    scale = np.linalg.norm(directions, axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    h = np.ascontiguousarray(directions / scale[:, None, None])
    # Scale Sigma by its smallest eigenvalue, not its norm: the multiplier t has to
    # reach 1/lambda_min(Sigma) before the pencil H + t Sigma is positive definite, so
    # this is the normalisation that puts the starting point at t = O(1).
    sig_scale = np.linalg.eigvalsh(sigma)[:, 0]
    sig_scale = np.where(sig_scale > 0, sig_scale, 1.0)
    sigma = np.ascontiguousarray(sigma / sig_scale[:, None, None])
    caps = np.ascontiguousarray(caps / sig_scale)
    lam_min = np.linalg.eigvalsh(sigma)[:, 0]
    if lam_min.min() <= 0:
        raise ValueError("Sigma must be positive definite for the near-optimal set.")
    nu, t = _feasible_start(h, sigma[owner], lam_min[owner])
    nu = np.ascontiguousarray(nu)
    t = np.ascontiguousarray(t)
    # Start the barrier where its two terms are comparable at the starting point,
    # rather than at a fixed value that may be far off the problem's scale.
    mu = np.maximum((np.abs(nu.sum(axis=1)) + caps[owner] * t) / (m + 1), 1e-3)
    ok = np.zeros(k, dtype=np.bool_)
    mgp_kernels.barrier_nearopt(
        h, sigma, caps, owner, mu.copy(), float(mu_end), float(shrink),
        float(center_tol), float(final_tol), int(max_iter), float(DIVERGENCE_BOUND),
        float(HESSIAN_JITTER), nu, t, mu, ok,
    )
    value = (nu.sum(axis=1) - caps[owner] * t) * scale
    if ok.all() or not backstop:
        return value, ok
    # The program is feasible and bounded, so what the barrier could not finish a
    # general solver can. Only the objective varies within a replication, so one task
    # per distinct covariance serves all of its stragglers.
    from mosek_face import MosekError, NearOptimalSolver, licensed_mosek

    stuck = np.flatnonzero(~ok)
    for j in np.unique(owner[stuck]):
        rows = stuck[owner[stuck] == j]
        if licensed_mosek():
            try:
                with NearOptimalSolver(sigma[j] * sig_scale[j], caps[j] * sig_scale[j]) as solver:
                    got, bad = solver.values(directions[rows])
            except MosekError:
                got, bad = _open_solver(directions[rows], sigma[j], caps[j])
        else:
            got, bad = _open_solver(directions[rows], sigma[j], caps[j])
        value[rows] = np.where(bad, value[rows], got)
        ok[rows] = ~bad
    return value, ok


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
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            values[k] = problem.value * scale
    return values, ~np.isfinite(values)


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
    k, m, _ = directions.shape
    scale = np.linalg.norm(directions, axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    # The value is homogeneous of degree one in (H, cap) jointly with Sigma held fixed,
    # so normalising the direction and the cap together only rescales the answer.
    h = directions / scale[:, None, None]
    # Substituting H = s h and (nu, t) = s (nu', t') leaves the constraint unchanged and
    # the objective scaled by s, so the *cap* is carried through unscaled: dividing it
    # too would tighten the near-optimality constraint past feasibility. Sigma and the
    # cap are then rescaled together, which does leave the value alone.
    sig_scale = float(np.linalg.eigvalsh(sigma)[0]) or 1.0
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
    return value, ok
