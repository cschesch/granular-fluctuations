#!/usr/bin/env python3
"""Batched semidefinite-program solver for the Monte Carlo experiments.

Solves grouped and weighted granularity problems and reduced-face directional
problems. The current gradient bootstrap uses near_optimal.py instead.
Returns estimates and convergence diagnostics.
See NUMERICAL_METHODS.md for the derivation and tests/test_inference_mc.py
for comparisons with the reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mgp_kernels

# Path-following schedule. For the full basis and full nonnegativity mask,
# the exact central-path gap is 2 M mu. See NUMERICAL_METHODS.md for qualifications.
MU_START_SCALE = 0.05   # mu_0 = MU_START_SCALE * tr(Sigma)/M on the unit-norm problem
MU_END = 1e-10          # terminal barrier parameter on the unit-norm problem
MU_SHRINK = 0.02
CENTER_TOL = 3e-1       # Newton decrement^2 tolerated before shrinking mu
FINAL_TOL = 1e-11       # Newton decrement^2 required at mu = MU_END
MAX_ITER = 500
HESSIAN_JITTER = 1e-13
FEASIBILITY_TOL = 1e-5
# Stop iterates that exceed this numerical divergence threshold.
DIVERGENCE_BOUND = 1e3


@dataclass
class BatchSolution:
    """Solution of a stack of ``max w'nu s.t. V'(Sigma-Diag(nu))V >= 0`` problems."""

    value: np.ndarray        # (K,)      estimated optimal w'nu
    nu: np.ndarray           # (K, M)    terminal diagonal iterate
    dual: np.ndarray         # (K, M, M) Y = mu V G^{-1} V', or zeros if omitted
    mu: np.ndarray           # (K,)      terminal barrier parameter
    iterations: int          # batch steps, or iteration limit on the compiled path
    converged: np.ndarray    # (K,)      reached MU_END with a centered iterate
    barrier_count: np.ndarray  # (K,)    matrix-cone dimension plus scalar barriers
    feasible: np.ndarray     # (K,)      terminal primal, and requested dual, checks

    @property
    def relative_gap(self) -> np.ndarray:
        """Barrier-based relative-gap diagnostic; interpret with convergence and feasibility checks."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self.barrier_count * self.mu / np.where(self.value != 0, np.abs(self.value), 1.0)

    def acceptable(self, gap_tol: float | None = None) -> np.ndarray:
        """Require convergence, finite values and feasibility; optionally limit the gap diagnostic."""
        gap = self.relative_gap
        ok = (self.converged & self.feasible & np.isfinite(self.value) & np.isfinite(self.mu)
              & (self.mu >= 0) & np.isfinite(gap) & (gap >= 0))
        if gap_tol is not None:
            ok &= gap <= gap_tol
        return ok

    def common(self, sigma: np.ndarray) -> np.ndarray:
        """C = Sigma - Diag(nu), the estimated common part for a full-basis solve."""
        m = self.nu.shape[1]
        return np.asarray(sigma, float) - self.nu[:, :, None] * np.eye(m)


def _as_batch(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2:
        sigma = sigma[None]
    if sigma.ndim != 3 or sigma.shape[-1] != sigma.shape[-2]:
        raise ValueError(f"Expected a (K,M,M) stack of square matrices, got {sigma.shape}.")
    return 0.5 * (sigma + np.swapaxes(sigma, -1, -2))


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def _positive_definite(matrices: np.ndarray) -> np.ndarray:
    """Check terminal barrier cones without letting one failed row abort a batch."""
    good = np.isfinite(matrices).all(axis=(1, 2))
    rows = np.flatnonzero(good)
    try:
        np.linalg.cholesky(matrices[rows])
    except np.linalg.LinAlgError:
        for row in rows:
            try:
                np.linalg.cholesky(matrices[row])
            except np.linalg.LinAlgError:
                good[row] = False
    return good


def _finish_solution(s, s_basis, basis, w, w_scale, mask, scale,
                     nu, mu, converged, iterations, need_dual):
    """Validate terminal iterates; omitted duals do not incur an extra inversion."""
    k, m = nu.shape
    mask_k = np.broadcast_to(mask, (k, m))
    if basis is None:
        q = m
        cone = s - nu[:, :, None] * np.eye(m)
    else:
        q = basis.shape[-1]
        basis_k = np.broadcast_to(basis, (k, m, q))
        cone = s_basis - np.einsum("ki,kia,kib->kab", nu, basis_k, basis_k)
    cone = _symmetrize(cone)
    feasible = (_positive_definite(cone) & np.isfinite(nu).all(axis=1)
                & np.isfinite(mu) & (mu >= 0)
                & np.all(~mask_k | (nu >= -FEASIBILITY_TOL), axis=1))
    y = np.zeros((k, m, m))
    if need_dual:
        rows = np.flatnonzero(feasible)
        inverse = np.empty_like(cone[rows])
        try:
            inverse[:] = np.linalg.inv(cone[rows])
        except np.linalg.LinAlgError:
            for j, row in enumerate(rows):
                try:
                    inverse[j] = np.linalg.inv(cone[row])
                except np.linalg.LinAlgError:
                    inverse[j] = np.nan
                    feasible[row] = False
        if basis is None:
            y[rows] = mu[rows, None, None] * inverse
        else:
            y[rows] = mu[rows, None, None] * np.einsum(
                "kia,kab,kjb->kij", basis_k[rows], inverse, basis_k[rows])
        y = _symmetrize(y)
        finite = np.isfinite(y).all(axis=(1, 2))
        feasible &= finite
        rows = np.flatnonzero(feasible)
        norm = np.maximum(np.linalg.norm(y[rows], axis=(1, 2)), 1.0)
        feasible[rows] &= np.linalg.eigvalsh(y[rows])[:, 0] >= -FEASIBILITY_TOL * norm
        diagonal = np.einsum("kii->ki", y)
        slack = diagonal - w
        tolerance = FEASIBILITY_TOL * np.maximum(np.abs(w), 1e-12)
        feasible &= np.all(np.where(mask_k, slack >= -tolerance, np.abs(slack) <= tolerance), axis=1)
    return BatchSolution(
        value=(nu @ w) * scale * w_scale, nu=nu * scale[:, None],
        dual=y * w_scale, mu=mu * scale * w_scale,
        iterations=iterations, converged=converged,
        barrier_count=np.asarray(q + mask_k.sum(axis=1)), feasible=feasible,
    )


def _feasible_start(
    s: np.ndarray, basis: np.ndarray | None, mask: np.ndarray, nu_init: np.ndarray | None
) -> np.ndarray:
    """Initialize nu, seeking V'(S-Diag(nu))V > 0 and nu > 0 on the mask."""
    k, m, _ = s.shape
    if basis is None:
        if nu_init is None:
            nu = 0.05 * np.einsum("kii->ki", s).copy()
        else:
            # Warm start. nu_init is optimal for a nearby Sigma and so generically
            # infeasible here; scale it back by the exact distance to the boundary,
            #   theta* = 1 / lambda_max(S^{-1/2} Diag(nu_init) S^{-1/2}),
            # in one shot rather than by halving until an eigenvalue check passes.
            nu = np.array(nu_init, dtype=float)
            w, v = np.linalg.eigh(s)
            inv_half = (v * (np.maximum(w, 1e-300) ** -0.5)[:, None, :]) @ np.swapaxes(v, -1, -2)
            probe = np.einsum("kij,kj,kjl->kil", inv_half, nu, inv_half)
            lam = np.linalg.eigvalsh(_symmetrize(probe))[:, -1]
            theta = np.where(lam > 0, 1.0 / np.maximum(lam, 1e-300), np.inf)
            nu = nu * np.minimum(1.0, 0.98 * theta)[:, None]
        if mask.any():
            floor = 1e-8 * np.abs(np.einsum("kii->ki", s)).max(axis=1, keepdims=True)
            nu = np.where(mask, np.maximum(nu, floor), nu)
        for _ in range(60):
            bad = np.linalg.eigvalsh(s - nu[:, :, None] * np.eye(m))[:, 0] <= 1e-13
            if not bad.any():
                break
            nu[bad] *= 0.5
        return nu
    # Reduced basis: V'V = I, so sum_i v_i v_i' = I and any sub-sum is <= I. Put the
    # masked (nu_i > 0) coordinates at a small positive eps and push the free ones down
    # until V'(S - Diag(nu))V >= I. A Heywood bucket of the grouped MGP -- where the
    # optimal D_i is zero and the dual only needs Y_ii >= 1 -- lands in the mask.
    g0 = _symmetrize(np.einsum("ia,kij,jb->kab", basis, s, basis))
    eps = 1e-4
    nu = np.zeros((k, m))
    if mask.any():
        nu[:, mask] = eps
        g0 = g0 - eps * np.einsum("ia,ib->ab", basis[mask], basis[mask])[None]
    shift = np.minimum(np.linalg.eigvalsh(g0)[:, 0] - 1.0, -1e-8)
    nu[:, ~mask] = shift[:, None]
    return nu


def _step_to_boundary(g_mat: np.ndarray, step: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Largest alpha with ``G - alpha V'Diag(step)V`` still positive definite.

    Equal to ``1/lambda_max(G^{-1/2} V'Diag(step)V G^{-1/2})``, and infinite when that
    largest eigenvalue is not positive. Built from an eigendecomposition rather than a
    Cholesky so that an iterate that has already slipped out of the cone returns a zero
    step (freezing it for the divergence check) instead of raising.
    """
    delta = np.einsum("ki,ia,ib->kab", step, basis, basis)
    w, v = np.linalg.eigh(g_mat)
    ok = w[:, 0] > 0
    inv_half = (v * (np.where(w > 0, w, 1.0) ** -0.5)[:, None, :]) @ np.swapaxes(v, -1, -2)
    reduced = _symmetrize(np.einsum("kai,kij,kjb->kab", inv_half, delta, inv_half))
    lam = np.linalg.eigvalsh(reduced)[:, -1]
    return np.where(~ok, 0.0, np.where(lam > 0, 1.0 / np.maximum(lam, 1e-300), np.inf))


def solve_batch(
    sigma: np.ndarray,
    *,
    nonneg: bool | np.ndarray = True,
    basis: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    nu_init: np.ndarray | None = None,
    mu_init: float | None = None,
    guard: bool = False,
    need_dual: bool = True,
    mu_end: float = MU_END,
    shrink: float = MU_SHRINK,
    center_tol: float = CENTER_TOL,
    final_tol: float = FINAL_TOL,
    max_iter: int = MAX_ITER,
) -> BatchSolution:
    """Solve ``max w'nu s.t. V'(Sigma_k - Diag(nu))V >= 0`` for every k at once.

    ``sigma`` is (K,M,M) (a single (M,M) is promoted). ``nonneg`` is True/False or
    a length-M boolean mask of coordinates constrained to nu_i >= 0. Reduced
    compiled problems also accept one (K,M) mask and (K,M,q) basis per problem.
    ``basis`` otherwise is (M,q) with orthonormal columns; None means the identity.
    ``weights`` is a positive length-M objective vector; None means all ones.
    ``nu_init`` and ``mu_init`` warm-start the same optimization problem.
    For a shared reduced basis, ``guard`` uses the NumPy path and caps steps
    inside the estimated cone boundary. ``need_dual=False`` skips dual
    reconstruction when only the value is needed. ``acceptable()`` checks
    convergence and terminal feasibility; a small ``relative_gap`` alone does not.
    """
    sig = _as_batch(sigma)
    k, m, _ = sig.shape
    eye = np.eye(m)

    scale = np.linalg.norm(sig, axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    s = sig / scale[:, None, None]

    if isinstance(nonneg, np.ndarray):
        mask = np.asarray(nonneg, dtype=bool)
        if mask.shape not in {(m,), (k, m)}:
            raise ValueError(f"nonneg mask must have shape ({m},) or ({k},{m}), got {mask.shape}.")
    else:
        mask = np.ones(m, dtype=bool) if nonneg else np.zeros(m, dtype=bool)
    any_mask = bool(mask.any())
    if mask.ndim == 2 and not (mgp_kernels.NUMBA_AVAILABLE and not guard):
        raise ValueError("A per-problem nonnegativity mask needs the compiled kernels.")

    if weights is None:
        w = np.ones(m)
        w_scale = 1.0
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != (m,) or not np.all(w > 0):
            raise ValueError(f"weights must be a positive length-{m} vector.")
        w_scale = float(w.mean())
        w = w / w_scale

    if basis is not None:
        basis = np.asarray(basis, dtype=float)
        if basis.ndim == 2:
            if basis.shape[0] != m:
                raise ValueError(f"basis must be ({m},q), got {basis.shape}.")
            s_basis = _symmetrize(np.einsum("ia,kij,jb->kab", basis, s, basis))
        elif basis.ndim == 3:
            if basis.shape[:2] != (k, m):
                raise ValueError(f"per-problem basis must be ({k},{m},q), got {basis.shape}.")
            s_basis = _symmetrize(np.einsum("kia,kij,kjb->kab", basis, s, basis))
        else:
            raise ValueError(f"basis must be 2- or 3-dimensional, got {basis.shape}.")
        if basis.shape[-1] == 0:
            raise ValueError("basis must have at least one column.")

    if mgp_kernels.NUMBA_AVAILABLE and not guard:
        # The kernels repair an infeasible start themselves, with scalar Cholesky
        # retries; the batched eigendecomposition here costs as much as the solve.
        nu = (0.05 * np.einsum("kii->ki", s).copy() if nu_init is None
              else np.array(nu_init, dtype=float) / scale[:, None])
    else:
        nu = _feasible_start(
            s, basis, mask, None if nu_init is None else nu_init / scale[:, None]
        )

    if mu_init is None:
        mu0 = MU_START_SCALE * np.abs(np.einsum("kii->ki", s)).sum(axis=1) / m
        mu0 = np.maximum(mu0, mu_end)
    else:
        mu0 = np.full(k, float(mu_init))

    if mgp_kernels.NUMBA_AVAILABLE and not guard:
        return _solve_batch_compiled(
            s, s_basis if basis is not None else None, basis, w, w_scale, mask, scale,
            nu, mu0, mu_end, shrink, center_tol, final_tol, max_iter, m, need_dual,
        )

    guard = bool(guard) and basis is not None   # boundary guard for reduced problems
    live = np.arange(k)
    s_live = s if basis is None else s_basis
    nu_live, mu_live = nu, mu0
    nu_out = np.empty((k, m))
    mu_out = np.empty(k)
    ok_out = np.zeros(k, dtype=bool)
    iterations = 0

    for _ in range(max_iter):
        if basis is None:
            g_mat = s_live - nu_live[:, :, None] * eye
            p = _symmetrize(np.linalg.inv(g_mat))
        else:
            g_mat = s_live - np.einsum("ki,ia,ib->kab", nu_live, basis, basis)
            p = _symmetrize(np.einsum("ia,kab,jb->kij", basis, np.linalg.inv(g_mat), basis))

        grad = np.einsum("kii->ki", p) - (1.0 / mu_live)[:, None] * w
        hess = p * p
        if any_mask:
            inv_nu = np.where(mask, 1.0 / np.where(mask, nu_live, 1.0), 0.0)
            grad = grad - inv_nu
            hess = hess + (inv_nu**2)[:, :, None] * eye
        hess = hess + HESSIAN_JITTER * np.einsum("kii->k", hess)[:, None, None] * eye

        if guard:
            # Jacobi-precondition: the 1/nu_i^2 entries span many orders of magnitude
            # once coordinates approach a Heywood boundary. Only worth its two extra
            # passes on the retry path, where the system is that badly scaled.
            jac = np.sqrt(np.einsum("kii->ki", hess))
            step = np.linalg.solve(
                hess / (jac[:, :, None] * jac[:, None, :]), (-grad / jac)[:, :, None]
            )[:, :, 0] / jac
        else:
            step = np.linalg.solve(hess, -grad[:, :, None])[:, :, 0]
        dec2 = np.maximum(-np.einsum("ki,ki->k", grad, step), 0.0)
        dec = np.sqrt(dec2)
        alpha = np.where(dec > 0.25, 1.0 / (1.0 + dec), 1.0)
        if guard:
            alpha = np.minimum(alpha, 0.99 * _step_to_boundary(g_mat, step, basis))
        if any_mask:
            falling = np.where(step < 0.0, step, -1.0)
            ratio = np.where((step < 0.0) & mask, -nu_live / falling, np.inf)
            alpha = np.minimum(alpha, 0.99 * ratio.min(axis=1))
        nu_live = nu_live + alpha[:, None] * step
        iterations += 1

        at_end = mu_live <= mu_end
        centered = dec2 < np.where(at_end, final_tol, center_tol)
        done = centered & at_end
        # Retire nonfinite or oversized iterates. This detects numerical failure,
        # not whether the underlying optimization problem is unbounded.
        gone = ~(np.abs(nu_live).max(axis=1) <= DIVERGENCE_BOUND)   # also catches NaN
        mu_live = np.where(centered & ~at_end, np.maximum(mu_live * shrink, mu_end), mu_live)
        retire = done | gone
        if retire.any():
            idx = live[retire]
            nu_out[idx] = nu_live[retire]
            mu_out[idx] = mu_live[retire]
            ok_out[idx] = done[retire]
            keep = ~retire
            live, s_live, nu_live, mu_live = live[keep], s_live[keep], nu_live[keep], mu_live[keep]
            if live.size == 0:
                break

    if live.size:  # hit max_iter: report the last iterate and flag it
        nu_out[live] = nu_live
        mu_out[live] = mu_live

    return _finish_solution(s, s_basis if basis is not None else None, basis,
                            w, w_scale, mask, scale, nu_out, mu_out, ok_out,
                            iterations, need_dual)


def _run_full_kernel(s, w, mask, nu_out, mu_out, ok_out, tail):
    """Dispatch the full-basis kernel, using lane blocks for larger batches.

    Lane blocks put the batch on the innermost axis and pad to a multiple of
    ``LANES`` by repeating the last problem. Small batches use the scalar kernel.
    """
    k, m, _ = s.shape
    lanes = mgp_kernels.LANES
    if k < 4 * lanes:
        mgp_kernels.barrier_full(
            np.ascontiguousarray(s), np.ascontiguousarray(w),
            np.ascontiguousarray(mask), mu_out.copy(), *tail, nu_out, mu_out, ok_out
        )
        return
    pad = (-k) % lanes
    sig = np.concatenate([s, np.repeat(s[-1:], pad, axis=0)]) if pad else s
    nu = np.concatenate([nu_out, np.repeat(nu_out[-1:], pad, axis=0)]) if pad else nu_out
    mu = np.concatenate([mu_out, np.repeat(mu_out[-1:], pad)]) if pad else mu_out
    nb = len(sig) // lanes
    sig_b = np.ascontiguousarray(sig.reshape(nb, lanes, m, m).transpose(0, 2, 3, 1))
    nu_b = np.ascontiguousarray(nu.reshape(nb, lanes, m).transpose(0, 2, 1))
    mu_b = np.ascontiguousarray(mu.reshape(nb, lanes))
    ok_b = np.zeros((nb, lanes), dtype=np.bool_)
    mgp_kernels.barrier_full_lanes(
        sig_b, np.ascontiguousarray(w), np.ascontiguousarray(mask),
        mu_b.copy(), *tail, nu_b, mu_b, ok_b
    )
    nu_out[:] = nu_b.transpose(0, 2, 1).reshape(-1, m)[:k]
    mu_out[:] = mu_b.reshape(-1)[:k]
    ok_out[:] = ok_b.reshape(-1)[:k]


def _run_reduced_kernel(s_basis, basis, w, mask, nu_out, mu_out, ok_out, tail):
    """Dispatch the reduced-basis kernel with the same lane layout as the full kernel."""
    k, m, q = basis.shape
    lanes = mgp_kernels.LANES
    if k < 4 * lanes:
        mgp_kernels.barrier_reduced(
            np.ascontiguousarray(s_basis), np.ascontiguousarray(basis),
            np.ascontiguousarray(w), np.ascontiguousarray(mask),
            mu_out.copy(), *tail, nu_out, mu_out, ok_out
        )
        return
    pad = (-k) % lanes
    rep = lambda a: np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]) if pad else a
    sb, bs, nu, mu = rep(s_basis), rep(basis), rep(nu_out), rep(mu_out)
    mk = rep(mask)
    nb = len(sb) // lanes
    sb_b = np.ascontiguousarray(sb.reshape(nb, lanes, q, q).transpose(0, 2, 3, 1))
    bs_b = np.ascontiguousarray(bs.reshape(nb, lanes, m, q).transpose(0, 2, 3, 1))
    mk_b = np.ascontiguousarray(mk.reshape(nb, lanes, m).transpose(0, 2, 1))
    nu_b = np.ascontiguousarray(nu.reshape(nb, lanes, m).transpose(0, 2, 1))
    mu_b = np.ascontiguousarray(mu.reshape(nb, lanes))
    ok_b = np.zeros((nb, lanes), dtype=np.bool_)
    mgp_kernels.barrier_reduced_lanes(
        sb_b, bs_b, np.ascontiguousarray(w), mk_b, mu_b.copy(), *tail, nu_b, mu_b, ok_b
    )
    nu_out[:] = nu_b.transpose(0, 2, 1).reshape(-1, m)[:k]
    mu_out[:] = mu_b.reshape(-1)[:k]
    ok_out[:] = ok_b.reshape(-1)[:k]


def _solve_batch_compiled(s, s_basis, basis, w, w_scale, mask, scale, nu, mu0,
                          mu_end, shrink, center_tol, final_tol, max_iter, m, need_dual):
    """Run the same iteration through the compiled kernels of ``mgp_kernels``."""
    nu_out = np.ascontiguousarray(nu, dtype=float)
    mu_out = np.ascontiguousarray(mu0, dtype=float)
    ok_out = np.zeros(len(nu_out), dtype=np.bool_)
    k = len(nu_out)
    args = (np.ascontiguousarray(w), mu_out.copy(),
            float(mu_end), float(shrink), float(center_tol), float(final_tol),
            int(max_iter), float(DIVERGENCE_BOUND), float(HESSIAN_JITTER),
            nu_out, mu_out, ok_out)
    if basis is None:
        _run_full_kernel(s, w, mask, nu_out, mu_out, ok_out, args[2:9])
    else:
        basis_k = basis if basis.ndim == 3 else np.broadcast_to(basis, (k,) + basis.shape)
        mask_k = mask if mask.ndim == 2 else np.broadcast_to(mask, (k, m))
        _run_reduced_kernel(s_basis, basis_k, w, mask_k, nu_out, mu_out, ok_out, args[2:9])
    return _finish_solution(s, s_basis, basis, w, w_scale, mask, scale,
                            nu_out, mu_out, ok_out, int(max_iter), need_dual)


def mgp_batch(sigma: np.ndarray, **kwargs) -> BatchSolution:
    """Grouped MGP ``Gamma_bar(Sigma) = max{tr D : D >= 0 diagonal, Sigma - D >= 0}``."""
    return solve_batch(sigma, nonneg=True, basis=None, **kwargs)


def common_rank(common: np.ndarray, rel_tol: float = 1e-3) -> np.ndarray:
    """Numerical rank of each C in a stack, matching ``mgp_core.numerical_rank``."""
    eig = np.linalg.eigvalsh(_symmetrize(common))
    top = eig.max(axis=1)
    return (eig >= rel_tol * top[:, None]).sum(axis=1)


def null_basis(common: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal basis of the complement of the top-``rank`` eigenspace of C.

    Complementary slackness makes every dual optimizer vanish on col(C*), so the
    dual optimal face lives in the span of these ``M - rank`` directions.
    """
    eig, vec = np.linalg.eigh(_symmetrize(common))
    return vec[:, : common.shape[0] - rank]


def face_is_bounded(corank: int, heywood: int, m: int) -> bool:
    """Conservative dimension screen for an estimated reduced dual face.

    The face is ``F = {Y = V Z V' >= 0 : Y_ii = 1 on the active diagonal, >= 1 elsewhere}``
    with V an orthonormal basis of the estimated null space of C*, of dimension
    ``corank``. A recession direction is an ``R >= 0`` in F's recession cone: ``R_ii = 0``
    on the active coordinates, which for ``R >= 0`` forces ``R e_i = 0`` there, so
    ``col(R)`` lies in ``col(V) intersect span{e_i : i Heywood}``. Any nonzero u in that
    intersection gives ``R = u u'``, so F is unbounded exactly when the intersection is
    nontrivial -- generically when ``corank + heywood > m``.

    This function accepts only ``corank + heywood < m``, rejecting the equality
    boundary as well. Counts alone do not test the actual intersection for a
    nongeneric basis; passing this screen is not a boundedness certificate.

    With the exact optimal face and Sigma > 0, a recession direction R would
    satisfy <R, Sigma> = 0, forcing R = 0. Estimating the null space and active
    coordinates can enlarge that face. The current gradient bootstrap avoids
    this construction by using a near-optimal objective cap.
    """
    return corank + heywood < m


def directional_derivative_batch(
    directions: np.ndarray,
    *,
    ystar: np.ndarray,
    basis: np.ndarray | None,
    nonneg: bool | np.ndarray = False,
    mu_end: float = MU_END,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate directional responses on a supplied dual face for a stack of H_b.

    With a singleton dual face (the regular case) the min is the linear map <Y*, H>
    and ``basis=None`` selects that path. Otherwise the face is
    ``{Y = V Z V' >= 0 : diag(Y) = 1 where D_i > 0, >= 1 elsewhere}``, whose Lagrangian
    dual is the ``nonneg``-masked member of this module's family evaluated at H.

    ``basis`` may be a single (M,q) face shared by every direction, or an (K,M,q) stack
    with one face per direction -- which is how a whole chunk of replications is solved
    in one call. A stacked basis must be screened with ``face_is_bounded`` by the caller.

    Returns the value and a boolean flag marking directions where the face was
    rejected or the program did not solve, so the single-dual value <Y*, H> was
    substituted. That substitution need not equal the directional derivative at
    a nonregular point. This routine is separate from the current near-optimal
    gradient bootstrap.
    """
    directions = _as_batch(directions)
    ystar = np.asarray(ystar, dtype=float)
    if ystar.ndim == 3 and len(ystar) != len(directions):
        # One Y* per replication, one direction block per replication.
        ystar = np.repeat(ystar, len(directions) // len(ystar), axis=0)
    flat = directions.reshape(len(directions), -1)
    single = ((flat * ystar.reshape(len(ystar), -1)).sum(axis=1) if ystar.ndim == 3
              else flat @ ystar.ravel())
    if basis is None:
        return single, np.zeros(len(directions), dtype=bool)
    if basis.ndim == 2:
        # A shared face: screen it here. Per-problem bases arrive pre-screened, since
        # boundedness is a property of the replication, not of the direction.
        heywood = int(np.sum(nonneg)) if isinstance(nonneg, np.ndarray) else 0
        if not face_is_bounded(basis.shape[1], heywood, basis.shape[0]):
            return single, np.ones(len(directions), dtype=bool)
    sol = solve_batch(directions, nonneg=nonneg, basis=basis, mu_end=mu_end,
                      need_dual=False)
    value, failed = sol.value.copy(), ~sol.acceptable()
    if failed.any():
        # Retry with a tighter barrier target and more iterations. A shared face
        # also permits a guarded retry. Remaining failures return the selected-dual
        # value and a failure flag below.
        idx = np.flatnonzero(failed)
        retry = solve_batch(
            directions[idx],
            nonneg=nonneg[idx] if isinstance(nonneg, np.ndarray) and nonneg.ndim == 2 else nonneg,
            basis=basis[idx] if basis.ndim == 3 else basis,
            mu_end=mu_end * 1e-2, max_iter=2 * MAX_ITER, need_dual=False,
        )
        value[idx] = retry.value
        failed[idx] = ~retry.acceptable()
        if failed.any() and basis.ndim == 2:
            idx = np.flatnonzero(failed)
            guarded = solve_batch(directions[idx], nonneg=nonneg, basis=basis, guard=True)
            value[idx] = guarded.value
            failed[idx] = ~guarded.acceptable()
    return np.where(failed, single, value), failed


def firm_level_mgp(sigma: np.ndarray, sizes: np.ndarray, **kwargs) -> BatchSolution:
    """Weighted MGP ``Gamma(Sigma, S) = max{S'DS : D >= 0 diagonal, Sigma - D >= 0}``.

    Uses the raw covariance and places size weights in the objective, avoiding
    the additional scale dispersion of a size-weighted covariance.
    """
    sizes = np.asarray(sizes, dtype=float).ravel()
    return solve_batch(sigma, nonneg=True, weights=sizes**2, **kwargs)
