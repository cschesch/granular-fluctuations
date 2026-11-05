#!/usr/bin/env python3
"""Batched interior-point solver for the small grouped-MGP semidefinite programs.

``mgp_core.solve_grouped_mgp_dual`` calls cvxpy once per covariance, which costs
1--25 ms of canonicalization and solver setup for an M x M problem that carries
only O(M^3) of arithmetic. A Monte Carlo that resamples the grouped covariance
hundreds of times per replication needs millions of such solves, so this module
solves a whole *stack* of them at once with numpy's batched LAPACK.

Every program the paper needs is the same parametric family

    max  w'nu   s.t.   V'(Sigma - Diag(nu)) V >= 0,   nu_i >= 0  (i in a mask),

with objective weights ``w > 0`` and an orthonormal basis ``V`` of R^M
(``basis=None`` means V = I, ``weights=None`` means w = 1):

* ``V = I``, ``w = 1`` and the full nonnegativity mask is the grouped MGP itself,
  ``Gamma_bar(Sigma) = max{tr D : D diagonal >= 0, Sigma - D >= 0}``, whose
  Lagrangian dual is ``min{<Y,Sigma> : Y >= 0, diag(Y) >= 1}`` (draft eq. 4);
* ``V = I`` and ``w = S o S`` is the *weighted* firm-level MGP of draft eq. (3),
  ``Gamma(Sigma_0, S) = max{S'DS : D diagonal >= 0, Sigma_0 - D >= 0}``. Solving it
  in this form rather than in the size-weighted coordinates of eq. (13) keeps the
  variables O(1) when sizes are power-law, where Diag(S) Sigma_0 Diag(S) spans
  eight orders of magnitude and the barrier stalls;
* ``V`` spanning the orthogonal complement of the common subspace col(C*) and an
  empty mask is the Lagrangian dual of the directional derivative
  ``Gamma_bar'(Sigma; H) = min{<Y,H> : Y in Y*(Sigma)}`` of Lemma 2, the object
  the Fang--Santos gradient bootstrap pushes each resampling direction through.

The method is a long-step path-following barrier method on ``nu``. Writing
G(nu) = V'(Sigma - Diag(nu))V and P(nu) = V G(nu)^{-1} V', the barrier

    phi_t(nu) = -t w'nu - log det G(nu) - sum_{i in mask} log nu_i

has gradient ``-t w + diag(P) - 1/nu`` and Hessian ``P o P + diag(1/nu^2)``: the
Hadamard square of P, so one matrix inverse and one linear solve per Newton step
and nothing else. Damped Newton steps of length 1/(1+lambda) in the Newton
decrement lambda keep every iterate strictly feasible without a line search
(Nesterov--Nemirovski self-concordance), which is what makes the whole step
vectorizable over the stack. Problems that finish drop out of the batch.

That self-concordance guarantee is exact-arithmetic only. On the full-basis
problems it holds numerically too, but the directional-derivative problems are
badly conditioned -- buckets at a Heywood boundary D_i = 0 send the 1/nu_i^2 block
of the Hessian to infinity -- and a damped step computed from an ill-conditioned
Newton system can leave the cone, after which nothing recovers. The reduced-basis
path therefore also caps each step at the exact distance to the boundary,
``alpha < 1/lambda_max(G^{-1} V'Diag(step)V)``, which costs one symmetric
eigendecomposition of a q x q block per iteration and makes strict feasibility a
numerical invariant rather than a theorem.

Some of those reduced problems are genuinely unbounded rather than merely hard.
The dual optimal face is bounded when it is cut out by the exact null space of
C*, but the estimator identifies that null space at a relative tolerance (1e-3 in
``mgp_core.numerical_rank``), and the enlarged face this produces can have
recession directions once some diagonal constraints are slack -- see
``face_is_bounded`` for the exact condition, which is cheap enough to check before
solving anything. Where it is missed, the iterates run off to infinity,
``converged`` is False, and the caller is expected to fall back exactly as
``gradient_bootstrap.ShareDirectionalDerivative`` does on a solver failure.

At the barrier parameter mu = 1/t the matrix ``Y = mu P`` satisfies ``Y >= 0``
and ``diag(Y) >= w`` with duality gap exactly ``2 M mu``, so it converges to the
dual optimizer Y* (the gradient of Gamma_bar, and the influence-function weight
in ``mgp_core``) and the returned value is certified to within 2 M mu of the
optimum.

``test_inference_mc.py`` checks values, Y and the recovered common part against
cvxpy/MOSEK. This module is used by the Monte Carlo only: the estimates in the
paper's tables still go through ``mgp_core``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mgp_kernels

# Path-following schedule. The value is certified to 2 M mu whatever the schedule, so
# (shrink, center_tol) only trade Newton steps per stage against number of stages;
# these are the fastest of the combinations tried. Long steps with loose re-centering
# win: the damped-Newton phase after each mu update is what the iterations are spent on.
MU_START_SCALE = 0.05   # mu_0 = MU_START_SCALE * tr(Sigma)/M on the unit-norm problem
MU_END = 1e-10          # stop at duality gap 2 M mu on the unit-norm problem
MU_SHRINK = 0.02
CENTER_TOL = 3e-1       # Newton decrement^2 tolerated before shrinking mu
FINAL_TOL = 1e-11       # Newton decrement^2 required at mu = MU_END
MAX_ITER = 500
HESSIAN_JITTER = 1e-13
# On the unit-norm problem any bounded face has ||Y||_F <= M, so |1'nu| <= M at the
# optimum; |nu| past this is an unbounded program, and bailing early keeps a
# hopeless reduced face from costing hundreds of Newton steps.
DIVERGENCE_BOUND = 1e3


@dataclass
class BatchSolution:
    """Solution of a stack of ``max w'nu s.t. V'(Sigma-Diag(nu))V >= 0`` problems."""

    value: np.ndarray        # (K,)      optimal w'nu
    nu: np.ndarray           # (K, M)    optimal diagonal
    dual: np.ndarray         # (K, M, M) Y = mu V G^{-1} V' -> Y*
    mu: np.ndarray           # (K,)      terminal barrier parameter (gap = 2 M mu)
    iterations: int          # Newton steps taken on the whole batch
    converged: np.ndarray    # (K,)      reached MU_END with a centered iterate

    @property
    def relative_gap(self) -> np.ndarray:
        """Certified relative duality gap ``2 M mu / value`` of each solution."""
        m = self.nu.shape[1]
        return 2.0 * m * self.mu / np.where(self.value != 0, np.abs(self.value), 1.0)

    def common(self, sigma: np.ndarray) -> np.ndarray:
        """C = Sigma - Diag(nu), the common part at the optimum (basis=None case)."""
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


def _feasible_start(
    s: np.ndarray, basis: np.ndarray | None, mask: np.ndarray, nu_init: np.ndarray | None
) -> np.ndarray:
    """A strictly feasible nu: V'(S-Diag(nu))V > 0, and nu > 0 on the mask."""
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
    a length-M boolean mask of coordinates constrained to nu_i >= 0. ``basis`` is
    an (M,q) matrix with orthonormal columns; None means the identity.
    ``weights`` is a positive length-M objective vector; None means all ones.
    ``nu_init`` and ``mu_init`` warm-start the path from a nearby solution
    (bootstrap draws around the full-sample optimum), which is exact, only faster.
    ``guard`` caps every step at the exact distance to the boundary; it costs about
    2x per iteration and changes nothing on problems that already converge, so it is
    a retry option rather than a default. ``need_dual=False`` skips reconstructing
    ``Y = mu P``, which costs about as much as the solve itself and is pure waste for
    the resampling draws, where only the optimal value is read.
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

    guard = bool(guard) and basis is not None   # see the module docstring
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
        # An unbounded program (an enlarged dual face with a recession direction) shows
        # up as iterates running away; retire those with converged = False.
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

    nu_out = np.where(np.isfinite(nu_out), nu_out, 0.0)
    if basis is None:
        y = mu_out[:, None, None] * np.linalg.inv(s - nu_out[:, :, None] * eye)
    else:
        g_mat = s_basis - np.einsum("ki,ia,ib->kab", nu_out, basis, basis)
        y = mu_out[:, None, None] * np.einsum("ia,kab,jb->kij", basis, np.linalg.inv(g_mat), basis)

    return BatchSolution(
        value=(nu_out @ w) * scale * w_scale,
        nu=nu_out * scale[:, None],
        dual=_symmetrize(y) * w_scale,
        mu=mu_out * scale * w_scale,
        iterations=iterations,
        converged=ok_out,
    )


def _run_full_kernel(s, w, mask, nu_out, mu_out, ok_out, tail):
    """Dispatch the full-basis kernel, lane-blocked when the batch is worth blocking.

    ``barrier_full_lanes`` wants the batch in the innermost axis and a multiple of
    ``LANES`` problems; the transpose is two passes over an array the solve then reads
    forty times, and the padding repeats the last problem. It returns the same numbers
    as the scalar kernel to machine precision (``test_inference_mc`` asserts 1e-12) at
    about 1.6x the speed, so the scalar path is kept only for batches too small to be
    worth reshaping.
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
    """Dispatch the reduced-basis kernel, lane-blocked when the batch is worth blocking.

    Same trade as ``_run_full_kernel``: two transposes against forty vectorised Newton
    steps. This is the hotter of the two kernels -- the directional derivatives are the
    largest single cost in the experiment -- and the layout buys about 1.7x on it.
    """
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
        y = (mu_out[:, None, None] * np.linalg.inv(s - nu_out[:, :, None] * np.eye(m))
             if need_dual else np.zeros((len(nu_out), m, m)))
    else:
        basis_k = basis if basis.ndim == 3 else np.broadcast_to(basis, (k,) + basis.shape)
        mask_k = mask if mask.ndim == 2 else np.broadcast_to(mask, (k, m))
        _run_reduced_kernel(s_basis, basis_k, w, mask_k, nu_out, mu_out, ok_out, args[2:9])
        if need_dual:
            g_mat = s_basis - np.einsum("ki,kia,kib->kab", nu_out, basis_k, basis_k)
            y = mu_out[:, None, None] * np.einsum(
                "kia,kab,kjb->kij", basis_k, np.linalg.inv(g_mat), basis_k
            )
        else:
            y = np.zeros((len(nu_out), m, m))
    return BatchSolution(
        value=(nu_out @ w) * scale * w_scale,
        nu=nu_out * scale[:, None],
        dual=_symmetrize(y) * w_scale,
        mu=mu_out * scale * w_scale,
        iterations=int(max_iter),
        converged=ok_out,
    )


def mgp_batch(sigma: np.ndarray, **kwargs) -> BatchSolution:
    """Grouped MGP ``Gamma_bar(Sigma) = max{tr D : D >= 0 diagonal, Sigma - D >= 0}``."""
    return solve_batch(sigma, nonneg=True, basis=None, **kwargs)


def mgp_shares_batch(sigma: np.ndarray, **kwargs) -> tuple[np.ndarray, BatchSolution]:
    """Grouped shares ``gamma_bar = Gamma_bar(Sigma)/(1'Sigma 1)`` and the full solution."""
    sol = mgp_batch(sigma, **kwargs)
    return sol.value / _as_batch(sigma).sum(axis=(-2, -1)), sol


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
    """Whether the reduced dual face has no recession direction, so its min is finite.

    The face is ``F = {Y = V Z V' >= 0 : Y_ii = 1 on the active diagonal, >= 1 elsewhere}``
    with V an orthonormal basis of the estimated null space of C*, of dimension
    ``corank``. A recession direction is an ``R >= 0`` in F's recession cone: ``R_ii = 0``
    on the active coordinates, which for ``R >= 0`` forces ``R e_i = 0`` there, so
    ``col(R)`` lies in ``col(V) intersect span{e_i : i Heywood}``. Any nonzero u in that
    intersection gives ``R = u u'``, so F is unbounded exactly when the intersection is
    nontrivial -- generically when ``corank + heywood > m``.

    The test here is the strict one, ``corank + heywood < m``. At equality the
    intersection is generically trivial and the face is formally bounded, but only just:
    its recession cone is trivial by a dimension count that fails in every neighbourhood,
    the minimum is attained arbitrarily far out, and neither the compiled solver nor
    MOSEK returns it reliably (MOSEK reports "solution may be inaccurate"). Those
    replications are therefore reported as fallbacks rather than solved badly, which is
    also what ``gradient_bootstrap.ShareDirectionalDerivative`` ends up doing.

    With the exact null space of C* none of this binds (a recession direction R would
    have <R, Sigma> = 0 with Sigma > 0, forcing R = 0). It binds only because the corank
    is read off at a relative tolerance, which overstates it.
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
    """``Gamma_bar'(Sigma; H_b) = min{<Y,H_b> : Y in Y*(Sigma)}`` for a stack of H_b.

    With a singleton dual face (the regular case) the min is the linear map <Y*, H>
    and ``basis=None`` selects that path. Otherwise the face is
    ``{Y = V Z V' >= 0 : diag(Y) = 1 where D_i > 0, >= 1 elsewhere}``, whose Lagrangian
    dual is the ``nonneg``-masked member of this module's family evaluated at H.

    ``basis`` may be a single (M,q) face shared by every direction, or an (K,M,q) stack
    with one face per direction -- which is how a whole chunk of replications is solved
    in one call. A stacked basis must be screened with ``face_is_bounded`` by the caller.

    Returns the derivative and a boolean flag marking the directions where the face
    program did not solve, and the single-dual value <Y*, H> was substituted -- the
    fallback ``gradient_bootstrap.ShareDirectionalDerivative`` also takes, and one the
    caller should report rather than hide.
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
    value, failed = sol.value.copy(), ~sol.converged
    if failed.any():
        # Retry the stragglers. A tighter barrier target with more iterations rescues
        # most of them for the price of a handful of solves; for a shared face the
        # exact step-to-boundary cap is available as well. Whatever still fails is the
        # caller's problem -- it re-solves those through cvxpy, or falls back.
        idx = np.flatnonzero(failed)
        retry = solve_batch(
            directions[idx],
            nonneg=nonneg[idx] if isinstance(nonneg, np.ndarray) and nonneg.ndim == 2 else nonneg,
            basis=basis[idx] if basis.ndim == 3 else basis,
            mu_end=mu_end * 1e-2, max_iter=2 * MAX_ITER, need_dual=False,
        )
        value[idx] = retry.value
        failed[idx] = ~retry.converged
        if failed.any() and basis.ndim == 2:
            idx = np.flatnonzero(failed)
            guarded = solve_batch(directions[idx], nonneg=nonneg, basis=basis, guard=True)
            value[idx] = guarded.value
            failed[idx] = ~guarded.converged
    return np.where(failed, single, value), failed


def firm_level_mgp(sigma: np.ndarray, sizes: np.ndarray, **kwargs) -> BatchSolution:
    """Weighted MGP ``Gamma(Sigma, S) = max{S'DS : D >= 0 diagonal, Sigma - D >= 0}``.

    The draft's eq. (3) at the raw (not size-weighted) covariance, so the diagonal
    variables stay O(1) even when the sizes are power-law.
    """
    sizes = np.asarray(sizes, dtype=float).ravel()
    return solve_batch(sigma, nonneg=True, weights=sizes**2, **kwargs)
