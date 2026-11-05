#!/usr/bin/env python3
"""Numba implementations of the batched solver in fast_mgp.py.

Each kernel solves one problem at a time, with parallel execution across
problems. Steps are shortened when necessary to preserve positive
definiteness. See NUMERICAL_METHODS.md for the algorithm and certificate.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range


_JIT = dict(cache=True, fastmath=True, nogil=True)


@njit("boolean(f8[:,::1], f8[:,::1], i8)", inline="always", **_JIT)
def _cholesky(a, out, n):
    """out <- lower Cholesky factor of a (n x n); False if a is not positive definite."""
    for i in range(n):
        for j in range(i + 1):
            total = a[i, j]
            for k in range(j):
                total -= out[i, k] * out[j, k]
            if i == j:
                if total <= 0.0:
                    return False
                out[i, j] = np.sqrt(total)
            else:
                out[i, j] = total / out[j, j]
        for j in range(i + 1, n):
            out[i, j] = 0.0
    return True


@njit("void(f8[:,::1], f8[:,::1], f8[::1], i8)", inline="always", **_JIT)
def _chol_inverse(chol, out, scratch, n):
    """out <- A^{-1} given its lower Cholesky factor, by solving A x = e_j column-wise."""
    for j in range(n):
        for i in range(n):
            total = 1.0 if i == j else 0.0
            for k in range(i):
                total -= chol[i, k] * scratch[k]
            scratch[i] = total / chol[i, i]
        for i in range(n - 1, -1, -1):
            total = scratch[i]
            for k in range(i + 1, n):
                total -= chol[k, i] * out[k, j]
            out[i, j] = total / chol[i, i]


@njit("boolean(f8[:,::1], f8[::1], f8[::1], f8[:,::1], f8[::1], i8)", inline="always", **_JIT)
def _spd_solve(a, rhs, out, chol, scratch, n):
    """out <- a^{-1} rhs for symmetric positive definite a; False if the Cholesky fails."""
    if not _cholesky(a, chol, n):
        return False
    for i in range(n):
        total = rhs[i]
        for k in range(i):
            total -= chol[i, k] * scratch[k]
        scratch[i] = total / chol[i, i]
    for i in range(n - 1, -1, -1):
        total = scratch[i]
        for k in range(i + 1, n):
            total -= chol[k, i] * out[k]
        out[i] = total / chol[i, i]
    return True


# --------------------------------------------------------------------------- #
# Full-basis kernel: max w'nu s.t. Sigma - Diag(nu) >= 0                        #
# --------------------------------------------------------------------------- #
@njit(parallel=True, **_JIT)
def barrier_full(sigma, weights, mask, mu0, mu_end, shrink, center_tol, final_tol,
                 max_iter, divergence, jitter, nu_out, mu_out, ok_out):
    k_total, m, _ = sigma.shape
    for k in prange(k_total):
        cmat = np.empty((m, m))
        ainv = np.empty((m, m))
        hess = np.empty((m, m))
        chol = np.empty((m, m))
        work = np.empty((m, m))
        grad = np.empty(m)
        step = np.empty(m)
        nu = np.empty(m)
        trial = np.empty(m)
        scratch = np.empty(m)

        for i in range(m):
            nu[i] = nu_out[k, i]
        # Repair the start in place: halve until Sigma - Diag(nu) is positive definite.
        # nu = 0 always works because Sigma > 0, so this terminates.
        for _ in range(60):
            for i in range(m):
                for j in range(m):
                    cmat[i, j] = sigma[k, i, j]
                cmat[i, i] -= nu[i]
            if _cholesky(cmat, chol, m):
                break
            for i in range(m):
                nu[i] *= 0.5
        mu = mu0[k]
        ok = False

        for _ in range(max_iter):
            for i in range(m):
                for j in range(m):
                    cmat[i, j] = sigma[k, i, j]
                cmat[i, i] -= nu[i]
            if not _cholesky(cmat, chol, m):
                break
            _chol_inverse(chol, ainv, scratch, m)

            inv_mu = 1.0 / mu
            for i in range(m):
                g = ainv[i, i] - inv_mu * weights[i]
                if mask[i]:
                    g -= 1.0 / nu[i]
                grad[i] = g
            trace_h = 0.0
            for i in range(m):
                for j in range(m):
                    hess[i, j] = ainv[i, j] * ainv[i, j]
                if mask[i]:
                    hess[i, i] += 1.0 / (nu[i] * nu[i])
                trace_h += hess[i, i]
            damp = jitter * trace_h
            for i in range(m):
                hess[i, i] += damp
            for i in range(m):
                scratch[i] = -grad[i]
            if not _spd_solve(hess, scratch, step, work, trial, m):
                break

            dec2 = 0.0
            for i in range(m):
                dec2 -= grad[i] * step[i]
            if dec2 < 0.0:
                dec2 = 0.0
            alpha = 1.0 if dec2 <= 0.0625 else 1.0 / (1.0 + np.sqrt(dec2))
            # Fraction to the nu >= 0 boundary on the sign-constrained coordinates.
            for i in range(m):
                if mask[i] and step[i] < 0.0:
                    cap = -0.99 * nu[i] / step[i]
                    if cap < alpha:
                        alpha = cap

            # Take the step, halving until the iterate is back inside the cone. In
            # exact arithmetic the damped step never leaves it; on the face problems
            # it can, and the Cholesky above is the cheapest possible detector.
            moved = False
            halved = False
            for _ in range(50):
                bad = False
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
                    if mask[i] and trial[i] <= 0.0:
                        bad = True
                if not bad:
                    for i in range(m):
                        for j in range(m):
                            work[i, j] = sigma[k, i, j]
                        work[i, i] -= trial[i]
                    if _cholesky(work, chol, m):
                        moved = True
                        break
                alpha *= 0.5
                halved = True
            if not moved:
                break
            if halved:
                # Feasible alphas form an interval containing 0, so backing off keeps
                # feasibility; landing hard against the boundary wrecks the next Newton
                # system, which is what a bare halving search does.
                alpha *= 0.9
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
            big = 0.0
            for i in range(m):
                nu[i] = trial[i]
                if abs(nu[i]) > big:
                    big = abs(nu[i])
            if big > divergence or not np.isfinite(big):
                break

            at_end = mu <= mu_end
            if dec2 < (final_tol if at_end else center_tol):
                if at_end:
                    ok = True
                    break
                mu = mu * shrink
                if mu < mu_end:
                    mu = mu_end

        for i in range(m):
            nu_out[k, i] = nu[i]
        mu_out[k] = mu
        ok_out[k] = ok


# --------------------------------------------------------------------------- #
# Reduced-basis kernel: max w'nu s.t. V'(Sigma - Diag(nu))V >= 0                #
# --------------------------------------------------------------------------- #
@njit(parallel=True, **_JIT)
def barrier_reduced(sigma_b, basis, weights, mask, mu0, mu_end, shrink, center_tol,
                    final_tol, max_iter, divergence, jitter, nu_out, mu_out, ok_out):
    k_total = sigma_b.shape[0]
    m, q = basis.shape[1], basis.shape[2]
    for k in prange(k_total):
        gmat = np.empty((q, q))
        ginv = np.empty((q, q))
        chol_q = np.empty((q, q))
        scratch_q = np.empty(q)
        pmat = np.empty((m, m))
        hess = np.empty((m, m))
        work = np.empty((m, m))
        grad = np.empty(m)
        step = np.empty(m)
        nu = np.empty(m)
        trial = np.empty(m)
        scratch = np.empty(m)

        # Start from nu_i = eps on the sign-constrained coordinates and nu_i = -shift on
        # the free ones. V has orthonormal columns, so lowering the free coordinates adds
        # shift * (I - sum_{masked} v_i v_i') to G; doubling shift reaches a positive
        # definite G whenever one exists, without an eigendecomposition.
        shift = 1.0
        for i in range(m):
            nu[i] = 1e-4 if mask[k, i] else -shift
        for _ in range(60):
            for a in range(q):
                for b in range(q):
                    total = sigma_b[k, a, b]
                    for i in range(m):
                        total -= nu[i] * basis[k, i, a] * basis[k, i, b]
                    gmat[a, b] = total
            if _cholesky(gmat, chol_q, q):
                break
            shift *= 2.0
            for i in range(m):
                if not mask[k, i]:
                    nu[i] = -shift
        mu = mu0[k]
        ok = False

        for _ in range(max_iter):
            for a in range(q):
                for b in range(q):
                    total = sigma_b[k, a, b]
                    for i in range(m):
                        total -= nu[i] * basis[k, i, a] * basis[k, i, b]
                    gmat[a, b] = total
            if not _cholesky(gmat, chol_q, q):
                break
            _chol_inverse(chol_q, ginv, scratch_q, q)
            for i in range(m):
                for a in range(q):
                    total = 0.0
                    for b in range(q):
                        total += ginv[a, b] * basis[k, i, b]
                    scratch_q[a] = total
                for j in range(i + 1):
                    total = 0.0
                    for a in range(q):
                        total += basis[k, j, a] * scratch_q[a]
                    pmat[i, j] = total
                    pmat[j, i] = total

            inv_mu = 1.0 / mu
            for i in range(m):
                g = pmat[i, i] - inv_mu * weights[i]
                if mask[k, i]:
                    g -= 1.0 / nu[i]
                grad[i] = g
            trace_h = 0.0
            for i in range(m):
                for j in range(m):
                    hess[i, j] = pmat[i, j] * pmat[i, j]
                if mask[k, i]:
                    hess[i, i] += 1.0 / (nu[i] * nu[i])
                trace_h += hess[i, i]
            damp = jitter * trace_h
            for i in range(m):
                hess[i, i] += damp
            for i in range(m):
                scratch[i] = -grad[i]
            if not _spd_solve(hess, scratch, step, work, trial, m):
                break

            dec2 = 0.0
            for i in range(m):
                dec2 -= grad[i] * step[i]
            if dec2 < 0.0:
                dec2 = 0.0
            alpha = 1.0 if dec2 <= 0.0625 else 1.0 / (1.0 + np.sqrt(dec2))
            for i in range(m):
                if mask[k, i] and step[i] < 0.0:
                    cap = -0.99 * nu[i] / step[i]
                    if cap < alpha:
                        alpha = cap

            moved = False
            halved = False
            for _ in range(50):
                bad = False
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
                    if mask[k, i] and trial[i] <= 0.0:
                        bad = True
                if not bad:
                    for a in range(q):
                        for b in range(q):
                            total = sigma_b[k, a, b]
                            for i in range(m):
                                total -= trial[i] * basis[k, i, a] * basis[k, i, b]
                            gmat[a, b] = total
                    if _cholesky(gmat, chol_q, q):
                        moved = True
                        break
                alpha *= 0.5
                halved = True
            if not moved:
                break
            if halved:
                # See barrier_full: back off from the boundary once a halving search
                # has been needed, or the next Newton system is unusable.
                alpha *= 0.9
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
            big = 0.0
            for i in range(m):
                nu[i] = trial[i]
                if abs(nu[i]) > big:
                    big = abs(nu[i])
            if big > divergence or not np.isfinite(big):
                break

            at_end = mu <= mu_end
            if dec2 < (final_tol if at_end else center_tol):
                if at_end:
                    ok = True
                    break
                mu = mu * shrink
                if mu < mu_end:
                    mu = mu_end

        for i in range(m):
            nu_out[k, i] = nu[i]
        mu_out[k] = mu
        ok_out[k] = ok


# --------------------------------------------------------------------------- #
# Lane-blocked kernel: the batch in the innermost axis, so LLVM vectorises it   #
# --------------------------------------------------------------------------- #
# ``barrier_full`` above solves one M x M problem per thread in scalar code. Every
# operation in it -- Cholesky, triangular inverse, the Hadamard square, the Newton
# solve -- is a chain of dependent scalar FMAs on ten-element rows, which leaves the
# vector units idle and the pipeline stalled on latency: measured against the flop
# count it reaches about 7% of scalar peak.
#
# This version keeps the identical algorithm but transposes the layout so that a block
# of LANES problems sits in the innermost, contiguous axis. Every scalar of the scalar
# kernel becomes a length-LANES vector, so the arithmetic vectorises, and -- more
# importantly at this size -- the dependent chains of different problems interleave, so
# there is real instruction-level parallelism to hide the latency of the divides and
# square roots. Nothing about the mathematics changes, which is why
# ``test_inference_mc`` can require the two kernels to agree to solver tolerance.
#
# The cost is that control flow must become arithmetic: a lane whose Cholesky fails
# cannot break out of the loop on its own, so failure is a mask and the block runs until
# every lane is done. With the iteration counts of neighbouring problems within a few
# steps of one another that waste is small, and much smaller than what the vectorisation
# buys back.
LANES = 8


@njit(parallel=True, **_JIT)
def barrier_full_lanes(sigma, weights, mask, mu0, mu_end, shrink, center_tol, final_tol,
                       max_iter, divergence, jitter, nu_out, mu_out, ok_out):
    """As ``barrier_full``, with ``sigma`` shaped (nblocks, M, M, LANES)."""
    nblocks, m, _, _ = sigma.shape
    for blk in prange(nblocks):
        cmat = np.empty((m, m, LANES))
        ainv = np.empty((m, m, LANES))
        chol = np.empty((m, m, LANES))
        hess = np.empty((m, m, LANES))
        work = np.empty((m, m, LANES))
        rinv = np.empty((m, LANES))
        grad = np.empty((m, LANES))
        step = np.empty((m, LANES))
        nu = np.empty((m, LANES))
        trial = np.empty((m, LANES))
        col = np.empty((m, LANES))
        mu = np.empty(LANES)
        alpha = np.empty(LANES)
        dec2 = np.empty(LANES)
        live = np.empty(LANES, dtype=np.bool_)
        good = np.empty(LANES, dtype=np.bool_)

        for l in range(LANES):
            for i in range(m):
                nu[i, l] = nu_out[blk, i, l]
            mu[l] = mu0[blk, l]
            live[l] = True
            ok_out[blk, l] = False

        # ---- repair the start: halve until Sigma - Diag(nu) is positive definite ----
        for _ in range(60):
            for l in range(LANES):
                good[l] = True
            for i in range(m):
                for j in range(i + 1):
                    for l in range(LANES):
                        tot = sigma[blk, i, j, l] - (nu[i, l] if i == j else 0.0)
                        for t in range(j):
                            tot -= chol[i, t, l] * chol[j, t, l]
                        if i == j:
                            if tot <= 0.0:
                                good[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            chol[i, j, l] = v
                            rinv[i, l] = 1.0 / v
                        else:
                            chol[i, j, l] = tot * rinv[j, l]
            done = True
            for l in range(LANES):
                if not good[l]:
                    done = False
                    for i in range(m):
                        nu[i, l] *= 0.5
            if done:
                break

        for _ in range(max_iter):
            any_live = False
            for l in range(LANES):
                if live[l]:
                    any_live = True
            if not any_live:
                break

            # ---- C = Sigma - Diag(nu), and its Cholesky ----
            for l in range(LANES):
                good[l] = True
            for i in range(m):
                for j in range(i + 1):
                    for l in range(LANES):
                        tot = sigma[blk, i, j, l] - (nu[i, l] if i == j else 0.0)
                        cmat[i, j, l] = tot
                        cmat[j, i, l] = tot
                        for t in range(j):
                            tot -= chol[i, t, l] * chol[j, t, l]
                        if i == j:
                            if tot <= 0.0:
                                good[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            chol[i, j, l] = v
                            rinv[i, l] = 1.0 / v
                        else:
                            chol[i, j, l] = tot * rinv[j, l]
            for l in range(LANES):
                if not good[l]:
                    live[l] = False

            # ---- A = C^{-1}, column by column through the Cholesky factor ----
            for jc in range(m):
                for i in range(m):
                    for l in range(LANES):
                        tot = 1.0 if i == jc else 0.0
                        for t in range(i):
                            tot -= chol[i, t, l] * col[t, l]
                        col[i, l] = tot * rinv[i, l]
                for i in range(m - 1, -1, -1):
                    for l in range(LANES):
                        tot = col[i, l]
                        for t in range(i + 1, m):
                            tot -= chol[t, i, l] * ainv[t, jc, l]
                        ainv[i, jc, l] = tot * rinv[i, l]

            # ---- gradient and Hessian ----
            for l in range(LANES):
                for i in range(m):
                    g = ainv[i, i, l] - weights[i] / mu[l]
                    if mask[i]:
                        g -= 1.0 / nu[i, l]
                    grad[i, l] = g
            for i in range(m):
                for j in range(m):
                    for l in range(LANES):
                        hess[i, j, l] = ainv[i, j, l] * ainv[i, j, l]
            for l in range(LANES):
                tr = 0.0
                for i in range(m):
                    if mask[i]:
                        hess[i, i, l] += 1.0 / (nu[i, l] * nu[i, l])
                    tr += hess[i, i, l]
                d = jitter * tr
                for i in range(m):
                    hess[i, i, l] += d

            # ---- Newton step: Cholesky solve of the Hessian ----
            for i in range(m):
                for j in range(i + 1):
                    for l in range(LANES):
                        tot = hess[i, j, l]
                        for t in range(j):
                            tot -= work[i, t, l] * work[j, t, l]
                        if i == j:
                            if tot <= 0.0:
                                live[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            work[i, j, l] = v
                            rinv[i, l] = 1.0 / v
                        else:
                            work[i, j, l] = tot * rinv[j, l]
            for i in range(m):
                for l in range(LANES):
                    tot = -grad[i, l]
                    for t in range(i):
                        tot -= work[i, t, l] * col[t, l]
                    col[i, l] = tot * rinv[i, l]
            for i in range(m - 1, -1, -1):
                for l in range(LANES):
                    tot = col[i, l]
                    for t in range(i + 1, m):
                        tot -= work[t, i, l] * step[t, l]
                    step[i, l] = tot * rinv[i, l]

            # ---- damped step, capped at the nu >= 0 boundary ----
            for l in range(LANES):
                d2 = 0.0
                for i in range(m):
                    d2 -= grad[i, l] * step[i, l]
                if d2 < 0.0:
                    d2 = 0.0
                dec2[l] = d2
                a = 1.0 if d2 <= 0.0625 else 1.0 / (1.0 + np.sqrt(d2))
                for i in range(m):
                    if mask[i] and step[i, l] < 0.0:
                        cap = -0.99 * nu[i, l] / step[i, l]
                        if cap < a:
                            a = cap
                alpha[l] = a

            # ---- halve until back inside the cone (see barrier_full) ----
            for l in range(LANES):
                good[l] = False
            for _ in range(50):
                allgood = True
                for l in range(LANES):
                    if good[l] or not live[l]:
                        continue
                    allgood = False
                    bad = False
                    for i in range(m):
                        trial[i, l] = nu[i, l] + alpha[l] * step[i, l]
                        if mask[i] and trial[i, l] <= 0.0:
                            bad = True
                    if not bad:
                        for i in range(m):
                            for j in range(i + 1):
                                tot = cmat[i, j, l] - (
                                    alpha[l] * step[i, l] if i == j else 0.0
                                )
                                for t in range(j):
                                    tot -= work[i, t, l] * work[j, t, l]
                                if i == j:
                                    if tot <= 0.0:
                                        bad = True
                                        tot = 1.0
                                    work[i, j, l] = np.sqrt(tot)
                                else:
                                    work[i, j, l] = tot / work[j, j, l]
                            if bad:
                                break
                    if bad:
                        alpha[l] *= 0.5
                    else:
                        good[l] = True
                if allgood:
                    break
            for l in range(LANES):
                if live[l] and not good[l]:
                    live[l] = False

            # ---- accept, and advance mu ----
            for l in range(LANES):
                if not live[l]:
                    continue
                big = 0.0
                for i in range(m):
                    nu[i, l] = nu[i, l] + alpha[l] * step[i, l]
                    if abs(nu[i, l]) > big:
                        big = abs(nu[i, l])
                if big > divergence or not np.isfinite(big):
                    live[l] = False
                    continue
                at_end = mu[l] <= mu_end
                if dec2[l] < (final_tol if at_end else center_tol):
                    if at_end:
                        ok_out[blk, l] = True
                        live[l] = False
                    else:
                        mu[l] = mu[l] * shrink
                        if mu[l] < mu_end:
                            mu[l] = mu_end

        for l in range(LANES):
            for i in range(m):
                nu_out[blk, i, l] = nu[i, l]
            mu_out[blk, l] = mu[l]


@njit(parallel=True, **_JIT)
def barrier_reduced_lanes(sigma_b, basis, weights, mask, mu0, mu_end, shrink, center_tol,
                          final_tol, max_iter, divergence, jitter, nu_out, mu_out, ok_out):
    """As ``barrier_reduced``, lane-blocked: ``sigma_b`` is (nblocks, q, q, LANES) and
    ``basis`` is (nblocks, M, q, LANES). See ``barrier_full_lanes`` for the layout."""
    nblocks = sigma_b.shape[0]
    q = sigma_b.shape[1]
    m = basis.shape[1]
    for blk in prange(nblocks):
        gmat = np.empty((q, q, LANES))
        ginv = np.empty((q, q, LANES))
        cq = np.empty((q, q, LANES))
        rq = np.empty((q, LANES))
        colq = np.empty((q, LANES))
        pmat = np.empty((m, m, LANES))
        hess = np.empty((m, m, LANES))
        work = np.empty((m, m, LANES))
        rinv = np.empty((m, LANES))
        grad = np.empty((m, LANES))
        step = np.empty((m, LANES))
        nu = np.empty((m, LANES))
        trial = np.empty((m, LANES))
        col = np.empty((m, LANES))
        mu = np.empty(LANES)
        alpha = np.empty(LANES)
        dec2 = np.empty(LANES)
        live = np.empty(LANES, dtype=np.bool_)
        good = np.empty(LANES, dtype=np.bool_)
        shift = np.empty(LANES)

        for l in range(LANES):
            mu[l] = mu0[blk, l]
            live[l] = True
            shift[l] = 1.0
            ok_out[blk, l] = False
            for i in range(m):
                nu[i, l] = 1e-4 if mask[blk, i, l] else -1.0

        # ---- push the free coordinates down until V'(Sigma - Diag(nu))V is definite ----
        for _ in range(60):
            for l in range(LANES):
                good[l] = True
            for a in range(q):
                for bq in range(a + 1):
                    for l in range(LANES):
                        tot = sigma_b[blk, a, bq, l]
                        for i in range(m):
                            tot -= nu[i, l] * basis[blk, i, a, l] * basis[blk, i, bq, l]
                        for t in range(bq):
                            tot -= cq[a, t, l] * cq[bq, t, l]
                        if a == bq:
                            if tot <= 0.0:
                                good[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            cq[a, bq, l] = v
                            rq[a, l] = 1.0 / v
                        else:
                            cq[a, bq, l] = tot * rq[bq, l]
            done = True
            for l in range(LANES):
                if not good[l]:
                    done = False
                    shift[l] *= 2.0
                    for i in range(m):
                        if not mask[blk, i, l]:
                            nu[i, l] = -shift[l]
            if done:
                break

        for _ in range(max_iter):
            any_live = False
            for l in range(LANES):
                if live[l]:
                    any_live = True
            if not any_live:
                break

            # ---- G and its Cholesky ----
            for l in range(LANES):
                good[l] = True
            for a in range(q):
                for bq in range(a + 1):
                    for l in range(LANES):
                        tot = sigma_b[blk, a, bq, l]
                        for i in range(m):
                            tot -= nu[i, l] * basis[blk, i, a, l] * basis[blk, i, bq, l]
                        gmat[a, bq, l] = tot
                        gmat[bq, a, l] = tot
                        for t in range(bq):
                            tot -= cq[a, t, l] * cq[bq, t, l]
                        if a == bq:
                            if tot <= 0.0:
                                good[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            cq[a, bq, l] = v
                            rq[a, l] = 1.0 / v
                        else:
                            cq[a, bq, l] = tot * rq[bq, l]
            for l in range(LANES):
                if not good[l]:
                    live[l] = False

            # ---- G^{-1} then P = V G^{-1} V' ----
            for jc in range(q):
                for a in range(q):
                    for l in range(LANES):
                        tot = 1.0 if a == jc else 0.0
                        for t in range(a):
                            tot -= cq[a, t, l] * colq[t, l]
                        colq[a, l] = tot * rq[a, l]
                for a in range(q - 1, -1, -1):
                    for l in range(LANES):
                        tot = colq[a, l]
                        for t in range(a + 1, q):
                            tot -= cq[t, a, l] * ginv[t, jc, l]
                        ginv[a, jc, l] = tot * rq[a, l]
            for i in range(m):
                for a in range(q):
                    for l in range(LANES):
                        tot = 0.0
                        for bq in range(q):
                            tot += ginv[a, bq, l] * basis[blk, i, bq, l]
                        colq[a, l] = tot
                for j in range(i + 1):
                    for l in range(LANES):
                        tot = 0.0
                        for a in range(q):
                            tot += basis[blk, j, a, l] * colq[a, l]
                        pmat[i, j, l] = tot
                        pmat[j, i, l] = tot

            # ---- gradient, Hessian, Newton step ----
            for l in range(LANES):
                for i in range(m):
                    g = pmat[i, i, l] - weights[i] / mu[l]
                    if mask[blk, i, l]:
                        g -= 1.0 / nu[i, l]
                    grad[i, l] = g
            for i in range(m):
                for j in range(m):
                    for l in range(LANES):
                        hess[i, j, l] = pmat[i, j, l] * pmat[i, j, l]
            for l in range(LANES):
                tr = 0.0
                for i in range(m):
                    if mask[blk, i, l]:
                        hess[i, i, l] += 1.0 / (nu[i, l] * nu[i, l])
                    tr += hess[i, i, l]
                d = jitter * tr
                for i in range(m):
                    hess[i, i, l] += d
            for i in range(m):
                for j in range(i + 1):
                    for l in range(LANES):
                        tot = hess[i, j, l]
                        for t in range(j):
                            tot -= work[i, t, l] * work[j, t, l]
                        if i == j:
                            if tot <= 0.0:
                                live[l] = False
                                tot = 1.0
                            v = np.sqrt(tot)
                            work[i, j, l] = v
                            rinv[i, l] = 1.0 / v
                        else:
                            work[i, j, l] = tot * rinv[j, l]
            for i in range(m):
                for l in range(LANES):
                    tot = -grad[i, l]
                    for t in range(i):
                        tot -= work[i, t, l] * col[t, l]
                    col[i, l] = tot * rinv[i, l]
            for i in range(m - 1, -1, -1):
                for l in range(LANES):
                    tot = col[i, l]
                    for t in range(i + 1, m):
                        tot -= work[t, i, l] * step[t, l]
                    step[i, l] = tot * rinv[i, l]

            for l in range(LANES):
                d2 = 0.0
                for i in range(m):
                    d2 -= grad[i, l] * step[i, l]
                if d2 < 0.0:
                    d2 = 0.0
                dec2[l] = d2
                a = 1.0 if d2 <= 0.0625 else 1.0 / (1.0 + np.sqrt(d2))
                for i in range(m):
                    if mask[blk, i, l] and step[i, l] < 0.0:
                        cap = -0.99 * nu[i, l] / step[i, l]
                        if cap < a:
                            a = cap
                alpha[l] = a

            # ---- halve until the step lands back inside the cone ----
            for l in range(LANES):
                good[l] = False
            for _ in range(50):
                allgood = True
                for l in range(LANES):
                    if good[l] or not live[l]:
                        continue
                    allgood = False
                    bad = False
                    for i in range(m):
                        trial[i, l] = nu[i, l] + alpha[l] * step[i, l]
                        if mask[blk, i, l] and trial[i, l] <= 0.0:
                            bad = True
                    if not bad:
                        for a in range(q):
                            for bq in range(a + 1):
                                tot = gmat[a, bq, l]
                                for i in range(m):
                                    tot -= (alpha[l] * step[i, l]
                                            * basis[blk, i, a, l] * basis[blk, i, bq, l])
                                for t in range(bq):
                                    tot -= work[a, t, l] * work[bq, t, l]
                                if a == bq:
                                    if tot <= 0.0:
                                        bad = True
                                        tot = 1.0
                                    work[a, bq, l] = np.sqrt(tot)
                                else:
                                    work[a, bq, l] = tot / work[bq, bq, l]
                            if bad:
                                break
                    if bad:
                        alpha[l] *= 0.5
                    else:
                        good[l] = True
                if allgood:
                    break
            for l in range(LANES):
                if live[l] and not good[l]:
                    live[l] = False

            for l in range(LANES):
                if not live[l]:
                    continue
                big = 0.0
                for i in range(m):
                    nu[i, l] = nu[i, l] + alpha[l] * step[i, l]
                    if abs(nu[i, l]) > big:
                        big = abs(nu[i, l])
                if big > divergence or not np.isfinite(big):
                    live[l] = False
                    continue
                at_end = mu[l] <= mu_end
                if dec2[l] < (final_tol if at_end else center_tol):
                    if at_end:
                        ok_out[blk, l] = True
                        live[l] = False
                    else:
                        mu[l] = mu[l] * shrink
                        if mu[l] < mu_end:
                            mu[l] = mu_end

        for l in range(LANES):
            for i in range(m):
                nu_out[blk, i, l] = nu[i, l]
            mu_out[blk, l] = mu[l]


# --------------------------------------------------------------------------- #
# Near-optimal dual set: max 1'nu - cap t  s.t. H + t Sigma - Diag(nu) >= 0     #
# --------------------------------------------------------------------------- #
@njit(parallel=True, **_JIT)
def barrier_nearopt(direction, sigma, cap, owner, mu0, mu_end, shrink, center_tol,
                    final_tol, max_iter, divergence, jitter, nu_out, t_out, mu_out, ok_out):
    """Fang--Santos directional derivative over the delta-near-optimal dual set.

    The dual of ``min{<Y,H> : Y >= 0, diag(Y) >= 1, <Y,Sigma> <= cap}`` is

        max  1'nu - cap t   s.t.   H + t Sigma - Diag(nu) >= 0,  nu >= 0,  t >= 0,

    which is the barrier family of ``barrier_full`` with one extra direction added to the
    matrix inequality. Writing ``A = (H + t Sigma - Diag(nu))^{-1}``, the barrier

        phi = -(1'nu - cap t)/mu - log det G - sum log nu_m - log t

    has gradient ``(diag(A) - 1/mu - 1/nu,  cap/mu - <A,Sigma> - 1/t)`` and Hessian

        [ A o A + diag(1/nu^2)   -diag(A Sigma A) ]
        [ -diag(A Sigma A)'      <A Sigma A, Sigma> + 1/t^2 ],

    an (M+1)-dimensional system. ``sigma`` and ``cap`` are indexed by ``owner``, which
    says which replication each direction belongs to, so one call carries every direction
    of a whole chunk of replications without duplicating their covariances.

    No rank, no null space, and -- because ``<Y,Sigma> >= lambda_min(Sigma) tr(Y)`` caps
    the trace whenever ``Sigma > 0`` -- no unbounded face to fall back from.
    """
    k_total, m, _ = direction.shape
    n = m + 1
    for k in prange(k_total):
        own = owner[k]
        gmat = np.empty((m, m))
        ainv = np.empty((m, m))
        asig = np.empty((m, m))
        asa = np.empty((m, m))
        chol = np.empty((m, m))
        hess = np.empty((n, n))
        work = np.empty((n, n))
        gtry = np.empty((m, m))
        grad = np.empty(n)
        step = np.empty(n)
        scratch = np.empty(n)
        scratch2 = np.empty(n)
        nu = np.empty(m)
        trial = np.empty(m)

        for i in range(m):
            nu[i] = nu_out[k, i]
        t = t_out[k]
        mu = mu0[k]
        ok = False

        for _ in range(max_iter):
            for i in range(m):
                for j in range(m):
                    gmat[i, j] = direction[k, i, j] + t * sigma[own, i, j]
                gmat[i, i] -= nu[i]
            if not _cholesky(gmat, chol, m):
                break
            _chol_inverse(chol, ainv, scratch, m)
            for i in range(m):
                for j in range(m):
                    total = 0.0
                    for p in range(m):
                        total += ainv[i, p] * sigma[own, p, j]
                    asig[i, j] = total
            for i in range(m):
                for j in range(m):
                    total = 0.0
                    for p in range(m):
                        total += asig[i, p] * ainv[p, j]
                    asa[i, j] = total

            inv_mu = 1.0 / mu
            trace_as = 0.0
            for i in range(m):
                grad[i] = ainv[i, i] - inv_mu - 1.0 / nu[i]
                trace_as += asig[i, i]
            quad = 0.0
            for i in range(m):
                for j in range(m):
                    quad += asa[i, j] * sigma[own, i, j]
            grad[m] = cap[own] * inv_mu - trace_as - 1.0 / t

            trace_h = 0.0
            for i in range(m):
                for j in range(m):
                    hess[i, j] = ainv[i, j] * ainv[i, j]
                hess[i, i] += 1.0 / (nu[i] * nu[i])
                hess[i, m] = -asa[i, i]
                hess[m, i] = -asa[i, i]
                trace_h += hess[i, i]
            hess[m, m] = quad + 1.0 / (t * t)
            trace_h += hess[m, m]
            damp = jitter * trace_h
            for i in range(n):
                hess[i, i] += damp

            for i in range(n):
                scratch[i] = -grad[i]
            if not _spd_solve(hess, scratch, step, work, scratch2, n):
                break

            dec2 = 0.0
            for i in range(n):
                dec2 -= grad[i] * step[i]
            if dec2 < 0.0:
                dec2 = 0.0
            alpha = 1.0 if dec2 <= 0.0625 else 1.0 / (1.0 + np.sqrt(dec2))
            for i in range(m):
                if step[i] < 0.0:
                    cand = -0.99 * nu[i] / step[i]
                    if cand < alpha:
                        alpha = cand
            if step[m] < 0.0:
                cand = -0.99 * t / step[m]
                if cand < alpha:
                    alpha = cand

            moved = False
            halved = False
            for _ in range(50):
                bad = False
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
                    if trial[i] <= 0.0:
                        bad = True
                t_try = t + alpha * step[m]
                if t_try <= 0.0:
                    bad = True
                if not bad:
                    for i in range(m):
                        for j in range(m):
                            gtry[i, j] = direction[k, i, j] + t_try * sigma[own, i, j]
                        gtry[i, i] -= trial[i]
                    if _cholesky(gtry, chol, m):
                        moved = True
                        break
                alpha *= 0.5
                halved = True
            if not moved:
                break
            if halved:
                alpha *= 0.9
                for i in range(m):
                    trial[i] = nu[i] + alpha * step[i]
                t_try = t + alpha * step[m]

            big = abs(t_try)
            for i in range(m):
                nu[i] = trial[i]
                if abs(nu[i]) > big:
                    big = abs(nu[i])
            t = t_try
            if big > divergence or not np.isfinite(big):
                break

            at_end = mu <= mu_end
            if dec2 < (final_tol if at_end else center_tol):
                if at_end:
                    ok = True
                    break
                mu = mu * shrink
                if mu < mu_end:
                    mu = mu_end

        for i in range(m):
            nu_out[k, i] = nu[i]
        t_out[k] = t
        mu_out[k] = mu
        ok_out[k] = ok
