#!/usr/bin/env python3
"""Direct-to-MOSEK solver for the near-optimal-set SDP, used for numerical retries.

The directional derivative of the grouped ceiling is

    min <Y,H>   s.t.   Y >= 0,   diag(Y) >= 1,   <Y,Sigma_hat> <= cap,

with ``cap = (1 + delta_T) Gamma_hat`` (see ``near_optimal``). The barrier method
in ``mgp_kernels.barrier_nearopt`` solves the great majority of these, but its dual can
stall on badly conditioned grouped covariances. For positive-definite ``Sigma_hat``
and a cap above the optimum, the set is nonempty and bounded:
``<Y,Sigma_hat> >= lambda_min(Sigma_hat) tr(Y)`` bounds the trace. This does not
guarantee numerical success; unresolved directions retain a failure flag.

Only the objective varies across the directions of one replication, so the task is built
once and each direction rewrites ``bar c`` against a fixed basis of elementary symmetric
matrices. Building a fresh matrix per direction instead would grow the task by one
matrix per solve and make MOSEK re-analyze what it has already seen; going through cvxpy
would rebuild the constraint data every time, which measured about thirty times slower.
"""

from __future__ import annotations

from functools import cache

import numpy as np

# mgp_core first: it sets MOSEKLM_LICENSE_FILE before the MOSEK library is loaded.
from mgp_core import installed_solvers

try:
    import mosek

    MOSEK_AVAILABLE = True
    MosekError = mosek.Error
except ImportError:  # pragma: no cover - environment dependent
    MOSEK_AVAILABLE = False
    MosekError = RuntimeError


@cache
def licensed_mosek() -> bool:
    """Whether CVXPY can use the optional commercial solver in this run."""
    if not MOSEK_AVAILABLE or "MOSEK" not in installed_solvers():
        return False
    try:
        with mosek.Env() as env:
            env.checkoutlicense(mosek.feature.pton)
    except MosekError:
        return False
    return True


def _lower_triplets(a: np.ndarray, tol: float = 0.0):
    """Lower-triangular (row, col, value) triplets of a symmetric matrix, MOSEK's format."""
    q = a.shape[0]
    rows, cols = np.tril_indices(q)
    vals = a[rows, cols]
    keep = np.abs(vals) > tol
    return rows[keep].tolist(), cols[keep].tolist(), vals[keep].tolist()


class NearOptimalSolver:
    """min <Y,H> over the near-optimal dual set, for many directions at one Sigma_hat."""

    def __init__(self, sigma: np.ndarray, cap: float):
        if not MOSEK_AVAILABLE:  # pragma: no cover - environment dependent
            raise RuntimeError("MOSEK is required for the near-optimal backstop.")
        sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
        self.m = m = sigma.shape[0]
        self._env = mosek.Env()
        self._task = mosek.Task(self._env, 0, 0)
        task = self._task
        task.appendbarvars([m])
        task.appendcons(m + 1)
        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        task.putintparam(mosek.iparam.num_threads, 1)
        for i in range(m):
            sym = task.appendsparsesymmat(m, [i], [i], [1.0])
            task.putbaraij(i, 0, [sym], [1.0])
            task.putconbound(i, mosek.boundkey.lo, 1.0, +0.0)
        sym = task.appendsparsesymmat(m, *_lower_triplets(sigma))
        task.putbaraij(m, 0, [sym], [1.0])
        task.putconbound(m, mosek.boundkey.up, -0.0, float(cap))
        self._rows, self._cols = np.tril_indices(m)
        self._obj_syms = [
            task.appendsparsesymmat(m, [int(r)], [int(c)], [1.0])
            for r, c in zip(self._rows, self._cols)
        ]

    def value(self, direction: np.ndarray) -> float | None:
        """The derivative for one direction, or None if MOSEK does not reach optimality."""
        h = 0.5 * (np.asarray(direction, dtype=float) + np.asarray(direction, dtype=float).T)
        task = self._task
        task.putbarcj(0, self._obj_syms, h[self._rows, self._cols].tolist())
        task.optimize()
        if task.getsolsta(mosek.soltype.itr) != mosek.solsta.optimal:
            return None
        return float(task.getprimalobj(mosek.soltype.itr))

    def values(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty(len(directions))
        failed = np.zeros(len(directions), dtype=bool)
        for k, h in enumerate(directions):
            got = self.value(h)
            if got is None:
                out[k], failed[k] = np.nan, True
            else:
                out[k] = got
        return out, failed

    def close(self) -> None:
        self._task.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
