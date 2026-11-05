#!/usr/bin/env python3
"""Gradient-bootstrap inference for grouped granularity bounds.

Resample the grouped covariance and estimate the share's directional
response over a near-optimal set of dual solutions. Both independent
and moving-block resampling are supported.

See near_optimal.py for the derivative solver and NUMERICAL_METHODS.md
for the method and its assumptions.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import mgp_core before cvxpy: it sets MOSEKLM_LICENSE_FILE, which the MOSEK C
# library reads once when cvxpy/mosek first initializes at import.
from mgp_core import (
    sample_cov, solve_grouped_mgp_dual, solve_primal_mtfa,
    numerical_rank, moving_block_indices, installed_solvers, ledermann_bound,
    atomic_write_pickle,
)

from near_optimal import SOLVER_POLICY_VERSION, delta_rate, near_optimal_derivative, require_solved


class ShareDirectionalDerivative:
    """Estimate directional responses at a fixed sample covariance.

    Minimize each direction over dual solutions within a relative objective
    tolerance delta_T = c * sqrt(log(T) / T) of the estimated optimum. Apply
    the quotient rule to convert value responses into share responses.

    The near-optimal set requires a positive-definite covariance.
    Rank and regularity measures are diagnostics, not inputs to this set.
    """

    def __init__(self, sigma: np.ndarray, solver: str, n_obs: int,
                 delta_scale: float = 1.0, rel_tol: float = 1e-3):
        self.m = sigma.shape[0]
        self.solver = solver if solver in installed_solvers() else "CLARABEL"
        self.sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
        self.a = float(self.sigma.sum())
        mgp = solve_grouped_mgp_dual(self.sigma, solver)
        y = np.asarray(mgp["Y"], dtype=float)
        self.ystar = 0.5 * (y + y.T)
        self.gamma_bar = float(mgp["value"])
        self.g_hat = self.gamma_bar / self.a
        self.n_obs = int(n_obs)
        self.delta = delta_rate(self.n_obs, delta_scale)
        self.cap = self.gamma_bar * (1.0 + self.delta)
        # Diagnostics only.
        c, _ = solve_primal_mtfa(self.sigma, solver)
        eig = np.linalg.eigvalsh(0.5 * (c + c.T))[::-1]
        self.r = numerical_rank(eig, rel_tol)
        self.regular = self.r >= ledermann_bound(self.m)
        self._face_solves = 0
        # Count unsuccessful returned directions, not general-solver retries.
        self._face_fallbacks = 0

    def levels(self, directions: np.ndarray) -> np.ndarray:
        """Estimate the SDP-value directional derivative for a batch of directions."""
        directions = np.asarray(directions, dtype=float)
        value, ok = near_optimal_derivative(directions, self.sigma, self.cap)
        self._face_solves += int(ok.sum())
        self._face_fallbacks += int((~ok).sum())
        return require_solved(value, ok, "Gradient-bootstrap derivative")

    def level(self, hmat: np.ndarray) -> float:
        return float(self.levels(np.asarray(hmat, dtype=float)[None])[0])

    def shares(self, directions: np.ndarray) -> np.ndarray:
        directions = np.asarray(directions, dtype=float)
        trace = directions.sum(axis=(-2, -1))
        stats = self.levels(directions) / self.a - self.g_hat * trace / self.a
        if not np.isfinite(stats).all():
            raise RuntimeError("Nonfinite gradient-bootstrap share derivative")
        return stats

    def share(self, hmat: np.ndarray) -> float:
        return float(self.shares(np.asarray(hmat, dtype=float)[None])[0])


def _share_from_cov(sigma: np.ndarray, solver: str) -> float:
    mgp = solve_grouped_mgp_dual(sigma, solver)
    one = np.ones(sigma.shape[0])
    return float(mgp["value"] / (one @ sigma @ one))


def _ci(g_hat, d, t, lo=0.05, hi=0.95):
    """Invert the gradient-bootstrap law: [g_hat - q_hi/sqrt(T), g_hat - q_lo/sqrt(T)]."""
    rt = np.sqrt(t)
    return float(g_hat - np.quantile(d, hi) / rt), float(g_hat - np.quantile(d, lo) / rt)


def moving_block_gradient_ci(
    x: np.ndarray,
    b: int,
    block_length: int,
    rng: np.random.Generator,
    solver: str,
    lo: float = 0.05,
    hi: float = 0.95,
) -> dict[str, object]:
    """Moving-block gradient interval without ordinary-bootstrap SDP refits."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a T by M matrix")
    t, m = x.shape
    if b < 2:
        raise ValueError("b must be at least two")
    if not 1 <= block_length <= t:
        raise ValueError("block_length must lie between one and T")

    sigma_hat = sample_cov(x)
    dd = ShareDirectionalDerivative(sigma_hat, solver, t)
    directions = np.empty((b, m, m), dtype=float)
    for draw in range(b):
        idx = moving_block_indices(t, block_length, rng)
        directions[draw] = np.sqrt(t) * (sample_cov(x[idx]) - sigma_hat)
    stats = dd.shares(directions)
    ci_low, ci_high = _ci(dd.g_hat, stats, t, lo=lo, hi=hi)
    return {
        "g_hat": dd.g_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ucl95": ci_high,
        "directional_stats": stats,
        "B": int(b),
        "block_length": int(block_length),
        "common_rank": int(dd.r),
        "regular": bool(dd.regular),
        "delta_T": float(dd.delta),
        "fallbacks": int(dd._face_fallbacks),
    }


def bootstrap_family(x: np.ndarray, b: int, block_length: int, rng: np.random.Generator,
                     solver: str, progress_every: int = 100,
                     checkpoint_path: Path | None = None,
                     checkpoint_every: int | None = None) -> dict:
    """Plain + gradient (shared iid draws) and block + block-gradient (shared block draws)."""
    t, m = x.shape
    sig_hat = sample_cov(x)
    dd = ShareDirectionalDerivative(sig_hat, solver, t)
    g_hat = dd.g_hat
    progress_every = max(1, int(progress_every))
    checkpoint_every = progress_every if checkpoint_every is None else max(1, int(checkpoint_every))

    idx_iid = [rng.choice(t, size=t, replace=True) for _ in range(b)]
    idx_blk = [moving_block_indices(t, block_length, rng) for _ in range(b)]
    shape = "regular" if dd.regular else "non-regular"
    print(f"[gradient_bootstrap] B={b}, block_length={block_length}, M={m}, "
          f"min over near-optimal set delta_T={dd.delta:.4f} (T={t}); "
          f"diagnostics: common rank r={dd.r}, k_bar={ledermann_bound(m):.1f} ({shape}) "
          f"(shared draws)", flush=True)

    # Draws are only reusable if they were produced by the SAME derivative estimator.
    # Stamp the construction, data and actual resampling indices into every row.
    # Matching dimensions and tuning alone cannot establish that a checkpoint
    # belongs to the current sample and random stream.
    signature = hashlib.sha256()
    signature.update(np.asarray(x.shape, dtype='<i8').tobytes())
    signature.update(np.ascontiguousarray(x, dtype='<f8').tobytes())
    for indices in idx_iid + idx_blk:
        signature.update(np.asarray(indices, dtype='<i8').tobytes())
    estimator_id = (
        f"nearopt_relative_delta={dd.delta:.10g}"
        f"|input_and_draws_sha256={signature.hexdigest()}"
    )
    rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_pickle(checkpoint_path)
        if not existing.empty:
            stamped = existing.get("estimator_id")
            if stamped is None:
                print(f"[gradient_bootstrap] discarding unstamped checkpoint "
                      f"{checkpoint_path} (predates the near-optimal estimator)", flush=True)
                existing = existing.iloc[0:0]
            else:
                bad = int((stamped != estimator_id).sum())
                if bad:
                    print(f"[gradient_bootstrap] discarding {bad} checkpointed draws from a "
                          f"different estimator (want {estimator_id})", flush=True)
                existing = existing[stamped == estimator_id]
        if not existing.empty:
            if not {"solver_policy", "directional_solved"}.issubset(existing.columns):
                print("[gradient_bootstrap] recomputing checkpoint without solver validation", flush=True)
                existing = existing.iloc[0:0]
            else:
                verified = (existing["solver_policy"].eq(SOLVER_POLICY_VERSION)
                            & existing["directional_solved"].eq(True)
                            & np.isfinite(existing[["g_share", "directional_stat"]]).all(axis=1))
                if not verified.all():
                    print(f"[gradient_bootstrap] recomputing {int((~verified).sum())} unverified draws", flush=True)
                existing = existing.loc[verified]
        if not existing.empty:
            existing = existing[
                existing["family"].isin(["iid", "block"])
                & pd.to_numeric(existing["draw"], errors="coerce").between(1, b)
            ]
            keep_block = (
                existing["family"].ne("block")
                | pd.to_numeric(existing["block_length"], errors="coerce").eq(block_length)
            )
            existing = existing[keep_block]
            existing = existing.drop_duplicates(["family", "draw"], keep="last")
            rows_by_key = {
                (str(row["family"]), int(row["draw"])): row.to_dict()
                for _, row in existing.iterrows()
            }
            if rows_by_key:
                n_iid = sum(1 for family, _ in rows_by_key if family == "iid")
                n_block = sum(1 for family, _ in rows_by_key if family == "block")
                print(
                    f"[gradient_bootstrap] resuming from {checkpoint_path} "
                    f"(iid {n_iid}/{b}, block {n_block}/{b})",
                    flush=True,
                )

    start = time.time()
    solved_now = 0
    dirty = False

    def save_checkpoint() -> None:
        ordered = [
            rows_by_key[key]
            for key in sorted(rows_by_key, key=lambda item: (0 if item[0] == "iid" else 1, item[1]))
        ]
        atomic_write_pickle(pd.DataFrame(ordered), checkpoint_path)
        print(
            f"[gradient_bootstrap] checkpointed {len(ordered)}/{2 * b} draw-family rows "
            f"-> {checkpoint_path}",
            flush=True,
        )

    def run_family(family: str, idx_list: list[np.ndarray], family_block_length: int | None) -> None:
        nonlocal dirty, solved_now
        for j, idx in enumerate(idx_list, start=1):
            key = (family, j)
            if key not in rows_by_key:
                sig_b = sample_cov(x[idx])
                g = _share_from_cov(sig_b, solver)
                d = dd.share(np.sqrt(t) * (sig_b - sig_hat))
                if not np.isfinite(g):
                    raise RuntimeError("Nonfinite bootstrap point estimate")
                rows_by_key[key] = {
                    "family": family,
                    "draw": j,
                    "g_share": g,
                    "directional_stat": d,
                    "block_length": family_block_length if family == "block" else np.nan,
                    "directional_common_rank": dd.r,
                    "directional_regular": dd.regular,
                    "estimator_id": estimator_id,
                    "solver_policy": SOLVER_POLICY_VERSION,
                    "directional_solved": True,
                }
                solved_now += 1
                dirty = True
            row = rows_by_key[key]
            if j == 1 or j % progress_every == 0 or j == b:
                family_rows = [
                    rows_by_key[(family, k)]
                    for k in range(1, j + 1)
                    if (family, k) in rows_by_key
                ]
                elapsed = time.time() - start
                print(
                    f"[gradient_bootstrap:{family}] draw {j:>4}/{b}: "
                    f"mean_g={np.mean([r['g_share'] for r in family_rows]):.4f}, "
                    f"mean_d={np.mean([r['directional_stat'] for r in family_rows]):+.4f}, "
                    f"last_d={float(row['directional_stat']):+.4f}, "
                    f"face_solves={dd._face_solves}, solved_now={solved_now}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if checkpoint_path is not None and dirty and (j == 1 or j % checkpoint_every == 0 or j == b):
                save_checkpoint()
                dirty = False

    try:
        run_family("iid", idx_iid, None)
        run_family("block", idx_blk, block_length)
    finally:
        if checkpoint_path is not None and rows_by_key:
            save_checkpoint()

    rows = pd.DataFrame(rows_by_key.values())
    iid_rows = rows[rows["family"] == "iid"].sort_values("draw")
    block_rows = rows[rows["family"] == "block"].sort_values("draw")
    if len(iid_rows) < b or len(block_rows) < b:
        raise RuntimeError(
            f"Gradient bootstrap checkpoint incomplete: iid {len(iid_rows)}/{b}, "
            f"block {len(block_rows)}/{b}."
        )
    g_iid = iid_rows["g_share"].to_numpy(dtype=float)
    d_iid = iid_rows["directional_stat"].to_numpy(dtype=float)
    g_blk = block_rows["g_share"].to_numpy(dtype=float)
    d_blk = block_rows["directional_stat"].to_numpy(dtype=float)
    print(f"[gradient_bootstrap] near-optimal solves={dd._face_solves}, "
          f"fallbacks={dd._face_fallbacks}", flush=True)

    boot_lo, boot_hi = float(np.quantile(g_iid, 0.05)), float(np.quantile(g_iid, 0.95))
    block_lo, block_hi = float(np.quantile(g_blk, 0.05)), float(np.quantile(g_blk, 0.95))
    grad_lo, grad_hi = _ci(g_hat, d_iid, t)
    bgrad_lo, bgrad_hi = _ci(g_hat, d_blk, t)
    return {
        "g_hat": g_hat,
        "bootstrap_mean": float(np.mean(g_iid)), "bootstrap_low": boot_lo, "bootstrap_high": boot_hi,
        "block_mean": float(np.mean(g_blk)), "block_low": block_lo, "block_high": block_hi,
        "gradient_low": grad_lo, "gradient_high": grad_hi,
        "block_gradient_low": bgrad_lo, "block_gradient_high": bgrad_hi,
        "gradient_ucl95": grad_hi, "block_gradient_ucl95": bgrad_hi,
        "directional_common_rank": dd.r,
        "directional_regular": dd.regular,
        "directional_delta_T": dd.delta,
        "directional_fallbacks": dd._face_fallbacks,
        "B": b, "block_length": block_length,
    }
