#!/usr/bin/env python3
"""Shared routines for estimating granularity bounds.

Includes panel preparation, LPT grouping, semidefinite optimization, inference,
and numerical diagnostics. Factor-rank tests live in factor_rank.py; plotting
helpers live in plot_style.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT / "output"

CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
RUN_OUTPUT_CACHE = CACHE_DIR / "run_outputs"
RUN_OUTPUT_CACHE.mkdir(exist_ok=True)
PANEL_CACHE_DIR = CACHE_DIR / "clean_panels"
PANEL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

LICENSE = SCRIPT_DIR / "mosek.lic"
if LICENSE.exists():
    os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(LICENSE))

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm


Z95 = float(norm.ppf(0.95))
ANALYSIS_BOOTSTRAP_DRAWS = 1000
_DUAL_PROBLEM_CACHE: dict[int, tuple[cp.Problem, cp.Parameter, cp.Variable]] = {}
_INSTALLED_SOLVERS: set[str] | None = None


def atomic_write_pickle(obj: object, path: Path) -> None:
    """Write a pickle through a temporary sibling so interrupted runs leave a valid file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.to_pickle(obj, tmp)
    tmp.replace(path)


def _content_fingerprint(p: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the complete input, including changes outside its endpoints."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def source_signature(paths: list[Path]) -> list[dict[str, object]]:
    out = []
    for path in paths:
        p = Path(path)
        out.append(
            {
                "path": str(p.resolve()),
                "size": int(p.stat().st_size),
                "fingerprint": _content_fingerprint(p),
            }
        )
    return out


def clean_panel_cache_path(prefix: str, params: dict[str, object], sources: list[Path]) -> Path:
    """Key cleaned data on full input hashes and the caller's cleaning parameters.

    Increment params['version'] when the panel's cleaning rules change.
    """
    payload = {
        "prefix": prefix,
        "params": params,
        "sources": source_signature(sources),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    safe_prefix = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in prefix)
    return PANEL_CACHE_DIR / f"{safe_prefix}_{digest}.pkl"


def read_clean_panel_cache(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    print(f"[cache] reading clean panel {path}", flush=True)
    return pd.read_pickle(path)


def write_clean_panel_cache(path: Path, payload: dict[str, object]) -> None:
    atomic_write_pickle(payload, path)
    print(f"[cache] wrote clean panel {path}", flush=True)


def make_panel_rng(base_seed: int, name: str) -> np.random.Generator:
    """Create a panel-specific random stream independent of run order."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    name_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), name_int]))


def choose_solver(requested: str) -> str:
    if requested != "auto":
        return requested.upper()
    installed = installed_solvers()
    # Prefer CLARABEL, then MOSEK, then SCS.
    for solver in ("CLARABEL", "MOSEK", "SCS"):
        if solver in installed:
            return solver
    raise RuntimeError(f"No supported SDP solver installed. Installed: {sorted(installed)}")


def installed_solvers() -> set[str]:
    global _INSTALLED_SOLVERS
    if _INSTALLED_SOLVERS is None:
        _INSTALLED_SOLVERS = set(cp.installed_solvers())
    return _INSTALLED_SOLVERS


def align_sizes_to_panel(y: pd.DataFrame, sizes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop units with missing or non-positive sizes and sort by decreasing size.

    Shared by every consumer of a ``SourcePanel``-style (units x time, sizes)
    pair so the unit ordering fed into LPT is identical across the run scripts
    and the diagnostic scripts.
    """
    sizes = sizes.reindex(y.index).astype(float)
    keep = sizes.gt(0) & np.isfinite(sizes)
    y = y.loc[keep]
    sizes = sizes.loc[keep]
    y = y.loc[sizes.sort_values(ascending=False).index]
    return y, sizes.loc[y.index]


def make_panel_contiguous(d: pd.DataFrame, time_var: str, unit_var: str) -> pd.DataFrame:
    """Keep, for each unit observed at the last date, its final contiguous block.

    Units absent at the last date are dropped; for the rest, only the trailing
    run of consecutive observations ending at the last date is retained.
    """
    if tuple(d.index.names) != (time_var, unit_var):
        raise ValueError(f"Expected index ({time_var}, {unit_var}), got {d.index.names}.")
    if not d.index.is_unique:
        raise ValueError("Panel index is not unique.")
    if d.isna().any(axis=None):
        raise ValueError("Panel has missing values before contiguity pass.")
    if not d.index.is_monotonic_increasing:
        d = d.sort_index()

    dates = d.index.unique(time_var).sort_values()
    last_date = dates[-1]
    last_units = d.loc[last_date].index.unique(unit_var)
    d = d[d.index.get_level_values(unit_var).isin(last_units)].copy()

    time_index = d.index.get_level_values(time_var)
    unit_index = d.index.get_level_values(unit_var)
    reverse_rank = pd.Series(np.arange(len(dates) - 1, -1, -1), index=dates)
    contiguous = time_index.map(reverse_rank) == d.groupby(level=unit_var).cumcount(ascending=False)
    first_date = pd.Series(time_index[contiguous], index=unit_index[contiguous]).groupby(level=0, sort=False).min()
    return d.loc[time_index >= unit_index.map(first_date)]


def lpt_buckets(job_sizes: np.ndarray, m: int) -> tuple[list[list[int]], np.ndarray]:
    """Longest-processing-time scheduling: assign each job to the lightest bucket."""
    order = np.argsort(job_sizes)[::-1]
    loads = np.zeros(m)
    buckets: list[list[int]] = [[] for _ in range(m)]
    for i in order:
        j = int(np.argmin(loads))
        buckets[j].append(int(i))
        loads[j] += float(job_sizes[i])
    return buckets, loads


def get_dual_problem(m: int) -> tuple[cp.Problem, cp.Parameter, cp.Variable]:
    if m not in _DUAL_PROBLEM_CACHE:
        sigma = cp.Parameter((m, m), symmetric=True)
        y = cp.Variable((m, m), symmetric=True)
        objective = cp.Minimize(cp.sum(cp.multiply(sigma, y)))
        constraints = [y >> 0, cp.diag(y) >= 1]
        prob = cp.Problem(objective, constraints)
        _DUAL_PROBLEM_CACHE[m] = (prob, sigma, y)
    return _DUAL_PROBLEM_CACHE[m]


def solve_grouped_mgp_dual(sigma: np.ndarray, solver: str) -> dict[str, object]:
    sigma = 0.5 * (sigma + sigma.T)
    m = sigma.shape[0]
    prob, sigma_param, y_var = get_dual_problem(m)
    # Solve a unit-scaled problem for conditioning: Ystar is scale-invariant and the
    # value scales linearly, so feeding Sigma/s and reading the value off the original
    # Sigma is exact but far better conditioned at the tiny grouped-covariance scale.
    scale = float(np.linalg.norm(sigma)) or 1.0
    sigma_param.value = sigma / scale

    attempted = []
    installed = installed_solvers()
    # Chosen solver first, then the other accurate cones, with SCS only as last resort.
    solvers = [solver] + [s for s in ("CLARABEL", "MOSEK", "SCS")
                          if s != solver and s in installed]

    last_error = None
    for candidate in solvers:
        attempted.append(candidate)
        try:
            prob.solve(solver=candidate, warm_start=True, verbose=False)
        except Exception as exc:  # pragma: no cover - solver/environment dependent
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and y_var.value is not None:
            y = np.asarray(y_var.value, dtype=float)
            y = 0.5 * (y + y.T)
            value = float(np.sum(y * sigma))
            eig_y = np.linalg.eigvalsh(y)
            return {
                "value": value,
                "Y": y,
                "status": prob.status,
                "solver": candidate,
                "attempted": "+".join(attempted),
                "dual_min_eig": float(eig_y.min()),
                "dual_min_diag": float(np.diag(y).min()),
            }
        last_error = prob.status

    raise RuntimeError(f"MGP dual solve failed after {attempted}: {last_error}")


def sample_cov(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False, ddof=1)


def mean_pairwise_correlation(values: np.ndarray) -> float:
    """Average off-diagonal correlation of a units x time panel.

    Off-diagonal only: the mean of the full correlation matrix (as reported by
    some replication packages) mixes in the N unit diagonal entries and is
    mechanically 1/N + (1-1/N) times this number.
    """
    correlation = np.corrcoef(np.asarray(values, dtype=float))
    return float(np.nanmean(correlation[np.triu_indices_from(correlation, k=1)]))


def mean_ar1(values: np.ndarray) -> float:
    """Average first-order autocorrelation across the units of a units x time panel."""
    rows = np.asarray(values, dtype=float)
    return float(np.nanmean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in rows]))


DAILY_RETURN_MAX_GAP = 3


def fill_short_internal_return_gaps(
    frame: pd.DataFrame, max_gap: int = DAILY_RETURN_MAX_GAP, *, carry_forward: bool = False,
) -> pd.DataFrame:
    """Set complete internal missing-return runs of at most max_gap dates to zero.

    Set carry_forward=True for wealth levels before taking log differences.
    Columns must be the ordered observation calendar. Runs at either endpoint and
    runs longer than the limit remain missing; several eligible runs may be
    filled for the same unit. Observed returns are never changed.
    """
    if max_gap < 0 or int(max_gap) != max_gap:
        raise ValueError("max_gap must be a nonnegative integer")
    values = frame.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    for row in values:
        missing = np.isnan(row)
        boundaries = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(np.int8)))
        for start, stop in zip(boundaries[::2], boundaries[1::2]):
            if start > 0 and stop < len(row) and stop - start <= max_gap:
                row[start:stop] = row[start - 1] if carry_forward else 0.0
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def winsorize(frame: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Pooled percentile winsorization of a growth panel at [q, 1-q].

    Replaces the earlier ad-hoc absolute clip; q=0.01 is the standard Compustat
    1st/99th-percentile rule. Quantiles are taken over the pooled (unit, time)
    values of the balanced panel.
    """
    flat = frame.to_numpy(dtype=float).ravel()
    lo, hi = np.nanquantile(flat, [q, 1.0 - q])
    return frame.clip(lower=lo, upper=hi)


def standardize_panel(frame: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """The paper's final cleaning for log-growth and return panels.

    Winsorize at the pooled [q, 1-q] percentiles of the balanced panel, then
    subtract the pooled grand mean. The network applications do not use it:
    they analyze the primitive shocks exactly as the replication package or
    structural recovery delivers them. Additive-flow applications also bypass
    this function to preserve the exact flow total.
    """
    clipped = winsorize(frame, q)
    grand_mean = float(np.nanmean(clipped.to_numpy(dtype=float)))
    return clipped - grand_mean


def analysis_block_length(t: int, overlap_horizon: int = 1) -> int:
    """Paper-wide moving-block rule, bounded by the observed time dimension."""
    if t < 1:
        raise ValueError("t must be positive")
    if overlap_horizon < 1:
        raise ValueError("overlap_horizon must be positive")
    return min(int(t), max(int(overlap_horizon), int(np.ceil(t ** (1.0 / 3.0)))))


def remove_periodic_means(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Residualize each unit series on periodic fixed effects."""
    if period < 2:
        raise ValueError("period must be at least two")
    if isinstance(frame.columns, pd.PeriodIndex) and period == 4:
        phase = frame.columns.quarter.to_numpy()
    elif isinstance(frame.columns, pd.PeriodIndex) and period == 12:
        phase = frame.columns.month.to_numpy()
    else:
        phase = np.arange(frame.shape[1]) % period
    adjusted = frame.astype(float).copy()
    for value in np.unique(phase):
        columns = adjusted.columns[phase == value]
        adjusted.loc[:, columns] = adjusted.loc[:, columns].sub(
            adjusted.loc[:, columns].mean(axis=1), axis=0
        )
    return adjusted


def rule_m(n_units: int, frequency: str = "") -> int:
    """Paper-wide grouping dimension: M=10, or M=30 for daily data.

    The number of units is an upper bound when an application has fewer units.
    """
    daily = "daily" in str(frequency).lower()
    return int(min(int(n_units), 30 if daily else 10))


def rule_summary_stem(stem: str) -> str:
    """Return ``f"{stem}_M{m}"`` for the cached paper-specification run.

    Reads N and M back from the ``*_summary.pkl`` files under
    RUN_OUTPUT_CACHE, so consumers never hard-code an M.
    """
    import pandas as pd
    hits = []
    for path in sorted(RUN_OUTPUT_CACHE.glob(f"{stem}_M*_summary.pkl")):
        s = pd.read_pickle(path).iloc[0]
        if int(s["M"]) == rule_m(int(s["N"]), str(s.get("frequency", ""))):
            hits.append(path.name[: -len("_summary.pkl")])
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected one rule-M summary for {stem!r} in {RUN_OUTPUT_CACHE}, found {hits}"
        )
    return hits[0]


def ledermann_bound(m: int) -> float:
    """Ledermann bound k_bar(M); the grouped GLB is regular when rank(C) >= k_bar(M)."""
    return (2 * m + 1 - np.sqrt(8 * m + 1)) / 2.0


def solve_primal_mtfa(sigma: np.ndarray, solver: str) -> tuple[np.ndarray, float]:
    """Recover the common part C = Sigma - D at the MTFA optimum (D diagonal, >= 0)."""
    sigma = 0.5 * (sigma + sigma.T)
    m = sigma.shape[0]
    scale = float(np.linalg.norm(sigma)) or 1.0
    sigma_unit = sigma / scale
    d = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(d)), [sigma_unit - cp.diag(d) >> 0])

    candidates = [solver]
    installed = installed_solvers()
    for fallback in ("CLARABEL", "SCS"):
        if fallback != solver and fallback in installed:
            candidates.append(fallback)
    last_error = None
    for candidate in candidates:
        try:
            prob.solve(solver=candidate, verbose=False)
        except Exception as exc:  # pragma: no cover - solver/environment dependent
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and d.value is not None:
            d_opt = np.asarray(d.value, dtype=float).ravel()
            c = scale * (sigma_unit - np.diag(d_opt))
            return 0.5 * (c + c.T), float(scale * prob.value)
        last_error = prob.status
    raise RuntimeError(f"Primal MTFA solve failed: {last_error}")


def numerical_rank(eigs: np.ndarray, rel_tol: float = 1e-3) -> int:
    pos = eigs[eigs > 0]
    if pos.size == 0:
        return 0
    return int(np.sum(eigs >= rel_tol * eigs.max()))


def mtfa_gap_diagnostics(sigma: np.ndarray, common: np.ndarray, rel_tol: float = 1e-3) -> dict[str, float]:
    """Plug-in diagnostics for the deterministic looseness bound.

    The population bound in Proposition 8 depends on the true grouped diagonal
    ``D0`` and common subspace. Empirically we report the same ingredients at the
    MTFA decomposition recovered from ``sigma``. These are diagnostics for how
    favorable the observed active decomposition is, not a second confidence
    statement.
    """
    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    common = 0.5 * (np.asarray(common, dtype=float) + np.asarray(common, dtype=float).T)
    m = sigma.shape[0]
    eig, vec = np.linalg.eigh(common)
    order = np.argsort(eig)[::-1]
    eig = eig[order]
    vec = vec[:, order]
    r = numerical_rank(eig, rel_tol)

    d = np.diag(sigma - common).astype(float)
    d_pos = np.clip(d, 0.0, None)
    d_bar = float(d_pos.mean())
    d_max = float(d_pos.max()) if d_pos.size else float("nan")
    d_ratio = d_max / d_bar if d_bar > 0 else float("inf")

    if r <= 0:
        h_max = 0.0
    elif d_bar <= 0:
        h_max = float("nan")
    elif np.any(d_pos <= 0):
        # Heywood boundary: some bucket's recovered idiosyncratic variance is zero,
        # so the whitened subspace concentrates there and h_max -> 1 in the limit.
        h_max = 1.0
    else:
        rmat = vec[:, :r] * np.sqrt(np.maximum(eig[:r], 0.0))
        w = rmat / np.sqrt(d_pos)[:, None]
        q, _ = np.linalg.qr(w, mode="reduced")
        h_max = float(np.sum(q[:, :r] ** 2, axis=1).max())

    if not np.isfinite(h_max) or h_max >= 1.0:
        gap_factor = float("inf")
    else:
        gap_factor = d_ratio * (1.0 - r / m) / (1.0 - h_max)

    return {
        "gap_dmax_over_dbar": float(d_ratio),
        "gap_hmax": float(h_max),
        "gap_bound_factor": float(gap_factor),
    }


def regularity_report(sigma: np.ndarray, solver: str, rel_tol: float = 1e-3) -> dict[str, object]:
    """Regular/non-regular and tight/loose diagnostics for one grouped covariance.

    Recovers the common part C = Sigma - D, takes its rank r at relative tolerance
    rel_tol, and compares to the Ledermann bound k_bar(M) (regular iff r >= k_bar)
    and to the tightness corank threshold 2 sqrt(M) (tight iff M - r > 2 sqrt(M)).
    rank_gap_ratio is the eigenvalue ratio at the rank cutoff: large means r is
    cleanly pinned, ~1 means a smooth spectrum with no well-defined rank.
    """
    m = sigma.shape[0]
    c, primal_value = solve_primal_mtfa(sigma, solver)
    eig = np.linalg.eigvalsh(c)[::-1]
    r = numerical_rank(eig, rel_tol)
    kbar = ledermann_bound(m)
    corank_threshold = 2.0 * np.sqrt(m)
    gap = float(eig[r - 1] / eig[r]) if 0 < r < m and eig[r] > 0 else float("inf")
    gap_diag = mtfa_gap_diagnostics(sigma, c, rel_tol)
    return {
        "ledermann_kbar": float(kbar),
        "common_rank": int(r),
        "regular": bool(r >= kbar),
        "corank": int(m - r),
        "tight": bool((m - r) > corank_threshold),
        "corank_threshold_2sqrtM": float(corank_threshold),
        "rank_gap_ratio": gap,
        "common_share_of_trace": float(eig.clip(min=0).sum() / np.trace(sigma)),
        "primal_value": primal_value,
        **gap_diag,
    }



def build_lpt_grouped_data(
    y: np.ndarray, shares: np.ndarray, m: int
) -> tuple[np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    """Group units into M LPT buckets on squared shares and aggregate.

    Returns the grouped T x M observations x_{m,t} = sum_{i in B_m} S_i y_{i,t}.
    Flow callers supply size-normalized outcomes, just like growth panels.
    """
    buckets, loads = lpt_buckets(shares**2, m)
    g = np.zeros((m, len(shares)))
    for j, bucket in enumerate(buckets):
        g[j, bucket] = shares[bucket]
    x_grouped = y.T @ g.T
    return x_grouped, buckets, loads, g


def mgp_share(x: np.ndarray, solver: str) -> dict[str, object]:
    t, m = x.shape
    sigma = sample_cov(x)
    mgp = solve_grouped_mgp_dual(sigma, solver)
    y_dual = np.asarray(mgp["Y"], dtype=float)
    one = np.ones(m)
    sigma_a2 = float(one @ sigma @ one)
    gamma = float(mgp["value"])
    share = gamma / sigma_a2

    u = x - x.mean(axis=0, keepdims=True)
    q = np.einsum("ti,ij,tj->t", u, y_dual, u)
    aggregate = u @ one
    r_if = (q - share * aggregate**2) / sigma_a2
    se = float(np.std(r_if, ddof=1) / np.sqrt(t))

    return {
        "mgp_share": share,
        "mgp_level": gamma,
        "aggregate_variance": sigma_a2,
        "se_share": se,
        "normal_ucl95": share + Z95 * se,
        "solver": mgp["solver"],
        "status": mgp["status"],
        "dual_min_eig": mgp["dual_min_eig"],
        "dual_min_diag": mgp["dual_min_diag"],
    }


def moving_block_indices(t: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 1:
        return rng.choice(t, size=t, replace=True)
    if block_length > t:
        raise ValueError(f"Block length {block_length} exceeds sample length {t}.")
    n_blocks = int(np.ceil(t / block_length))
    starts = rng.integers(0, t - block_length + 1, size=n_blocks)
    idx = starts[:, None] + np.arange(block_length)[None, :]
    return idx.ravel()[:t]


def resample_mgp(
    x: np.ndarray,
    b: int,
    solver: str,
    rng: np.random.Generator,
    mode: str,
    progress_every: int,
    block_length: int | None = None,
    extra_stats=None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int | None = None,
) -> pd.DataFrame:
    if b <= 0:
        return pd.DataFrame()

    t, m = x.shape
    sigma = sample_cov(x) if mode == "gaussian_covariance" else None
    progress_every = max(1, int(progress_every))
    checkpoint_every = progress_every if checkpoint_every is None else max(1, int(checkpoint_every))

    existing_by_draw: dict[int, dict[str, object]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_pickle(checkpoint_path)
        if not existing.empty:
            if "mode" in existing:
                existing = existing[existing["mode"] == mode]
            if mode == "moving_block_bootstrap":
                if "block_length" not in existing:
                    existing = existing.iloc[0:0]
                else:
                    existing = existing[
                        pd.to_numeric(existing["block_length"], errors="coerce").eq(block_length)
                    ]
            existing = existing[pd.to_numeric(existing["draw"], errors="coerce").between(1, b)]
            existing = existing.drop_duplicates("draw", keep="last")
            existing_by_draw = {
                int(row["draw"]): row.to_dict()
                for _, row in existing.iterrows()
            }
            if existing_by_draw:
                print(
                    f"[{mode}] resuming from {checkpoint_path} "
                    f"({len(existing_by_draw)}/{b} completed)",
                    flush=True,
                )

    draws: list[dict[str, object]] = []
    start = time.time()
    solved = 0
    dirty = False
    extra = f", block_length={block_length}" if mode == "moving_block_bootstrap" else ""
    print(f"[{mode}] starting {b} resamples{extra}", flush=True)
    for j in range(1, b + 1):
        if mode == "bootstrap":
            idx = rng.choice(t, size=t, replace=True)
            xb = x[idx]
        elif mode == "moving_block_bootstrap":
            if block_length is None:
                raise ValueError("moving_block_bootstrap requires block_length.")
            idx = moving_block_indices(t, block_length, rng)
            xb = x[idx]
        elif mode == "gaussian_covariance":
            idx = None
            xb = rng.multivariate_normal(mean=np.zeros(m), cov=sigma, size=t)
        else:
            raise ValueError(f"Unknown resampling mode {mode!r}")

        if j in existing_by_draw:
            row = dict(existing_by_draw[j])
        else:
            row = mgp_share(xb, solver)
            if extra_stats is not None:
                row.update(extra_stats(xb, idx))
            row.update(
                {
                    "mode": mode,
                    "draw": j,
                    "block_length": block_length if mode == "moving_block_bootstrap" else np.nan,
                }
            )
            existing_by_draw[j] = dict(row)
            solved += 1
            dirty = True
        draws.append(row)

        if j == 1 or j % progress_every == 0 or j == b:
            elapsed = time.time() - start
            print(
                f"[{mode}] draw {j:>4}/{b}: "
                f"mean={np.mean([d['mgp_share'] for d in draws]):.4f}, "
                f"last={row['mgp_share']:.4f}, solved_now={solved}, elapsed={elapsed:.1f}s",
                flush=True,
            )
        if checkpoint_path is not None and dirty and (j == 1 or j % checkpoint_every == 0 or j == b):
            checkpoint = pd.DataFrame(
                [existing_by_draw[k] for k in sorted(existing_by_draw)]
            )
            atomic_write_pickle(checkpoint, checkpoint_path)
            print(
                f"[{mode}] checkpointed {len(checkpoint)}/{b} draws -> {checkpoint_path}",
                flush=True,
            )
            dirty = False

    out = pd.DataFrame(draws)
    if checkpoint_path is not None:
        atomic_write_pickle(out, checkpoint_path)
    return out


def summarize_distribution(values: pd.Series, prefix: str) -> dict[str, float]:
    if values.empty:
        return {}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)),
        f"{prefix}_p05": float(values.quantile(0.05)),
        f"{prefix}_p50": float(values.quantile(0.50)),
        f"{prefix}_p95": float(values.quantile(0.95)),
    }



def default_subsample_length(t: int, m: int, exponent: float = 2.0 / 3.0) -> int:
    """Subsample size l_T ~ T^exponent (Politis--Romano rate).

    The subsample covariance may be singular when T is short relative to M; the
    MGP SDP can still be evaluated on that covariance. Capping below T keeps the
    rule usable for short or diagnostic panels where an M+2 floor would make
    subsampling impossible.
    """
    if t <= 2:
        raise ValueError(f"Subsampling needs T>2, got T={t}.")
    return int(min(t - 1, max(3, np.floor(t ** exponent))))


def subsample_indices(t: int, ell: int, n_sub: int, rng: np.random.Generator,
                      overlapping: bool = True) -> list[np.ndarray]:
    """Subsample index sets of size ell.

    overlapping=True returns contiguous blocks (Politis--Romano subsampling for a
    dependent series): all T-ell+1 blocks if there are at most n_sub of them, else
    a random subset of n_sub block start points. overlapping=False returns random
    size-ell subsets without replacement (the i.i.d. case of ST00 Assumption 1).
    """
    if ell > t:
        raise ValueError(f"Subsample size {ell} exceeds T={t}.")
    if overlapping:
        starts = np.arange(t - ell + 1)
        if len(starts) > n_sub:
            starts = np.sort(rng.choice(starts, size=n_sub, replace=False))
        return [np.arange(s, s + ell) for s in starts]
    return [np.sort(rng.choice(t, size=ell, replace=False)) for _ in range(n_sub)]


def subsample_mgp(x: np.ndarray, ell: int, n_sub: int, solver: str,
                  rng: np.random.Generator, progress_every: int,
                  g_hat: float | None = None, overlapping: bool = True,
                  checkpoint_path: Path | None = None,
                  checkpoint_every: int | None = None) -> tuple[pd.DataFrame, float]:
    """Subsampling distribution of the grouped share (Politis--Romano / Fang--Santos).

    For each subsample b of size ell, recompute the grouped share g_{l,b} on the
    subsample covariance and form the recentered, rescaled statistic
        L_{l,b} = sqrt(l) (g_{l,b} - g_hat).
    The empirical law of {L_{l,b}} is consistent for the limit law of
    sqrt(T)(g_hat - g) for the directionally differentiable share functional, in
    both the regular and non-regular cases (where the bootstrap is inconsistent).
    Returns the per-subsample draws and the full-sample share g_hat.
    """
    t, m = x.shape
    if ell < 2:
        raise ValueError(f"Subsample size ell={ell} must be at least 2.")
    if g_hat is None:
        g_hat = float(mgp_share(x, solver)["mgp_share"])
    idx_list = subsample_indices(t, ell, n_sub, rng, overlapping)
    progress_every = max(1, int(progress_every))
    checkpoint_every = progress_every if checkpoint_every is None else max(1, int(checkpoint_every))
    existing_by_subsample: dict[int, dict[str, object]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_pickle(checkpoint_path)
        if not existing.empty:
            existing = existing[pd.to_numeric(existing["subsample"], errors="coerce").between(1, len(idx_list))]
            existing = existing.drop_duplicates("subsample", keep="last")
            existing_by_subsample = {
                int(row["subsample"]): row.to_dict()
                for _, row in existing.iterrows()
            }
            if existing_by_subsample:
                print(
                    f"[subsample] resuming from {checkpoint_path} "
                    f"({len(existing_by_subsample)}/{len(idx_list)} completed)",
                    flush=True,
                )
    rows: list[dict[str, object]] = []
    start = time.time()
    solved = 0
    dirty = False
    kind = "overlapping blocks" if overlapping else "random subsets"
    print(f"[subsample] ell={ell}, n={len(idx_list)} ({kind})", flush=True)
    for j, idx in enumerate(idx_list, 1):
        if j in existing_by_subsample:
            row = dict(existing_by_subsample[j])
        else:
            g_b = float(mgp_share(x[idx], solver)["mgp_share"])
            row = {
                "subsample": j,
                "ell": ell,
                "g_sub": g_b,
                "L": float(np.sqrt(ell) * (g_b - g_hat)),
            }
            existing_by_subsample[j] = dict(row)
            solved += 1
            dirty = True
        rows.append(row)
        if j == 1 or j % progress_every == 0 or j == len(idx_list):
            print(
                f"[subsample] {j:>4}/{len(idx_list)} solved_now={solved} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        if checkpoint_path is not None and dirty and (j == 1 or j % checkpoint_every == 0 or j == len(idx_list)):
            checkpoint = pd.DataFrame(
                [existing_by_subsample[k] for k in sorted(existing_by_subsample)]
            )
            atomic_write_pickle(checkpoint, checkpoint_path)
            print(
                f"[subsample] checkpointed {len(checkpoint)}/{len(idx_list)} draws -> {checkpoint_path}",
                flush=True,
            )
            dirty = False
    out = pd.DataFrame(rows)
    if checkpoint_path is not None:
        atomic_write_pickle(out, checkpoint_path)
    return out, float(g_hat)


def subsampling_summary(draws: pd.DataFrame, g_hat: float, t: int, alpha: float = 0.10) -> dict[str, float]:
    """Two-sided (1-alpha) subsampling CI and one-sided UCL for the share.

    With w_q the empirical q-quantile of L = sqrt(l)(g_l - g_hat),
        CI(1-alpha) = [g_hat - w_{1-alpha/2}/sqrt(T), g_hat - w_{alpha/2}/sqrt(T)],
        UCL(1-alpha) = g_hat - w_{alpha}/sqrt(T).
    With alpha=0.10 the endpoints are the reported 5 and 95 percent points.
    """
    if draws.empty:
        return {}
    L = draws["L"].to_numpy(dtype=float)
    rt = np.sqrt(t)
    w_lo = float(np.quantile(L, alpha / 2.0))
    w_hi = float(np.quantile(L, 1.0 - alpha / 2.0))
    w_med = float(np.quantile(L, 0.5))
    w_a = float(np.quantile(L, alpha))
    return {
        "subsample_ell": int(draws["ell"].iloc[0]),
        "subsample_n": int(len(draws)),
        "subsample_p05": float(g_hat - w_hi / rt),
        "subsample_p50": float(g_hat - w_med / rt),
        "subsample_p95": float(g_hat - w_lo / rt),
        "subsample_ucl95": float(g_hat - w_a / rt),
    }
