"""Alternative firm-to-bucket partitions, compared against LPT.

Section~\\ref{sec:grouping} of the draft motivates LPT (longest-processing-time
scheduling on squared sizes) as a way to "equalize the sum of squared sizes
across buckets, while mixing units across buckets" -- but never checks LPT
against the alternatives it is implicitly compared to. This module supplies
those alternatives, all consumed by ``run_grouping_schemes.py``:

  core_fringe_buckets   the naive scheme the draft's prose argues against:
                        the M-1 largest units as singletons, everyone else
                        in one leftover bucket.
  round_robin_buckets   deterministic but not load-aware: rank units by size
                        and deal them into buckets 1..M, 1..M, ... in turn.
  lpt_buckets           mgp_core's incumbent (imported, not reimplemented).
  exact_h2_buckets      the H^2-optimal partition, certified by CP-SAT.

All take a ``sizes`` array of *shares* (not squared) and return
``(buckets, loads)`` in ``mgp_core.lpt_buckets``'s own format: ``buckets`` is
a length-M list of index lists, ``loads`` is the array of bucket sums of
SQUARED sizes, i.e. ``loads[m] = sum_{i in B_m} sizes[i]**2``.  ``exact_h2_buckets``
appends a third element, its solve report.
"""

from __future__ import annotations

import numpy as np

from mgp_core import lpt_buckets as _lpt_buckets_squared

__all__ = [
    "lpt_buckets",
    "core_fringe_buckets",
    "round_robin_buckets",
    "exact_h2_buckets",
    "bucket_loads",
    "cv_of_loads",
    "h2_objective",
]


def bucket_loads(sizes: np.ndarray, buckets: list[list[int]]) -> np.ndarray:
    """Sum of squared sizes in each bucket, matching ``lpt_buckets``'s own ``loads``."""
    sq = np.asarray(sizes, dtype=float) ** 2
    return np.array([float(sq[b].sum()) for b in buckets])


def cv_of_loads(loads: np.ndarray) -> float:
    """Population CV of bucket loads: the balance measure the draft calls CV(S-tilde)."""
    loads = np.asarray(loads, dtype=float)
    mean = float(loads.mean())
    return float(np.std(loads, ddof=0) / mean) if mean > 0 else float("nan")


def h2_objective(loads: np.ndarray) -> float:
    """sum_m load_m^2, what ``exact_h2_buckets`` minimizes.

    The total load sum_i S_i^2 is fixed by the data, so minimizing this is the
    same problem as minimizing Var_m(load_m), i.e. as minimizing CV(S-tilde).
    """
    return float(np.sum(np.asarray(loads, dtype=float) ** 2))


def _rank_order(sizes: np.ndarray) -> np.ndarray:
    """Indices sorted by decreasing size (robust to unsorted input)."""
    return np.argsort(sizes)[::-1]


def lpt_buckets(sizes: np.ndarray, m: int) -> tuple[list[list[int]], np.ndarray]:
    """mgp_core's incumbent, wrapped to the same (unsquared sizes) signature as
    the other schemes here -- mgp_core.lpt_buckets itself takes squared job
    sizes directly (matching its other caller, ``build_lpt_grouped_data``).
    """
    buckets, _ = _lpt_buckets_squared(np.asarray(sizes, dtype=float) ** 2, m)
    buckets = [list(b) for b in buckets]
    return buckets, bucket_loads(sizes, buckets)


def core_fringe_buckets(sizes: np.ndarray, m: int) -> tuple[list[list[int]], np.ndarray]:
    """The naive alternative to LPT: M-1 singleton "core" buckets, one "fringe" bucket.

    The M-1 largest units each get their own bucket; every other unit, however
    small or large the residual mass, is pooled into the M-th bucket. This is
    the scheme the draft's prose contrasts LPT with.
    """
    order = _rank_order(sizes)
    buckets: list[list[int]] = [[int(i)] for i in order[: m - 1]]
    buckets.append([int(i) for i in order[m - 1 :]])
    return buckets, bucket_loads(sizes, buckets)


def round_robin_buckets(sizes: np.ndarray, m: int) -> tuple[list[list[int]], np.ndarray]:
    """Deal ranked units into buckets 1..M, 1..M, ... : mixing without load-awareness.

    Rank 1 (largest) into bucket 1, rank 2 into bucket 2, ..., rank M+1 back
    into bucket 1, and so on. Mixes sizes across buckets exactly as LPT does,
    but ignores the running load entirely, isolating what LPT's greedy
    load-matching step buys over mere deterministic mixing.
    """
    order = _rank_order(sizes)
    buckets: list[list[int]] = [[] for _ in range(m)]
    for rank, i in enumerate(order):
        buckets[rank % m].append(int(i))
    return buckets, bucket_loads(sizes, buckets)


def exact_h2_buckets(
    sizes: np.ndarray,
    m: int,
    *,
    scale: float | None = None,
    time_limit: float = 300.0,
    workers: int = 8,
    hint: list[list[int]] | str | None = "lpt",
    max_lost_mass: float = 1e-6,
) -> tuple[list[list[int]], np.ndarray, dict]:
    """The exact H^2-minimizing partition, via OR-Tools CP-SAT.

    Minimizing sum_m load_m^2 over an assignment of N units to M buckets is a
    genuine mixed-integer quadratic program: N*M booleans, M integer loads,
    and M quadratic terms load_m^2 (``CpModel.AddMultiplicationEquality``).
    It looks intractable at N ~ 1000 (10^956 partitions), but is not: with a
    long-tailed size distribution, only the handful of largest units make a
    real combinatorial choice, so CP-SAT's LP relaxation plus its integer
    search drive the certified gap to ~1e-16 in a couple of minutes,
    cold-started (``hint`` only speeds this up).

    Squared sizes are scaled to integers (CP-SAT needs integer arithmetic).
    The scale is chosen automatically so that ``m`` copies of the squared-load
    domain ``(scale * sum(sizes**2))**2`` -- CP-SAT sums *all* variable domains
    for its own overflow check, not just the largest one -- stay safely under
    int64.  A naive fixed scale is a real trap here, not a cosmetic one: at a
    coarse scale, a long tail of tiny squared sizes rounds to exactly zero, and
    CP-SAT then finds a genuine, correctly PROVEN optimum -- of a discretized
    problem that has silently thrown away most of the tail, free to dump every
    zeroed-out unit into one bucket at no cost to the (wrong) objective it was
    handed.  This is checked, not just avoided by a bigger default:
    ``max_lost_mass`` caps the total squared size rounded away to zero, as a
    share of the true total, and the function raises rather than silently
    return a partition solved for the wrong problem.

    Returns ``(buckets, loads, info)`` in the usual format, plus a solve report
    so a caller never has to take "close enough" on faith:

      ``status``     CP-SAT's own status name,
      ``optimal``    whether it formally reached OPTIMAL,
      ``gap``        the relative gap between the best integer solution found
                     and CP-SAT's own proven lower bound, which is a rigorous
                     bound whether or not the search formally terminates,
      ``lost_mass``  the rounded-to-zero share checked above,
      ``h2``         the returned partition's objective in exact float
                     arithmetic (so the caller can compare it with LPT's).
    """
    from ortools.sat.python import cp_model

    sizes = np.asarray(sizes, dtype=float)
    sq = sizes ** 2
    n = sq.size
    true_total = float(sq.sum())
    if scale is None:
        # CP-SAT rejects a model if its variable domains' sizes cannot be summed
        # without overflowing int64 (~9.22e18); m squared-load terms of size
        # (scale*total)^2 dominate that sum, so budget scale*total accordingly,
        # with a 10x safety margin under the true int64 ceiling.
        safe_budget = (2 ** 63 - 1) / 10.0
        scale = np.sqrt(safe_budget / m) / true_total
    w = np.round(sq * scale).astype(np.int64)
    lost_mass = float(sq[w == 0].sum()) / true_total if true_total > 0 else 0.0
    if lost_mass > max_lost_mass:
        raise RuntimeError(
            f"exact_h2_buckets: scale={scale:.3e} rounds {int((w == 0).sum())}/{n} units to "
            f"zero, losing {lost_mass:.2e} of the true total mass (> max_lost_mass="
            f"{max_lost_mass:.1e}). Pass an explicit finer ``scale``."
        )
    total = int(w.sum())

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(m)] for i in range(n)]
    for row in x:
        model.AddExactlyOne(row)
    loads = []
    for j in range(m):
        load = model.NewIntVar(0, total, f"load_{j}")
        model.Add(load == sum(int(w[i]) * x[i][j] for i in range(n)))
        loads.append(load)
    # Buckets are otherwise interchangeable: break that M!-fold symmetry by
    # forcing loads sorted descending, which is also a modest CP-SAT speedup.
    for j in range(m - 1):
        model.Add(loads[j] >= loads[j + 1])
    squares = []
    for j in range(m):
        square = model.NewIntVar(0, total * total, f"sq_{j}")
        model.AddMultiplicationEquality(square, [loads[j], loads[j]])
        squares.append(square)
    model.Minimize(sum(squares))

    if hint == "lpt":
        hint, _ = lpt_buckets(sizes, m)
    if hint is not None:
        assign = np.empty(n, dtype=int)
        for j, bucket in enumerate(hint):
            assign[bucket] = j
        hint_loads = np.array([w[assign == j].sum() for j in range(m)])
        # Respect the descending-load symmetry break, or the hint is infeasible.
        remap = {int(old): new for new, old in enumerate(np.argsort(-hint_loads))}
        for i in range(n):
            model.AddHint(x[i][remap[assign[i]]], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status_name not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"CP-SAT could not find a feasible partition (status={status_name}).")
    bound = solver.BestObjectiveBound()
    objective = solver.ObjectiveValue()
    gap = float((objective - bound) / bound) if bound > 0 else float("nan")

    assignment = np.array(
        [next(j for j in range(m) if solver.Value(x[i][j])) for i in range(n)]
    )
    buckets = [list(np.flatnonzero(assignment == j)) for j in range(m)]
    loads_out = bucket_loads(sizes, buckets)
    info = {
        "status": status_name,
        "optimal": status_name == "OPTIMAL",
        "gap": gap,
        "scale": float(scale),
        "lost_mass": lost_mass,
        "h2": h2_objective(loads_out),
    }
    return buckets, loads_out, info


def certified_balance_buckets(sizes: np.ndarray, m: int, *, tolerance: float=1e-10):
    """Return LPT when its objective meets a rigorous continuous lower bound.

    If q_(i) are squared unit sizes in decreasing order and L_(m) are the
    decreasing bucket loads of any partition, sum_{j<=k} L_(j) >=
    sum_{i<=k} q_(i): the k largest jobs occupy at most k buckets. The
    least-dispersed profile satisfying these prefix inequalities consists of
    the large q_i followed by equal tail loads (water filling). This profile
    is majorized by every attainable bucket-load vector. Convexity of x^2
    therefore makes its squared norm a lower bound on every partition's H².

    LPT is an actual partition. If its upper bound meets this lower bound to
    the requested tolerance, it is a certified optimal witness at that
    precision; no integer program or choice among tied optima is needed.
    """
    bound,k=continuous_balance_lower_bound(sizes,m)
    buckets,loads=lpt_buckets(sizes,m)
    objective=h2_objective(loads)
    gap=max(0.,float(objective/bound-1))
    if gap>tolerance:
        raise RuntimeError(f'LPT balance gap {gap:.3g} exceeds tolerance {tolerance:g}.')
    return buckets,loads,dict(status='CONTINUOUS_BOUND_CERTIFICATE',
                             optimal=True,gap=gap,scale=np.nan,lost_mass=0.,h2=objective,
                             lower_bound=bound,fixed_large_jobs=k)


def continuous_balance_lower_bound(sizes: np.ndarray,m: int):
    """Majorization lower bound on squared bucket-load dispersion; see above."""
    sizes=np.asarray(sizes,dtype=float)
    q=np.sort(sizes**2)[::-1]
    if not 1<=m<=len(q):
        raise ValueError('Require 1 <= M <= N.')
    remaining=float(q.sum()); k=0
    while k<m-1 and q[k] > remaining/(m-k):
        remaining-=q[k]; k+=1
    return float(np.sum(q[:k]**2)+remaining**2/(m-k)),k
