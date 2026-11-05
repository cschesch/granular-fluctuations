"""Firm-to-bucket partitions used by the grouping comparison.

Core/fringe and round-robin are compared with LPT. The optimal row uses the
fixed paper allocation or certifies LPT against a continuous lower bound.
Every scheme accepts unit size shares and returns buckets and squared-size
loads, with an additional certificate for the optimal partition.
"""

from __future__ import annotations

import numpy as np

from mgp_core import lpt_buckets as _lpt_buckets_squared


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
    """Sum of squared bucket loads, the balance objective.

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
