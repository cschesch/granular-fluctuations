"""Shared paths and the grouped bound for Atalay's recovered productivity shocks."""

import numpy as np
import pandas as pd

from mgp_core import (
    ROOT, align_sizes_to_panel, build_lpt_grouped_data, mean_pairwise_correlation,
)
from applications.networks import network_observed_stats


DERIVED = ROOT / "code/.cache/derived/atalay2017"
PROFILE_ROOT = DERIVED / "elasticity_profile"
SURFACE_ROOT = DERIVED / "elasticity_surface"
SURFACE_SLABS = (667, 999, 1333)
# The deposited model is singular at exactly one.
UNITY = 0.999


def grouped_ceiling(shocks: np.ndarray, network_v: np.ndarray, m: int, solver: str) -> dict[str, float]:
    """Section~5's grouped primitive-shock bound for one structural filter.

    ``shocks`` is T x N primitive innovations, ``network_v`` the N x N matrix
    with aggregate-output variance ``trace(V Sigma_eps)``.  Total dynamic
    influence is ``V_i = sqrt(V_ii)``, exactly as in ``run_atalay_epsilon.py``;
    units are sorted by decreasing influence as everywhere else in the paper, so
    the LPT grouping is the same construction.
    """
    n = network_v.shape[0]
    if shocks.shape[1] != n:
        raise ValueError(f"shocks have {shocks.shape[1]} columns, V is {n}x{n}.")
    ids = pd.Index([f"shock_{i + 1:02d}" for i in range(n)], name="shock_id")
    y = pd.DataFrame(shocks.T, index=ids)
    sizes = pd.Series(np.sqrt(np.maximum(np.diag(network_v), 0.0)), index=ids)
    y, sizes = align_sizes_to_panel(y, sizes)
    order = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    if sorted(order) != list(range(n)):
        raise ValueError("Influence sort dropped or duplicated an industry.")
    v_sorted = network_v[np.ix_(order, order)]

    shares = (sizes / sizes.sum()).to_numpy(float)
    y_np = y.to_numpy(float)
    x_grouped, *_ = build_lpt_grouped_data(y_np, shares, m)
    observed = network_observed_stats(
        x_grouped, y_np.T, v_sorted, float(sizes.sum() ** 2), solver
    )
    return {
        "mgp_share": float(observed["mgp_share"]),
        "mgp_level": float(observed["mgp_level"]),
        "aggregate_variance": float(observed["aggregate_variance"]),
        "se_share": float(observed["se_share"]),
        "influence_hhi": float(np.sum(shares**2)),
        "top1_influence_share": float(shares[0]),
        "mean_pairwise_corr": mean_pairwise_correlation(y_np),
    }
