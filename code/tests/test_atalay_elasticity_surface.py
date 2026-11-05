#!/usr/bin/env python3
"""Checks for the 3-D Atalay (eps_D, eps_M, eps_Q) elasticity surface.

The surface export re-runs the deposited, unmodified filter over 11,934 parameter
vectors, so the things worth pinning are that (i) the cell sitting at Atalay's
own benchmark calibration reproduces the deposited benchmark export bit-for-bit,
(ii) the ceilings read off that cell -- Kalman, which the surface uses, and
Equation-(11), which Section~5 quotes -- are both what they should be, and
(iii) the surface agrees with the one-dimensional profile everywhere the two
grids touch, which is the eps_D = 1 slab along eps_Q at eps_M = unity.

The full-data regression runs after `make all` has prepared and priced the grid.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from mgp_core import choose_solver
from applications.atalay.run_atalay_elasticity_profile import PROFILE_CSV, UNITY, grouped_ceiling
from scipy.io import loadmat

from applications.atalay.run_atalay_elasticity_surface import (
    BENCHMARK,
    SURFACE_CSV,
    SURFACE_ROOT,
    load_slabs,
    preferred,
)
from applications.atalay.run_atalay_epsilon import DEFAULT_SHOCKS_CSV

# Section 5's headline Atalay column, reproduced by test_atalay_epsilon.py, and
# the Kalman counterpart at the same parameters, from the eps_M profile.
SECTION5_CEILING = 0.3487879683352339
BENCHMARK_KALMAN_CEILING = 0.3521901844304733
# The surface grid is coarser than the profile grid, so cells the two share are
# compared at the solver's own tolerance rather than bit-for-bit.
CEILING_TOL = 1e-6


def _cell(slab: dict, value_m: float, value_q: float) -> tuple[int, int]:
    eps_m = np.atleast_1d(np.asarray(slab["eps_M_grid"], dtype=float))
    eps_q = np.atleast_1d(np.asarray(slab["eps_Q_grid"], dtype=float))
    i = int(np.argmin(np.abs(eps_m - value_m)))
    j = int(np.argmin(np.abs(eps_q - value_q)))
    if abs(eps_m[i] - value_m) > 1e-6 or abs(eps_q[j] - value_q) > 1e-6:
        raise AssertionError(
            f"({value_m}, {value_q}) is not on the grid; nearest is "
            f"({eps_m[i]}, {eps_q[j]})."
        )
    return i, j


def main() -> int:
    required = [SURFACE_CSV, PROFILE_CSV, DEFAULT_SHOCKS_CSV]
    required.extend(SURFACE_ROOT / f"atalay_surface_epsD_{value}.mat" for value in (250, 500, 750, 900, 999, 1250))
    if not all(path.is_file() for path in required):
        print("[surface-test] SKIP full surface regression. Run make all with Stata and MATLAB to prepare and price the public grid.")
        return 0
    solver = choose_solver("auto")
    # ``load_slabs`` returns paths, so that the pricer can farm them out to workers.
    slabs = [loadmat(path, squeeze_me=True) for path in load_slabs(SURFACE_ROOT)]
    slabs.sort(key=lambda slab: float(np.asarray(slab["eps_D"])))
    print(f"[surface-test] {len(slabs)} slabs at eps_D = "
          + ", ".join(f"{float(np.asarray(s['eps_D'])):g}" for s in slabs))

    # (i) and (ii): the benchmark cell against the deposited export and Section 5.
    benchmark = [s for s in slabs if np.isclose(float(np.asarray(s["eps_D"])), UNITY)]
    if not benchmark:
        raise AssertionError("No eps_D = 1 slab: the benchmark cell cannot be checked.")
    slab = benchmark[0]
    i, j = _cell(slab, BENCHMARK["eps_M"], UNITY)
    if not bool(np.atleast_2d(slab["ok"])[i, j]):
        raise AssertionError("The benchmark cell did not solve in the export.")
    shocks = np.asarray(slab["shocks"], dtype=float)[i, j]
    deposited = pd.read_csv(DEFAULT_SHOCKS_CSV).select_dtypes("number").to_numpy()[:, -30:]
    gap = float(np.max(np.abs(shocks - deposited)))
    assert gap < 1e-10, f"Benchmark cell differs from the deposited export by {gap:.2e}."
    print(f"[surface-test] benchmark cell matches the deposited shocks to {gap:.2e}")

    network = 0.5 * (np.asarray(slab["V"], dtype=float)[i, j]
                     + np.asarray(slab["V"], dtype=float)[i, j].T)
    ceiling = grouped_ceiling(shocks, network, 10, solver)["mgp_share"]
    gap = abs(ceiling - SECTION5_CEILING)
    assert gap < CEILING_TOL, (
        f"Benchmark Equation-(11) ceiling {ceiling:.6f} differs from Section 5's "
        f"{SECTION5_CEILING:.6f} by {gap:.2e}."
    )
    print(f"[surface-test] benchmark Equation-(11) ceiling {ceiling:.4%} reproduces Section 5")

    # The surface itself is the Kalman filter everywhere, so pin that too.
    kalman = np.asarray(slab["shocks_kf"], dtype=float)[i, j]
    ceiling_kf = grouped_ceiling(kalman, network, 10, solver)["mgp_share"]
    gap = abs(ceiling_kf - BENCHMARK_KALMAN_CEILING)
    assert gap < CEILING_TOL, (
        f"Benchmark Kalman ceiling {ceiling_kf:.6f} differs from the profile's "
        f"{BENCHMARK_KALMAN_CEILING:.6f} by {gap:.2e}."
    )
    print(f"[surface-test] benchmark Kalman ceiling {ceiling_kf:.4%} reproduces the profile")

    # (iii): where the surface and the one-dimensional eps_Q profile overlap.
    surface = preferred(pd.read_csv(SURFACE_CSV))
    assert surface["M"].eq(10).all(), "Surface must use the paper-wide M=10."
    # The profile CSV indexes its sweep by ``value``; align it to the surface's
    # named axis before merging. Only the eps_Q sweep touches the surface, and only
    # at eps_M = eps_D = 1, where the recursion fails everywhere and both sides use
    # the Kalman filter.
    profile = pd.read_csv(PROFILE_CSV)
    profile = profile[profile["sweep"].eq("eps_Q")].rename(columns={"value": "eps_Q"})
    overlap = surface[
        np.isclose(surface["eps_D"], UNITY) & np.isclose(surface["eps_M"], UNITY)
    ].merge(profile[["eps_Q", "mgp_share_kf"]].rename(
        columns={"mgp_share_kf": "profile_share"}), on="eps_Q")
    if overlap.empty:
        print("[surface-test] no (eps_D, eps_M) = (1, 1) cells to compare with the profile")
    else:
        gap = float((overlap["mgp_share_preferred"] - overlap["profile_share"]).abs().max())
        assert gap < CEILING_TOL, (
            f"Surface and profile disagree by {gap:.2e} on {len(overlap)} shared eps_Q cells."
        )
        print(f"[surface-test] surface matches the eps_Q profile on {len(overlap)} shared "
              f"cells to {gap:.2e}")

    # (iv): the drawn surface is the Kalman ceiling at every cell, and the two
    # filters stay close where both are defined.
    priced = surface[surface["mgp_share_preferred"].notna()]
    assert priced["mgp_share_preferred"].equals(priced["mgp_share_kf"]), (
        "The drawn surface is not the Kalman ceiling everywhere."
    )
    both = priced.dropna(subset=["mgp_share"])
    spread = (both["mgp_share_kf"] - both["mgp_share"]).abs()
    print(f"[surface-test] {len(priced)} priced cells, all Kalman; on the "
          f"{len(both)} where the recursion is also defined the two differ by "
          f"{spread.median():.2%} at the median")

    print("[surface-test] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
