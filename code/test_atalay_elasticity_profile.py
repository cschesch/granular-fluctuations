#!/usr/bin/env python3
"""Checks for the Atalay eps_M/eps_Q primitive-granularity profiles.

Both profiles re-run the deposited Atalay filter at many elasticities, so the
things worth pinning are that (i) the eps_M grid point at Atalay's own
calibration reproduces the deposited benchmark export bit-for-bit, (ii) the
ceiling we read off it is the number Section~5 already quotes, (iii) both
grids reproduce Atalay's own published one-factor shares at his benchmark and
unit-elasticity parameterizations, (iv) the eps_Q grid's Equation-(11)
recursion is undefined everywhere (eps_M held at unity puts it outside
Atalay's own conditioning screen at every point), and (v) the
reverse-identification inversion behaves on both sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mgp_core import choose_solver
from applications.atalay.run_atalay_epsilon import DEFAULT_NETWORK_V_CSV, DEFAULT_SHOCKS_CSV
from applications.atalay.run_atalay_elasticity_profile import (
    PROFILE_CSV,
    PROFILE_ROOT,
    SWEEPS,
    UNITY,
    crossings,
    grouped_ceiling,
    identified_sets,
    load_grid,
)


# Section 5's headline Atalay column, reproduced by test_atalay_epsilon.py.
SECTION5_CEILING = 0.3487879683352339
# Atalay's published one-factor shares: (eps_M, eps_D, eps_Q) = (0.1, 1, 1)
# gives 0.83 with no "kf" marker (Table 2); (1, 1, 1) gives 0.21 with "kf".
PUBLISHED_BENCHMARK_ONE_FACTOR = 0.832264871355539
PUBLISHED_UNIT_ONE_FACTOR = 0.21


def grid_point(index: pd.DataFrame, profile_dir, value: float) -> dict:
    hit = index[np.isclose(index["value"], value)]
    assert len(hit) == 1, f"expected exactly one grid point at value={value}, found {len(hit)}"
    row = hit.iloc[0]
    assert row["status"] == "ok", f"grid point value={value} failed: {row['status']}"
    return loadmat(profile_dir / str(row["file"]), squeeze_me=True)


def main() -> int:
    profile = pd.read_csv(PROFILE_CSV)
    indexes = {}
    for sweep in SWEEPS:
        profile_dir = PROFILE_ROOT / sweep
        index = load_grid(profile_dir, sweep)
        indexes[sweep] = index
        assert index["value"].is_monotonic_increasing
        assert index["value"].gt(0).all(), f"{sweep} must be a positive elasticity"
        assert not np.isclose(index["value"], 1.0).any(), (
            f"the model is singular at {sweep}=1 exactly"
        )

    # 1. The eps_M benchmark grid point is the deposited export, to machine
    #    precision -- this pins the whole eps_M sweep to Section 5's inputs.
    benchmark = float(SWEEPS["eps_M"]["benchmark"])
    payload = grid_point(indexes["eps_M"], PROFILE_ROOT / "eps_M", benchmark)
    deposited = pd.read_csv(DEFAULT_SHOCKS_CSV)
    deposited_shocks = deposited[[c for c in deposited if c.startswith("shock_")]].to_numpy(float)
    deposited_v = pd.read_csv(DEFAULT_NETWORK_V_CSV, header=None).to_numpy(float)
    shocks = np.asarray(payload["shocks"], dtype=float)
    network_v = np.asarray(payload["V"], dtype=float)
    assert np.allclose(shocks, deposited_shocks, rtol=0, atol=1e-12)
    assert np.allclose(network_v, deposited_v, rtol=0, atol=1e-12)
    assert np.array_equal(np.asarray(payload["years"], dtype=int), deposited["year"].to_numpy(int))

    # 2. Reading the ceiling off that grid point gives Section 5's number.
    ceiling = grouped_ceiling(shocks, 0.5 * (network_v + network_v.T), 10, choose_solver("auto"))
    assert np.isclose(ceiling["mgp_share"], SECTION5_CEILING, rtol=0, atol=1e-9), ceiling["mgp_share"]

    # 3. Both grids reproduce Atalay's own published decomposition where it is
    #    reported: eps_M=0.10 (no "kf") and eps_M=eps_Q=1 (both "kf").
    assert np.isclose(
        float(payload["one_factor_share"]), PUBLISHED_BENCHMARK_ONE_FACTOR, atol=1e-12
    )
    unit_m = grid_point(indexes["eps_M"], PROFILE_ROOT / "eps_M", UNITY)
    assert float(unit_m["min_abs_eig_filter_matrix_1_A"]) < 0.01, (
        "eps_M=1 should fail Atalay's own conditioning screen"
    )
    assert np.isclose(float(unit_m["one_factor_share_kf"]), PUBLISHED_UNIT_ONE_FACTOR, atol=0.01)
    unit_q = grid_point(indexes["eps_Q"], PROFILE_ROOT / "eps_Q", UNITY)
    assert float(unit_q["min_abs_eig_filter_matrix_1_A"]) < 0.01
    assert np.isclose(float(unit_q["one_factor_share_kf"]), PUBLISHED_UNIT_ONE_FACTOR, atol=0.01)

    # 4. On the eps_Q sweep eps_M is held at unity throughout, so Atalay's own
    #    conditioning screen fails at every grid point: the Equation-(11)
    #    recursion is undefined on the whole sweep, matching every eps_M=1 row
    #    of his Table 2 carrying a "kf" superscript.
    q_all = pd.read_csv(PROFILE_CSV)
    q_all = q_all[q_all["sweep"].eq("eps_Q")]
    assert not q_all["eq11_screen_passes"].any(), (
        "expected Atalay's Equation-(11) screen to fail everywhere on the eps_Q sweep"
    )
    assert q_all["mgp_share"].isna().all()
    assert q_all["mgp_share_kf"].notna().all()

    # 5. The cached profile and its inversion are internally consistent.
    assert len(profile) == sum(len(idx) for idx in indexes.values())
    for sweep in SWEEPS:
        d = profile[profile["sweep"].eq(sweep)]
        assert d["value"].is_monotonic_increasing
        eq11 = d.dropna(subset=["mgp_share"])
        assert eq11["eq11_screen_passes"].all(), (
            f"a ceiling was computed on {sweep} where Atalay's screen fails"
        )
        assert d.loc[~d["eq11_screen_passes"], "mgp_share"].isna().all()
        assert d["mgp_share_kf"].notna().all(), f"Kalman filter undefined somewhere on {sweep}"
        assert d["mgp_share"].max(skipna=True) < 1.0 or eq11.empty
        assert d["mgp_share_kf"].max() < 1.0

    m_profile = profile[profile["sweep"].eq("eps_M")].dropna(subset=["mgp_share"])
    assert np.isclose(
        m_profile.loc[np.isclose(m_profile["value"], benchmark), "mgp_share"].iloc[0],
        SECTION5_CEILING,
        atol=1e-9,
    )

    for sweep in SWEEPS:
        for column in ("mgp_share", "mgp_share_kf"):
            sets = identified_sets(profile, sweep, column)
            d = profile[profile["sweep"].eq(sweep)].dropna(subset=[column])
            for _, row in sets.iterrows():
                inside = d[d[column] >= row["threshold"]]
                assert bool(row["attained"]) == (not inside.empty)
                assert int(row["n_grid_points"]) == len(inside)
                if not inside.empty:
                    assert np.isclose(row["grid_min_value"], inside["value"].min())
                    assert np.isclose(row["grid_max_value"], inside["value"].max())
                assert (row["threshold"] <= row["profile_max"]) == bool(row["attained"])

    # 6. The crossing helper: exact, interior, and no-crossing cases.
    x = np.array([0.0, 1.0, 2.0])
    assert crossings(x, np.array([0.2, 0.6, 0.4]), 0.5) == [0.75, 1.5]
    assert crossings(x, np.array([0.5, 0.6, 0.4]), 0.5) == [0.0, 1.5]
    assert crossings(x, np.array([0.2, 0.3, 0.4]), 0.5) == []

    print(
        f"Atalay eps_M/eps_Q profile checks passed ({len(profile)} grid points "
        f"total; benchmark ceiling {SECTION5_CEILING:.4f}; eps_Q sweep has no "
        f"usable Equation-(11) points, as expected)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
