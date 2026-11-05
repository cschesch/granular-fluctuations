"""Check the published three-slab Kalman surface and its one-factor comparison."""

import numpy as np
import pandas as pd
from scipy.io import loadmat

from applications.atalay.common import SURFACE_ROOT, SURFACE_SLABS, UNITY, grouped_ceiling
from applications.atalay.run_atalay_elasticity_profile import PROFILE_CSV
from applications.atalay.run_atalay_elasticity_surface import SURFACE_CSV
from applications.atalay.run_atalay_epsilon import DEFAULT_SHOCKS_CSV
from mgp_core import choose_solver
from tests.support import require_files


TEST_GROUP = "integration"
TEST_CACHE_INPUTS = (
    ".cache/derived/atalay2017/elasticity_surface/*.mat",
    ".cache/derived/atalay2017/elasticity_surface/one_factor_*.csv",
    ".cache/derived/atalay2017/atalay_benchmark_*.csv",
    ".cache/run_outputs/atalay_elasticity_profile.csv",
    ".cache/run_outputs/atalay_surface_comparison.csv",
)
BENCHMARK_DIRECT = 0.3487879683352339
BENCHMARK_KALMAN = 0.3521901844304733
BENCHMARK_ONE_FACTOR_KALMAN = 0.8364535577959672
CEILING_TOL = 1e-6


def cell_index(grid, value):
    hits = np.flatnonzero(np.isclose(grid, value, rtol=0, atol=1e-10))
    assert len(hits) == 1, f"Expected one grid point at {value}, found {len(hits)}"
    return int(hits[0])


def main() -> int:
    require_files(SURFACE_CSV, PROFILE_CSV, DEFAULT_SHOCKS_CSV,
                  *(SURFACE_ROOT / f"atalay_surface_epsD_{tag}.mat" for tag in SURFACE_SLABS),
                  *(SURFACE_ROOT / f"one_factor_{tag}.csv" for tag in SURFACE_SLABS))
    surface = pd.read_csv(SURFACE_CSV)
    assert surface["M"].eq(10).all()
    assert len(surface) == 3 * 39 * 51
    assert not surface.duplicated(["eps_D", "eps_M", "eps_Q"]).any()
    assert np.allclose(sorted(surface.eps_D.unique()), [2 / 3, UNITY, 4 / 3], rtol=0, atol=1e-12)
    assert np.isfinite(surface[["one_factor", "mgp_share_kf"]]).all().all()
    assert surface["fit_limit"].isin([False, True]).all()
    solver = choose_solver("auto")

    for tag in SURFACE_SLABS:
        slab = loadmat(SURFACE_ROOT / f"atalay_surface_epsD_{tag}.mat", squeeze_me=True)
        d = float(slab["eps_D"])
        m_grid = np.atleast_1d(slab["eps_M_grid"])
        q_grid = np.atleast_1d(slab["eps_Q_grid"])
        assert (len(m_grid), len(q_grid), int(slab["rows_done"])) == (39, 51, 39)
        assert (np.diff(m_grid) > 0).all() and (np.diff(q_grid) > 0).all()
        assert np.asarray(slab["ok"]).all()
        assert np.array_equal(np.asarray(slab["years"])[[0, -1]], [1965, 2013])
        block = surface[np.isclose(surface.eps_D, d)].set_index(["eps_M", "eps_Q"]).sort_index()
        grid = pd.MultiIndex.from_product([m_grid, q_grid], names=["eps_M", "eps_Q"])
        assert np.allclose(np.asarray(block.index.tolist()), np.asarray(grid.tolist()),
                           rtol=0, atol=1e-12)

        # Keep the original MATLAB fit values and iteration-limit flags intact.
        fits = pd.read_csv(SURFACE_ROOT / f"one_factor_{tag}.csv").set_index(["eps_M", "eps_Q"]).sort_index()
        assert fits.index.equals(block.index)
        assert np.allclose(block.one_factor, fits.one_factor, rtol=0, atol=1e-14)
        assert np.array_equal(block.fit_limit.astype(bool), fits.fit_limit.astype(bool))

        # Re-estimate representative cells from shocks, including the benchmark.
        selected = [(0, 0), (cell_index(m_grid, .1), cell_index(q_grid, UNITY)),
                    (len(m_grid) - 1, len(q_grid) - 1)]
        for i, j in selected:
            v = np.asarray(slab["V"], dtype=float)[i, j]
            v = .5 * (v + v.T)
            shocks = np.asarray(slab["shocks_kf"], dtype=float)[i, j]
            actual = grouped_ceiling(shocks, v, 10, solver)["mgp_share"]
            expected = float(block.iloc[i * len(q_grid) + j].mgp_share_kf)
            assert np.isclose(actual, expected, rtol=0, atol=CEILING_TOL)

        if tag == 999:
            i, j = cell_index(m_grid, .1), cell_index(q_grid, UNITY)
            shocks = np.asarray(slab["shocks"], dtype=float)[i, j]
            deposited = pd.read_csv(DEFAULT_SHOCKS_CSV).filter(like="shock_").to_numpy(float)
            assert np.allclose(shocks, deposited, rtol=0, atol=1e-10)
            v = np.asarray(slab["V"], dtype=float)[i, j]
            direct = grouped_ceiling(shocks, .5 * (v + v.T), 10, solver)["mgp_share"]
            assert np.isclose(direct, BENCHMARK_DIRECT, rtol=0, atol=CEILING_TOL)
            point = block.iloc[i * len(q_grid) + j]
            assert np.isclose(point.mgp_share_kf, BENCHMARK_KALMAN, rtol=0, atol=CEILING_TOL)
            assert np.isclose(point.one_factor, BENCHMARK_ONE_FACTOR_KALMAN, rtol=0, atol=1e-12)

    # Both outputs use the same Kalman calculation where their grids overlap.
    profile = pd.read_csv(PROFILE_CSV)
    profile = profile[profile.sweep.eq("eps_Q")].rename(columns={"value": "eps_Q"})
    overlap = surface[np.isclose(surface.eps_D, UNITY) & np.isclose(surface.eps_M, UNITY)].merge(
        profile[["eps_Q", "mgp_share_kf"]], on="eps_Q", suffixes=("_surface", "_profile"))
    assert not overlap.empty
    assert np.allclose(overlap.mgp_share_kf_surface, overlap.mgp_share_kf_profile,
                       rtol=0, atol=CEILING_TOL)
    print(f"Atalay surface checks passed: {len(surface)} cells across three calibrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
