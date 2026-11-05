#!/usr/bin/env python3
"""Fast construction and inversion checks for the Carvalho--Gabaix proxy."""

from __future__ import annotations

import numpy as np

from mgp_core import build_lpt_grouped_data, choose_solver, mgp_share
from applications.carvalho_gabaix.run_carvalho_gabaix_proxy import (
    EXPECTED_SOURCE_HASHES,
    Calibration,
    invert_measured_tfp,
    load_historical_panel,
    validate_source_files,
)

from tests.support import require_files

TEST_GROUP = "integration"
TEST_CACHE_INPUTS = ()


def main() -> int:
    require_files(*EXPECTED_SOURCE_HASHES)
    validate_source_files()
    calibration = Calibration()
    assert np.isclose(calibration.beta, 30.0 / 11.0)
    assert np.isclose(calibration.phi_capital, 10.0 / 33.0)
    assert np.isclose(calibration.gdp_multiplier_mu, 4.5)
    # The published "Resulting Values" of online-appendix Table 5, plus the c_A = 1.19
    # quoted in its Section C. These are what tie the implementation to the paper.
    # The paper's 1.19 is c_A evaluated at its own rounded beta = 2.7 (giving 1.1935);
    # carrying the exact beta = 30/11 through gives 1.1955.
    assert np.round(calibration.beta, 1) == 2.7
    assert np.round(calibration.phi_capital, 1) == 0.3
    assert abs(calibration.measured_tfp_own_loading - 1.19) < 0.01
    assert np.isclose(1.0 + 0.43 * 0.5 * (2.0 / 3.0) * 0.5 * 2.7, 1.1935)

    historical, diagnostics = load_historical_panel()
    historical, error = invert_measured_tfp(historical, calibration)
    assert historical["industry_code"].nunique() == 60
    assert historical["year"].nunique() == 53
    assert diagnostics["mfp_validation_correlation"] > 0.999999999
    assert diagnostics["mfp_validation_max_abs_difference_pct_point"] < 5e-5
    assert error < 1e-12

    primitive = historical.pivot(
        index="industry_code", columns="year", values="primitive_productivity_shock"
    ).sort_index()
    influence = (
        historical.groupby("industry_code")["fixed_influence_weight"].first().reindex(primitive.index)
    )
    grouped, _, _, _ = build_lpt_grouped_data(
        primitive.to_numpy(float), (influence / influence.sum()).to_numpy(float), 10
    )
    result = mgp_share(grouped, choose_solver("auto"))
    assert np.isclose(result["mgp_share"], 0.2764757871623982, rtol=0, atol=1e-9)

    print("Carvalho--Gabaix public-proxy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
