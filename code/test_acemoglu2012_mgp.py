#!/usr/bin/env python3
"""Fast source, replication, panel, and headline checks for Acemoglu et al."""

from __future__ import annotations

import numpy as np

from mgp_core import build_lpt_grouped_data, choose_solver, mgp_share
from applications.acemoglu.run_acemoglu2012_mgp import (
    load_nber_manufacturing_panel,
    validate_source_files,
    verify_original_replication_calculations,
)


def main() -> int:
    validate_source_files()
    original = verify_original_replication_calculations()
    assert original["detailed_io_dimensions"] == [483, 524, 529, 510, 476, 474, 417]
    assert original["summary_io_dimensions"] == [84, 84, 80, 89, 89, 127, 128]
    # Published Table II of the paper itself, 1972--2002 (n_d and n_s above are its
    # parenthetical sector counts). The 1992 detailed entry prints as 0.093 there
    # against our 0.09249, which is the table's own rounding.
    assert np.allclose(
        np.round(original["detailed_influence_l2_norms"], 3),
        [0.098, 0.091, 0.088, 0.088, 0.092, 0.090, 0.094],
    )
    assert np.allclose(
        np.round(original["summary_influence_l2_norms"], 3),
        [0.139, 0.137, 0.149, 0.133, 0.137, 0.115, 0.119],
    )

    panel, diagnostics = load_nber_manufacturing_panel()
    assert diagnostics["balanced_industries"] == 451
    assert diagnostics["dropped_sic_codes"] == [2411, 2711, 2721, 2731, 2741, 2771, 3292, 3732]
    assert panel["year"].nunique() == 47
    assert not panel.isna().any().any()
    assert np.isclose(panel.groupby("sic")["fixed_influence_weight"].first().sum(), 1.0)

    primitive = panel.pivot(
        index="sic", columns="year", values="primitive_productivity_shock"
    ).sort_index()
    influence = panel.groupby("sic")["fixed_influence_weight"].first().reindex(primitive.index)
    grouped, _, _, _ = build_lpt_grouped_data(
        primitive.to_numpy(float), influence.to_numpy(float), 10
    )
    result = mgp_share(grouped, choose_solver("auto"))
    assert np.isclose(result["mgp_share"], 0.03724451368618636, rtol=0, atol=1e-9)
    print("Acemoglu et al. replication and manufacturing-MGP checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
