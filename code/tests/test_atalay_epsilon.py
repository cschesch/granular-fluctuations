#!/usr/bin/env python3
"""Check rebuilt Atalay inputs against MATLAB arrays and the paper's estimator anchors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mgp_core import align_sizes_to_panel, build_lpt_grouped_data, choose_solver, sample_cov
from applications.atalay.run_atalay_epsilon import (
    DEFAULT_NETWORK_V_CSV,
    DEFAULT_SHOCKS_CSV,
    DERIVED,
    load_atalay_panel,
)
from applications.networks import matrix_inner, network_observed_stats


WORKSPACE = DERIVED / "benchmark/atalay_eps_M_001.mat"
# Columns 9 and 13 of the package's own results row for the benchmark parameters.
PACKAGE_MEAN_CORRELATION_WITH_DIAGONAL = 0.0569212034435663
PACKAGE_ONE_FACTOR_SHARE = 0.832264871355539


def main() -> int:
    if not all(path.is_file() for path in (WORKSPACE, DEFAULT_SHOCKS_CSV, DEFAULT_NETWORK_V_CSV)):
        print("[atalay-test] SKIP benchmark regression. Run applications.atalay.prepare --section benchmark first.")
        return 0
    workspace = loadmat(WORKSPACE, squeeze_me=True, struct_as_record=False)

    # 1. The CSV exports are the untouched MATLAB objects.
    shocks = pd.read_csv(DEFAULT_SHOCKS_CSV)
    filtered = np.asarray(workspace["shocks"], dtype=float)
    exported = shocks[[c for c in shocks if c.startswith("shock_")]].to_numpy(float)
    assert np.allclose(exported, filtered, rtol=0, atol=1e-13)
    assert np.array_equal(shocks["year"].to_numpy(int), np.asarray(workspace["years"], dtype=int))

    network_v = pd.read_csv(DEFAULT_NETWORK_V_CSV, header=None).to_numpy(float)
    assert np.allclose(network_v, np.asarray(workspace["V"], dtype=float), rtol=0, atol=1e-14)

    # 2. The package's own reported numbers, recomputed from the exports.
    psi = np.asarray(workspace["psi_A"], dtype=float)
    sigma_fitted = np.asarray(workspace["sigma_Matrix_A"], dtype=float)
    sigma_eps = sample_cov(filtered)
    correlation = np.corrcoef(filtered, rowvar=False)
    assert np.isclose(correlation.mean(), PACKAGE_MEAN_CORRELATION_WITH_DIAGONAL, atol=1e-12)
    one_factor_share = matrix_inner(network_v, np.diag(psi)) / matrix_inner(network_v, sigma_fitted)
    assert np.isclose(one_factor_share, PACKAGE_ONE_FACTOR_SHARE, atol=1e-12)

    # 3. That share is not a decomposition of the recovered covariance: the fitted
    #    one-factor Sigma is a long way from the sample one, and its Psi is not even
    #    admissible (Sigma_eps - Psi is indefinite), so it is not in the MGP feasible
    #    set our ceiling maximizes over. On the sample denominator it exceeds one.
    residual = sigma_eps - np.diag(psi)
    min_eig = float(np.linalg.eigvalsh(0.5 * (residual + residual.T)).min())
    scale = float(np.abs(np.linalg.eigvalsh(0.5 * (residual + residual.T))).max())
    assert min_eig < -1e-3 * scale, "Atalay's one-factor Psi is admissible; revisit the comparison."
    psi_on_sample = matrix_inner(network_v, np.diag(psi)) / matrix_inner(network_v, sigma_eps)
    assert psi_on_sample > 1.0

    # 4. The panel and the headline estimate quoted in Section 5.
    m = 10
    panel, network_v_loaded = load_atalay_panel(DEFAULT_SHOCKS_CSV, DEFAULT_NETWORK_V_CSV, m)
    assert np.allclose(panel.sizes.to_numpy(float), np.sqrt(np.diag(network_v_loaded)), atol=1e-13)
    y, sizes = align_sizes_to_panel(panel.y, panel.sizes)
    assert y.shape == (30, 49)
    order_pos = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    network_v_sorted = network_v_loaded[np.ix_(order_pos, order_pos)]
    shares = (sizes / sizes.sum()).to_numpy(float)
    x_grouped, *_ = build_lpt_grouped_data(y.to_numpy(float), shares, m)
    observed = network_observed_stats(
        x_grouped,
        y.to_numpy(float).T,
        network_v_sorted,
        float(sizes.sum() ** 2),
        choose_solver("auto"),
    )
    assert np.isclose(observed["mgp_share"], 0.3487879683344252, rtol=0, atol=1e-9)

    print(
        "Atalay export, replication, and MGP checks passed "
        f"(one-factor share {one_factor_share:.4f} on the fitted covariance, "
        f"{psi_on_sample:.4f} on the sample covariance; grouped ceiling "
        f"{observed['mgp_share']:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
