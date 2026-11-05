#!/usr/bin/env python3
"""Unpropagated BEA--BLS sector-output application for the overview figure."""

import numpy as np
import pandas as pd

import gradient_bootstrap as GB
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    RUN_OUTPUT_CACHE,
    analysis_block_length,
    choose_solver,
    lpt_buckets,
    sample_cov,
)
from applications.volatility.run_bea_granular_ceiling import build_inputs, grouping_matrix


def load_baseline():
    """Measured TFP growth and Domar weights from its pre-growth base year."""
    _, tfp_frame, weights_frame, _ = build_inputs()
    weight_year = int(tfp_frame.columns.min()) - 1
    return tfp_frame, weights_frame.loc[weight_year].to_numpy(dtype=float), weight_year


def main() -> None:
    tfp_frame, weights, weight_year = load_baseline()
    x = tfp_frame.to_numpy(dtype=float)
    t, n = x.shape[1], x.shape[0]
    m = min(n, 10)
    block_length = analysis_block_length(t)
    buckets, _ = lpt_buckets(weights**2, m)
    grouped = x.T @ grouping_matrix(weights, buckets).T
    bootstrap = GB.moving_block_gradient_ci(
        grouped, b=ANALYSIS_BOOTSTRAP_DRAWS, block_length=block_length,
        rng=np.random.default_rng(20260907), solver=choose_solver("auto"),
    )

    covariance = sample_cov(x.T)
    aggregate_variance = float(weights @ covariance @ weights)
    variances = np.diag(covariance)
    row = {
        "dataset": "bea_bls_sector_tfp_no_network",
        "N": n,
        "T": t,
        "M": m,
        "HHI": float(weights @ weights),
        "block_length": block_length,
        "bootstrap_b": ANALYSIS_BOOTSTRAP_DRAWS,
        "mgp_share": bootstrap["g_hat"],
        "moving_block_gradient_bootstrap_p05": bootstrap["ci_low"],
        "moving_block_gradient_bootstrap_p95": bootstrap["ci_high"],
        "naive": float(weights**2 @ variances / aggregate_variance),
        "naive_homoscedastic": float(weights @ weights * variances.mean() / aggregate_variance),
        "aggregate_variance": aggregate_variance,
        "start_date": int(tfp_frame.columns.min()),
        "last_date": int(tfp_frame.columns.max()),
        "weight_date": weight_year,
        "notes": f"BEA--BLS gross-output TFP with fixed {weight_year} Domar weights; no I/O propagation or lags.",
    }
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    out = RUN_OUTPUT_CACHE / f"bea_bls_sector_tfp_no_network_M{m}_summary.pkl"
    pd.DataFrame([row]).to_pickle(out)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
