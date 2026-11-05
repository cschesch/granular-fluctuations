#!/usr/bin/env python3
"""Grouped-MGP estimate on firm shocks recovered with Stella's dynamic inverse.

The baseline uses the exact decision-rule equations in Stella's deposited GAUSS
code, solved by the independently benchmarked Octave/gensys port.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    ROOT,
    RUN_OUTPUT_CACHE,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    sample_cov,
)
from applications.foerster.run_foerster_epsilon import (
    cap_dynamic_share_report,
    network_direct_diagonal_share,
    network_gradient_family,
    network_observed_stats,
    network_variance_matrix,
)
from applications.stella.stella_provenance import archive_existing, file_fingerprints, verify_fingerprints


CALIBRATION = ROOT / "code/.cache/firm_networks/stella_literal"
PAPER_CALIBRATION = CALIBRATION.with_name("stella_revtq")
DYNAMIC = CALIBRATION / "dynamic_solution"
DIAGNOSTICS = ROOT / "output/stella_firm_network_diagnostics.json"


def load_growth(calibration: Path = CALIBRATION) -> pd.DataFrame:
    path = calibration / "sales_growth_sax12.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare_stella_sales.py first.")
    growth = pd.read_csv(path, index_col=0)
    growth.index = pd.PeriodIndex(growth.index, freq="Q")
    growth.columns = growth.columns.astype(str).str.zfill(6)
    return growth


def load_dynamic_matrices(n: int, directory: Path = DYNAMIC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    provenance = directory / "benchmark_and_solver_diagnostics.json"
    if provenance.exists():
        recorded = json.loads(provenance.read_text()).get("input_file_fingerprints", {})
        if recorded:
            verify_fingerprints(recorded, ROOT)
    matrices = tuple(
        np.loadtxt(directory / filename, delimiter=",")
        for filename in ("pii_a.csv", "xi.csv", "cap_rho.csv")
    )
    for name, matrix in zip(("pii_a", "xi", "cap_rho"), matrices, strict=True):
        if matrix.shape != (n, n):
            raise ValueError(f"{name} has shape {matrix.shape}, expected {(n, n)}")
    return matrices  # type: ignore[return-value]


def recover_dynamic_shocks(
    sales_growth: np.ndarray,
    pii_a: np.ndarray,
    xi: np.ndarray,
    cap_rho: np.ndarray,
) -> np.ndarray:
    """Literal translation of Stella's transformation_data.gss recurrence."""
    inverse_pii = np.linalg.inv(pii_a)
    ay0 = inverse_pii
    ay1 = -inverse_pii @ cap_rho
    ae1 = -inverse_pii @ xi
    shocks = np.zeros_like(sales_growth)
    dy = np.zeros_like(sales_growth)
    # GAUSS dy[3:T] uses sales[3:T] and sales[2:T-1]. Rows 1--2 are
    # initialization values, then eps[t]=dy[t]+eps[t-1] ae1'.
    dy[2:] = sales_growth[2:] @ ay0.T + sales_growth[1:-1] @ ay1.T
    for t in range(1, len(shocks)):
        shocks[t] = dy[t] + shocks[t - 1] @ ae1.T
    return shocks


def load_firm_ids(calibration: Path) -> pd.Index:
    mapping = pd.read_csv(calibration / "firm_mapping.csv", dtype={"gvkey": str})
    return pd.Index(mapping["gvkey"].str.zfill(6))


def load_dynamic_panel(calibration: Path, ids: pd.Index):
    growth = load_growth(calibration).reindex(columns=ids)
    matrices = load_dynamic_matrices(len(ids), calibration / "dynamic_solution")
    full_shocks = recover_dynamic_shocks(growth.to_numpy(float), *matrices)
    leading = int((np.abs(full_shocks).sum(axis=1) == 0).cumprod().sum())
    if leading != 2:
        raise ValueError(f"Expected two Stella initialization rows; found {leading}.")
    nominal = pd.read_csv(calibration / "nominal_saleq_levels.csv", index_col=0)
    nominal.index = nominal.index.astype(str).str.zfill(6)
    nominal.columns = pd.PeriodIndex(nominal.columns, freq="Q")
    # Fix weights immediately before the first usable innovation.
    initial_weights = nominal.loc[ids, growth.index[2] - 1].to_numpy(float)
    network_v = network_variance_matrix(initial_weights, *matrices)
    return growth, full_shocks, matrices, initial_weights, network_v


def main() -> None:
    parser = argparse.ArgumentParser()
    # Paper-wide grouping dimension; other values are sensitivities only.
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION)
    parser.add_argument("--output-stem", default=None,
                        help="separate cache/output prefix for a reconstructed-sample sensitivity")
    args = parser.parse_args()
    if args.M <= 0:
        raise ValueError("M must be positive.")

    calibration = args.calibration_dir.resolve()
    if calibration != CALIBRATION.resolve() and not args.output_stem:
        raise ValueError("A separate calibration requires --output-stem to preserve baseline results.")
    output_stem = args.output_stem or "stella_firm_network"
    if not output_stem.replace("_", "").isalnum():
        raise ValueError("--output-stem must contain only letters, numbers and underscores.")
    ids = load_firm_ids(calibration)
    calibration_diag = json.loads((calibration / "diagnostics.json").read_text())
    if calibration_diag.get("input_file_fingerprints"):
        verify_fingerprints(calibration_diag["input_file_fingerprints"], ROOT)
    sales_diagnostics_path = calibration / "sales_preparation_sax12_diagnostics.json"
    sales_diagnostics = json.loads(sales_diagnostics_path.read_text()) if sales_diagnostics_path.exists() else {}
    if sales_diagnostics.get("input_file_fingerprints"):
        verify_fingerprints(sales_diagnostics["input_file_fingerprints"], ROOT)
    solver = choose_solver(args.solver)

    growth, full_shocks, matrices, initial_weights, network_v = load_dynamic_panel(calibration, ids)
    pii_a, xi, cap_rho = matrices
    leading_zero_rows = 2
    shocks = full_shocks[leading_zero_rows:]
    periods = growth.index[leading_zero_rows:]
    weight_period = periods[0] - 1
    reconstructed_growth = (
        full_shocks[2:] @ pii_a.T
        + growth.to_numpy(float)[1:-1] @ cap_rho.T
        + full_shocks[1:-1] @ xi.T
    )
    recurrence_error = float(
        np.max(np.abs(reconstructed_growth - growth.to_numpy(float)[2:]))
    )
    fixed_weights = initial_weights / initial_weights.sum()
    observed_aggregate_growth = growth.to_numpy(float)[leading_zero_rows:] @ fixed_weights
    inversion_note = "Exact deposited dynamic recurrence using benchmarked gensys decision-rule matrices"
    dynamic_diag = {
        "leading_zero_initialization_rows_dropped": leading_zero_rows,
        "weight_period": str(weight_period),
        "spectral_radius_cap_rho": float(np.max(np.abs(np.linalg.eigvals(cap_rho)))),
        "condition_pii_a": float(np.linalg.cond(pii_a)),
        "dynamic_recurrence_max_abs_error": recurrence_error,
    }

    analysis_shocks = np.asarray(shocks, dtype=float)
    block_length = analysis_block_length(analysis_shocks.shape[0])
    network_sizes = np.sqrt(np.maximum(np.diag(network_v), 0.0))
    if not np.all(network_sizes > 0):
        raise ValueError("Dynamic network contains a zero-exposure primitive.")
    exposure_shares = network_sizes / network_sizes.sum()
    weight_scale = float(network_sizes.sum() ** 2)
    grouped, _, loads, _ = build_lpt_grouped_data(analysis_shocks.T, exposure_shares, args.M)
    observed = network_observed_stats(grouped, analysis_shocks, network_v, weight_scale, solver)
    sigma = sample_cov(analysis_shocks)
    direct = network_direct_diagonal_share(sigma, network_v)
    homoskedastic = float(
        np.diag(network_v).sum() * np.diag(sigma).mean()
        / observed["aggregate_variance"]
    )
    implied_aggregate_variance = float(np.sum(network_v * sigma))
    observed_aggregate_variance = float(np.var(observed_aggregate_growth, ddof=1))
    dynamic_diag.update({
        "stationary_implied_aggregate_variance": implied_aggregate_variance,
        "observed_fixed_weight_aggregate_variance": observed_aggregate_variance,
        "stationary_vs_observed_aggregate_variance_relative_gap": (
            implied_aggregate_variance / observed_aggregate_variance - 1.0
        ),
    })
    gradient = network_gradient_family(
        grouped,
        analysis_shocks,
        network_v,
        weight_scale,
        b=args.bootstrap_b,
        block_length=block_length,
        rng=make_panel_rng(args.seed, "stella_dynamic_sax12_gradient"),
        solver=solver,
    )

    diagnostics = {
        **calibration_diag,
        **dynamic_diag,
        "sample": f"{periods[0]}--{periods[-1]}",
        "N": len(ids),
        "T": len(shocks),
        "M": args.M,
        "block_length": block_length,
        "inversion": "dynamic",
        "sales_variant": "sax12",
        "inversion_description": inversion_note,
        "exposure_HHI": float(np.sum(exposure_shares**2)),
        "top1_exposure_share": float(exposure_shares.max()),
        "top10_exposure_share": float(np.sort(exposure_shares)[-10:].sum()),
        "official_archive": "data/stella2015/source.zip",
        "solver_validation": "output/stella_dynamic_solver_benchmark.json",
        "input_revenue_field": sales_diagnostics.get("input_revenue_field", "saleq"),
        "remaining_outcome_gap": f"Uses current-vintage {sales_diagnostics.get('input_revenue_field', 'saleq')} adjusted with {sales_diagnostics.get('seasonal_adjustment_engine', 'X-13')}/X-11. The sax12 variant reproduces the released seasonal specification; the historical Compustat vintage and exact original executable build remain unavailable.",
        "seasonal_adjustment_engine": sales_diagnostics.get("seasonal_adjustment_engine", "X-13"),
        "replication_scope": f"Exact deposited calibration accounts, model equations, and shock recurrence applied to a current-vintage reconstructed {len(ids)}-firm panel and customer/sector mapping.",
        "sample_mode": calibration_diag["sample_mode"],
        "calibration_directory": str(calibration.relative_to(ROOT)),
        "run_input_fingerprints": file_fingerprints(
            [calibration / name for name in (
                "firm_mapping.csv", "nominal_saleq_levels.csv", "diagnostics.json",
                "sales_growth_sax12.csv")]
            + [calibration / "dynamic_solution" / name for name in
               ("pii_a.csv", "xi.csv", "cap_rho.csv")], ROOT),
    }
    diagnostic_path = DIAGNOSTICS if output_stem == "stella_firm_network" else ROOT / "output" / f"{output_stem}_sax12_diagnostics.json"
    archive_existing(diagnostic_path)
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    pd.DataFrame(shocks, index=periods, columns=ids).to_pickle(
        calibration / "epsilon_dynamic_sax12.pkl"
    )

    stem = output_stem
    summary = {
        "dataset": f"{stem}_M{args.M}",
        "table_outcome": "Stella-style network-recovered firm output shocks",
        "frequency": "quarterly",
        "start_date": str(periods[0]),
        "last_date": str(periods[-1]),
        "N": len(ids),
        "T": shocks.shape[0],
        "M": args.M,
        "block_length": block_length,
        "block_bootstrap_b": args.bootstrap_b,
        "HHI": diagnostics["exposure_HHI"],
        "top1_share": diagnostics["top1_exposure_share"],
        "top10_share": diagnostics["top10_exposure_share"],
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        "direct_diagonal": direct,
        "naive": direct,
        "naive_homoscedastic": homoskedastic,
        **{key: value for key, value in observed.items() if key not in {"solver", "status"}},
        **gradient,
        "inversion": "dynamic",
        "sales_variant": "sax12",
        "input_revenue_field": diagnostics["input_revenue_field"],
        "seasonal_adjustment_engine": diagnostics["seasonal_adjustment_engine"],
        "sample_mode": calibration_diag["sample_mode"],
        "calibration_directory": diagnostics["calibration_directory"],
        "run_input_fingerprints": diagnostics["run_input_fingerprints"],
        "notes": diagnostics["replication_scope"] + " " + inversion_note + ".",
    }
    cap_dynamic_share_report(summary)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_OUTPUT_CACHE / f"{stem}_M{args.M}_summary.pkl"
    archive_existing(summary_path)
    pd.DataFrame([summary]).to_pickle(summary_path)
    print(pd.Series(summary).to_string())
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
