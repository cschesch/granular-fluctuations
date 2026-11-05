#!/usr/bin/env python3
"""Grouped-MGP estimate on Foerster-Sarte-Watson benchmark epsilon shocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    OUTPUT_DIR,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    plot_lpt_squared_bucket_sizes,
    regularity_report,
    sample_cov,
)
from applications.panels import SourcePanel
from applications.networks import (
    cap_dynamic_share_report,
    network_gradient_family,
    network_observed_stats,
    network_variance_matrix,
)
from run_multi_source import lpt_diagnostics, output_prefix
from .prepare import EXPORTS as FOERSTER_OUT

DEFAULT_EPSILON_CSV = FOERSTER_OUT / "eps_q_98_0_1972_2007benchmark_wide.csv"
DEFAULT_WEIGHTS_CSV = FOERSTER_OUT / "l4wq_q_98_0_1972_2007.csv"
DEFAULT_COLUMNS_CSV = FOERSTER_OUT / "eps_q_98_0_1972_2007benchmark_columns.csv"
DEFAULT_PII_A_CSV = FOERSTER_OUT / "pii_a_98_benchmark.csv"
DEFAULT_XI_CSV = FOERSTER_OUT / "xi_98_benchmark.csv"
DEFAULT_CAP_RHO_CSV = FOERSTER_OUT / "cap_rho_98_benchmark.csv"
DEFAULT_RESULTS_DIR = OUTPUT_DIR


def cal_to_quarter(value: float) -> str:
    year = int(np.floor(float(value) + 1e-8))
    quarter = int(round((float(value) - year) * 4.0)) + 1
    if quarter < 1 or quarter > 4:
        raise ValueError(f"Invalid Foerster quarterly date: {value}")
    return f"{year}Q{quarter}"


def load_foerster_shocks_and_weights(
    epsilon_csv: Path, weights_csv: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    export = pd.read_csv(epsilon_csv)
    eps = export.filter(regex=r"^eps_")
    if eps.empty or not (eps.iloc[0] == 0).all():
        raise ValueError("Expected a Foerster epsilon export starting at initialization.")
    weights = pd.read_csv(weights_csv, header=None)
    if weights.shape != (len(eps), eps.shape[1] + 1) or not np.allclose(
        weights.iloc[:, 0], export["cal"]
    ):
        raise ValueError("Foerster weights do not align with the epsilon export.")
    eps.index = pd.PeriodIndex(export["cal"].map(cal_to_quarter), freq="Q")
    # factor_io.gss drops the first quarter with missing output growth via
    # packr(e~dy). Its second quarter retains the zero initial innovation.
    sizes = weights.iloc[:, 1:].mean(axis=0)
    sizes.index = eps.columns
    if not sizes.gt(0).all():
        raise ValueError("Foerster mean IP weights must be positive.")
    return eps.iloc[1:], sizes


def load_foerster_epsilon_panel(
    epsilon_csv: Path,
    weights_csv: Path,
    columns_csv: Path,
    m: int,
) -> SourcePanel:
    if not epsilon_csv.exists():
        raise FileNotFoundError(f"Missing epsilon export: {epsilon_csv}")
    if not weights_csv.exists():
        raise FileNotFoundError(
            f"Missing Foerster L4 weights export: {weights_csv}. "
            "Run python -m applications.foerster.prepare first."
        )
    if not columns_csv.exists():
        raise FileNotFoundError(f"Missing epsilon column map: {columns_csv}")

    eps, sizes = load_foerster_shocks_and_weights(epsilon_csv, weights_csv)
    eps_cols = eps.columns
    periods = eps.index.astype(str)
    y = eps.T.set_axis(periods.rename("quarter"), axis=1)
    y.index.name = "epsilon_column"

    meta = pd.read_csv(columns_csv)
    labels = (
        meta.set_index("epsilon_column")["description"]
        .astype(str)
        .str.strip()
        .reindex(eps_cols)
    )
    labels = labels.fillna(pd.Series(eps_cols, index=eps_cols))

    return SourcePanel(
        name=f"foerster_epsilon_M{m}",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=periods[0],
        last_date=periods[-1],
        frequency="quarterly",
        size_name="Mean IP relative-importance weight over 1972–2007",
        value_name="Foerster-Sarte-Watson benchmark epsilon innovation",
        coverage=1.0,
        notes=(
            "Benchmark 1998-NAICS L4 epsilon export from the localized Foerster et al. "
            "replication package; retains the 1972Q2 zero initial innovation "
            f"and the authors' mean IP weights; grouped with M={m}."
        ),
        weight_timing="sample_mean",
        weight_date="1972Q1–2007Q4",
    )


def read_network_matrix(path: Path, n: int, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Foerster network matrix export: {path}. "
            "Run python -m applications.foerster.prepare first."
        )
    mat = pd.read_csv(path, header=None).to_numpy(float)
    if mat.shape != (n, n):
        raise ValueError(f"{name} matrix has shape {mat.shape}, expected {(n, n)}.")
    return mat


def run_network_exposure_panel(
    panel: SourcePanel,
    network_sizes: pd.Series,
    network_v: np.ndarray,
    m: int,
    block_length: int,
    bootstrap_b: int,
    solver: str,
    gradient_rng: np.random.Generator,
    out_root: Path,
) -> dict[str, object]:
    y, sizes = align_sizes_to_panel(panel.y, network_sizes)
    order_pos = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    network_v_sorted = network_v[np.ix_(order_pos, order_pos)]

    shares = (sizes / sizes.sum()).to_numpy(dtype=float)
    weight_scale = float(sizes.sum() ** 2)
    y_np = y.to_numpy(dtype=float)
    eps_obs = y_np.T
    x_grouped, buckets, loads, _ = build_lpt_grouped_data(y_np, shares, m)
    observed = network_observed_stats(x_grouped, eps_obs, network_v_sorted, weight_scale, solver)
    sigma_grouped = sample_cov(x_grouped)
    regularity = regularity_report(sigma_grouped, solver)
    hhi = float(np.sum(shares**2))

    print(
        f"\n[foerster_epsilon_network_M{m}] N={y.shape[0]:,}, T={y.shape[1]:,}, M={m}, "
        f"start={panel.start_date}, last={panel.last_date}, block={block_length}",
        flush=True,
    )
    print(
        f"[foerster_epsilon_network_M{m}] observed share={observed['mgp_share']:.4f}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}, exposure HHI={hhi:.6g}, "
        f"denom={observed['aggregate_variance']:.6g}",
        flush=True,
    )

    gradient = network_gradient_family(
        x_grouped, eps_obs, network_v_sorted, weight_scale,
        b=bootstrap_b, block_length=block_length,
        rng=gradient_rng, solver=solver,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(f"foerster_epsilon_network_M{m}")
    firm_diag, bucket_diag = lpt_diagnostics(y.index, shares, panel.labels, buckets, loads)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    plot_lpt_squared_bucket_sizes(
        firm_diag.rename(columns={"unit_id": "gvkey"}),
        bucket_diag,
        out_root / f"lpt_{prefix}",
        ylabel=r"Bucket squared-influence load: $\sum_{i\in B_m} V_i^2$",
    )
    summary = {
        "dataset": f"foerster_epsilon_network_M{m}",
        "table_column": f"Network exposure M={m}",
        "table_outcome": "network-propagated epsilon bound",
        "frequency": panel.frequency,
        "value_name": "Foerster epsilon innovations with FSW network exposure denominator",
        "size_name": "FSW network exposure L_i",
        "weight_timing": panel.weight_timing,
        "weight_date": str(panel.weight_date),
        "start_date": str(panel.start_date),
        "last_date": str(panel.last_date),
        "N": y.shape[0],
        "T": y.shape[1],
        "M": m,
        "bootstrap_b": bootstrap_b,
        "block_length": block_length,
        "coverage": panel.coverage,
        "HHI": hhi,
        "top1_share": float(shares[0]),
        "top10_share": float(shares[: min(10, len(shares))].sum()),
        "top50_share": float(shares[: min(50, len(shares))].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        **{k: v for k, v in observed.items() if k not in {"solver", "status", "dual_min_eig", "dual_min_diag"}},
        "ledermann_kbar": regularity["ledermann_kbar"],
        "common_rank": regularity["common_rank"],
        "regular": regularity["regular"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
        **gradient,
        "network_weight_sum": float(sizes.sum()),
        "network_denominator_variance": float(observed["aggregate_variance"]),
        "notes": (
            "FSW observable-space upper bound: numerator uses network exposure "
            "L_i=sqrt(sum_h (w'H_h e_i)^2), LPT grouped on L_i^2; denominator is "
            "model-implied aggregate IP variance <V,Sigma_epsilon>."
        ),
    }
    cap_dynamic_share_report(summary)
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-csv", type=Path, default=DEFAULT_EPSILON_CSV)
    parser.add_argument("--weights-csv", type=Path, default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--columns-csv", type=Path, default=DEFAULT_COLUMNS_CSV)
    parser.add_argument("--pii-a-csv", type=Path, default=DEFAULT_PII_A_CSV)
    parser.add_argument("--xi-csv", type=Path, default=DEFAULT_XI_CSV)
    parser.add_argument("--cap-rho-csv", type=Path, default=DEFAULT_CAP_RHO_CSV)
    parser.add_argument("--M", type=int, default=10,
                        help="LPT groups; default 10.")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if args.M <= 0:
        raise ValueError("M must be positive.")

    solver = choose_solver(args.solver)
    panel = load_foerster_epsilon_panel(args.epsilon_csv, args.weights_csv, args.columns_csv, args.M)
    block_length = analysis_block_length(panel.y.shape[1])
    n = panel.y.shape[0]
    pii_a = read_network_matrix(args.pii_a_csv, n, "Pi_a")
    xi = read_network_matrix(args.xi_csv, n, "Xi")
    cap_rho = read_network_matrix(args.cap_rho_csv, n, "cap_rho")
    network_v = network_variance_matrix(panel.sizes.to_numpy(float), pii_a, xi, cap_rho)
    network_sizes = pd.Series(np.sqrt(np.maximum(np.diag(network_v), 0.0)), index=panel.y.index)
    network_summary = run_network_exposure_panel(
        panel=panel,
        network_sizes=network_sizes,
        network_v=network_v,
        m=args.M,
        block_length=block_length,
        bootstrap_b=args.bootstrap_b,
        solver=solver,
        gradient_rng=make_panel_rng(args.seed, f"{panel.name}_network_gradient"),
        out_root=args.out_dir,
    )

    display = pd.DataFrame([network_summary])
    cols = [
        "dataset",
        "N",
        "T",
        "M",
        "HHI",
        "mgp_share",
        "moving_block_gradient_bootstrap_p05",
        "moving_block_gradient_bootstrap_p95",
    ]
    print("\nFoerster epsilon summary:", flush=True)
    print(display[cols].round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
