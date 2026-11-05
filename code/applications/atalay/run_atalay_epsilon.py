#!/usr/bin/env python3
"""Grouped-MGP estimate on Atalay's benchmark primitive productivity shocks.

The inputs are exports from an unmodified run of the deposited Atalay (2017)
``filter_full.m`` at parameters [0.999, 0.999, 0.10, 0.999, 2, 0, 1, 4].
No code in the deposited replication package is imported or modified here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factor_rank import factor_rank_tests
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    ROOT,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    build_lpt_grouped_data,
    choose_solver,
    default_subsample_length,
    make_panel_rng,
    regularity_report,
    sample_cov,
)
from applications.foerster.run_foerster_epsilon import (
    cap_dynamic_share_report,
    network_gradient_family,
    network_observed_stats,
    resample_network_mgp,
    subsample_network_mgp,
)
from applications.panels import SourcePanel
from run_multi_source import output_prefix


DERIVED = ROOT / "code/.cache/derived/atalay2017"
DEFAULT_SHOCKS_CSV = DERIVED / "atalay_benchmark_productivity_shocks.csv"
DEFAULT_NETWORK_V_CSV = DERIVED / "atalay_benchmark_network_variance_matrix.csv"

INDUSTRY_LABELS = [
    "Agriculture",
    "Mining except oil and gas",
    "Oil and gas extraction",
    "Construction",
    "Food products",
    "Textiles",
    "Apparel and leather",
    "Wood products",
    "Furniture",
    "Paper",
    "Printing and publishing",
    "Chemicals",
    "Petroleum and coal products",
    "Plastics and rubber",
    "Nonmetallic mineral products",
    "Primary metals",
    "Fabricated metal products",
    "Machinery",
    "Electrical machinery",
    "Motor vehicles",
    "Other transportation equipment",
    "Instruments",
    "Miscellaneous manufacturing",
    "Transportation",
    "Communications",
    "Utilities",
    "Wholesale and retail trade",
    "Finance, insurance, and real estate",
    "Other services",
    "Government",
]


def load_atalay_panel(shocks_csv: Path, network_v_csv: Path, m: int) -> tuple[SourcePanel, np.ndarray]:
    if not shocks_csv.exists():
        raise FileNotFoundError(f"Missing Atalay primitive-shock export: {shocks_csv}")
    if not network_v_csv.exists():
        raise FileNotFoundError(f"Missing Atalay propagation export: {network_v_csv}")

    shocks = pd.read_csv(shocks_csv)
    shock_cols = [col for col in shocks if col.startswith("shock_")]
    # The column order carries three positional assumptions at once: INDUSTRY_LABELS,
    # the rows/columns of the network variance matrix, and the reindexing in
    # run_atalay_panel. Pin it here rather than letting any of them drift silently.
    expected_cols = [f"shock_{i:02d}" for i in range(1, len(INDUSTRY_LABELS) + 1)]
    if shock_cols != expected_cols:
        raise ValueError(
            f"{shocks_csv} must hold {expected_cols[0]}..{expected_cols[-1]} in Atalay's "
            f"Table A1 industry order, found {shock_cols}."
        )
    years = shocks["year"].astype(int)
    if not years.is_monotonic_increasing or years.duplicated().any():
        raise ValueError("Atalay shock years must be unique and increasing.")

    network_v = pd.read_csv(network_v_csv, header=None).to_numpy(float)
    if network_v.shape != (len(shock_cols), len(shock_cols)):
        raise ValueError(f"Atalay network V has shape {network_v.shape}, expected (30, 30).")
    network_v = 0.5 * (network_v + network_v.T)
    min_eig = float(np.linalg.eigvalsh(network_v).min())
    if min_eig < -1e-9 * max(float(np.linalg.norm(network_v, 2)), 1.0):
        raise ValueError(f"Atalay network V is not positive semidefinite (min eigenvalue {min_eig:g}).")

    ids = pd.Index(shock_cols, name="shock_id")
    y = shocks[shock_cols].T.astype(float)
    y.index = ids
    y.columns = pd.Index(years.astype(str), name="year")
    y = y.sub(float(y.to_numpy().mean()))
    sizes = pd.Series(np.sqrt(np.maximum(np.diag(network_v), 0.0)), index=ids)
    if not sizes.gt(0).all():
        raise ValueError("Atalay total-influence weights must be strictly positive.")
    labels = pd.Series(INDUSTRY_LABELS, index=ids)

    panel = SourcePanel(
        name=f"atalay_epsilon_network_M{m}",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=int(years.iloc[0]),
        last_date=int(years.iloc[-1]),
        frequency="annual",
        size_name="Atalay dynamic aggregate-output influence L_i",
        value_name="Atalay benchmark primitive productivity innovation",
        coverage=1.0,
        notes=(
            "Exports from the unmodified Atalay (2017) U.S. benchmark filter, "
            "epsilon_D=epsilon_Q=epsilon_X=0.999 and epsilon_M=0.10; four-period burn-in."
        ),
        weight_timing="structural_total_influence_v1",
        weight_date="benchmark calibration",
    )
    return panel, network_v


def run_atalay_panel(
    panel: SourcePanel,
    network_v: np.ndarray,
    *,
    m: int,
    block_length: int,
    bootstrap_b: int,
    block_bootstrap_b: int,
    subsample_n: int,
    solver: str,
    seed: int,
    progress_every: int,
    run_profile: str,
) -> dict[str, object]:
    y, sizes = align_sizes_to_panel(panel.y, panel.sizes)
    order_pos = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    if sorted(order_pos) != list(range(network_v.shape[0])):
        raise ValueError("Atalay panel ordering is not a permutation of the network rows.")
    network_v = network_v[np.ix_(order_pos, order_pos)]

    shares = (sizes / sizes.sum()).to_numpy(float)
    weight_scale = float(sizes.sum() ** 2)
    y_np = y.to_numpy(float)
    eps_obs = y_np.T
    x_grouped, _, loads, _ = build_lpt_grouped_data(y_np, shares, m)
    observed = network_observed_stats(x_grouped, eps_obs, network_v, weight_scale, solver)
    sigma_grouped = sample_cov(x_grouped)
    regularity = regularity_report(sigma_grouped, solver)
    hhi = float(np.sum(shares**2))

    try:
        _, rank_summary = factor_rank_tests(
            x_grouped,
            alpha=0.05,
            progress_label=f"atalay_network_M{m}",
        )
        factor_lb_j = rank_summary.get("factor_lb_J", np.nan)
        regular_certified_j = bool(rank_summary.get("regular_certified_J", False))
    except Exception as exc:
        print(f"[{panel.name}] J-test skipped: {type(exc).__name__}: {exc}", flush=True)
        factor_lb_j, regular_certified_j = np.nan, False

    print(
        f"\n[{panel.name}] N={y.shape[0]}, T={y.shape[1]}, M={m}, "
        f"sample={panel.start_date}--{panel.last_date}, block={block_length}",
        flush=True,
    )
    print(
        f"[{panel.name}] observed share={observed['mgp_share']:.4f}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}, exposure HHI={hhi:.6g}, "
        f"denom={observed['aggregate_variance']:.6g}",
        flush=True,
    )

    rng = make_panel_rng(seed, panel.name)
    gradient_rng = make_panel_rng(seed, f"{panel.name}_gradient")
    bootstrap = resample_network_mgp(
        x_grouped,
        eps_obs,
        network_v,
        weight_scale,
        b=bootstrap_b,
        solver=solver,
        rng=rng,
        mode="bootstrap",
        progress_every=progress_every,
    )
    block_bootstrap = resample_network_mgp(
        x_grouped,
        eps_obs,
        network_v,
        weight_scale,
        b=block_bootstrap_b,
        solver=solver,
        rng=rng,
        mode="moving_block_bootstrap",
        progress_every=progress_every,
        block_length=block_length,
    )
    gradient = network_gradient_family(
        x_grouped,
        eps_obs,
        network_v,
        weight_scale,
        b=min(bootstrap_b, block_bootstrap_b),
        block_length=block_length,
        rng=gradient_rng,
        solver=solver,
    )

    ell = default_subsample_length(y.shape[1], m)
    if subsample_n > 0 and ell < y.shape[1]:
        subsamples, sub_ci = subsample_network_mgp(
            x_grouped,
            eps_obs,
            network_v,
            weight_scale,
            ell=ell,
            n_sub=subsample_n,
            solver=solver,
            rng=rng,
            progress_every=progress_every,
            g_hat=float(observed["mgp_share_uncapped"]),
        )
    else:
        subsamples, sub_ci = pd.DataFrame(), {}

    prefix = output_prefix(panel.name)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat([bootstrap, block_bootstrap], ignore_index=True).to_pickle(
        RUN_OUTPUT_CACHE / f"{prefix}_draws_resampling.pkl"
    )
    if not subsamples.empty:
        subsamples.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_draws_subsampling.pkl")

    boot_vals = bootstrap["mgp_share_uncapped"] if not bootstrap.empty else pd.Series(dtype=float)
    block_vals = block_bootstrap["mgp_share_uncapped"] if not block_bootstrap.empty else pd.Series(dtype=float)
    summary = {
        "dataset": panel.name,
        "table_column": f"Network exposure M={m}",
        "table_outcome": "network-propagated primitive-shock bound",
        "run_profile": run_profile,
        "frequency": panel.frequency,
        "value_name": panel.value_name,
        "size_name": panel.size_name,
        "weight_timing": panel.weight_timing,
        "weight_date": str(panel.weight_date),
        "start_date": str(panel.start_date),
        "last_date": str(panel.last_date),
        "N": y.shape[0],
        "T": y.shape[1],
        "M": m,
        "bootstrap_b": bootstrap_b,
        "block_bootstrap_b": block_bootstrap_b,
        "requested_subsample_n": subsample_n,
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
        "factor_lb_J": factor_lb_j,
        "regular_certified_J": regular_certified_j,
        "regular": regularity["regular"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
        "bootstrap_mean": float(boot_vals.mean()) if not boot_vals.empty else np.nan,
        "bootstrap_p05": float(boot_vals.quantile(0.05)) if not boot_vals.empty else np.nan,
        "bootstrap_p50": float(boot_vals.quantile(0.50)) if not boot_vals.empty else np.nan,
        "bootstrap_p95": float(boot_vals.quantile(0.95)) if not boot_vals.empty else np.nan,
        "moving_block_bootstrap_mean": float(block_vals.mean()) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p05": float(block_vals.quantile(0.05)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p50": float(block_vals.quantile(0.50)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p95": float(block_vals.quantile(0.95)) if not block_vals.empty else np.nan,
        **gradient,
        **sub_ci,
        "network_weight_sum": float(sizes.sum()),
        "network_denominator_variance": float(observed["aggregate_variance"]),
        "notes": (
            "Atalay benchmark observable-space upper bound: numerator uses total dynamic "
            "aggregate-output influence L_i and denominator <V,Sigma_omega>."
        ),
    }
    cap_dynamic_share_report(summary)
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shocks-csv", type=Path, default=DEFAULT_SHOCKS_CSV)
    parser.add_argument("--network-v-csv", type=Path, default=DEFAULT_NETWORK_V_CSV)
    parser.add_argument("--M", type=int, default=10,
                        help="LPT groups; default 10.")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--block-bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--subsample-n", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--run-profile", choices=["medium", "full"], default="full")
    args = parser.parse_args()
    if args.run_profile == "medium":
        args.subsample_n = 0

    solver = choose_solver(args.solver)
    panel, network_v = load_atalay_panel(args.shocks_csv, args.network_v_csv, args.M)
    block_length = analysis_block_length(panel.y.shape[1])
    summary = run_atalay_panel(
        panel,
        network_v,
        m=args.M,
        block_length=block_length,
        bootstrap_b=args.bootstrap_b,
        block_bootstrap_b=args.block_bootstrap_b,
        subsample_n=args.subsample_n,
        solver=solver,
        seed=args.seed,
        progress_every=args.progress_every,
        run_profile=args.run_profile,
    )
    display = pd.DataFrame([summary])
    cols = [
        "dataset",
        "N",
        "T",
        "M",
        "mgp_share",
        "moving_block_gradient_bootstrap_p05",
        "moving_block_gradient_bootstrap_p95",
    ]
    print("\nAtalay epsilon summary:", flush=True)
    print(display[cols].round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
