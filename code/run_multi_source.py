#!/usr/bin/env python3
"""Grouped-MGP estimates on the non-quarterly data sources for draft Table 1.

Datasets, each produced by a single default run as flat files in ``../output/``:
  - Compustat annual sales and annual employment over the last 50 years.
  - Gabaix-style Compustat labor productivity, annually (sales/employees) and
    quarterly (quarterly sales/annual employees).
  - BLS CES employment leaves, made additive by synthetic residual nodes, at M=10.
  - BLS CPI item inflation leaves with December 1967 weights,
    made additive by synthetic residual nodes, at M=10.
  - CRSP daily stock returns over the last 20*252 trading days, at M=30.
  - Forbes/Internet Archive daily billionaire wealth growth over the full
    2019--2025 archived real-time panel, at M=30.

The common estimator is the paper's weighted-contribution grouping,
``x_{m,t} = sum_{i in B_m} S_i y_{i,t}``, so the grouped MGP is the unweighted
fixed-M SDP. The estimation core is imported from ``mgp_core.py``; each panel
gets its own deterministic random stream so a single run reproduces every row.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    OUTPUT_DIR,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    analysis_block_length,
    build_lpt_grouped_data,
    choose_solver,
    default_subsample_length,
    make_panel_rng,
    mgp_share,
    regularity_report,
    resample_mgp,
    sample_cov,
    subsample_mgp,
    subsampling_summary,
)
from wrds_inputs import ANALYSIS_LAST_QUARTER, wrds_path
from project_paths import DERIVED_CACHE, ROOT, available, data_file
from mosek_face import licensed_mosek

from applications.bls_ces.panel import load_bls_ces_panel
from applications.bls_cpi.panel import load_bls_cpi_panel
from applications.compustat.panel import load_compustat_annual_panel, load_compustat_quarterly_tfp_panel
from applications.crsp.panel import load_crsp_panel
from applications.forbes.panel import (
    FORBES_ARCHIVE_DAILY,
    load_forbes_billionaire_panels,
)
from applications.panels import (
    GDPDEF_PATH,
    WEIGHT_TIMING,
    SourcePanel,
)
import gradient_bootstrap as GB

RESULTS_DIR = OUTPUT_DIR
DAILY_HIGH_M_THRESHOLD = 30
DATASETS = ("bls", "cpi", "crsp", "compustat_annual", "compustat_tfp", "forbes_billionaires")


# Short output stems for each source panel. The `_M{m}` (and Forbes `_T{t}`)
# suffix is carried over unchanged, so the map does not need to know which M
# a run used.
OUTPUT_STEMS = {
    "compustat_annual_sales": "compustat_ann_sales",
    "compustat_annual_employment": "compustat_ann_emp",
    "compustat_annual_tfp": "compustat_ann_tfp",
    "compustat_quarterly_tfp": "compustat_qtr_tfp",
    "bls_ces_employment": "bls_ces",
    "bls_cpi_1967_inflation": "cpi1967",
    "crsp_daily_returns": "crsp",
    "forbes_billionaire_wealth_daily": "forbes",
    "forbes_billionaire_wealth_daily_full": "forbes_full",
    "foerster_epsilon_network": "foerster_net",
    "foerster_epsilon": "foerster_eps",
}
_SUFFIX = re.compile(r"((?:_T\d+)?_M\d+)$")


def output_prefix(name: str) -> str:
    m = _SUFFIX.search(name)
    stem, suffix = (name[: m.start()], m.group(1)) if m else (name, "")
    return OUTPUT_STEMS.get(stem, stem) + suffix


def lpt_diagnostics(unit_ids: pd.Index, shares: np.ndarray, labels: pd.Series, buckets: list[list[int]], loads: np.ndarray):
    assignment = np.empty(len(shares), dtype=int)
    for bucket_id, bucket in enumerate(buckets):
        assignment[bucket] = bucket_id
    unit_id_text = pd.Series([str(x) for x in unit_ids], index=unit_ids)
    label_values = labels.reindex(unit_ids)
    label_values = label_values.where(label_values.notna(), unit_id_text)
    firm_diag = pd.DataFrame(
        {
            "unit_id": unit_id_text.to_numpy(),
            "bucket": assignment,
            "name": label_values.to_numpy(),
            "size_share": shares,
            "size_share_sq": shares**2,
        }
    )
    bucket_diag = (
        firm_diag.groupby("bucket")
        .agg(
            n_firms=("unit_id", "size"),
            total_size_share=("size_share", "sum"),
            total_sq_share=("size_share_sq", "sum"),
            max_size_share=("size_share", "max"),
        )
        .reset_index()
    )
    bucket_diag["lpt_load"] = loads[bucket_diag["bucket"].to_numpy()]
    return firm_diag, bucket_diag


def run_panel(
    panel: SourcePanel,
    m: int,
    block_length: int,
    bootstrap_b: int,
    block_bootstrap_b: int,
    subsample_n: int,
    solver: str,
    rng: np.random.Generator,
    gradient_rng: np.random.Generator,
    out_root: Path,
    progress_every: int,
    run_profile: str,
) -> dict[str, object]:
    if panel.weight_timing != WEIGHT_TIMING:
        raise ValueError(
            f"{panel.name} has weight_timing={panel.weight_timing!r}; expected {WEIGHT_TIMING!r}."
        )
    requested_bootstrap_b = bootstrap_b
    requested_block_bootstrap_b = block_bootstrap_b
    requested_subsample_n = subsample_n
    y, sizes = align_sizes_to_panel(panel.y, panel.sizes)
    shares = (sizes / sizes.sum()).to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    print(
        f"\n[{panel.name}] N={y.shape[0]:,}, T={y.shape[1]:,}, M={m}, "
        f"start={panel.start_date}, last={panel.last_date}, block={block_length}",
        flush=True,
    )
    x_grouped, buckets, loads, _ = build_lpt_grouped_data(y_np, shares, m)
    observed = mgp_share(x_grouped, solver)
    regularity = regularity_report(sample_cov(x_grouped), solver)
    hhi = float(np.sum(shares**2))
    print(
        f"[{panel.name}] observed share={observed['mgp_share']:.4f}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}, HHI={hhi:.6g}",
        flush=True,
    )
    prefix = output_prefix(panel.name)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    # A source refresh must not resume resampling draws from the previous panel.
    stamp = RUN_OUTPUT_CACHE / f"{prefix}_resampling_input.sha256"
    fingerprint = hashlib.sha256(np.ascontiguousarray(x_grouped).tobytes()).hexdigest()
    if not stamp.exists() or stamp.read_text().strip() != fingerprint:
        stale = list(RUN_OUTPUT_CACHE.glob(f"{prefix}_draws_*.pkl"))
        if stale:
            archive = RUN_OUTPUT_CACHE / "archive" / f"{prefix}_{time.time_ns()}"
            archive.mkdir(parents=True, exist_ok=True)
            for path in stale:
                path.rename(archive / path.name)
        stamp.write_text(fingerprint + "\n")
    bootstrap = resample_mgp(
        x_grouped,
        b=bootstrap_b,
        solver=solver,
        rng=rng,
        mode="bootstrap",
        progress_every=progress_every,
        checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_bootstrap.pkl",
        checkpoint_every=progress_every,
    )
    block_bootstrap = resample_mgp(
        x_grouped,
        b=block_bootstrap_b,
        solver=solver,
        rng=rng,
        mode="moving_block_bootstrap",
        progress_every=progress_every,
        block_length=block_length,
        checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_moving_block_bootstrap.pkl",
        checkpoint_every=progress_every,
    )
    draws = pd.concat([bootstrap, block_bootstrap], ignore_index=True)

    gradient_family: dict[str, object] = {}
    gradient_b = min(bootstrap_b, block_bootstrap_b)
    if gradient_b > 0:
        gradient_family = GB.bootstrap_family(
            x_grouped,
            b=gradient_b,
            block_length=block_length,
            rng=gradient_rng,
            solver=solver,
            progress_every=progress_every,
            checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_gradient.pkl",
            checkpoint_every=progress_every,
        )

    ell = default_subsample_length(y.shape[1], m)
    if subsample_n > 0 and ell < y.shape[1]:
        subsamples, _ = subsample_mgp(
            x_grouped, ell=ell, n_sub=subsample_n, solver=solver, rng=rng,
            progress_every=progress_every, g_hat=observed["mgp_share"], overlapping=True,
            checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_subsampling.pkl",
            checkpoint_every=progress_every,
        )
        sub_ci = subsampling_summary(subsamples, observed["mgp_share"], y.shape[1])
    else:
        print(f"[{panel.name}] subsampling skipped (n={subsample_n}, ell={ell}, T={y.shape[1]}).", flush=True)
        subsamples = pd.DataFrame()
        sub_ci = {}

    out_root.mkdir(parents=True, exist_ok=True)
    firm_diag, bucket_diag = lpt_diagnostics(y.index, shares, panel.labels, buckets, loads)
    draws.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_draws_resampling.pkl")
    if not subsamples.empty:
        subsamples.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_draws_subsampling.pkl")
    firm_diag.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_units.pkl")
    bucket_diag.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_buckets.pkl")
    boot_vals = bootstrap["mgp_share"] if not bootstrap.empty else pd.Series(dtype=float)
    block_vals = block_bootstrap["mgp_share"] if not block_bootstrap.empty else pd.Series(dtype=float)
    summary = {
        "dataset": panel.name,
        "run_profile": run_profile,
        "solver_backend": solver,
        "frequency": panel.frequency,
        "value_name": panel.value_name,
        "size_name": panel.size_name,
        "weight_timing": panel.weight_timing,
        "weight_date": str(panel.weight_date),
        "clean_panel_path": panel.clean_panel_path,
        "start_date": str(panel.start_date),
        "last_date": str(panel.last_date),
        "N": y.shape[0],
        "T": y.shape[1],
        "M": m,
        "bootstrap_b": bootstrap_b,
        "block_bootstrap_b": block_bootstrap_b,
        "requested_bootstrap_b": requested_bootstrap_b,
        "requested_block_bootstrap_b": requested_block_bootstrap_b,
        "requested_subsample_n": requested_subsample_n,
        "block_length": block_length,
        "coverage": panel.coverage,
        "HHI": hhi,
        "top1_share": float(shares[0]),
        "top10_share": float(shares[: min(10, len(shares))].sum()),
        "top50_share": float(shares[: min(50, len(shares))].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        # sigma_x: equal-weighted mean over units of each unit's time-series std (raw,
        # ungrouped); sigma_y: std of the aggregate = sqrt(1' Sigma_tilde 1).
        "sigma_x": float(np.nanmean(np.nanstd(y_np, axis=1, ddof=1))),
        "sigma_y": float(np.sqrt(observed["aggregate_variance"])),
        "mgp_share": observed["mgp_share"],
        "mgp_level": observed["mgp_level"],
        "aggregate_variance": observed["aggregate_variance"],
        "se_share": observed["se_share"],
        "normal_ucl95": observed["normal_ucl95"],
        "ledermann_kbar": regularity["ledermann_kbar"],
        "common_rank": regularity["common_rank"],
        "regular": regularity["regular"],
        "corank": regularity["corank"],
        "tight": regularity["tight"],
        "corank_threshold_2sqrtM": regularity["corank_threshold_2sqrtM"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
        "gap_dmax_over_dbar": regularity["gap_dmax_over_dbar"],
        "gap_hmax": regularity["gap_hmax"],
        "gap_bound_factor": regularity["gap_bound_factor"],
        "bootstrap_mean": float(boot_vals.mean()) if not boot_vals.empty else np.nan,
        "bootstrap_p05": float(boot_vals.quantile(0.05)) if not boot_vals.empty else np.nan,
        "bootstrap_p50": float(boot_vals.quantile(0.50)) if not boot_vals.empty else np.nan,
        "bootstrap_p95": float(boot_vals.quantile(0.95)) if not boot_vals.empty else np.nan,
        "moving_block_bootstrap_mean": float(block_vals.mean()) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p05": float(block_vals.quantile(0.05)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p50": float(block_vals.quantile(0.50)) if not block_vals.empty else np.nan,
        "moving_block_bootstrap_p95": float(block_vals.quantile(0.95)) if not block_vals.empty else np.nan,
        "gradient_bootstrap_p05": gradient_family.get("gradient_low", np.nan),
        "gradient_bootstrap_p95": gradient_family.get("gradient_high", np.nan),
        "moving_block_gradient_bootstrap_p05": gradient_family.get("block_gradient_low", np.nan),
        "moving_block_gradient_bootstrap_p95": gradient_family.get("block_gradient_high", np.nan),
        "gradient_bootstrap_common_rank": gradient_family.get("directional_common_rank", np.nan),
        "gradient_bootstrap_regular": gradient_family.get("directional_regular", np.nan),
        **sub_ci,
        "notes": panel.notes,
    }
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    print(
        f"[{panel.name}] boot p95={summary['bootstrap_p95']:.4f}, "
        f"block p95={summary['moving_block_bootstrap_p95']:.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
    )
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--block-bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--solver", default="auto")
    parser.add_argument(
        "--run-profile",
        choices=["medium", "full"],
        default="full",
        help="medium skips subsampling; full runs the expensive diagnostics.",
    )
    parser.add_argument("--refresh-panel-cache", action="store_true", help="Rebuild and overwrite cleaned-panel caches.")
    parser.add_argument("--no-panel-cache", dest="use_panel_cache", action="store_false", help="Do not read or write cleaned-panel caches.")
    parser.set_defaults(use_panel_cache=True)
    parser.add_argument("--crsp-t", type=int, default=20 * 252)
    parser.add_argument(
        "--crsp-m",
        nargs="+",
        type=int,
        default=None,
        help="CRSP bucket counts. Default: 30 for the daily headline panel.",
    )
    parser.add_argument("--annual-t", type=int, default=50)
    parser.add_argument("--annual-m", type=int, default=10)
    parser.add_argument("--quarterly-tfp-t", type=int, default=100)
    parser.add_argument("--quarterly-tfp-m", type=int, default=10)
    parser.add_argument("--quarterly-tfp-last-date", default=ANALYSIS_LAST_QUARTER)
    parser.add_argument(
        "--bls-m",
        nargs="+",
        type=int,
        default=None,
        help="BLS CES bucket counts. Default: 10.",
    )
    parser.add_argument(
        "--cpi-m",
        nargs="+",
        type=int,
        default=None,
        help="BLS CPI bucket counts. Default: 10.",
    )
    parser.add_argument("--forbes-archive-csv", type=Path, default=FORBES_ARCHIVE_DAILY)
    parser.add_argument(
        "--forbes-t",
        nargs="+",
        type=int,
        default=None,
        help="Optional trailing Forbes wealth-growth windows in archive snapshots. Default: full archive window.",
    )
    parser.add_argument(
        "--forbes-m",
        nargs="+",
        type=int,
        default=None,
        help="Forbes billionaire bucket counts. Default: 30 for the daily headline panel.",
    )
    parser.add_argument("--subsample-n", type=int, default=300,
                        help="Number of subsamples for the subsampling CI.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.run_profile == "medium":
        args.subsample_n = 0
    solver = choose_solver(args.solver)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}", flush=True)
    print(
        f"solver={solver}, profile={args.run_profile}, bootstrap_b={args.bootstrap_b}, "
        f"block_bootstrap_b={args.block_bootstrap_b}, subsample_n={args.subsample_n}",
        flush=True,
    )

    bls_dir = ROOT / "data/bls_ces"
    cpi_dir = DERIVED_CACHE / "bls_cpi/panels"
    annual = wrds_path("compustat_annual")
    def ready(dataset, *inputs, stems):
        if dataset not in args.datasets:
            return False
        if not args.available_only or available(", ".join(stems), *inputs):
            return True
        for stem in stems:
            (RUN_OUTPUT_CACHE / f"{stem}_summary.pkl").unlink(missing_ok=True)
            for path in RUN_OUTPUT_CACHE.glob(f"{stem}_M*_summary.pkl"):
                path.unlink()
        return False

    jobs: list[tuple[SourcePanel, int, int]] = []

    def add_grouped_jobs(panel: SourcePanel, bucket_counts: list[int]) -> None:
        for m in dict.fromkeys(bucket_counts):
            grouped_panel = replace(
                panel, name=f"{panel.name}_M{m}",
                notes=f"{panel.notes} Grouped with M={m}.",
            )
            jobs.append((grouped_panel, m, analysis_block_length(panel.y.shape[1])))

    if ready("bls", *(data_file(bls_dir / name) for name in ("ce.data.0.AllCESSeries.txt", "ce.industry.txt", "tree.csv")),
             stems=("bls_ces",)):
        bls = load_bls_ces_panel(args.use_panel_cache, args.refresh_panel_cache)
        add_grouped_jobs(bls, args.bls_m if args.bls_m is not None else [10])
    if ready("crsp", wrds_path("crsp_daily_stocks_ciz"), wrds_path("crsp_security_history_ciz"), stems=("crsp",)):
        crsp = load_crsp_panel(args.crsp_t, args.use_panel_cache, args.refresh_panel_cache)
        add_grouped_jobs(crsp, args.crsp_m if args.crsp_m is not None else [30])
    if ready("cpi", cpi_dir / "cpi_1967_leaves.csv", cpi_dir / "cpi_1967_inflation.parquet", stems=("cpi1967",)):
        cpi = load_bls_cpi_panel(args.use_panel_cache, args.refresh_panel_cache)
        add_grouped_jobs(cpi, args.cpi_m if args.cpi_m is not None else [10])
    if ready("compustat_annual", annual, stems=("compustat_ann_sales", "compustat_ann_emp")):
        jobs.append((
            load_compustat_annual_panel("sale", args.annual_t, args.use_panel_cache, args.refresh_panel_cache),
            args.annual_m,
            analysis_block_length(args.annual_t),
        ))
        jobs.append((
            load_compustat_annual_panel("emp", args.annual_t, args.use_panel_cache, args.refresh_panel_cache),
            args.annual_m,
            analysis_block_length(args.annual_t),
        ))
    if ready("compustat_tfp", annual, GDPDEF_PATH, wrds_path("compustat_quarterly_sales"),
             stems=("compustat_qtr_tfp",)):
        jobs.append((
            load_compustat_quarterly_tfp_panel(
                args.quarterly_tfp_t,
                args.quarterly_tfp_last_date,
                args.use_panel_cache,
                args.refresh_panel_cache,
            ),
            args.quarterly_tfp_m,
            analysis_block_length(args.quarterly_tfp_t),
        ))
    if ready("compustat_tfp", annual, GDPDEF_PATH, stems=("compustat_ann_tfp",)):
        jobs.append((
            load_compustat_annual_panel("tfp", args.annual_t, args.use_panel_cache, args.refresh_panel_cache),
            args.annual_m,
            analysis_block_length(args.annual_t),
        ))
    if ready("forbes_billionaires", args.forbes_archive_csv, stems=("forbes_full",)):
        forbes_panels = load_forbes_billionaire_panels(
            args.forbes_t,
            args.forbes_archive_csv,
            args.use_panel_cache,
            args.refresh_panel_cache,
        )
        for panel in forbes_panels:
            add_grouped_jobs(panel, args.forbes_m if args.forbes_m is not None else [30])

    summaries = []
    for panel, m, block_length in jobs:
        is_daily = panel.frequency == "daily"
        is_daily_high_m = is_daily and m >= DAILY_HIGH_M_THRESHOLD
        job_solver = "MOSEK" if is_daily_high_m and licensed_mosek() else solver
        if is_daily_high_m:
            print(
                f"[{panel.name}] daily high-M profile: solver={job_solver}, "
                f"bootstrap/block/subsample={args.bootstrap_b}/{args.block_bootstrap_b}/{args.subsample_n}",
                flush=True,
            )
        summaries.append(
            run_panel(
                panel=panel,
                m=m,
                block_length=block_length,
                bootstrap_b=args.bootstrap_b,
                block_bootstrap_b=args.block_bootstrap_b,
                subsample_n=args.subsample_n,
                solver=job_solver,
                rng=make_panel_rng(args.seed, panel.name),
                gradient_rng=make_panel_rng(args.seed, f"{panel.name}_gradient"),
                out_root=out_root,
                progress_every=args.progress_every,
                run_profile=args.run_profile,
            )
        )

    summary = pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=["dataset"])
    summary_path = RUN_OUTPUT_CACHE / "multi_source_summary_all.pkl"
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    # A partial --datasets run must not clobber unrelated rows, but it must remove
    # every stale M variant of the datasets being replaced.
    if set(args.datasets) != set(DATASETS) and summary_path.exists():
        previous = pd.read_pickle(summary_path)
        selected_prefixes = {
            "bls": ("bls_ces_employment_",),
            "cpi": ("bls_cpi_",),
            "crsp": ("crsp_daily_returns_",),
            "compustat_annual": ("compustat_annual_",),
            "compustat_tfp": ("compustat_quarterly_tfp", "compustat_annual_tfp"),
            "forbes_billionaires": ("forbes_billionaire_wealth_daily_",),
        }
        prefixes = tuple(
            prefix for dataset in args.datasets for prefix in selected_prefixes[dataset]
        )
        previous = previous[~previous["dataset"].astype(str).str.startswith(prefixes)]
        summary = pd.concat([previous, summary], ignore_index=True)
    summary.to_pickle(summary_path)
    display_cols = [
        "dataset",
        "N",
        "T",
        "M",
        "start_date",
        "last_date",
        "coverage",
        "HHI",
        "mgp_share",
        "bootstrap_p95",
        "moving_block_bootstrap_p95",
    ]
    print("\nSummary:", flush=True)
    if not summary.empty:
        print(summary[display_cols].round(6).to_string(index=False), flush=True)
    print(f"\nSaved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
