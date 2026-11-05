#!/usr/bin/env python3
"""Grouped-MGP estimates on the non-quarterly data sources for draft Table 1.

Datasets, each saved as a summary under ``.cache/run_outputs/``:
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
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    analysis_block_length,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    mgp_share,
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
    solver: str,
    gradient_rng: np.random.Generator,
    progress_every: int,
) -> dict[str, object]:
    if panel.weight_timing != WEIGHT_TIMING:
        raise ValueError(
            f"{panel.name} has weight_timing={panel.weight_timing!r}; expected {WEIGHT_TIMING!r}."
        )
    y, sizes = align_sizes_to_panel(panel.y, panel.sizes)
    shares = (sizes / sizes.sum()).to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    print(
        f"\n[{panel.name}] N={y.shape[0]:,}, T={y.shape[1]:,}, M={m}, "
        f"start={panel.start_date}, last={panel.last_date}, block={block_length}",
        flush=True,
    )
    x_grouped, _, loads, _ = build_lpt_grouped_data(y_np, shares, m)
    observed = mgp_share(x_grouped, solver)
    hhi = float(np.sum(shares**2))
    print(
        f"[{panel.name}] observed share={observed['mgp_share']:.4f}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}, HHI={hhi:.6g}",
        flush=True,
    )
    prefix = output_prefix(panel.name)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    # The inference checkpoint validates its own data and resampling indices.
    gradient_family: dict[str, object] = {}
    if bootstrap_b > 0:
        gradient_family = GB.bootstrap_family(
            x_grouped,
            b=bootstrap_b,
            block_length=block_length,
            rng=gradient_rng,
            solver=solver,
            progress_every=progress_every,
            checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_gradient.pkl",
            checkpoint_every=progress_every,
        )

    summary = {
        "dataset": panel.name,
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
        "gradient_bootstrap_p05": gradient_family.get("gradient_low", np.nan),
        "gradient_bootstrap_p95": gradient_family.get("gradient_high", np.nan),
        "moving_block_gradient_bootstrap_p05": gradient_family.get("block_gradient_low", np.nan),
        "moving_block_gradient_bootstrap_p95": gradient_family.get("block_gradient_high", np.nan),
        "gradient_bootstrap_common_rank": gradient_family.get("directional_common_rank", np.nan),
        "gradient_bootstrap_regular": gradient_family.get("directional_regular", np.nan),
        "notes": panel.notes,
    }
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    print(
        f"[{panel.name}] moving-block gradient UCL={summary['moving_block_gradient_bootstrap_p95']:.4f}",
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
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--solver", default="auto")
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
    args = parser.parse_args()

    solver = choose_solver(args.solver)
    print(f"solver={solver}, bootstrap_b={args.bootstrap_b}", flush=True)

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
                f"bootstrap_b={args.bootstrap_b}",
                flush=True,
            )
        summaries.append(
            run_panel(
                panel=panel,
                m=m,
                block_length=block_length,
                bootstrap_b=args.bootstrap_b,
                solver=job_solver,
                gradient_rng=make_panel_rng(args.seed, f"{panel.name}_gradient"),
                progress_every=args.progress_every,
            )
        )

    summary = pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=["dataset"])
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
        "moving_block_gradient_bootstrap_p05",
        "moving_block_gradient_bootstrap_p95",
    ]
    print("\nSummary:", flush=True)
    if not summary.empty:
        print(summary[display_cols].round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
