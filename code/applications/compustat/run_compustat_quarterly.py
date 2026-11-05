#!/usr/bin/env python3
"""Grouped-MGP estimate on quarterly Compustat sales growth (draft column 1 + Figure 1).

Pipeline:

1. Load quarterly Compustat USA data.
2. Keep domestic firms, drop oil/energy/finance SICs, and use positive sales.
3. Build quarterly log sales growth, remove unit quarter effects, and keep the
   last contiguous panel.
4. Choose the window (default: trailing 100 quarters ending 2025Q4).
5. Group firms by LPT on S_i^2 and solve the grouped MGP.
6. Run the time bootstrap and moving-block bootstrap of the grouped observations.

Outputs are written as flat files in ``../output/`` (overwritten each run). The
LPT squared-bucket-size plot saved there is the draft's Figure 1.

The estimation core (LPT, dual SDP, share, resampling, plotting) lives in
``mgp_core.py`` and is shared with ``run_multi_source.py``.
"""

from __future__ import annotations

from wrds_inputs import wrds_path, ANALYSIS_LAST_QUARTER

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    EXCLUDED_COMPUSTAT_SIC,
    analysis_block_length,
    ROOT,
    OUTPUT_DIR,
    RUN_OUTPUT_CACHE,
    build_lpt_grouped_data,
    choose_solver,
    clean_panel_cache_path,
    default_subsample_length,
    make_panel_contiguous,
    make_panel_rng,
    mgp_share,
    plot_lpt_squared_bucket_sizes,
    regularity_report,
    remove_periodic_means,
    resample_mgp,
    read_clean_panel_cache,
    sample_cov,
    subsample_mgp,
    subsampling_summary,
    summarize_distribution,
    standardize_panel,
    write_clean_panel_cache,
)

import numpy as np
import pandas as pd
import gradient_bootstrap as GB

DATA_DIR = ROOT / "data" / "compustat"
RESULTS_DIR = OUTPUT_DIR
OUTPUT_PREFIX = "compustat_qtr"


@dataclass(frozen=True)
class PanelData:
    wide: pd.DataFrame
    sizes: pd.Series
    names: pd.Series
    sic2: pd.Series
    panel: pd.DataFrame
    full_sales_by_date: pd.Series
    start_date: pd.Period
    last_date: pd.Period
    start_last_sales_coverage: float
    selection_rule: str
    weight_timing: str
    weight_date: pd.Period


def parse_t_arg(text: str) -> int | str:
    lowered = str(text).lower()
    if lowered in {"all", "coverage"}:
        return lowered
    out = int(text)
    if out <= 1:
        raise ValueError("--T must be an integer greater than 1, 'coverage', or 'all'.")
    return out


def clean_name(name: object) -> str:
    out = str(name)
    suffixes = [
        " INC",
        " CORP",
        " CO",
        " PLC",
        " LTD",
        " -CL A",
        " -CL B",
        " -OLD",
        " -ADR",
        "/DE",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out.strip()


def choose_start_by_last_sales_coverage(
    d: pd.DataFrame,
    dates: pd.PeriodIndex,
    threshold: float,
) -> tuple[pd.Period, float]:
    """Earliest start date whose continuous survivors cover threshold of last sales."""
    last_date = dates[-1]
    last_sizes = d.loc[last_date, "saleq"]
    total_last_size = float(last_sizes.sum())
    rows = []
    for date in dates:
        units = d.loc[date].index.unique("gvkey")
        coverage = float(last_sizes.reindex(units).sum() / total_last_size)
        rows.append((date, coverage))

    coverage_by_date = pd.Series(dict(rows))
    eligible = coverage_by_date.loc[coverage_by_date >= threshold]
    if eligible.empty:
        raise ValueError(f"No start date reaches last-sales coverage threshold {threshold:.2%}.")
    start_date = eligible.index[0]
    return start_date, float(eligible.iloc[0])


def compustat_panel_cache_path(args: argparse.Namespace) -> Path:
    params = {
        "version": 6,
        "weight_timing": "pre_shock_t_minus_1_v1",
        "last_date": str(args.last_date),
        "T": str(args.T),
        "coverage_threshold": float(args.coverage_threshold),
    }
    return clean_panel_cache_path(
        "compustat_quarterly_sales",
        params,
        [Path(args.compustat_csv)],
    )


def load_compustat_panel(args: argparse.Namespace) -> PanelData:
    use_cache = bool(getattr(args, "use_panel_cache", True))
    refresh_cache = bool(getattr(args, "refresh_panel_cache", False))
    cache_path = compustat_panel_cache_path(args)
    if use_cache and not refresh_cache:
        payload = read_clean_panel_cache(cache_path)
        if payload is not None:
            panel = PanelData(**payload)
            if panel.weight_timing != "pre_shock_t_minus_1_v1":
                raise ValueError(f"Unexpected Compustat weight timing: {panel.weight_timing!r}")
            return panel

    print(f"[load] reading {args.compustat_csv}", flush=True)
    usecols = ["gvkey", "datadate", "datacqtr", "curcdq", "sic", "saleq", "conm", "loc", "fic"]
    d = pd.read_csv(args.compustat_csv, usecols=usecols)
    print(f"[load] raw rows={len(d):,}", flush=True)

    d = d[(d["loc"] == "USA") & (d["fic"] == "USA") & d["curcdq"].eq("USD")].copy()
    d = d.dropna(subset=["datacqtr"])
    d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
    d["datadate"] = pd.PeriodIndex(d["datacqtr"], freq="Q")
    d["sic"] = pd.to_numeric(d["sic"], errors="coerce")
    d["saleq"] = pd.to_numeric(d["saleq"], errors="coerce")
    d = d.dropna(subset=["gvkey", "datadate", "sic", "saleq", "conm"])
    d["sic"] = d["sic"].astype(int)
    d = d.drop_duplicates(subset=["gvkey", "datadate"], keep="last")

    d = d.loc[~d["sic"].isin(EXCLUDED_COMPUSTAT_SIC)].copy()

    d["saleq"] = d["saleq"].clip(lower=0).replace(0, np.nan)
    d = d.dropna(subset=["saleq"])
    d = d.loc[d["datadate"] <= pd.Period(args.last_date, freq="Q")].copy()
    d["sic2"] = d["sic"].astype(str).str[:2].astype(int)
    d = d.sort_values(["gvkey", "datadate"])
    d["lagged_datadate"] = d.groupby("gvkey", sort=False)["datadate"].shift()
    d["lagged_saleq"] = d.groupby("gvkey", sort=False)["saleq"].shift()
    d = d.loc[d["lagged_datadate"].eq(d["datadate"] - 1)].copy()
    d["saleq_logdiff"] = np.log(d["saleq"] / d["lagged_saleq"])
    d = d.set_index(["datadate", "gvkey"]).sort_index()
    print(f"[filter] rows after domestic/nonenergy/nonfinancial/positive-sales filters={len(d):,}", flush=True)

    full_sales_by_date = d.groupby(level="datadate")["saleq"].sum()
    d = d.dropna(subset=["saleq_logdiff", "lagged_saleq", "saleq", "sic", "sic2", "conm"])
    d = make_panel_contiguous(d, "datadate", "gvkey")
    dates = d.index.unique("datadate")
    if args.T == "all":
        start_date = dates[0]
        start_coverage = choose_start_by_last_sales_coverage(d, dates, 0.0)[1]
        selection_rule = "all_contiguous_history"
    elif args.T == "coverage":
        start_date, start_coverage = choose_start_by_last_sales_coverage(
            d=d,
            dates=dates,
            threshold=args.coverage_threshold,
        )
        selection_rule = f"last_sales_coverage_at_least_{args.coverage_threshold:.2f}"
    else:
        if len(dates) < args.T:
            raise ValueError(f"Only {len(dates)} balanced dates available, fewer than requested T={args.T}.")
        start_date = dates[-args.T]
        last_sizes = d.loc[dates[-1], "saleq"]
        start_units = d.loc[start_date].index.unique("gvkey")
        start_coverage = float(last_sizes.reindex(start_units).sum() / last_sizes.sum())
        selection_rule = f"last_{args.T}_quarters"
    last_date = dates[-1]
    wide = d.loc[start_date:last_date, "saleq_logdiff"].unstack("datadate")
    wide = wide.dropna(axis=0)

    wide = remove_periodic_means(wide, 4)
    wide = standardize_panel(wide)
    start = d.loc[start_date].loc[wide.index]
    sizes = start["lagged_saleq"].astype(float).sort_values(ascending=False)
    wide = wide.loc[sizes.index]
    start = start.loc[sizes.index]
    names = start["conm"].map(clean_name)
    sic2 = start["sic2"].astype(int)

    print(
        f"[panel] balanced sample: N={wide.shape[0]:,}, T={wide.shape[1]:,}, "
        f"start={start_date}, last={last_date}, last-sales coverage={start_coverage:.3f}",
        flush=True,
    )
    panel = PanelData(
        wide=wide,
        sizes=sizes,
        names=names,
        sic2=sic2,
        panel=d,
        full_sales_by_date=full_sales_by_date,
        start_date=start_date,
        last_date=last_date,
        start_last_sales_coverage=start_coverage,
        selection_rule=selection_rule,
        weight_timing="pre_shock_t_minus_1_v1",
        weight_date=start_date - 1,
    )
    if use_cache:
        write_clean_panel_cache(cache_path, vars(panel))
    return panel


def grouped_sigma_default(m: int) -> tuple[np.ndarray, int]:
    """Grouped covariance and T for the draft's default quarterly configuration.

    Shared by the diagnostic scripts (compute_bias, check_regularity) so they
    rebuild exactly the Table 1 quarterly column.
    """
    args = argparse.Namespace(
        compustat_csv=wrds_path("compustat_quarterly_sales"),
        last_date=ANALYSIS_LAST_QUARTER,
        T=100,
        coverage_threshold=0.8,
    )
    panel = load_compustat_panel(args)
    if panel.weight_timing != "pre_shock_t_minus_1_v1":
        raise ValueError(f"Unexpected Compustat weight timing: {panel.weight_timing!r}")
    shares = panel.sizes.to_numpy(dtype=float)
    shares = shares / shares.sum()
    y = panel.wide.to_numpy(dtype=float)
    y = y - y.mean()
    x_grouped, *_ = build_lpt_grouped_data(y, shares, m)
    return sample_cov(x_grouped), int(panel.wide.shape[1])


def make_bucket_diagnostics(
    panel: PanelData,
    shares: np.ndarray,
    buckets: list[list[int]],
    loads: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment = np.empty(len(shares), dtype=int)
    for j, bucket in enumerate(buckets):
        assignment[bucket] = j

    firm_diag = pd.DataFrame(
        {
            "gvkey": panel.wide.index,
            "bucket": assignment,
            "name": panel.names.to_numpy(),
            "sic2": panel.sic2.to_numpy(),
            "start_saleq": panel.sizes.to_numpy(dtype=float),
            "size_share": shares,
            "size_share_sq": shares**2,
        }
    )
    bucket_diag = (
        firm_diag.groupby("bucket")
        .agg(
            n_firms=("gvkey", "size"),
            total_size_share=("size_share", "sum"),
            total_sq_share=("size_share_sq", "sum"),
            max_size_share=("size_share", "max"),
        )
        .reset_index()
    )
    bucket_diag["lpt_load"] = loads[bucket_diag["bucket"].to_numpy()]

    sic2_mix = pd.crosstab(
        firm_diag["bucket"],
        firm_diag["sic2"],
        values=firm_diag["size_share_sq"],
        aggfunc="sum",
    ).fillna(0.0)
    sic2_mix = sic2_mix.div(sic2_mix.sum(axis=1), axis=0)
    sic2_effective = 1.0 / (sic2_mix.pow(2).sum(axis=1))
    bucket_diag["effective_sic2_by_sq_share"] = sic2_effective.to_numpy()

    top_firms = firm_diag.sort_values("size_share", ascending=False).head(50)
    return firm_diag, bucket_diag, top_firms


def make_attrition_summary(panel: PanelData) -> pd.DataFrame:
    last_sizes = panel.panel.loc[panel.last_date, "saleq"]
    total_last_size = float(last_sizes.sum())
    rows = []
    for date, group in panel.panel.groupby(level="datadate"):
        units = group.index.get_level_values("gvkey")
        current_sales = float(group["saleq"].sum())
        full_sales = float(panel.full_sales_by_date.get(date, np.nan))
        last_share = float(last_sizes.reindex(units).sum() / total_last_size)
        rows.append(
            {
                "datadate": str(date),
                "n_firms": int(len(units)),
                "current_sales_share_of_filtered_full": current_sales / full_sales if full_sales else np.nan,
                "last_date_sales_share_of_survivors": last_share,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    default_csv = wrds_path("compustat_quarterly_sales")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compustat-csv", type=Path, default=default_csv)
    parser.add_argument("--last-date", default=ANALYSIS_LAST_QUARTER)
    parser.add_argument("--T", default="100", help="Trailing quarters, 'coverage', or 'all'.")
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--block-bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--subsample-n", type=int, default=300,
                        help="Number of subsamples for the subsampling CI.")
    parser.add_argument("--gaussian-b", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--solver", default="auto", help="auto, MOSEK, CLARABEL, or SCS")
    parser.add_argument(
        "--run-profile",
        choices=["fast", "medium", "full"],
        default="full",
        help="fast/medium skip subsampling; full runs every diagnostic.",
    )
    parser.add_argument("--refresh-panel-cache", action="store_true", help="Rebuild and overwrite the cleaned-panel cache.")
    parser.add_argument("--no-panel-cache", dest="use_panel_cache", action="store_false", help="Do not read or write cleaned-panel caches.")
    parser.set_defaults(use_panel_cache=True)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.T = parse_t_arg(args.T)
    if args.run_profile in {"fast", "medium"}:
        args.subsample_n = 0
    if args.T == "coverage":
        t_label = f"coverage{int(round(100 * args.coverage_threshold))}"
    else:
        t_label = str(args.T)
    start = time.time()
    solver = choose_solver(args.solver)
    rng = make_panel_rng(args.seed, "compustat_quarterly_sales")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Compustat grouped MGP (quarterly sales)", flush=True)
    print(
        f"T={t_label}, M={args.M}, bootstrap_b={args.bootstrap_b}, "
        f"block_bootstrap_b={args.block_bootstrap_b}, "
        f"gaussian_b={args.gaussian_b}, subsample_n={args.subsample_n}, "
        f"profile={args.run_profile}, solver={solver}",
        flush=True,
    )
    panel = load_compustat_panel(args)
    block_length = analysis_block_length(panel.wide.shape[1])
    print(f"paper-wide block_length={block_length}", flush=True)

    sizes = panel.sizes.to_numpy(dtype=float)
    shares = sizes / sizes.sum()
    y = panel.wide.to_numpy(dtype=float)
    y = y - y.mean()
    x_grouped, buckets, loads, _ = build_lpt_grouped_data(y, shares, args.M)
    aggregate = shares @ y
    aggregate_from_groups = x_grouped.sum(axis=1)
    if not np.allclose(aggregate, aggregate_from_groups):
        raise RuntimeError("Grouped data do not aggregate back to the size-weighted aggregate.")

    observed = mgp_share(x_grouped, solver)
    regularity = regularity_report(sample_cov(x_grouped), solver)
    herfindahl = float(np.sum(shares**2))
    aggregate_variance_direct = float(np.var(aggregate, ddof=1))
    print(
        f"[observed] MGP share={observed['mgp_share']:.4f}, "
        f"level={observed['mgp_level']:.6g}, agg_var={observed['aggregate_variance']:.6g}, "
        f"normal_ucl95={observed['normal_ucl95']:.4f}",
        flush=True,
    )
    print(
        f"[weights] HHI={herfindahl:.6g}, top10_share={shares[:10].sum():.4f}, "
        f"top50_share={shares[:50].sum():.4f}, bucket_cv={np.std(loads) / np.mean(loads):.4f}",
        flush=True,
    )

    firm_diag, bucket_diag, top_firms = make_bucket_diagnostics(panel, shares, buckets, loads)
    attrition = make_attrition_summary(panel)

    bootstrap = resample_mgp(
        x_grouped,
        b=args.bootstrap_b,
        solver=solver,
        rng=rng,
        mode="bootstrap",
        progress_every=args.progress_every,
    )
    block_bootstrap = resample_mgp(
        x_grouped,
        b=args.block_bootstrap_b,
        solver=solver,
        rng=rng,
        mode="moving_block_bootstrap",
        progress_every=args.progress_every,
        block_length=block_length,
    )
    gaussian = resample_mgp(
        x_grouped,
        b=args.gaussian_b,
        solver=solver,
        rng=rng,
        mode="gaussian_covariance",
        progress_every=args.progress_every,
    )
    draws = pd.concat([bootstrap, block_bootstrap, gaussian], ignore_index=True)

    gradient_family: dict[str, object] = {}
    gradient_b = min(args.bootstrap_b, args.block_bootstrap_b)
    if gradient_b > 0:
        gradient_family = GB.bootstrap_family(
            x_grouped,
            b=gradient_b,
            block_length=block_length,
            rng=make_panel_rng(args.seed, "compustat_quarterly_sales_gradient"),
            solver=solver,
            progress_every=args.progress_every,
        )

    ell = default_subsample_length(panel.wide.shape[1], args.M)
    if args.subsample_n > 0 and ell < panel.wide.shape[1]:
        subsamples, _ = subsample_mgp(
            x_grouped, ell=ell, n_sub=args.subsample_n, solver=solver, rng=rng,
            progress_every=args.progress_every, g_hat=observed["mgp_share"], overlapping=True,
        )
        sub_ci = subsampling_summary(subsamples, observed["mgp_share"], panel.wide.shape[1])
    else:
        subsamples = pd.DataFrame()
        sub_ci = {}

    summary = {
        "dataset": "compustat_quarterly_sales",
        "run_profile": args.run_profile,
        "compustat_csv": str(args.compustat_csv),
        "clean_panel_path": (
            str(compustat_panel_cache_path(args).relative_to(ROOT)) if args.use_panel_cache else None
        ),
        "start_date": str(panel.start_date),
        "last_date": str(panel.last_date),
        "selection_rule": panel.selection_rule,
        "weight_timing": panel.weight_timing,
        "weight_date": str(panel.weight_date),
        "seasonal_adjustment": "unit quarter-of-year fixed effects",
        "start_last_sales_coverage": panel.start_last_sales_coverage,
        "coverage_threshold": args.coverage_threshold if args.T == "coverage" else np.nan,
        "N": panel.wide.shape[0],
        "T": panel.wide.shape[1],
        "M": args.M,
        "block_bootstrap_b": args.block_bootstrap_b,
        "subsample_n_requested": args.subsample_n,
        "block_length": block_length,
        "HHI": herfindahl,
        "top1_share": float(shares[0]),
        "top10_share": float(shares[:10].sum()),
        "top50_share": float(shares[:50].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        "bucket_load_min": float(loads.min()),
        "bucket_load_max": float(loads.max()),
        "aggregate_variance_direct": aggregate_variance_direct,
        # sigma_x: equal-weighted mean over units of each unit's time-series std (raw,
        # ungrouped); sigma_y: std of the aggregate = sqrt(1' Sigma_tilde 1).
        "sigma_x": float(np.nanmean(np.nanstd(y, axis=1, ddof=1))),
        "sigma_y": float(np.sqrt(observed["aggregate_variance"])),
        **observed,
        **regularity,
        **summarize_distribution(bootstrap["mgp_share"] if not bootstrap.empty else pd.Series(dtype=float), "bootstrap"),
        **summarize_distribution(
            block_bootstrap["mgp_share"] if not block_bootstrap.empty else pd.Series(dtype=float),
            "moving_block_bootstrap",
        ),
        **summarize_distribution(gaussian["mgp_share"] if not gaussian.empty else pd.Series(dtype=float), "gaussian_covariance"),
        "gradient_bootstrap_p05": gradient_family.get("gradient_low", np.nan),
        "gradient_bootstrap_p95": gradient_family.get("gradient_high", np.nan),
        "moving_block_gradient_bootstrap_p05": gradient_family.get("block_gradient_low", np.nan),
        "moving_block_gradient_bootstrap_p95": gradient_family.get("block_gradient_high", np.nan),
        "gradient_bootstrap_common_rank": gradient_family.get("directional_common_rank", np.nan),
        "gradient_bootstrap_regular": gradient_family.get("directional_regular", np.nan),
        **sub_ci,
        "seconds": time.time() - start,
    }
    summary_df = pd.DataFrame([summary])

    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    summary_df.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_summary.pkl")
    draws.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_draws_resampling.pkl")
    if not subsamples.empty:
        subsamples.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_draws_subsampling.pkl")
    firm_diag.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_diag_firms.pkl")
    bucket_diag.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_diag_buckets.pkl")
    top_firms.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_top_firms.pkl")
    attrition.to_pickle(RUN_OUTPUT_CACHE / f"{OUTPUT_PREFIX}_attrition.pkl")
    plot_lpt_squared_bucket_sizes(
        firm_diag,
        bucket_diag,
        out_dir / f"lpt_{OUTPUT_PREFIX}",
        ylabel="Bucket squared-size load",
        figsize_scale=(2 / 3, 0.9),
    )

    display_cols = [
        "N",
        "T",
        "M",
        "start_date",
        "last_date",
        "start_last_sales_coverage",
        "HHI",
        "top10_share",
        "mgp_share",
        "normal_ucl95",
        "bootstrap_mean",
        "bootstrap_p05",
        "bootstrap_p95",
        "moving_block_bootstrap_mean",
        "moving_block_bootstrap_p05",
        "moving_block_bootstrap_p95",
    ]
    display_cols = [col for col in display_cols if col in summary_df.columns]
    print("\nSummary:", flush=True)
    print(summary_df[display_cols].round(6).to_string(index=False), flush=True)
    print(f"\nSaved outputs in: {out_dir}", flush=True)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
