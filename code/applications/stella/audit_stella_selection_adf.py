#!/usr/bin/env python3
"""Reproduce Stella's pre-balance firm-level stationarity screen.

The released Stata program seasonally adjusts every firm in the union of the
quarterly top-500 samples, forms real quarterly log revenue growth, and runs
``dfuller`` with its defaults (constant and zero augmentation lags).  It drops
firms whose reported p-value exceeds one percent before imposing the balanced
1970--2011 window.  The dedicated licensed extract supplies ``revtq`` directly and the recovered
X-12 executable is preferred; historical-vintage differences remain.  X-13 failures are left unclassified and retained for this screen;
they are separately flagged and neither can enter the eventual balanced panel.
The published sax12 estimation options are translated literally.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from mgp_core import ROOT
from applications.stella.prepare_stella_calibration import (
    QUARTERS,
    SELECTION_ADF,
    QFILE,
    load_quarterly,
    released_screen_panel,
)
from applications.stella.prepare_stella_sales import adjust_one, locate_x13
from applications.stella.paths import CALIBRATION
from applications.panels import load_gdp_deflator
from applications.stella.stella_provenance import archive_existing, file_fingerprints
from applications.stella.stella_quarterly_inputs import REVENUE_FIELD, input_description


DIAGNOSTICS = CALIBRATION / "selection_adf_diagnostics.json"


def evaluate(
    gvkey: str,
    series: pd.Series,
    deflator: pd.Series,
    executable: Path,
    seasonal_spec: str = "sax12",
    save_adjusted: bool = False,
) -> dict:
    start = series.index[0]
    result = {
        "gvkey": gvkey,
        "start": str(start),
        "end": str(series.index[-1]),
        "nlevels": int(len(series)),
        "adf_stat": np.nan,
        "pvalue": np.nan,
        "status": "x13_or_adf_failed_unclassified",
        "error": "",
    }
    try:
        _, adjusted, _ = adjust_one(gvkey, series.to_numpy(float), start, executable, seasonal_spec)
        price = deflator.reindex(series.index).to_numpy(float)
        if not np.isfinite(price).all() or np.any(price <= 0):
            raise ValueError("GDP deflator does not cover the firm's full history")
        growth = np.diff(np.log(adjusted) - np.log(price))
        statistic, pvalue, *_ = adfuller(
            growth, maxlag=0, regression="c", autolag=None
        )
        result.update(
            adf_stat=float(statistic),
            pvalue=float(pvalue),
            status="ok",
        )
        if save_adjusted:
            result["_adjusted_levels"] = adjusted
    except Exception as exc:
        result["error"] = str(exc)[-1000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x12", "--x13", dest="x13", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics-path", type=Path, default=None)
    parser.add_argument("--save-adjusted", action="store_true", help="cache private adjusted histories for descriptive-moment audits")
    args = parser.parse_args()

    executable = locate_x13(args.x13)
    engine = "X-12" if "x12" in executable.name.lower() else "X-13"
    output = args.output or SELECTION_ADF
    diagnostics_path = args.diagnostics_path or DIAGNOSTICS
    selected, union_n = released_screen_panel(load_quarterly())
    selected = selected.sort_values(["gvkey", "quarter"])
    deflator = load_gdp_deflator("quarterly")
    work = {
        gvkey: group.set_index("quarter")["saleq"].sort_index()
        for gvkey, group in selected.groupby("gvkey")
    }
    if len(work) != union_n:
        raise ValueError("Top-500-union panel and reported firm count disagree.")

    # Preserve the released data-specific tid exception literally on this
    # current sorted GVKEY universe; never move it to a series that fails.
    no_outlier_gvkey = sorted(work)[548] if len(work) >= 549 else None
    rows: list[dict] = []
    adjusted_rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate, gvkey, series, deflator, executable,
                        "sax12_no_outlier" if gvkey == no_outlier_gvkey else "sax12",
                        args.save_adjusted): gvkey
            for gvkey, series in work.items()
        }
        for future in as_completed(futures):
            result = future.result()
            adjusted = result.pop("_adjusted_levels", None)
            if adjusted is not None:
                adjusted_rows.append(pd.DataFrame({"gvkey": result["gvkey"], "quarter": work[result["gvkey"]].index.astype(str), "adjusted_nominal_saleq": adjusted}))
            rows.append(result)
    frame = pd.DataFrame(rows).sort_values("gvkey").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_existing(output)
    archive_existing(diagnostics_path)
    frame.to_csv(output, index=False)
    if adjusted_rows:
        pd.concat(adjusted_rows, ignore_index=True).sort_values(["gvkey", "quarter"]).to_parquet(output.with_name("selection_sax12_adjusted_levels.parquet"), index=False)

    rejected = frame["pvalue"].gt(0.01)
    failures = frame["status"].ne("ok")
    balanced_before_adf = (
        selected.loc[selected["quarter"].isin(QUARTERS)]
        .pivot(index="gvkey", columns="quarter", values="saleq")
        .reindex(columns=QUARTERS)
        .dropna()
        .index
    )
    balanced_rows = frame["gvkey"].isin(balanced_before_adf)
    payload = {
        "paper_rule": "Stata dfuller on real seasonally adjusted q/q log revenue growth; default intercept and zero augmentation lags; reject firm when p>0.01",
        "reconstruction_rule": f"statsmodels ADF with constant, maxlag=0, autolag=None on real {engine}/X-11 q/q log {REVENUE_FIELD} growth; MacKinnon approximate p-value",
        "seasonal_adjustment_engine": engine,
        **input_description(),
        "input_file_fingerprints": file_fingerprints([QFILE, executable], ROOT),
        "seasonal_specification": "sax12",
        "literal_tid549_current_gvkey": no_outlier_gvkey,
        "tid549_interpretation": "Released tid=549 no-outlier exception applied literally to the current sorted firm list; original GVKEY at that tid is unavailable. It is not reassigned based on adjustment failures.",
        "known_input_differences": [
            "Current Compustat vintage rather than the author's historical extract",
            "Census X-12 v0.3 rebuilt from original source" if engine == "X-12" else "Current Census X-13 substitutes for the paper's X-12 executable",
            "Statsmodels' current MacKinnon response surface substitutes for Stata 12.1's implementation",
        ],
        "top500_union_N": int(len(frame)),
        "successful_tests": int((~failures).sum()),
        "failed_tests_unclassified_and_retained": int(failures.sum()),
        "rejected_at_one_percent": int(rejected.sum()),
        "retained_after_adf": int((~rejected).sum()),
        "maximum_pvalue": float(frame["pvalue"].max()),
        "balanced_candidates_before_adf": int(len(balanced_before_adf)),
        "balanced_candidates_rejected_by_adf": int((balanced_rows & rejected).sum()),
        "maximum_pvalue_among_balanced_candidates": float(frame.loc[balanced_rows, "pvalue"].max()),
        "minimum_adf_statistic": float(frame.loc[~failures, "adf_stat"].min()),
        "maximum_adf_statistic": float(frame.loc[~failures, "adf_stat"].max()),
        "balanced_classification_margin_note": "Classification is reported from this run; no sample count is imposed.",
        "failure_relevance_note": f"{int(failures.sum())} {engine}/ADF failures ({int((balanced_rows & failures).sum())} balanced candidates); {int(rejected.sum())} ADF rejections ({int((balanced_rows & rejected).sum())} balanced candidates).",
        "output": str(output),
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2) + "\n")
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
