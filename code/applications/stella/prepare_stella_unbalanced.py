#!/usr/bin/env python3
"""Reconstruct Stella's unbalanced, top-500-union statistical-analysis inputs.

All row-level files are licensed derivatives and remain in the private cache.
The original vintage and identities remain unavailable. This uses current
revtq, the recovered X-12 engine and published sax12 options, and source ADF/weight timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT
from applications.stella.prepare_stella_calibration import load_quarterly, released_screen_panel
from applications.panels import load_gdp_deflator
from applications.stella.stella_provenance import archive_existing, file_fingerprints
from applications.stella.stella_quarterly_inputs import input_description, verify_input_provenance

DEFAULT_SOURCE = ROOT / "code/.cache/firm_networks/stella_literal"
DEFAULT_OUT = DEFAULT_SOURCE.with_name("stella_unbalanced")
LEVEL_PERIODS = pd.period_range("1962Q1", "2011Q4", freq="Q")


def top500_membership(d: pd.DataFrame) -> pd.DataFrame:
    """Recreate dated membership before the union and stationarity screens."""
    x = d.loc[d["quarter"].isin(LEVEL_PERIODS)].copy()
    nonpositive = x["saleq"].le(0).groupby(x["gvkey"]).max()
    x = x.loc[~x["gvkey"].isin(nonpositive[nonpositive].index) & x["saleq"].notna()]
    x = x.groupby(["gvkey", "quarter"], as_index=False)["saleq"].mean().sort_values(["gvkey", "quarter"])
    gap = x.groupby("gvkey")["quarter"].apply(
        lambda q: max((b.ordinal-a.ordinal for a, b in zip(q.iloc[:-1], q.iloc[1:])), default=1))
    x = x.loc[x["gvkey"].isin(gap[gap <= 1].index)].copy()
    tenure = x.groupby("gvkey").size()
    x["lag_sale"] = x.groupby("gvkey")["saleq"].shift()
    rows = []
    for quarter, group in x.groupby("quarter"):
        eligible = group.loc[group["gvkey"].map(tenure).ge(41)]
        if len(group) > 500:
            # Stata egen rank() leaves missing lag_sale unranked.
            eligible = eligible.loc[eligible["lag_sale"].notna()].nlargest(500, "lag_sale")
        rows.extend({"gvkey": gvkey, "quarter": quarter, "top500": 1} for gvkey in eligible["gvkey"])
    return pd.DataFrame(rows)


def build(source_dir: Path, output_dir: Path) -> dict:
    adf_path = source_dir / "selection_adf_pvalues_sax12.csv"
    adjusted_path = source_dir / "selection_sax12_adjusted_levels.parquet"
    provenance_path = adf_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if provenance.get("input_file_fingerprints"):
        verify_input_provenance(provenance)
    engine = provenance.get("seasonal_adjustment_engine", "X-13")
    adf = pd.read_csv(adf_path, dtype={"gvkey": str}).set_index("gvkey")
    adjusted = pd.read_parquet(adjusted_path)
    adjusted["quarter"] = pd.PeriodIndex(adjusted["quarter"], freq="Q")
    quarterly = load_quarterly()
    legacy_selected, legacy_union_n = released_screen_panel(quarterly, strict_lag_ranks=False)
    membership = top500_membership(quarterly)
    source_union = set(membership["gvkey"])
    legacy_union = set(legacy_selected["gvkey"])
    if not source_union.issubset(legacy_union) or not source_union.issubset(adf.index):
        raise ValueError("Source-faithful membership is not covered by adjustment/ADF histories")
    source_adf = adf.loc[sorted(source_union)]
    passed = source_adf.index[~source_adf["pvalue"].gt(.01)]
    available = pd.Index(sorted(set(passed) & set(adjusted["gvkey"])))
    omitted = sorted(set(passed)-set(available))
    levels = adjusted.loc[adjusted["gvkey"].isin(available)].pivot(
        index="gvkey", columns="quarter", values="adjusted_nominal_saleq").reindex(index=available, columns=LEVEL_PERIODS)
    assert len(available) == len(passed)-len(omitted)
    price = load_gdp_deflator("quarterly").reindex(LEVEL_PERIODS)
    if price.isna().any() or not price.gt(0).all():
        raise ValueError("GDP deflator does not cover full source sample")
    real = levels.div(price, axis=1) * 100
    growth = np.log(real).diff(axis=1).iloc[:, 1:].T
    growth = growth.sub(growth.mean(axis=0), axis=1)
    shares = levels.div(levels.sum(axis=0), axis=1)
    lagged = shares.iloc[:, :-1].T.copy()
    lagged.index = growth.index
    current = shares.iloc[:, 1:].T
    mean = pd.DataFrame({"mean_sales_share": shares.mean(axis=1),
                         "sd_sales_share": shares.std(axis=1, ddof=1)})

    top500 = membership.pivot(index="gvkey", columns="quarter", values="top500").reindex(
        index=available, columns=growth.index).fillna(0).astype(int)
    sic = quarterly.loc[quarterly["gvkey"].isin(available)].groupby("gvkey")["sic"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan).reindex(available)
    metadata = mean.assign(sic=sic, sic_1digit=np.floor(sic/1000), tid=np.arange(1, len(available)+1))
    records = []
    for gvkey in available:
        observed = growth[gvkey].dropna()
        for quarter, value in observed.items():
            records.append({"gvkey": gvkey, "tid": int(metadata.loc[gvkey, "tid"]),
                            "quarter": str(quarter), "time": quarter.ordinal-growth.index[0].ordinal+1,
                            "rev_growth": value, "sic_1digit": metadata.loc[gvkey, "sic_1digit"],
                            "lw_it": lagged.loc[quarter, gvkey], "w_sd": mean.loc[gvkey, "sd_sales_share"],
                            "w_bar": mean.loc[gvkey, "mean_sales_share"], "w_it": current.loc[quarter, gvkey]})
    long = pd.DataFrame(records)
    # Source weights are defined before growth differencing; mean weights can
    # therefore sum to more than one for an entry/exit panel, as in the deposit.
    np.testing.assert_allclose(shares.sum(axis=0).to_numpy(), 1, atol=1e-12)
    assert np.isfinite(long[["rev_growth", "lw_it", "w_bar", "w_sd", "w_it"]]).all().all()
    np.testing.assert_allclose(growth.mean(axis=0).to_numpy(), 0, atol=1e-14)
    assert top500.shape == (len(available), 199)
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("*.csv"):
        archive_existing(previous)
    archive_existing(output_dir / "diagnostics.json")
    levels.to_csv(output_dir / "nominal_saleq_sax12_adjusted.csv", index_label="gvkey")
    growth.to_csv(output_dir / "sales_growth_sax12.csv", index_label="quarter")
    top500.to_csv(output_dir / "top500_index.csv", index_label="gvkey")
    lagged.to_csv(output_dir / "lagged_sales_shares.csv", index_label="quarter")
    current.to_csv(output_dir / "current_sales_shares.csv", index_label="quarter")
    metadata.to_csv(output_dir / "mean_sales_shares.csv", index_label="gvkey")
    long.to_csv(output_dir / "source_dfm_data.csv", index=False)
    payload = {
        "scope": "Current-vintage unbalanced statistical-factor inputs; no artificial dimension matching",
        **input_description(),
        "input": str(adjusted_path), "adf_input": str(adf_path),
        "seasonal_specification": f"Published sax12 estimation options, Census {engine} executable",
        "seasonal_adjustment_engine": engine,
        "legacy_top500_union_N": legacy_union_n,
        "pre_adf_top500_union_N": len(source_union), "after_source_spec_adf_N": len(passed),
        "legacy_firms_excluded_by_source_missing_lag_rank_rule": sorted(legacy_union-source_union),
        "omitted_without_successful_adjusted_history": omitted, "N": len(available),
        "level_T": len(LEVEL_PERIODS), "growth_T": len(growth),
        "sample_growth": [str(growth.index[0]), str(growth.index[-1])],
        "observed_growth_cells": int(growth.notna().sum().sum()),
        "missing_growth_cells": int(growth.isna().sum().sum()),
        "growth_mean_max_absolute": float(abs(growth.mean(axis=0)).max()),
        "top500_firms_per_growth_quarter_min": int(top500.sum(axis=0).min()),
        "top500_firms_per_growth_quarter_max": int(top500.sum(axis=0).max()),
        "weights": "Quarter-specific real/nominal adjusted sales shares, with lagged weights from previous level quarter. Fixed w_bar is mean over each firm's observed levels, before differencing.",
        "index_timing": "Top500 membership in growth quarter t, ranked by nominal saleq at t-1, before ADF; missing lag_sales are unranked when period population exceeds 500, as in Stata egen rank. Consistent with index_top5001-index_top500199 in the deposited workflow.",
        "remaining_differences": ["current Compustat vintage" + (" and X-13 executable" if engine == "X-13" else ""), "original SIC concordance absent", "one original tid549 special X12 treatment cannot be identified", "failed adjustment is excluded, not zero-filled"],
        "private_licensed_derivatives": True,
        "input_file_fingerprints": file_fingerprints([adf_path, adjusted_path], ROOT),
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2)+"\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
