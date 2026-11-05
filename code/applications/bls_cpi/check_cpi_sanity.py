#!/usr/bin/env python3
"""Sanity checks for the CPI weight series and fixed-weight CPI panels."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from project_paths import DERIVED_CACHE


BCPI = DERIVED_CACHE / "bls_cpi"


def norm(value: object) -> str:
    s = str(value).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_price_wide() -> pd.DataFrame:
    price = pd.read_parquet(BCPI / "cpi_price_long.parquet")
    return price.pivot(index="date", columns="item_code", values="value").sort_index()


def check_weight_series() -> pd.DataFrame:
    weights = pd.read_csv(BCPI / "cpi_weight_series_1967_onward.csv")
    wide = pd.read_parquet(BCPI / "cpi_weight_series_1967_onward_wide.parquet")

    years = set(range(1967, int(weights["year"].max()) + 1))
    require(set(weights["year"].unique()) == years, "Long weight series has missing years.")
    require(set(wide.index) == years, "Wide weight series has missing years.")
    require(
        weights.groupby(["year", "item_key"]).size().le(1).all(),
        "Long weight series has duplicate (year, item_key) rows.",
    )
    require(not wide.columns.duplicated().any(), "Wide weight series has duplicate columns.")
    require(not wide.isna().all().any(), "Wide weight series has all-NA columns.")
    require(not wide.isna().all(axis=1).any(), "Wide weight series has all-NA years.")

    roots = weights[(weights["section"] == "expenditure") & (weights["item_name"] == "all items")]
    require(roots.groupby("year").size().eq(1).all(), "Expected exactly one expenditure All-items root per year.")
    require(roots["year"].nunique() == len(years), "Missing expenditure All-items root years.")
    require(np.allclose(roots["cpi_u"].dropna(), 100.0), "All-items CPI-U roots are not 100.")

    bad_bounds = weights[(weights["cpi_u"].notna()) & ((weights["cpi_u"] < 0) | (weights["cpi_u"] > 100.000001))]
    require(bad_bounds.empty, "CPI-U weights outside [0, 100].")

    print(
        f"[weights] long={weights.shape}, wide={wide.shape}, years={min(years)}-{max(years)}, "
        f"missing cpi_u rows={weights['cpi_u'].isna().sum()}"
    )
    return weights


def panel_aggregate_gap(tag: str, baseline: str, leaves: pd.DataFrame, infl: pd.DataFrame, price_wide: pd.DataFrame) -> pd.Series:
    allitems = (
        np.log(price_wide["SA0"].reindex(pd.period_range(pd.Period(baseline, "M"), infl.index.max(), freq="M")))
        .diff()
        .loc[infl.index]
    )
    shares = leaves.set_index("label")["size_share"]
    aggregate = infl.mul(shares, axis=1).sum(axis=1)
    gap = aggregate - allitems
    require(allitems.notna().all(), f"{tag}: official All-items comparison series has missing values.")
    return gap


def check_panel(tag: str, baseline: str, weights: pd.DataFrame, price_wide: pd.DataFrame) -> None:
    leaves = pd.read_csv(BCPI / "panels" / f"cpi_{tag}_leaves.csv")
    infl = pd.read_parquet(BCPI / "panels" / f"cpi_{tag}_inflation.parquet")

    require(len(leaves) == infl.shape[1], f"{tag}: leaf count does not match inflation columns.")
    require(list(leaves["label"]) == list(infl.columns), f"{tag}: leaf labels do not align with columns.")
    require(leaves["label"].is_unique, f"{tag}: duplicate leaf labels.")
    require(leaves.loc[leaves["item_code"] != "RESIDUAL", "item_code"].is_unique, f"{tag}: duplicate explicit item codes.")
    require(np.isclose(leaves["size_share"].sum(), 1.0), f"{tag}: size shares do not sum to 1.")
    require((leaves["weight"] > 0).all(), f"{tag}: nonpositive weights.")
    require((leaves["size_share"] > 0).all(), f"{tag}: nonpositive size shares.")
    require(infl.index.equals(pd.period_range(infl.index.min(), infl.index.max(), freq="M")), f"{tag}: non-contiguous months.")
    require(not infl.isna().any(axis=None), f"{tag}: missing inflation values.")
    require(np.isfinite(infl.to_numpy()).all(), f"{tag}: non-finite inflation values.")
    require((infl.var(axis=0) > 1e-18).all(), f"{tag}: zero-variance inflation columns.")

    level_index = pd.period_range(pd.Period(baseline, "M"), infl.index.max(), freq="M")
    for _, row in leaves[leaves["item_code"] != "RESIDUAL"].iterrows():
        code = row["item_code"]
        require(code in price_wide.columns, f"{tag}: missing price code {code}.")
        missing = price_wide[code].reindex(level_index).isna().sum()
        require(missing == 0, f"{tag}: price code {code} has {missing} missing level observations.")

    if tag == "1967":
        year_weights = weights[weights["year"] == 1967]
        for _, row in leaves[leaves["kind"] == "leaf"].iterrows():
            source_name = row.get("source_item_name", row["label"])
            hits = year_weights[(year_weights["section"] == "expenditure") & (year_weights["item_name"] == norm(source_name))]
            require(len(hits) > 0, f"1967: no weight-series match for {row['label']}.")
            require(np.isclose((hits["cpi_u"] - row["weight"]).abs().min(), 0.0), f"1967: weight mismatch for {row['label']}.")

    gap = panel_aggregate_gap(tag, baseline, leaves, infl, price_wide)
    max_gap = float(gap.abs().max())
    require(max_gap < 1e-12, f"1967: backed-out residual aggregate gap too large: {max_gap}.")

    weighted_abs = infl.abs().mul(leaves.set_index("label")["size_share"], axis=1).max().max()
    print(
        f"[panel {tag}] leaves={len(leaves)}, months={len(infl)}, residual_share="
        f"{leaves.loc[leaves['kind'].eq('residual'), 'size_share'].sum():.3f}, "
        f"aggregate_gap_max={max_gap:.6g}, max_weighted_abs_contribution={weighted_abs:.6g}"
    )


def main() -> int:
    weights = check_weight_series()
    price_wide = load_price_wide()
    check_panel("1967", "1967-12", weights, price_wide)
    print("[cpi-sanity] all hard checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
