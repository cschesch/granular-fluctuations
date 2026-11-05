"""Balanced BLS CPI inflation panel."""

from __future__ import annotations

import pandas as pd

from mgp_core import standardize_panel
from project_paths import DERIVED_CACHE

from applications.panels import (
    SourcePanel,
    WEIGHT_TIMING,
    load_source_panel_cache,
    save_source_panel_cache,
    source_panel_cache_path,
)


def load_bls_cpi_panel(use_cache: bool = True, refresh_cache: bool = False) -> SourcePanel:
    vintage = "1967"
    panel_dir = DERIVED_CACHE / "bls_cpi" / "panels"
    leaves_path = panel_dir / f"cpi_{vintage}_leaves.csv"
    infl_path = panel_dir / f"cpi_{vintage}_inflation.parquet"
    if not leaves_path.exists() or not infl_path.exists():
        raise FileNotFoundError(
            "Missing CPI panel files; run the prepare stage of paper_applications.py first."
        )
    cache_path = source_panel_cache_path(
        f"bls_cpi_{vintage}_inflation",
        {"vintage": vintage, "weight_timing": WEIGHT_TIMING, "seasonal_adjustment": "BLS_CUSR"},
        [leaves_path, infl_path],
    )
    cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
    if cached is not None:
        return cached

    leaves = pd.read_csv(leaves_path)
    infl = pd.read_parquet(infl_path).sort_index()
    if infl.isna().any(axis=None):
        raise ValueError("CPI inflation panel has missing values.")

    meta = leaves.set_index("label")
    y = infl.T
    weight_period = pd.Period(f"{vintage}-12", freq="M")
    if weight_period >= pd.Period(y.columns[0], freq="M"):
        raise ValueError(f"CPI {vintage} weights must precede the first inflation observation.")
    sizes = meta["size_share"].reindex(y.index).astype(float)
    labels = pd.Series(
        [
            f"{label} [{code}]" if code and code != "nan" else label
            for label, code in zip(meta.index.astype(str), meta["item_code"].astype(str))
        ],
        index=meta.index,
    ).reindex(y.index)

    y = standardize_panel(y)
    y = y.sub(y.mean().mean())
    residual_share = float(meta.loc[meta["kind"].eq("residual"), "size_share"].sum())
    panel = SourcePanel(
        name=f"bls_cpi_{vintage}_inflation",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=y.columns[0],
        last_date=y.columns[-1],
        frequency="monthly",
        size_name=f"December {vintage} CPI-U relative importance (t-1)",
        value_name="monthly log CPI inflation",
        coverage=1.0,
        notes=(
            f"CPI-U expenditure arborescence using December {vintage} (t-1) weights; "
            "BLS seasonally adjusted CUSR0000 item indexes, without further seasonal adjustment; "
            f"{residual_share:.1%} of weight in backed-out residual leaves; "
            "log inflation winsorized at 1st/99th pct."
        ),
        weight_timing=WEIGHT_TIMING,
        weight_date=weight_period,
    )
    save_source_panel_cache(cache_path, panel, use_cache)
    return panel
