"""Balanced CRSP daily stock returns panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import DAILY_RETURN_MAX_GAP, fill_short_internal_return_gaps, standardize_panel

from applications.panels import (
    SourcePanel,
    WEIGHT_TIMING,
    load_source_panel_cache,
    save_source_panel_cache,
    source_panel_cache_path,
)


def load_crsp_panel(t: int, use_cache: bool = True, refresh_cache: bool = False) -> SourcePanel:
    from applications.crsp.prepare_crsp_daily import prepare
    source_path = prepare()
    cache_path = source_panel_cache_path(
        "crsp_daily_returns", {"t": int(t), "weight_timing": WEIGHT_TIMING,
                               "internal_zero_gap_max": DAILY_RETURN_MAX_GAP,
                               "cleaning": "pooled_1_99_grand_mean"}, [source_path]
    )
    cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
    if cached is not None:
        return cached

    d = pd.read_parquet(source_path, columns=["RET", "mkt_cap"])
    d = d.sort_index()
    dates = pd.Index(d.index.get_level_values("date").unique()).sort_values()
    if len(dates) < t + 1:
        raise ValueError(f"CRSP has only {len(dates)} dates, fewer than T+1={t + 1} needed for t-1 weights.")
    start_date = dates[-t]
    weight_date = dates[-t - 1]
    last_date = dates[-1]
    print(f"[crsp_daily_returns] heuristic window: {start_date.date()} to {last_date.date()} ({t} trading days)", flush=True)
    sub = d.loc[start_date:last_date].copy()
    sub["RET"] = pd.to_numeric(sub["RET"], errors="coerce")
    sub["mkt_cap"] = pd.to_numeric(sub["mkt_cap"], errors="coerce")
    raw_returns = sub["RET"].unstack("date").reindex(columns=dates[-t:])
    filled_returns = fill_short_internal_return_gaps(raw_returns)
    wide = filled_returns.dropna(axis=0)
    start_sizes = d.loc[weight_date, "mkt_cap"].reindex(wide.index).astype(float)
    keep = start_sizes.gt(0) & np.isfinite(start_sizes)
    wide = wide.loc[keep]
    start_sizes = start_sizes.loc[keep]
    imputed = raw_returns.reindex(index=wide.index).isna() & wide.notna()
    filled_cells = int(imputed.to_numpy().sum())
    filled_units = int(imputed.any(axis=1).sum())
    rescued_units = int(raw_returns.reindex(index=wide.index).isna().any(axis=1).sum())
    print(f"[crsp_daily_returns] zero-filled {filled_cells:,} internal return cells in "
          f"{filled_units:,} securities; rescued {rescued_units:,} otherwise incomplete securities", flush=True)
    last_sizes_all = d.loc[last_date, "mkt_cap"].astype(float)
    coverage = float(last_sizes_all.reindex(wide.index).sum() / last_sizes_all.sum())
    labels = pd.Series([str(x) for x in wide.index], index=wide.index)
    panel = SourcePanel(
        name="crsp_daily_returns",
        y=standardize_panel(wide),
        sizes=start_sizes,
        labels=labels,
        start_date=start_date.date(),
        last_date=last_date.date(),
        frequency="daily",
        size_name=f"market capitalization on prior trading date {weight_date.date()} (t-1)",
        value_name="daily stock return",
        coverage=coverage,
        notes=(f"Heuristic last {t}=20*252 trading-day window; fixed market-cap weights from the preceding trading date (t-1); "
               f"internal missing-return runs of at most {DAILY_RETURN_MAX_GAP} consecutive trading days are set to zero; "
               f"longer and endpoint gaps remain missing. "
               f"Filled {filled_cells} cells in {filled_units} retained securities, rescuing {rescued_units} securities. "
               "Pooled 1/99 winsorization and grand-mean centering after gap filling. "
               "No coverage-date optimization."),
        weight_timing=WEIGHT_TIMING,
        weight_date=weight_date.date(),
    )
    save_source_panel_cache(cache_path, panel, use_cache)
    return panel
