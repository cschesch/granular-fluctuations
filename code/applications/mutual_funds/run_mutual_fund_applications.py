#!/usr/bin/env python3
"""Daily CRSP mutual-fund returns and monthly net dollar flows by portfolio.

Dated CRSP mappings consolidate share classes. Daily simple returns use prior
month-end class assets; monthly flows use month-end assets and monthly returns.
Neither panel receives seasonal adjustment or linear detrending.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS, DAILY_RETURN_MAX_GAP, ROOT, choose_solver, clean_panel_cache_path,
    fill_short_internal_return_gaps,
    read_clean_panel_cache, write_clean_panel_cache,
)
from applications.additional.common import Application, estimate
from wrds_inputs import wrds_path

CACHE = ROOT / "code/.cache/run_outputs"
# First full calendar year after the broad 2010 CRSP portfolio-ID reset.
FIRST_LEVEL = pd.Period("2010-12", freq="M")
LAST_LEVEL = pd.Period("2025-12", freq="M")


def attach_portfolios(d: pd.DataFrame, headers: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach share classes to the portfolio valid on each observation date."""
    headers = headers[["crsp_fundno", "m_fund", "et_flag"]].drop_duplicates("crsp_fundno")
    d = d.merge(headers, on="crsp_fundno", how="left", validate="many_to_one")
    mapping = mapping[["crsp_fundno", "crsp_portno", "begdt", "enddt"]].copy()
    mapping["begdt"] = pd.to_datetime(mapping["begdt"], errors="coerce").fillna(pd.Timestamp.min)
    mapping["enddt"] = pd.to_datetime(mapping["enddt"], errors="coerce").fillna(pd.Timestamp.max)
    d = d.merge(mapping, on="crsp_fundno", how="inner", validate="many_to_many")
    d = d.loc[d["caldt"].between(d["begdt"], d["enddt"])].copy()
    if d.duplicated(["crsp_fundno", "caldt"]).any():
        raise ValueError("Overlapping CRSP portfolio mappings for the same share-class date")
    return d


def read_identifiers(header_path: Path, mapping_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    headers = pd.read_csv(header_path, usecols=["crsp_fundno", "m_fund", "et_flag", "first_offer_dt", "end_dt", "dead_flag", "delist_cd"], dtype=str)
    mapping = pd.read_csv(mapping_path, dtype=str)
    for d in (headers, mapping):
        d["crsp_fundno"] = d["crsp_fundno"].str.strip().str.zfill(6)
    return headers, mapping


def eligible_share_classes(d: pd.DataFrame) -> pd.DataFrame:
    """CRSP ET_FLAG is F for ETFs, N for ETNs, and blank for ordinary funds."""
    money_market = d.m_fund.fillna("").str.strip().eq("Y")
    exchange_traded = d.et_flag.fillna("").str.strip().isin(["F", "N"])
    return d.loc[d.crsp_portno.notna() & ~money_market & ~exchange_traded].copy()


def read_monthly(path: Path, value: str) -> pd.DataFrame:
    """Read actual monthly TNA/returns, not stale quarterly summary values."""
    pieces = []
    for chunk in pd.read_csv(path, usecols=["crsp_fundno", "caldt", value],
                             dtype=str, chunksize=1_000_000):
        chunk = chunk.loc[chunk.caldt.between(str(FIRST_LEVEL.start_time.date()),
                                             str(LAST_LEVEL.end_time.date()))].copy()
        if chunk.empty:
            continue
        chunk["crsp_fundno"] = chunk.crsp_fundno.str.strip().str.zfill(6)
        chunk["caldt"] = pd.to_datetime(chunk.caldt)
        chunk["month"] = chunk.caldt.dt.to_period("M")
        chunk[value] = pd.to_numeric(chunk[value], errors="coerce")
        pieces.append(chunk)
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated(["crsp_fundno", "month"]).any():
        raise ValueError(f"Duplicate share-class months in {path.name}")
    return out


def balanced_assets(tna: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.period_range(FIRST_LEVEL, LAST_LEVEL, freq="M")
    levels = tna.pivot_table(index="crsp_portno", columns="month", values="tna", aggfunc="sum")
    levels = levels.reindex(columns=months)
    levels = levels.loc[levels.gt(0).all(axis=1)]
    return tna.loc[tna.crsp_portno.isin(levels.index)].copy(), levels


def lagged_class_assets(tna: pd.DataFrame) -> pd.DataFrame:
    lag = tna[["crsp_fundno", "crsp_portno", "month", "tna"]].copy()
    lag["month"] += 1
    lag = lag.rename(columns={"tna": "lag_tna"})
    if lag.duplicated(["crsp_fundno", "month"]).any():
        raise ValueError("Multiple portfolio asset records for the same share-class month")
    return lag


def aggregate_class_returns(d: pd.DataFrame, period: str) -> pd.DataFrame:
    """Sum sufficient statistics, so splitting source chunks cannot change weights."""
    d = d.loc[d["class_return"].notna() & d["lag_tna"].gt(0)].copy()
    d["weighted_return"] = d["lag_tna"] * d["class_return"]
    return d.groupby(["crsp_portno", period], observed=True)[["weighted_return", "lag_tna"]].sum()


def documented_asset_portfolios(tna: pd.DataFrame, returns: pd.DataFrame,
                                 headers: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.Index, dict]:
    """Missing AUM is zero only outside documented class life or membership."""
    months = pd.period_range(FIRST_LEVEL, LAST_LEVEL, freq="M")
    keys = ["crsp_fundno", "crsp_portno"]
    m = mapping.loc[mapping.crsp_portno.isin(tna.crsp_portno.unique())].copy()
    m["beg"] = pd.to_datetime(m.begdt, errors="coerce").dt.to_period("M")
    m["end"] = pd.to_datetime(m.enddt, errors="coerce").dt.to_period("M")
    m = m.loc[(m.beg.isna() | m.beg.le(LAST_LEVEL)) & (m.end.isna() | m.end.ge(FIRST_LEVEL))]
    m = eligible_share_classes(m.merge(headers[["crsp_fundno", "m_fund", "et_flag"]], on="crsp_fundno", validate="many_to_one"))
    pairs = m[keys].drop_duplicates()
    index = pd.MultiIndex.from_frame(pairs)
    observed = np.zeros((len(pairs), len(months)), dtype=bool)
    asset_rows = index.get_indexer(pd.MultiIndex.from_frame(tna[keys]))
    if (asset_rows < 0).any():
        raise ValueError("Observed class AUM lacks a dated portfolio membership")
    observed[asset_rows, tna.month.astype("int64").to_numpy() - FIRST_LEVEL.ordinal] = True
    member = np.zeros_like(observed)
    for row in m.itertuples():
        lo = 0 if pd.isna(row.beg) else max(0, row.beg.ordinal - FIRST_LEVEL.ordinal)
        hi = len(months) if pd.isna(row.end) else min(len(months), row.end.ordinal - FIRST_LEVEL.ordinal + 1)
        member[index.get_loc((row.crsp_fundno, row.crsp_portno)), lo:hi] = True
    h = headers.set_index("crsp_fundno").reindex(pairs.crsp_fundno).reset_index(drop=True)
    first = pd.to_datetime(h.first_offer_dt, errors="coerce").dt.to_period("M")
    last = pd.to_datetime(h.end_dt, errors="coerce").dt.to_period("M")
    history = pd.concat([tna[["crsp_fundno", "month"]], returns[["crsp_fundno", "month"]]])
    history = history.groupby("crsp_fundno").month.agg(["min", "max"]).reindex(pairs.crsp_fundno).reset_index(drop=True)
    # Observed history overrides lifecycle dates that contradict it.
    first = first.where(history["min"].isna() | first.le(history["min"]), history["min"]).where(first.notna())
    last = last.where(history["max"].isna() | last.ge(history["max"]), history["max"]).where(last.notna())
    time = months.astype("int64").to_numpy()[None, :]
    before = first.notna().to_numpy()[:, None] & (time < first.astype("int64").to_numpy()[:, None])
    closed = h.dead_flag.eq("Y") & h.delist_cd.isin(["L", "M"]) & last.notna()
    after = closed.to_numpy()[:, None] & (time > last.astype("int64").to_numpy()[:, None])
    unknown = ~observed & member & ~before & ~after
    bad = pd.Series(unknown.any(axis=1), index=pairs.crsp_portno).groupby(level=0).any()
    return bad.index[~bad], {"dropped_unresolved_aum_portfolios": int(bad.sum()),
                             "unresolved_class_aum_months": int(unknown.sum())}


def build_portfolio_panels(tna: pd.DataFrame, returns: pd.DataFrame,
                           headers: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Class-first flows after documented-lifecycle AUM and peer-return screens."""
    valid_assets, asset_diagnostics = documented_asset_portfolios(tna, returns, headers, mapping)
    tna = tna.loc[tna.crsp_portno.isin(valid_assets)]
    levels = tna.pivot_table(index="crsp_portno", columns="month", values="tna", aggfunc="sum")
    levels = levels.reindex(columns=pd.period_range(FIRST_LEVEL, LAST_LEVEL, freq="M")).fillna(0.)
    lag = lagged_class_assets(tna)
    keys = ["crsp_fundno", "crsp_portno", "month"]
    groups = ["crsp_portno", "month"]
    months = levels.columns[1:]
    d = tna[keys + ["tna"]].merge(lag, on=keys, how="outer", validate="one_to_one")
    observed = returns.loc[returns.crsp_portno.isin(levels.index) & returns.month.isin(months), keys + ["mret"]]
    d = d.merge(observed, on=keys, how="outer", validate="one_to_one")
    d = d.loc[d.month.isin(months)].copy()
    d[["tna", "lag_tna"]] = d[["tna", "lag_tna"]].fillna(0.)
    observed = d.loc[d.mret.notna()]
    def peer_return(asset):
        total = (observed.mret * observed[asset]).groupby([observed.crsp_portno, observed.month]).sum()
        return total / observed.groupby(groups, observed=True)[asset].sum().replace(0., np.nan)
    # Prefer lagged AUM; at class turnover use current AUM, then equal observed-class weights.
    peer = peer_return("lag_tna").fillna(peer_return("tna")).fillna(observed.groupby(groups, observed=True).mret.mean())
    d = d.join(peer.rename("peer_return"), on=groups)
    d["mret"] = d.mret.fillna(d.peer_return)
    d["flow"] = d.tna - (1. + d.mret) * d.lag_tna
    flows = d.groupby(groups, observed=True).flow.sum(min_count=1).unstack("month").reindex(index=levels.index, columns=months)
    r = peer.unstack("month").reindex(index=levels.index, columns=months)
    # With no assets left, missing returns need not exclude the portfolio:
    # record liquidation as withdrawal of the previous assets, then zero flows.
    inactive = levels.loc[:, months].eq(0.)
    previous = levels.shift(1, axis=1).loc[:, months]
    inactive_missing = inactive & r.isna()
    flows = flows.mask(inactive_missing, -previous)
    zero_aum_missing = int(inactive_missing.to_numpy().sum())
    r = r.mask(inactive_missing, 0.)
    valid = flows.notna().all(axis=1) & r.notna().all(axis=1)
    invalid_months = int(flows.isna().to_numpy().sum())
    r, flows, levels = r.loc[valid], flows.loc[valid], levels.loc[valid]
    diagnostics = {
        **asset_diagnostics,
        "portfolios_with_positive_assets": int(valid.size),
        "zero_aum_missing_return_months": zero_aum_missing,
        "balanced_after_return_coverage": int(valid.sum()),
        "dropped_fund_months_no_class_return": invalid_months,
    }
    return r, flows, levels.loc[:, months].mean(axis=1), diagnostics


def daily_portfolio_returns(path: Path, tna: pd.DataFrame, headers: pd.DataFrame,
                            mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    classes, levels = balanced_assets(tna)
    lag = lagged_class_assets(classes)
    fund_ids = set(lag.crsp_fundno)
    pieces, dates = [], set()
    for j, chunk in enumerate(pd.read_csv(path, usecols=["crsp_fundno", "caldt", "dret"],
                                          dtype=str, chunksize=2_000_000), 1):
        chunk = chunk.loc[chunk.caldt.between(str((FIRST_LEVEL + 1).start_time.date()),
                                             str(LAST_LEVEL.end_time.date()))].copy()
        # Use the source's complete daily calendar, before selecting our cohort.
        dates.update(chunk.caldt.unique())
        chunk["crsp_fundno"] = chunk.crsp_fundno.str.strip().str.zfill(6)
        chunk = chunk.loc[chunk.crsp_fundno.isin(fund_ids)].copy()
        if not chunk.empty:
            chunk["caldt"] = pd.to_datetime(chunk.caldt)
            chunk["month"] = chunk.caldt.dt.to_period("M")
            chunk["class_return"] = pd.to_numeric(chunk.dret, errors="coerce")
            chunk = chunk.loc[chunk.class_return.ge(-1) & np.isfinite(chunk.class_return)]
            chunk = attach_portfolios(chunk[["crsp_fundno", "caldt", "month", "class_return"]], headers, mapping)
            chunk = chunk.merge(lag, on=["crsp_fundno", "crsp_portno", "month"],
                                how="inner", validate="many_to_one")
            pieces.append(aggregate_class_returns(chunk, "caldt"))
        if j % 10 == 0:
            print(f"[mutual funds] scanned {2*j:,} million daily rows", flush=True)
    stats = pd.concat(pieces).groupby(level=[0, 1]).sum()
    calendar = pd.DatetimeIndex(sorted(dates), name="caldt")
    r = (stats.weighted_return / stats.lag_tna).unstack("caldt").reindex(index=levels.index, columns=calendar)
    represented = stats.lag_tna.unstack("caldt").reindex(index=levels.index, columns=calendar)
    previous = levels.reindex(columns=calendar.to_period("M") - 1)
    previous.columns = calendar
    coverage = represented / previous
    before_fill = r.copy()
    r = fill_short_internal_return_gaps(r)
    valid = r.notna().all(axis=1)
    imputed = before_fill.loc[valid].isna() & r.loc[valid].notna()
    diagnostics = {
        "balanced_positive_monthly_assets": len(levels),
        "balanced_after_return_coverage": int(valid.sum()),
        "minimum_return_aum_coverage": float(coverage.loc[valid].where(before_fill.loc[valid].notna()).min().min()),
        "complete_before_gap_fill": int(before_fill.notna().all(axis=1).sum()),
        "portfolios_with_zero_filled_returns": int(imputed.any(axis=1).sum()),
        "zero_filled_return_cells": int(imputed.to_numpy().sum()),
        "max_internal_gap_trading_days": DAILY_RETURN_MAX_GAP,
    }
    if not valid.any():
        raise ValueError(f"No portfolios satisfy daily completeness over {len(calendar)} dates; "
                         f"minimum date-level coverage: {coverage.max(axis=0).nsmallest().to_dict()}")
    return r.loc[valid], levels.loc[valid, FIRST_LEVEL], diagnostics


def prepared_panels(*, assets: Path | None = None, monthly_returns: Path | None = None,
                    daily_returns: Path | None = None, rebuild: bool = False) -> dict:
    assets = assets or wrds_path("mutual_fund_monthly_assets")
    monthly_returns = monthly_returns or wrds_path("mutual_fund_monthly_returns")
    daily_returns = daily_returns or wrds_path("mutual_fund_daily_returns")
    header = wrds_path("mutual_fund_headers")
    mapping_path = wrds_path("mutual_fund_portfolio_map")
    cache = clean_panel_cache_path(
        "crsp_mutual_fund_daily_monthly",
        {"version": 8, "first": str(FIRST_LEVEL), "last": str(LAST_LEVEL),
         "monthly_cleaning": "class_flows_documented_lifecycle_peer_returns", "daily_coverage": "any_observed_class",
         "daily_internal_zero_gap_max": DAILY_RETURN_MAX_GAP,
         "return_weights": "December 2010 assets", "flow_rate_screen": "none",
         "flow_size_months": "outcome_months"},
        [assets, monthly_returns, daily_returns, header, mapping_path],
    )
    cached = None if rebuild else read_clean_panel_cache(cache)
    if cached is not None:
        return cached
    headers, mapping = read_identifiers(header, mapping_path)
    tna = read_monthly(assets, "mtna").rename(columns={"mtna": "tna"})
    tna = tna.loc[tna.tna.gt(0)].copy()
    tna = attach_portfolios(tna, headers, mapping)
    tna = eligible_share_classes(tna)
    tna = tna[["crsp_fundno", "crsp_portno", "caldt", "month", "tna"]]
    monthly = read_monthly(monthly_returns, "mret")
    monthly = monthly.loc[monthly.mret.ge(-1) & np.isfinite(monthly.mret)]
    monthly = eligible_share_classes(attach_portfolios(monthly, headers, mapping))
    _, flows, flow_sizes, flow_diagnostics = build_portfolio_panels(tna, monthly, headers, mapping)
    print(f"[mutual funds] monthly flow panel: {flows.shape}; {flow_diagnostics}", flush=True)
    daily_cache = clean_panel_cache_path("crsp_mutual_fund_daily",
        {"first": str(FIRST_LEVEL), "last": str(LAST_LEVEL), "coverage": "any_observed_class",
         "internal_zero_gap_max": DAILY_RETURN_MAX_GAP,
         "weight_month": str(FIRST_LEVEL)}, [assets, daily_returns, header, mapping_path])
    daily_panel = None if rebuild else read_clean_panel_cache(daily_cache)
    if daily_panel is None:
        daily, return_sizes, return_diagnostics = daily_portfolio_returns(daily_returns, tna, headers, mapping)
        daily_panel = dict(returns=daily, return_sizes=return_sizes, return_diagnostics=return_diagnostics)
        write_clean_panel_cache(daily_cache, daily_panel)
    print(f"[mutual funds] daily return panel: {daily_panel['returns'].shape}; "
          f"{daily_panel['return_diagnostics']}", flush=True)
    payload = dict(daily_panel, flows=flows, flow_sizes=flow_sizes, flow_diagnostics=flow_diagnostics)
    write_clean_panel_cache(cache, payload)
    return payload


def portfolio_applications(returns: pd.DataFrame, flows: pd.DataFrame, return_sizes: pd.Series,
                           flow_sizes: pd.Series) -> list[Application]:
    """One construction shared by estimation and all downstream diagnostics."""
    dates = returns.columns.to_timestamp() if isinstance(returns.columns, pd.PeriodIndex) else returns.columns
    return_sample = f"{dates[0].date()}--{dates[-1].date()}"
    return_labels = pd.Series(returns.index.astype(str), index=returns.index)
    flow_labels = pd.Series(flows.index.astype(str), index=flows.index)
    flow_sample = f"{flows.columns[0].year}M{flows.columns[0].month}--{flows.columns[-1].year}M{flows.columns[-1].month}"
    return [
        Application("Mutual fund portfolios: returns", "Daily total return",
                    "CRSP Survivor-Bias-Free US Mutual Fund Database via WRDS", "",
                    returns, return_sizes, return_labels, return_sample, "Daily", "December 2010 AUM",
                    "Daily simple share-class returns consolidated using dated CRSP portfolio mappings and "
                    "prior month-end class assets, renormalized over the observed classes each day. "
                    "At least one observed class suffices; internal all-class-missing runs of at most three "
                    "consecutive trading days are set to zero. Longer and endpoint gaps remain missing. "
                    "Positive monthly assets from December 2010 through December 2025. "
                    "Fixed December 31, 2010 AUM shares; no seasonal adjustment. Existing pooled 1/99 winsorization retained."),
        Application("Mutual fund portfolios: flows", "Monthly net dollar flow",
                    "CRSP Survivor-Bias-Free US Mutual Fund Database via WRDS", "",
                    flows, flow_sizes, flow_labels, flow_sample, "Monthly", "Full-sample mean AUM",
                    "Monthly class flows equal TNA_t-(1+r_t)TNA_{t-1} and are summed by dated portfolio mappings. "
                    "Absent class AUM is zero only before the documented offer date, after confirmed liquidation/merger and the last NAV month, or outside dated portfolio membership. Observed asset/return history extends conflicting lifecycle dates. Other missing AUM excludes the portfolio from the balanced panel. Observed returns are retained and missing returns use observed sibling "
                    "classes, weighted by lagged AUM (current AUM, then equal weights if necessary). "
                    "A positive-AUM fund-month with no observed class return is missing; the estimation panel is balanced. Zero-AUM months remain: absent returns imply withdrawal of lagged assets in the liquidation month and zero subsequent flows. "
                    "Signed dollar flows enter in levels without detrending, seasonal adjustment, or winsorization. "
                    "Flows divided by fixed mean AUM are grouped and aggregated with those same AUM shares. "
                    "Mean monthly AUM over January 2011–December 2025 flow observations determines sizes. "
                    "Portfolios with zero mean AUM over these months cannot be normalized and are excluded.", aggregation="sum"),
    ]


def load_applications() -> list[Application]:
    p = prepared_panels()
    return portfolio_applications(p["returns"], p["flows"], p["return_sizes"], p["flow_sizes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, help="Monthly TNA export")
    parser.add_argument("--monthly-returns", type=Path)
    parser.add_argument("--daily-returns", type=Path)
    parser.add_argument("--rebuild-panels", action="store_true")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    p = prepared_panels(assets=args.assets, monthly_returns=args.monthly_returns,
                        daily_returns=args.daily_returns, rebuild=args.rebuild_panels)
    apps = portfolio_applications(p["returns"], p["flows"], p["return_sizes"], p["flow_sizes"])
    rows = []
    for app, diagnostics in zip(apps, (p["return_diagnostics"], p["flow_diagnostics"]), strict=True):
        row, _ = estimate(app, choose_solver("auto"), args.bootstrap_b, 20260907)
        row.update(diagnostics)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_pickle(CACHE / "mutual_fund_applications_summary.pkl")
    print(out[["Entity", "Frequency", "N", "T", "M", "HHI", "MGP_share", "UCL95"]].to_string(index=False))


if __name__ == "__main__":
    main()
