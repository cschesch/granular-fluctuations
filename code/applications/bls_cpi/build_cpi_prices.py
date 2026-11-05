#!/usr/bin/env python3
"""Assemble the long CPI price-index panel (US city average, SA, monthly).

The ``cu.data.0.Current.txt`` file only covers 1997+, so for a long history we
use the BLS historical group files in ``data/bls_cpi/history/`` (downloaded from
``download.bls.gov/pub/time.series/cu/``: ``cu.data.1.AllItems``,
``cu.data.2.Summaries``, ``cu.data.11..18`` major groups, ``cu.data.20`` special
aggregates).  Together these carry every US-city-average series back to its
start (All items to 1913; many expenditure classes to the 1950s-60s).

Output: ``.cache/derived/bls_cpi/cpi_price_long.parquet`` -- tidy
(``item_code``, ``date`` [monthly Period], ``value``) for every CUSR0000 series
(US city average, seasonally adjusted), monthly observations only.

Also prints a history-vs-granularity summary: for a few "earliest start year"
cutoffs, how fine an *additive* partition of All items we can form using only
nodes whose price series reaches that far back.  This is the lever for "as long
a history as possible even if coarse": pick a cutoff, take the deepest tree node
on each branch that still has history past the cutoff.
"""

from __future__ import annotations


import pandas as pd

from project_paths import DERIVED_CACHE, ROOT


BCPI = ROOT / "data" / "bls_cpi"
HIST = BCPI / "history"
OUT = DERIVED_CACHE / "bls_cpi"


def load_prices() -> pd.DataFrame:
    files = list(HIST.glob("cu.data.*")) + [BCPI / "cu.data.0.Current.txt"]
    frames = []
    for f in files:
        d = pd.read_csv(f, sep="\t", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        d["series_id"] = d["series_id"].str.strip()
        # US city average, provider-seasonally-adjusted, monthly.
        d = d[d["series_id"].str.startswith("CUSR0000")]
        d = d[d["period"].str.strip().str.match(r"M(0[1-9]|1[0-2])$")]
        d["item_code"] = d["series_id"].str[8:]
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["date"] = pd.PeriodIndex(
            d["year"].str.strip() + "-" + d["period"].str.strip().str[1:], freq="M"
        )
        frames.append(d[["item_code", "date", "value"]].dropna(subset=["value"]))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["item_code", "date"])
    return out.sort_values(["item_code", "date"]).reset_index(drop=True)


def main() -> int:
    prices = load_prices()
    OUT.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(OUT / "cpi_price_long.parquet", index=False)
    n_codes = prices["item_code"].nunique()
    start = prices.groupby("item_code")["date"].min().dt.year
    print(f"[cpi-prices] {len(prices):,} monthly obs, {n_codes} US-SA item codes")
    print(f"[cpi-prices] overall span {prices['date'].min()} .. {prices['date'].max()}")

    # History-vs-granularity tradeoff menu. The natural finest partition is the
    # latest-year expenditure *leaves*; each unit's balanced history is bounded by
    # its price-series start. So for a panel that begins in year Y, you keep the
    # leaves with start<=Y (others drop into an uncovered residual). Coarser
    # partitions (parents) start earlier and so reach further back.
    keyed = pd.read_csv(OUT / "cpi_weights_keyed.csv")
    latest = int(keyed["year"].max())
    code_start = start.to_dict()
    leaves = keyed[(keyed["year"] == latest) & (keyed["section"] == "expenditure")
                   & (keyed["is_leaf"] == True)].copy()  # noqa: E712
    leaves["start_year"] = leaves["item_code"].map(code_start)

    print(f"\n[cpi-prices] tradeoff: finest partition = {len(leaves)} keyed "
          f"expenditure leaves ({latest} weights). Panel starting in year Y keeps "
          f"leaves whose price series reaches Y:")
    print(f"   {'Y':>5}  {'#leaves':>7}  {'CPI-U covered':>13}")
    for y in (2000, 1997, 1990, 1980, 1970, 1960):
        sub = leaves[leaves["start_year"].notna() & (leaves["start_year"] <= y)]
        print(f"   {y:>5}  {len(sub):>7}  {sub['cpi_u'].sum():>12.1f}%")

    # Coarse end of the menu: the 8 major expenditure groups (level-1 nodes).
    groups = keyed[(keyed["year"] == latest) & (keyed["section"] == "expenditure")
                   & (keyed["level"] == 1)].copy()
    groups["start_year"] = groups["item_code"].map(code_start)
    print(f"\n[cpi-prices] coarse end: {len(groups)} major groups, "
          f"earliest common start {int(groups['start_year'].max())} "
          f"(weights sum {groups['cpi_u'].sum():.1f}%):")
    for _, r in groups.sort_values("start_year").iterrows():
        print(f"     {int(r['start_year'])}  {r['item_code']:6s} {r['item_name_raw']} "
              f"({r['cpi_u']:.1f}%)")
    print("\n[cpi-prices] wrote: cpi_price_long.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
