"""Firm granularity and US GDP volatility."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import OUTPUT_DIR, ROOT, RUN_OUTPUT_CACHE, choose_solver
from applications.panels import load_gdp_deflator
from wrds_inputs import ANALYSIS_YEAR, wrds_path
from . import run_bea_granular_ceiling as bea
from .paths import build_paths

M = 10
GDPC1_PATH = ROOT / "data/fred/GDPC1.csv"
EXCLUDED_SIC = (
    {2911, 5172, 1311, 4922, 4923, 4924, 1389}
    | set(range(4901, 4941))
    | set(range(6001, 7001))
)


@dataclass(frozen=True)
class Panel:
    key: str
    label: str
    growth: np.ndarray          # N x T unit growth rates, the frozen covariance
    weights: pd.DataFrame       # years x N weight vectors
    labels: pd.Series           # N unit names
    weight_note: str
    unit_note: str


def real_gdp() -> pd.Series:
    """Annual real GDP, averaging the quarterly FRED series within calendar years."""
    d = pd.read_csv(GDPC1_PATH, parse_dates=["observation_date"])
    d["GDPC1"] = pd.to_numeric(d["GDPC1"], errors="coerce")
    d = d.dropna(subset=["GDPC1"])
    d["year"] = d["observation_date"].dt.year
    counts = d.groupby("year").size()
    annual = d.groupby("year")["GDPC1"].mean()
    return annual.loc[counts.index[counts.eq(4)]]


def firm_panel(start_year: int) -> Panel:
    """Balanced Compustat firms, Gabaix (2011)'s labor-productivity shock, with
    each year's real sales / real GDP as the weight."""
    source = wrds_path("compustat_annual")
    d = pd.read_csv(
        source,
        usecols=["gvkey", "datadate", "fyear", "sic", "sale", "emp", "conm", "loc", "fic"],
    )
    d = d.loc[d["loc"].eq("USA") & d["fic"].eq("USA")].copy()
    d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
    for column in ("fyear", "sic", "sale", "emp"):
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d = d.dropna(subset=["gvkey", "fyear", "sic", "sale", "emp", "conm"])
    d = d.loc[~d["sic"].astype(int).isin(EXCLUDED_SIC)]
    d = d.loc[d["sale"].gt(0) & d["emp"].gt(0)].copy()
    d["year"] = d["fyear"].astype(int)
    d = (
        d.sort_values(["gvkey", "year", "datadate"])
        .drop_duplicates(["gvkey", "year"], keep="last")
    )

    deflator = load_gdp_deflator("annual")
    deflator.index = deflator.index.year
    gdp = real_gdp()
    years = [y for y in range(start_year, ANALYSIS_YEAR + 1) if y in deflator.index and y in gdp.index]
    d = d.loc[d["year"].isin(years)].copy()
    # Compustat ``sale`` is millions of nominal dollars and GDPDEF is an index with
    # 2017 = 100, so deflate AND convert to the billions of chained 2017 dollars
    # GDPC1 is quoted in: without the 1000/100 the weights are ten times too large
    # (shares and the log-log HHI regression are scale-free, but the volatility
    # levels the figure plots against GDP volatility are not).
    d["real_sale"] = d["sale"] / 1000.0 * (100.0 / d["year"].map(deflator))
    d["productivity"] = np.log(d["real_sale"] / d["emp"])

    sales = d.pivot(index="gvkey", columns="year", values="real_sale").reindex(columns=years)
    productivity = d.pivot(index="gvkey", columns="year", values="productivity").reindex(columns=years)
    balanced = sales.notna().all(axis=1) & productivity.notna().all(axis=1)
    sales, productivity = sales.loc[balanced], productivity.loc[balanced]
    if sales.shape[0] < 50:
        raise ValueError(
            f"Only {sales.shape[0]} firms are balanced over {start_year}--{ANALYSIS_YEAR}; "
            "raise --firm-start."
        )
    growth = productivity.diff(axis=1).iloc[:, 1:]
    # Sales / GDP, as in Gabaix (2011): both sides are billions of chained 2017
    # dollars, so this is a genuine Domar-style level and the ceiling's units are
    # annualized aggregate volatility.
    weights = sales.T.div(pd.Series(gdp).reindex(years).to_numpy(), axis=0)
    labels = (
        d.drop_duplicates("gvkey").set_index("gvkey")["conm"].reindex(sales.index)
        .astype(str).str.title()
    )
    return Panel(
        key="firms",
        label=f"Compustat firms: labor-productivity growth, {start_year + 1}--{ANALYSIS_YEAR}",
        growth=growth.to_numpy(dtype=float),
        weights=weights,
        labels=labels,
        weight_note="weight = firm real sales / real GDP",
        unit_note=f"{sales.shape[0]} firms balanced from {start_year}",
    )


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--firm-start", type=int, default=1970)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    panel = firm_panel(args.firm_start)
    paths = build_paths(panel.growth, panel.weights, args.M, choose_solver(args.solver), args.references)
    figure_data = paths.rename(columns={"diagonal_volatility": "cg_diagonal_volatility"})
    bea.plot_figure1(figure_data.join(bea.gdp_volatility()), args.output_dir / "firm_granular_volatility",
                     frozen_paths=paths)
    paths.to_csv(RUN_OUTPUT_CACHE / "firm_granular_volatility_paths.csv")


if __name__ == "__main__":
    main()
