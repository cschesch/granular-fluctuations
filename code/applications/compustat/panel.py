"""Annual Compustat panels and quarterly labor productivity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import (
    make_panel_contiguous,
    remove_periodic_means,
    standardize_panel,
)
from wrds_inputs import ANALYSIS_LAST_QUARTER, ANALYSIS_YEAR, wrds_path

from applications.panels import (
    GDPDEF_PATH,
    SourcePanel,
    WEIGHT_TIMING,
    load_gdp_deflator,
    load_source_panel_cache,
    save_source_panel_cache,
    source_panel_cache_path,
)


# Gabaix-style exclusions shared by the annual and quarterly panels.
EXCLUDED_COMPUSTAT_SIC = {1311, 1389, 2911, 5172} | set(range(4901, 4941)) | set(range(6001, 7001))


def load_compustat_annual_panel(value: str, t: int, use_cache: bool = True, refresh_cache: bool = False) -> SourcePanel:
    # "tfp" is Gabaix (2011)'s granular-residual shock: labor productivity growth
    # dlog(real sale/emp). Every application uses the size observed immediately before
    # the first reported shock, so the fixed grouping weights are predetermined.
    if value not in {"sale", "emp", "tfp"}:
        raise ValueError(value)
    source_path = wrds_path("compustat_annual")
    sources = [source_path] + ([GDPDEF_PATH] if value == "tfp" else [])
    cache_path = source_panel_cache_path(
        f"compustat_annual_{value}",
        {"value": value, "t": int(t), "last_year": ANALYSIS_YEAR, "weight_timing": WEIGHT_TIMING},
        sources,
    )
    cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
    if cached is not None:
        return cached

    usecols = ["gvkey", "datadate", "fyear", "sic", "sale", "emp", "conm", "loc", "fic", "curcd"]
    d = pd.read_csv(source_path, usecols=usecols)
    d = d[(d["loc"] == "USA") & (d["fic"] == "USA") & d["curcd"].eq("USD")].copy()
    d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
    d["fyear"] = pd.to_numeric(d["fyear"], errors="coerce").astype("Int64")
    d = d.loc[d["fyear"].le(ANALYSIS_YEAR)].copy()
    d["sic"] = pd.to_numeric(d["sic"], errors="coerce")
    d["sale"] = pd.to_numeric(d["sale"], errors="coerce")
    d["emp"] = pd.to_numeric(d["emp"], errors="coerce")
    d = d.dropna(subset=["gvkey", "fyear", "sic", "sale", "emp", "conm"])
    d["sic"] = d["sic"].astype(int)
    d = d.loc[~d["sic"].isin(EXCLUDED_COMPUSTAT_SIC)].copy()
    d = d.loc[(d["sale"] > 0) & (d["emp"] > 0)].copy()
    d["year"] = pd.PeriodIndex(d["fyear"].astype(int).astype(str), freq="Y")
    d = d.sort_values(["gvkey", "year", "datadate"]).drop_duplicates(["gvkey", "year"], keep="last")
    g = d.groupby("gvkey", sort=False)
    d["lagged_fyear"] = g["fyear"].shift()
    d["lagged_sale"] = g["sale"].shift()
    d["lagged_emp"] = g["emp"].shift()
    if value == "tfp":
        deflator = load_gdp_deflator("annual")
        d["gdp_deflator"] = d["year"].map(deflator)
        d = d.dropna(subset=["gdp_deflator"]).copy()
        d["real_sale"] = d["sale"] / d["gdp_deflator"]
        g = d.groupby("gvkey", sort=False)
        d["lagged_real_sale"] = g["real_sale"].shift()
        d = d.loc[
            d["lagged_fyear"].eq(d["fyear"].astype(float) - 1.0)
            & d["lagged_real_sale"].gt(0)
            & d["lagged_emp"].gt(0)
        ].copy()
        d["prod_logdiff"] = np.log(d["real_sale"] / d["emp"]) - np.log(d["lagged_real_sale"] / d["lagged_emp"])
        growth_col = "prod"
        size_col = "lagged_real_sale"
        coverage_size_col = "real_sale"
    else:
        d = d.loc[
            d["lagged_fyear"].eq(d["fyear"].astype(float) - 1.0)
            & d[f"lagged_{value}"].gt(0)
        ].copy()
        d[f"{value}_logdiff"] = np.log(d[value] / d[f"lagged_{value}"])
        growth_col = value
        size_col = f"lagged_{value}"
        coverage_size_col = value
    d = d.set_index(["year", "gvkey"]).sort_index()
    full_last_sizes_by_year = d.groupby(level="year")[coverage_size_col].sum()
    keep = list(dict.fromkeys([size_col, coverage_size_col, f"{growth_col}_logdiff", "conm"]))
    d = d.dropna(subset=[f"{growth_col}_logdiff", size_col, "conm"])
    d = make_panel_contiguous(d[keep], "year", "gvkey")

    dates = d.index.unique("year").sort_values()
    if len(dates) < t:
        start_date = dates[0]
    else:
        start_date = dates[-t]
    last_date = dates[-1]
    y = d.loc[start_date:last_date, f"{growth_col}_logdiff"].unstack("year").dropna(axis=0)
    sizes = d.loc[start_date, size_col].reindex(y.index).astype(float)
    labels = d.loc[start_date, "conm"].reindex(y.index).astype(str)
    last_sizes = d.loc[last_date, coverage_size_col]
    coverage = float(last_sizes.reindex(y.index).sum() / full_last_sizes_by_year.loc[last_date])
    y = standardize_panel(y)
    y = y.sub(y.mean().mean())
    name_suffix = {"sale": "sales", "emp": "employment", "tfp": "tfp"}[value]
    panel = SourcePanel(
        name=f"compustat_annual_{name_suffix}",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=y.columns[0],
        last_date=y.columns[-1],
        frequency="annual",
        size_name=(
            "real sales in year preceding first growth (t-1)"
            if value == "tfp"
            else f"{value} in year preceding first growth (t-1)"
        ),
        value_name="log productivity (real sales/emp) growth" if value == "tfp" else f"log {value} growth",
        coverage=coverage,
        notes=(
            f"Trailing {y.shape[1]} annual observations; Gabaix-style real-sales labor productivity "
            "growth with fixed pre-growth (t-1) real-sales weights; log growth winsorized at 1st/99th pct."
            if value == "tfp"
            else f"Trailing {y.shape[1]} annual observations; fixed pre-growth (t-1) {value} weights; log growth winsorized at 1st/99th pct."
        ),
        weight_timing=WEIGHT_TIMING,
        weight_date=start_date - 1,
    )
    save_source_panel_cache(cache_path, panel, use_cache)
    return panel


def load_compustat_quarterly_tfp_panel(
    t: int,
    last_date: str = ANALYSIS_LAST_QUARTER,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> SourcePanel:
    """Gabaix labor-productivity growth on QUARTERLY sales with ANNUAL employees.

    Productivity level = real saleq / emp_{calendar year} (the annual head-count held
    constant within its year); the shock is QoQ dlog(productivity). The grouped panel
    uses fixed initial-period real quarterly sales weights, matching the other applications.
    """
    quarterly_path = wrds_path("compustat_quarterly_sales")
    emp_path = wrds_path("compustat_annual")
    cache_path = source_panel_cache_path(
        "compustat_quarterly_tfp",
        {"version": 2, "t": int(t), "last_date": str(last_date), "weight_timing": WEIGHT_TIMING,
         "seasonal_adjustment": "unit_quarter_fixed_effects"},
        [quarterly_path, emp_path, GDPDEF_PATH],
    )
    cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
    if cached is not None:
        return cached

    q = pd.read_csv(quarterly_path, usecols=["gvkey", "datadate", "datacqtr", "curcdq", "sic", "saleq", "conm", "loc", "fic"])
    q = q[(q["loc"] == "USA") & (q["fic"] == "USA") & q["curcdq"].eq("USD")].copy()
    q = q.dropna(subset=["datacqtr"])
    q["gvkey"] = q["gvkey"].astype(str).str.zfill(6)
    q["datadate"] = pd.PeriodIndex(q["datacqtr"], freq="Q")
    q["sic"] = pd.to_numeric(q["sic"], errors="coerce")
    q["saleq"] = pd.to_numeric(q["saleq"], errors="coerce")
    q = q.dropna(subset=["gvkey", "datadate", "sic", "saleq", "conm"])
    q = q.loc[q["datadate"] <= pd.Period(last_date, freq="Q")].copy()
    q["sic"] = q["sic"].astype(int)
    q = q.loc[~q["sic"].isin(EXCLUDED_COMPUSTAT_SIC) & (q["saleq"] > 0)].copy()
    q = q.drop_duplicates(subset=["gvkey", "datadate"], keep="last")
    q["year"] = q["datadate"].dt.year

    e = pd.read_csv(emp_path, usecols=["gvkey", "datadate", "emp"])
    e["gvkey"] = e["gvkey"].astype(str).str.zfill(6)
    e["year"] = pd.to_datetime(e["datadate"]).dt.year
    e["emp"] = pd.to_numeric(e["emp"], errors="coerce")
    e = e.loc[e["emp"] > 0].drop_duplicates(subset=["gvkey", "year"], keep="last")[["gvkey", "year", "emp"]]

    q = q.merge(e, on=["gvkey", "year"], how="inner")
    deflator = load_gdp_deflator("quarterly")
    q["gdp_deflator"] = q["datadate"].map(deflator)
    q = q.dropna(subset=["gdp_deflator"]).copy()
    q["real_saleq"] = q["saleq"] / q["gdp_deflator"]
    q = q.sort_values(["gvkey", "datadate"])
    g = q.groupby("gvkey", sort=False)
    q["quarter_ord"] = q["datadate"].astype("int64")
    q["lagged_quarter_ord"] = g["quarter_ord"].shift()
    q["lagged_real_saleq"] = g["real_saleq"].shift()
    q["lagged_emp"] = g["emp"].shift()
    q = q.loc[
        q["lagged_quarter_ord"].eq(q["quarter_ord"] - 1)
        & q["lagged_real_saleq"].gt(0)
        & q["lagged_emp"].gt(0)
    ].copy()
    q["prod_logdiff"] = np.log(q["real_saleq"] / q["emp"]) - np.log(q["lagged_real_saleq"] / q["lagged_emp"])
    q = q.set_index(["datadate", "gvkey"]).sort_index()
    full_sizes_by_date = q.groupby(level="datadate")["real_saleq"].sum()
    q = q.dropna(subset=["prod_logdiff", "real_saleq", "conm"])
    q = make_panel_contiguous(q[["real_saleq", "lagged_real_saleq", "prod_logdiff", "conm"]], "datadate", "gvkey")

    dates = q.index.unique("datadate").sort_values()
    start_date = dates[0] if len(dates) < t else dates[-t]
    last_date = dates[-1]
    y = q.loc[start_date:last_date, "prod_logdiff"].unstack("datadate").dropna(axis=0)
    y = remove_periodic_means(y, 4)
    y = standardize_panel(y)
    sizes = q.loc[start_date, "lagged_real_saleq"].reindex(y.index).astype(float)
    labels = q.loc[start_date, "conm"].reindex(y.index).astype(str)
    last_sizes = q.loc[last_date, "real_saleq"]
    coverage = float(last_sizes.reindex(y.index).sum() / full_sizes_by_date.loc[last_date])
    y = y.sub(y.mean().mean())
    panel = SourcePanel(
        name="compustat_quarterly_tfp",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=y.columns[0],
        last_date=y.columns[-1],
        frequency="quarterly",
        size_name="real saleq in quarter preceding first growth (t-1)",
        value_name="seasonally adjusted log productivity (real quarterly sales / annual emp) growth",
        coverage=coverage,
        notes=(
            f"Trailing {y.shape[1]} quarters; Gabaix-style real-sales labor productivity "
            "with annual employees held within calendar year and fixed pre-growth (t-1) real-sales weights; "
            "unit quarter-of-year effects removed; QoQ growth winsorized 1/99."
        ),
        weight_timing=WEIGHT_TIMING,
        weight_date=start_date - 1,
    )
    save_source_panel_cache(cache_path, panel, use_cache)
    return panel
