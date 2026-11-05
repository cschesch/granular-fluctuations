"""Public and private debt growth with fixed initial USD-debt weights."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from project_paths import DERIVED_CACHE, ROOT
from applications.common import Application

RAW = ROOT / "data/debt"
APPLICATION_CACHE = DERIVED_CACHE / "debt"
GMD_FILE = RAW / "gmd_2026_09.dta"
GDD_FILE = RAW / "GlobalDebtDatabase_2025_release.xlsx"
LAST = 2024

# Exact crosswalk for the GDD names that differ from the GMD. No fuzzy joins.
GDD_ALIASES = {
    "Antigua & Barbuda": "ATG", "Bahamas, The": "BHS", "Bosnia & Herzegovina": "BIH",
    "Brunei Darussalam": "BRN", "C.A.R.": "CAF", "China, Mainland": "CHN",
    "Congo, Dem. Rep. of": "COD", "Congo, Republic of": "COG", "Cote D'Ivoire": "CIV",
    "Gambia, The": "GMB", "Hong Kong SAR": "HKG", "Iran, I. Rep. Of": "IRN",
    "Korea, Republic of": "KOR", "Kyrgyz Republic": "KGZ", "Macedonia, FYR": "MKD",
    "Marshall Islands, Rep.": "MHL", "Slovak Republic": "SVK", "St. Kitts and Nevis": "KNA",
    "St. Lucia": "LCA", "St. Vincent and the Grenadines": "VCT", "Swaziland": "SWZ",
    "São Tomé and Príncipe": "STP", "Taiwan Province of China": "TWN", "Timor Leste": "TLS",
    "Trinidad & Tobago": "TTO", "Türkiye": "TUR", "U.A.E.": "ARE",
    "Venezuela, Republica Bolivariana de": "VEN", "West Bank and Gaza": "PSE",
}

# Exclusions concern unresolved currency units, not debt-growth outcomes or size.
# SLE: GDD GDP is in old leones, GMD FX in new leones (factor 1000).
# BGR: GDD is in lev; GMD is in euro after the 2026 changeover and its GDP
#      level also shows an unresolved additional conversion discrepancy.
# PSE: GDD is in shekels, while GMD uses a dollar GDP unit and FX=1.
# ZWE: multiple historical currency regimes; cross-source GDP-unit ratios have
#      a >20-fold discontinuity in 2007. Do not guess redenomination factors.
GDD_CURRENCY_EXCLUSIONS = {
    "SLE": "Old-leone GDP versus new-leone exchange rate",
    "BGR": "Lev/euro units and unresolved GMD conversion discrepancy",
    "PSE": "Shekel GDP versus GMD dollar unit and FX=1",
    "ZWE": "Unresolved historical redenomination/unit discontinuity",
}
GMD_CURRENCY_EXCLUSIONS = {
    "BGR": "Unresolved GDP currency-conversion discrepancy around euro changeover",
}

def read_inputs(include_gdd: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["ISO3", "countryname", "year", "gen_govdebt", "gen_govdebt_GDP",
               "nGDP", "nGDP_USD", "USDfx", "forecast_gen_govdebt",
               "forecast_nGDP", "forecast_USDfx"]
    gmd = pd.read_stata(GMD_FILE, columns=columns, convert_categoricals=False)
    gmd["year"] = gmd.year.astype(int)
    if gmd.duplicated(["ISO3", "year"]).any():
        raise ValueError("Duplicate GMD country-year observations")
    if not include_gdd:
        return gmd, pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data Validation extension is not supported")
        gdd = pd.read_excel(GDD_FILE, sheet_name="rawdata", header=2)
    required = {"ifscode", "country", "year", "gg_data", "cg_data", "tot_ls_data", "ngdp_data"}
    if not required <= set(gdd.columns):
        raise ValueError(f"Unexpected GDD rawdata schema: {list(gdd.columns)}")
    country_map = gmd[["countryname", "ISO3"]].drop_duplicates().set_index("countryname").ISO3.to_dict()
    country_map.update(GDD_ALIASES)
    gdd["ISO3"] = gdd.country.map(country_map)
    if gdd.ISO3.isna().any():
        raise ValueError(f"Unmapped GDD countries: {gdd.loc[gdd.ISO3.isna(), 'country'].unique()}")
    gdd["year"] = pd.to_numeric(gdd.year).astype(int)
    if gdd.duplicated(["ISO3", "year"]).any():
        raise ValueError("Duplicate country-year observations")
    if (int(gdd.year.min()), int(gdd.year.max())) != (1950, 2024):
        raise ValueError("GDD workbook does not cover the expected 1950–2024 release")
    for col in ["gg_data", "cg_data", "tot_ls_data", "ngdp_data"]:
        gdd[col] = pd.to_numeric(gdd[col], errors="coerce")
    return gmd, gdd


def positive_complete(frame: pd.DataFrame) -> pd.Series:
    return (np.isfinite(frame) & frame.gt(0)).all(axis=1)


def balance_usd(
    local: pd.DataFrame, fx: pd.DataFrame, exclusions: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop incomplete/uncertain conversions without imposing a size cutoff."""
    fx = fx.reindex(index=local.index, columns=local.columns)
    stock_ok, fx_ok = positive_complete(local), positive_complete(fx)
    uncertain = pd.Series(local.index.isin(exclusions), index=local.index)
    keep = stock_ok & fx_ok & ~uncertain
    audit = pd.DataFrame({"positive_complete_local_stock": stock_ok,
                          "positive_complete_fx": fx_ok,
                          "currency_units_excluded": uncertain,
                          "retained": keep}, index=local.index)
    audit["currency_reason"] = pd.Series(exclusions).reindex(local.index).fillna("")
    # Input local stocks are billions of the provider's currency unit; output
    # is USD billions. A constant common multiplier does not affect shares.
    return local.loc[keep].div(fx.loc[keep]), audit


def load_debt(private: bool) -> Application:
    """Annual 1971–2024 growth, with fixed 1970 USD-debt weights."""
    gmd, gdd = read_inputs(include_gdd=private)
    years = list(range(1970, LAST + 1))
    currency = gmd.assign(fx=gmd.USDfx.where(gmd.forecast_USDfx.eq(0)))
    fx = currency.pivot(index="ISO3", columns="year", values="fx").reindex(columns=years)
    if private:
        d = gdd[gdd.year.between(1970, LAST)].copy()
        ratio = d.pivot(index="ISO3", columns="year", values="tot_ls_data").reindex(columns=years)
        gdp = d.pivot(index="ISO3", columns="year", values="ngdp_data").reindex(columns=years)
        # GDD ratios are percentages and GDP is billions of national currency.
        local = ratio * gdp / 100.
        labels = d[["ISO3", "country"]].drop_duplicates().set_index("ISO3").country
        exclusions = GDD_CURRENCY_EXCLUSIONS
        definition = "Private nonfinancial, loans and debt securities"
        source = "IMF GDD workbook: debt and own nominal GDP; GMD 2026_09 annual-average FX only"
        entity, outcome = "Private nonfinancial debt by country", "Private nonfinancial-debt growth"
        audit_name = "gdd_private_currency_audit.csv"
    else:
        observed = gmd.forecast_gen_govdebt.eq(0) & gmd.forecast_nGDP.eq(0)
        d = gmd.assign(stock=gmd.gen_govdebt.where(observed) / 1000.)
        local = d.pivot(index="ISO3", columns="year", values="stock").reindex(columns=years)
        labels = d[["ISO3", "countryname"]].drop_duplicates().set_index("ISO3").countryname
        exclusions = GMD_CURRENCY_EXCLUSIONS
        definition = "General government"
        source = "Global Macro Database 2026_09: published debt, GDP and annual-average FX"
        entity, outcome = "Public debt by country", "General-government debt growth"
        audit_name = "gmd_gg_currency_audit.csv"
    usd, audit = balance_usd(local, fx, exclusions)
    usd = usd.sort_index()
    if len(usd) < 2:
        raise ValueError(f"Too few USD-convertible countries: {entity}")
    y = np.log(usd).diff(axis=1).iloc[:, 1:]
    app = Application(
        entity, outcome, source, definition, y, usd[1970], labels,
        "1971–2024 growth; 1970 initial stock and size weights", "Annual", "1970",
        "Positive complete USD stocks; fixed initial USD-debt weights. Annual log differences, "
        "pooled 1/99 winsorization and grand-mean centering through the existing Application constructor. "
        "No interpolation, size exclusion, currency-factor guessing, or within-window government-perimeter switches. "
        "FX is annual-average, so converted stocks are not end-year-FX valuations.",
    )
    APPLICATION_CACHE.mkdir(parents=True, exist_ok=True)
    audit.to_csv(APPLICATION_CACHE / audit_name)
    return app


def load_public_debt() -> Application:
    return load_debt(private=False)


def load_private_debt() -> Application:
    return load_debt(private=True)
