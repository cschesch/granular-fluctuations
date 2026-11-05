"""GDP panels for US metros, states, and counties."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .common import RAW, Application, balanced_log_growth


FIGSHARE_CITIES = RAW / "DatasetsGranularCities.xlsx"
BEA_STATES = RAW / "SAGDP.zip"
BEA_COUNTIES = RAW / "CAGDP1.zip"


def load_cities() -> Application:
    gdp = pd.read_excel(FIGSHARE_CITIES, sheet_name="US MR gdp")
    gdp = gdp.set_index(gdp.columns[0]).rename_axis(None).drop(columns=["average"], errors="ignore")
    gdp.columns = gdp.columns.astype(int)
    levels = gdp.apply(pd.to_numeric, errors="coerce")
    y = balanced_log_growth(levels, 2002, 2019)
    return Application(
        "Metropolitan areas", "GDP growth", "Esquierro & Da Silva Granular Cities Figshare package (BEA/Census inputs)",
        "Esquierro & Da Silva (2024) instead studies metropolitan GDP per capita using population weights.",
        y, levels.loc[y.index, 2002], pd.Series(y.index, index=y.index), "2002–2019 levels; 2003–2019 growth", "Annual", "2002",
        "All 357 U.S. metropolitan areas in the deposited balanced panel; fixed 2002 GDP weights.",
    )


def _bea_panel(path: Path, filename: str, geo_keep, first: int, last: int, entity: str) -> Application:
    with zipfile.ZipFile(path) as archive:
        d = pd.read_csv(archive.open(filename), encoding="latin1", dtype={"GeoFIPS": str})
    d["fips"] = d["GeoFIPS"].str.replace('"', '', regex=False).str.strip()
    d = d[d["fips"].map(geo_keep)]
    year_cols = [str(y) for y in range(first, last + 1)]
    real = d[d.LineCode.eq(1)].set_index("fips")[year_cols].apply(pd.to_numeric, errors="coerce")
    nominal = d[d.LineCode.eq(3)].set_index("fips")[year_cols].apply(pd.to_numeric, errors="coerce")
    real.columns = nominal.columns = range(first, last + 1)
    y = balanced_log_growth(real, first, last)
    names = d.drop_duplicates("fips").set_index("fips")["GeoName"].str.strip()
    return Application(entity, "Real GDP growth", f"BEA Regional Economic Accounts ({path.stem})",
        "Geographic granularity application using the canonical BEA regional accounts.", y, nominal.loc[y.index, first], names,
        f"{first}–{last} levels; {first+1}–{last} growth", "Annual", str(first),
        f"Balanced {entity.lower()} panel; fixed initial current-dollar GDP weights and chained-dollar GDP growth.")


def load_states() -> Application:
    states = {f"{x:02d}" for x in range(1, 57)} - {"03", "07", "14", "43", "52"}
    return _bea_panel(BEA_STATES, "SAGDP1__ALL_AREAS_1997_2025.csv",
                      lambda x: len(x) == 5 and x.endswith("000") and x[:2] in states, 1997, 2024, "US states")


def load_counties() -> Application:
    return _bea_panel(BEA_COUNTIES, "CAGDP1__ALL_AREAS_2001_2024.csv",
                      lambda x: len(x) == 5 and not x.endswith("000") and x != "00000", 2001, 2024, "US counties")
