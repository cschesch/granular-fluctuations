"""Metropolitan GDP from the Granular Cities publication data."""

import pandas as pd
from project_paths import ROOT
from applications.common import Application, balanced_log_growth


FIGSHARE_CITIES = ROOT / "data/granular_cities/DatasetsGranularCities.xlsx"


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
