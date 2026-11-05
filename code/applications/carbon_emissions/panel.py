"""Country carbon emissions from the Global Carbon Budget."""

import pandas as pd
from project_paths import ROOT
from applications.common import Application, balanced_log_growth


OWID_CO2 = ROOT / "data/carbon_emissions/owid_annual_co2_emissions_per_country.csv"


def load_country_co2() -> Application:
    d = pd.read_csv(OWID_CO2)
    d = d[d.Code.str.len().eq(3)]
    levels = d.pivot(index="Code", columns="Year", values="Annual CO₂ emissions")
    y = balanced_log_growth(levels, 1950, 2024)
    labels = d.drop_duplicates("Code").set_index("Code")["Entity"]
    return Application("Country CO2 emissions", "Territorial CO2-emissions growth", "Global Carbon Budget via Our World in Data",
        "Country decomposition of global fossil-fuel and cement CO2 emissions.", y, levels.loc[y.index, 1950], labels,
        "1950–2024 levels; 1951–2024 growth", "Annual", "1950",
        "Balanced countries with positive territorial CO2 emissions; fixed 1950 emissions weights.")
