"""Country GDP from the Maddison Project Database."""

import pandas as pd
from project_paths import ROOT
from applications.common import Application, balanced_log_growth


MADDISON_GDP = ROOT / "data/maddison/maddison_project_gdp_2023_owid.csv"


def load_countries() -> Application:
    frame = pd.read_csv(MADDISON_GDP)
    frame = frame[frame["Code"].fillna("").str.len().eq(3)]
    levels = frame.pivot(index="Code", columns="Year", values="GDP")
    y = balanced_log_growth(levels, 1950, 2022)
    names = frame.drop_duplicates("Code").set_index("Code")["Entity"]
    return Application(
        "Countries", "Real GDP growth", "Maddison Project Database 2023 via Our World in Data",
        "Closest literature benchmark: cross-country granular-residual work (Ebeke & Eklou, 2017); this panel instead treats countries as grains of world output.",
        y, levels.loc[y.index, 1950], names, "1950–2022 levels; 1951–2022 growth", "Annual", "1950",
        "Balanced country panel; GDP in constant 2011 international dollars and fixed 1950 weights.",
    )
