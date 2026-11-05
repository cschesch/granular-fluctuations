"""Country oil production from Our World in Data."""

import pandas as pd
from project_paths import ROOT
from applications.common import Application, balanced_log_growth


OWID_ENERGY = ROOT / "data/owid_energy/owid-energy-data.csv"


def load_oil() -> Application:
    d = pd.read_csv(OWID_ENERGY, usecols=["country", "iso_code", "year", "oil_production"])
    d = d[d.iso_code.str.len().eq(3) & ~d.iso_code.str.startswith("OWID", na=False)]
    levels = d.pivot(index="iso_code", columns="year", values="oil_production")
    y = balanced_log_growth(levels, 1965, 2024)
    labels = d.drop_duplicates("iso_code").set_index("iso_code")["country"]
    return Application("Oil-producing countries", "Oil-production growth", "Energy Institute Statistical Review via OWID",
        "Country production of an additive globally traded physical commodity.", y, levels.loc[y.index, 1965], labels,
        "1965–2024 levels; 1966–2024 growth", "Annual", "1965",
        "Balanced countries with positive oil production; fixed 1965 production weights. Oil production is reported in TWh.")
