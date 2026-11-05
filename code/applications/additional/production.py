"""Electricity, emissions, and oil production panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import remove_periodic_means
from wrds_inputs import ANALYSIS_LAST_QUARTER

from .common import RAW, Application, balanced_log_growth


EIA = RAW / "eia_annual_plant_generation_2001_2024.csv"
CAMPD = RAW / "epa_campd_facility_quarterly_1999q4_2024.csv"
OWID_ENERGY = RAW / "owid_energy_data.csv"
OWID_CO2 = RAW / "owid_annual_co2_emissions_per_country.csv"


def load_electricity() -> Application:
    d = pd.read_csv(EIA, dtype={"plantCode": str})
    d["generation"] = pd.to_numeric(d["generation"], errors="coerce")
    levels = d.pivot_table(index="plantCode", columns="period", values="generation", aggfunc="sum")
    y = balanced_log_growth(levels, 2001, 2024)
    labels = d.drop_duplicates("plantCode").set_index("plantCode")["plantName"]
    return Application("Electricity plants", "Net generation growth", "EIA-923 via EIA Open Data API",
        "Plant-level electricity provides a high-frequency physical-output analogue to firm sales.", y,
        levels.loc[y.index, 2001], labels, "2001–2024 levels; 2002–2024 growth", "Annual", "2001",
        "Balanced plants with positive annual net generation; fixed 2001 generation weights.")


def load_campd(column: str, entity: str, outcome: str) -> Application:
    d = pd.read_csv(CAMPD, dtype={"Facility ID": str})
    d["period"] = pd.PeriodIndex.from_fields(year=d["Year"], quarter=d["Quarter"], freq="Q")
    levels = d.pivot(index="Facility ID", columns="period", values=column)
    last_period = min(pd.Period(ANALYSIS_LAST_QUARTER, freq="Q"), d["period"].max())
    periods = pd.period_range("1999Q4", last_period, freq="Q")
    levels = levels.reindex(columns=periods).apply(pd.to_numeric, errors="coerce")
    levels = levels.loc[(levels > 0).all(axis=1)]
    y = np.log(levels).diff(axis=1).iloc[:, 1:]
    y = remove_periodic_means(y, 4)
    labels = d.drop_duplicates("Facility ID").set_index("Facility ID")["Facility Name"]
    return Application(entity, outcome, "EPA CAMPD apportioned quarterly emissions by facility",
        "CAMPD continuously monitored power-sector operations and emissions, aggregated to facilities.",
        y, levels.loc[y.index, periods[0]], labels,
        f"1999Q4–{last_period} levels; quarter-on-quarter growth 2000Q1–{last_period}",
        "Quarterly", "1999Q4",
        f"Balanced CAMPD facilities with positive {column.lower()} in all {len(periods)} quarters; fixed 1999Q4 weights. "
        "Quarter-on-quarter log growth with plant-specific quarter-of-year fixed effects removed before pooled 1/99 winsorization; "
        "CAMPD covers regulated generating units rather than every US generator.")


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
