"""Power-plant generation and emissions from EPA CAMPD."""

import numpy as np
import pandas as pd
from mgp_core import remove_periodic_means
from wrds_inputs import ANALYSIS_LAST_QUARTER
from project_paths import ROOT
from applications.common import Application


CAMPD = ROOT / "data/epa/epa_campd_facility_quarterly_1999q4_2024.csv"


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
