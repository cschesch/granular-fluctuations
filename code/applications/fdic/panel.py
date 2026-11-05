"""Bank lending from FDIC quarterly financials."""

import numpy as np
import pandas as pd
from mgp_core import remove_periodic_means
from project_paths import ROOT
from applications.common import Application


FDIC = ROOT / "data/fdic/fdic_quarterly_financials_1999q4_2025q4.csv"


def load_banks() -> Application:
    d = pd.read_csv(FDIC, dtype={"CERT": str, "REPDTE": str})
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["REPDTE"]), freq="Q")
    loans = d.pivot(index="CERT", columns="period", values="LNLSNET")
    periods = pd.period_range("1999Q4", "2025Q4", freq="Q")
    missing = periods.difference(loans.columns)
    if len(missing):
        raise ValueError(f"Missing FDIC quarters: {list(missing)}; refresh the public-data input")
    loans = loans.reindex(columns=periods).apply(pd.to_numeric, errors="coerce")
    loans = loans.loc[(loans > 0).all(axis=1)]
    # Match quarterly Compustat sales: QoQ log growth, then residualize each
    # bank on its own quarter-of-year fixed effects before pooled winsorization.
    y = np.log(loans).diff(axis=1).iloc[:, 1:]
    y = remove_periodic_means(y, 4)
    labels = d.sort_values("period").drop_duplicates("CERT", keep="last").set_index("CERT")["NAME"]
    return Application(
        "Banks", "Net loans and leases growth", "FDIC BankFind Suite quarterly financials",
        "Buch & Neugebauer (2011) and Blanco-Arroyo & Alfarano (2022).",
        y, loans.loc[y.index, pd.Period("1999Q4")], labels, "1999Q4–2025Q4 levels; quarter-on-quarter growth 2000Q1–2025Q4", "Quarterly", "1999Q4",
        "Balanced FDIC-insured-bank panel; quarter-on-quarter log growth with bank-specific quarter-of-year fixed effects removed before pooled 1/99 winsorization. Mergers and exits imply survivor selection; fixed 1999Q4 net-loan weights.",
    )
