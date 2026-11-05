"""Foreign holdings of US Treasury securities."""

import numpy as np
import pandas as pd
from project_paths import ROOT
from applications.common import Application


TIC_CURRENT = ROOT / "data/treasury/tic_slt_table3_2020_current.txt"
TIC_HISTORY = ROOT / "data/treasury/tic_slt3d_2011_2019.csv"


def load_tic() -> Application:
    columns = ["country", "code", "date", "holdings"]
    current = pd.read_csv(
        TIC_CURRENT, sep="\t", skiprows=9, dtype=str,
        names=[*columns, "net", "lt", "ltnet", "val", "st", "stnet"],
    )
    historical = pd.read_csv(
        TIC_HISTORY, skiprows=12, names=[*columns, "lt", "st"],
        dtype=str, engine="python", on_bad_lines="skip",
    )
    d = pd.concat([historical[columns], current[columns]], ignore_index=True)
    d["code_num"] = pd.to_numeric(d.code, errors="coerce")
    d["holdings"] = pd.to_numeric(d.holdings.str.replace(",", ""), errors="coerce")
    d = d[d.code_num.notna() & d.date.str.match(r"^20\d\d-\d\d$", na=False)]
    d["period"] = pd.PeriodIndex(d.date, freq="M")
    periods = pd.period_range("2011-12", "2025-12", freq="M")
    d = d[d.period.between(periods[0], periods[-1])]
    levels = d.sort_values("period").drop_duplicates(["code", "period"], keep="last")
    levels = levels.pivot(index="code", columns="period", values="holdings").reindex(columns=periods)
    labels = d.drop_duplicates("code").set_index("code")["country"]
    # TIC geographic totals and memo groups also have codes below 70000.
    country = (levels.index.astype(int) < 70000) & ~labels.reindex(levels.index).str.match(
        r"^(?:Total |Memo:|All |Other |.*Oil Exporters$)"
    )
    world = levels.loc["99996"]
    # Retain the two disjoint published organization groups separately,
    # excluding their combined total (79995). Do not construct a residual.
    organization = levels.index.isin(["72907", "76929"])
    levels = levels.loc[(country | organization) & levels.gt(0).all(axis=1)]
    n_countries = int((levels.index.astype(int) < 70000).sum())
    coverage = levels.sum(axis=0) / world
    # Monthly holdings changes: no overlapping annual differences and no
    # seasonal adjustment. Application retains the existing pooled winsorization.
    y = np.log(levels).diff(axis=1).iloc[:, 1:]
    return Application("Foreign Treasury holders", "US Treasury-holdings growth", "US Treasury International Capital, Table 3",
        "Non-overlapping country/territory holdings and separately reported organization groups.", y, levels.loc[y.index, periods[0]], labels,
        "2011M12–2025M12 levels; month-on-month growth 2012M1–2025M12", "Monthly", "2011M12",
        f"{n_countries} balanced positive country/territory series and {len(levels)-n_countries} separately reported organization groups "
        "(International and Total Regional Orgs.). Geographic totals, memo groups, and Total IROs are excluded; no residual is added. "
        f"Retained holdings cover {coverage.min():.2%}–{coverage.max():.2%} of TIC Grand Total across the sample. "
        "Month-on-month log growth without seasonal adjustment, with pooled 1/99 winsorization and fixed December 2011 holdings weights. "
        "Fixed-share winsorized growth is an approximation to world holdings growth. Holdings are custodial and may not identify ultimate owners.")
