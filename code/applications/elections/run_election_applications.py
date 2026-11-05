#!/usr/bin/env python3
"""Granularity of Democratic U.S. presidential votes across states and counties."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import ANALYSIS_BOOTSTRAP_DRAWS, ROOT, choose_solver
from applications.additional.common import Application, estimate


DATA = ROOT / "data/elections/algara_amlani_presidential_county_1868_2020.csv"
CACHE = ROOT / "code/.cache/run_outputs/election_applications_summary.pkl"


def read_returns() -> pd.DataFrame:
    d = pd.read_csv(DATA, dtype={"fips": "string", "sfips": "string"})
    d["votes"] = pd.to_numeric(d["democratic_raw_votes"], errors="coerce")
    d = d.loc[d["complete_county_cases"].eq(1) & d["votes"].gt(0)].copy()
    # The release contains two otherwise identical Jones County, TX rows in
    # 1884 with slightly different all-candidate totals.  Keep its first row;
    # missing-FIPS historical counties must remain distinct.
    duplicate = d["fips"].notna() & d.duplicated(["fips", "election_year"], keep="first")
    return d.loc[~duplicate]


def balanced_growth(levels: pd.DataFrame, first: int, last: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(range(first, last + 1, 4))
    levels = levels.reindex(columns=years)
    levels = levels.loc[(levels > 0).all(axis=1)]
    return np.log(levels).diff(axis=1).iloc[:, 1:], levels


def state_application(d: pd.DataFrame) -> tuple[Application, pd.DataFrame]:
    levels = d.groupby(["state", "election_year"])["votes"].sum().unstack()
    y, levels = balanced_growth(levels, 1912, 2020)
    app = Application(
        "Democratic votes by state", "Growth in Democratic presidential votes",
        "Amlani and Algara (2021), county presidential returns",
        "County-level Democratic vote counts are aggregated to states.",
        y, levels[1912], pd.Series(y.index, index=y.index),
        "1912–2020 election levels; 1916–2020 growth", "Presidential elections (four-year frequency)", "1912",
        "Balanced 48-state panel, excluding Alaska, Hawaii, and DC to retain the full 1912–2020 window; fixed 1912 Democratic-vote weights.",
    )
    return app, levels


def county_application(d: pd.DataFrame) -> tuple[Application, pd.DataFrame]:
    levels = d.dropna(subset=["fips"]).pivot(index="fips", columns="election_year", values="votes")
    y, levels = balanced_growth(levels, 1900, 2020)
    names = d.drop_duplicates("fips").set_index("fips")
    names = names["county_name"] + ", " + names["state"]
    coverage = levels[2020].sum() / d.loc[d["election_year"].eq(2020), "votes"].sum()
    app = Application(
        "Democratic votes by county", "Growth in Democratic presidential votes",
        "Amlani and Algara (2021), county presidential returns",
        "We use the source's county-level Democratic presidential vote count.",
        y, levels[1900], names,
        "1900–2020 election levels; 1904–2020 growth", "Presidential elections (four-year frequency)", "1900",
        f"Balanced historical-FIPS panel; fixed 1900 Democratic-vote weights. Survivors account for {coverage:.1%} of recorded 2020 Democratic votes. "
        "County creations, dissolutions, and boundary changes make this a fixed historical cohort; Alaska is absent from the source.",
    )
    return app, levels


def main() -> None:
    d = read_returns()
    solver = choose_solver("auto")
    rows = []
    for app, _ in (state_application(d), county_application(d)):
        row, _ = estimate(app, solver, bootstrap_b=ANALYSIS_BOOTSTRAP_DRAWS, seed=20260908)
        rows.append(row)
        print(f"{app.entity}: N={row['N']}, T={row['T']}, M={row['M']}, "
              f"HHI={row['HHI']:.5f}, ceiling={row['MGP_share']:.4f}, "
              f"CI=[{row['CI90_low']:.4f}, {row['CI90_high']:.4f}]", flush=True)
    frame = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE)


if __name__ == "__main__":
    main()
