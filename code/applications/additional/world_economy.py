"""Maddison country GDP and Kenneth French country equity returns."""

from __future__ import annotations

import pandas as pd

from .common import DERIVED, RAW, Application, balanced_log_growth


MADDISON_GDP = RAW / "maddison_project_gdp_2023_owid.csv"
FRENCH_COUNTRY_RETURNS = DERIVED / "french_country_market_returns_monthly_1975_2025.csv"
FRENCH_COUNTRY_WEIGHTS = DERIVED / "world_bank_country_market_cap_weights_1975.csv"


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


def load_country_equity_returns() -> Application:
    returns = pd.read_csv(FRENCH_COUNTRY_RETURNS, index_col="date", parse_dates=True).loc["1976-01-01":].T
    weights = pd.read_csv(FRENCH_COUNTRY_WEIGHTS).set_index("country")
    weights = weights.loc[returns.index]
    return Application(
        "Country equity markets", "Country stock-market return",
        "Kenneth French Data Library and World Bank WDI market capitalization",
        "A fixed-weight synthetic global-equity return across developed markets with 1975 capitalization data.",
        returns, weights["market_cap_current_usd"], pd.Series(returns.index, index=returns.index),
        "1976M1--2025M12", "Monthly", "1975",
        "French country market returns in US dollars; the US return is Mkt-RF plus RF. "
        "Fixed end-1975 World Bank market-capitalization weights; 12 markets have both observed capitalization in that vintage and complete returns from January 1976.",
    )
