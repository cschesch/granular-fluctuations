"""Country equity returns with fixed World Bank market-cap weights."""

import pandas as pd
from project_paths import DERIVED_CACHE
from applications.common import Application


FRENCH_COUNTRY_RETURNS = DERIVED_CACHE / "french/french_country_market_returns_monthly_1975_2025.csv"
FRENCH_COUNTRY_WEIGHTS = DERIVED_CACHE / "french/world_bank_country_market_cap_weights_1975.csv"


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
