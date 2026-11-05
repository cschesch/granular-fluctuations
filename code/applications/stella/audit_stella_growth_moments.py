"""Reconstruct the growth moments in Stella's Table 1 and Figure 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from applications.panels import load_gdp_deflator
from applications.stella.prepare_public import GDP_GROWTH


def growth_accounting(levels: pd.DataFrame) -> pd.DataFrame:
    previous = levels.iloc[:, :-1].to_numpy(float)
    current = levels.iloc[:, 1:].to_numpy(float)
    both = np.isfinite(previous) & np.isfinite(current)
    entering = ~np.isfinite(previous) & np.isfinite(current)
    exiting = np.isfinite(previous) & ~np.isfinite(current)
    denominator = np.nansum(previous, axis=0)
    total = 100 * (np.nansum(current, axis=0) - denominator) / denominator
    continuing = 100 * np.where(both, current - previous, 0).sum(axis=0) / denominator
    entrants = 100 * np.where(entering, current, 0).sum(axis=0) / denominator
    exits = -100 * np.where(exiting, previous, 0).sum(axis=0) / denominator
    np.testing.assert_allclose(total, continuing + entrants + exits, rtol=0, atol=1e-10)
    return pd.DataFrame({
        "aggregate_growth_pct": total,
        "continuing_contribution_pp": continuing,
        "entry_contribution_pp": entrants,
        "exit_contribution_pp": exits,
    }, index=levels.columns[1:])


def growth_moments(directory: Path) -> dict[str, float]:
    nominal = pd.read_csv(directory / "nominal_saleq_sax12_adjusted.csv", index_col=0)
    nominal.columns = pd.PeriodIndex(nominal.columns, freq="Q")
    price = load_gdp_deflator("quarterly").reindex(nominal.columns)
    if price.isna().any():
        raise ValueError("Missing GDP-deflator observation")
    real = nominal.div(price, axis=1) * 100
    moments = growth_accounting(real).std(ddof=1).to_dict()

    index = pd.read_csv(directory / "top500_index.csv", index_col=0)
    index.columns = pd.PeriodIndex(index.columns, freq="Q")
    full_index = pd.DataFrame(0, index=nominal.index, columns=nominal.columns)
    full_index.loc[:, index.columns] = index
    # The first-quarter population is below 500. The deposited growth index starts in Q2.
    first = nominal.iloc[:, 0].notna()
    assert int(first.sum()) <= 500
    full_index.iloc[:, 0] = first.astype(int)

    gdp = pd.read_csv(GDP_GROWTH, index_col="quarter")["real_gdp_log_growth_deposited"]
    gdp.index = pd.PeriodIndex(gdp.index, freq="Q")
    # The deposited script uses nominal sales and the article describes real sales.
    for label, levels in (("nominal", nominal), ("real", real)):
        aggregate = levels.where(full_index.eq(1)).sum(axis=0)
        moments[f"top500_{label}_sales_gdp_correlation"] = np.log(aggregate).diff().corr(gdp)
    return moments
