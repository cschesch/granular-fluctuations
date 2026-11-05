"""Synthetic checks for Stella's dated ranking and growth accounting."""
import numpy as np
import pandas as pd
from applications.stella.prepare_stella_calibration import released_screen_panel, screen_history, top500_membership
from applications.stella.audit_stella_growth_moments import growth_accounting


def test_missing_lag_ranks_and_growth_accounting():
    # Real source behavior: missing lag sales cannot occupy a top-500 rank.
    dates = pd.period_range("1962Q1", periods=41, freq="Q")
    synthetic = pd.DataFrame({
        "gvkey": np.repeat([f"{i:06d}" for i in range(1, 502)], len(dates)),
        "quarter": np.tile(dates, 501),
        "saleq": np.repeat(np.arange(1, 502), len(dates)),
    })
    strict, strict_n = released_screen_panel(synthetic)
    assert strict_n == 500
    assert "000001" not in set(strict["gvkey"])
    membership = top500_membership(screen_history(synthetic))
    assert set(membership["gvkey"]) == set(strict["gvkey"])
    assert dates[0] not in set(membership["quarter"])
    account = growth_accounting(pd.DataFrame([[100., 110., np.nan], [np.nan, 20., 25.],
                                             [50., 50., 50.]]))
    np.testing.assert_allclose(account.iloc[0], [20., 100*10/150, 100*20/150, 0.])
    np.testing.assert_allclose(account.iloc[1], [-100*105/180, 100*5/180, 0., -100*110/180])
