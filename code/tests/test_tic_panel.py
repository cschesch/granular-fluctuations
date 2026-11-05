"""TIC keeps countries and separate organization groups without overlapping totals."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications import common
from applications.treasury import panel as finance


class TicPanelTests(unittest.TestCase):
    def load_fixture(self, missing_org=False):
        periods = pd.period_range("2011-12", "2025-12", freq="M")
        records = []
        for k, period in enumerate(periods):
            # Incomplete countries are excluded, without adding a residual.
            for name, code, value in (
                ("Austria", "10189", 100 + k),
                ("Belgium", "10251", 20 if k == 0 else np.nan),
                ("Cyprus", "10405", 0),
                ("Memo: Euro Area", "16713", 150 + k),
                ("Total Europe", "19992", 200 + k),
                ("All Countries", "69995", 250 + k),
                ("International", "72907", np.nan if missing_org and k == 1 else 30),
                ("Total Regional Orgs.", "76929", 10),
                ("Total IROs", "79995", 40),
                ("Grand Total", "99996", 290 + k),
            ):
                records.append([name, code, str(period), value])
        data = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as directory:
            history, current = Path(directory)/"old.csv", Path(directory)/"new.txt"
            history.write_text("\n"*12 + data.loc[data[2].eq("2011-12")].to_csv(index=False, header=False))
            current.write_text("\n"*9 + data.to_csv(sep="\t", index=False, header=False))
            with patch.object(finance, "TIC_HISTORY", history), patch.object(
                finance, "TIC_CURRENT", current
            ), patch.object(common, "standardize_panel", lambda frame: frame):
                return finance.load_tic()

    def test_no_overlapping_totals_or_synthetic_residual(self):
        app = self.load_fixture()
        self.assertEqual(list(app.y.index), ["10189", "72907", "76929"])
        self.assertEqual(app.y.shape, (3, 168))
        np.testing.assert_array_equal(app.sizes, [100, 30, 10])
        reconstructed = np.exp(app.y.cumsum(axis=1)).mul(app.sizes, axis=0)
        np.testing.assert_allclose(reconstructed.sum(axis=0), 140 + np.arange(1, 169))
        np.testing.assert_allclose(app.y.loc["10189"], np.diff(np.log(100 + np.arange(169))))
        np.testing.assert_array_equal(app.y.loc[["72907", "76929"]], 0)
        self.assertEqual(app.dependence_horizon, 1)
        self.assertEqual(app.weight_date, "2011M12")

    def test_organizations_obey_the_same_balanced_positive_rule(self):
        app = self.load_fixture(missing_org=True)
        self.assertEqual(list(app.y.index), ["10189", "76929"])


if __name__ == "__main__":
    unittest.main()
