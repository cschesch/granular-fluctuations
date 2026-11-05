"""CAMPD start-date, lag alignment, and archived-source preservation checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.additional import common, download, production


class CampdPanelTests(unittest.TestCase):
    def test_growth_window_weights_and_balancing(self):
        periods = pd.period_range("1999Q4", "2024Q4", freq="Q")
        records = []
        for facility, base, slope in (("a", 100, 1), ("b", 500, 2),
                                      ("late", 300, 1), ("zero", 300, 1)):
            for k, period in enumerate(periods):
                if facility == "late" and period < pd.Period("2001Q1"):
                    continue
                value = 0 if facility == "zero" and k == 0 else base + slope*k
                records.append({"Facility ID": facility, "Facility Name": facility,
                                "Year": period.year, "Quarter": period.quarter,
                                "Gross Load (MWh)": value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"campd.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            # Isolate the timing construction from the unchanged pooled
            # winsorization policy, which has its own core invariant checks.
            with patch.object(production, "CAMPD", path), patch.object(
                common, "standardize_panel", lambda frame: frame
            ):
                app = production.load_campd("Gross Load (MWh)", "Power plants", "Gross-load growth")
        self.assertEqual(list(app.y.index), ["a", "b"])
        self.assertEqual(app.y.shape, (2, 100))
        self.assertEqual(str(app.y.columns[0]), "2000Q1")
        self.assertEqual(str(app.y.columns[-1]), "2024Q4")
        self.assertEqual(app.weight_date, "1999Q4")
        self.assertEqual(app.dependence_horizon, 1)
        np.testing.assert_array_equal(app.sizes, [100, 500])
        k = np.arange(1, 101)
        for facility, base, slope in (("a", 100, 1), ("b", 500, 2)):
            expected = pd.Series(np.log((base+slope*k)/(base+slope*(k-1))), index=periods[1:])
            expected -= expected.groupby(expected.index.quarter).transform("mean")
            np.testing.assert_allclose(app.y.loc[facility], expected, atol=1e-15)
        np.testing.assert_allclose(app.y.T.groupby(app.y.columns.quarter).mean(), 0.0, atol=1e-15)
        self.assertIn("all 101 quarters", app.notes)

    def test_extension_preserves_archive_and_reuses_download(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            legacy = raw/"epa_campd_facility_quarterly_2000_2024.csv"
            target = raw/"epa_campd_facility_quarterly_1999q4_2024.csv"
            old = pd.DataFrame({"Facility ID": ["a"]*100,
                                "Year": np.repeat(np.arange(2000, 2025), 4),
                                "Quarter": np.tile(np.arange(1, 5), 25),
                                "Gross Load (MWh)": np.arange(100)+100.0})
            old.to_csv(legacy, index=False)
            original = legacy.read_bytes()

            def fake_fetch(url, destination, **kwargs):
                self.assertIn("year=1999", url)
                self.assertIn("quarter=4", url)
                pd.DataFrame({"Facility ID": ["a"], "Year": [1999],
                              "Quarter": [4], "Gross Load (MWh)": [99.]}).to_csv(
                                  destination, index=False)

            with patch.object(download, "RAW", raw), patch.object(download, "CAMPD", target), \
                    patch.object(download, "fetch", side_effect=fake_fetch) as fetch:
                download.download_campd_data()
                download.download_campd_data()
                fetch.assert_called_once()
            self.assertEqual(legacy.read_bytes(), original)
            combined = pd.read_csv(target)
            self.assertEqual(len(combined), 101)
            pd.testing.assert_frame_equal(combined.loc[combined.Year.ge(2000)].reset_index(drop=True), old)


if __name__ == "__main__":
    unittest.main()
