"""Rank cache identity, failure reporting, and selective refreshes."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import check_factor_rank_figure1 as rank
from check_factor_rank_figure1 import panel_fingerprint


class RankPanelFingerprintTest(unittest.TestCase):
    def test_values_and_order_are_part_of_cache_identity(self):
        x = np.arange(12.0).reshape(4, 3)
        changed = x.copy()
        changed[0, 0] = 1
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(changed))
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(x[::-1]))
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(x.reshape(3, 4)))

    def test_array_storage_does_not_change_cache_identity(self):
        x = np.arange(12.0).reshape(4, 3)
        self.assertEqual(panel_fingerprint(x), panel_fingerprint(np.asfortranarray(x)))
        self.assertEqual(panel_fingerprint(x), panel_fingerprint(x.astype(int)))


class RankCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        paths = {
            "PANEL_CACHE": root / "panels",
            "RESULT_CACHE": root / "results",
            "SUMMARY_PKL": root / "summary.pkl",
            "SUMMARY_CSV": root / "summary.csv",
        }
        paths["PANEL_CACHE"].mkdir()
        paths["RESULT_CACHE"].mkdir()
        patches = patch.multiple(rank, **paths)
        patches.start()
        self.addCleanup(patches.stop)
        self.x = np.arange(60.0).reshape(20, 3)
        self.meta = dict(section="Test", item="Chosen", dataset="chosen", N=3, T=20, M=3)
        self.other = {**self.meta, "item": "Other", "dataset": "other"}
        pd.to_pickle({**self.meta, "x": self.x}, rank.PANEL_CACHE / "chosen.pkl")
        self.args = argparse.Namespace(
            only=["Chosen"], max_dimension=10, force=False, alpha=0.05,
            maxiter_ml=2000, maxiter_gmm=2000,
        )

    def test_matching_result_is_reused(self):
        pd.to_pickle(
            {**self.meta, "rank_test_status": "ok", "alpha": 0.05,
             "input_fingerprint": panel_fingerprint(self.x)},
            rank.RESULT_CACHE / "chosen.pkl",
        )
        with patch.object(rank, "factor_rank_tests") as fit:
            rank.run_tests([self.meta], self.args)
        fit.assert_not_called()

    def test_failure_is_saved_and_raised_without_losing_other_results(self):
        pd.to_pickle(
            {**self.other, "rank_test_status": "ok", "factor_lb_J": 1},
            rank.RESULT_CACHE / "other.pkl",
        )
        with patch.object(rank, "factor_rank_tests", side_effect=ValueError("bad panel")):
            with self.assertRaisesRegex(ValueError, "bad panel"):
                rank.run_tests([self.meta, self.other], self.args)
        saved = pd.read_pickle(rank.RESULT_CACHE / "chosen.pkl")
        self.assertEqual(saved["rank_test_status"], "error")
        self.assertIn("bad panel", saved["skip_reason"])
        summary = pd.read_pickle(rank.SUMMARY_PKL).set_index("item")
        self.assertEqual(summary.loc["Other", "factor_lb_J"], 1)

    def test_targeted_panel_refresh_retains_unrelated_metadata(self):
        pd.DataFrame([self.other]).to_pickle(rank.SUMMARY_PKL)
        with patch.object(rank, "_selected_builders", return_value=[]):
            manifest = rank.build_panel_cache(refresh=True, only=["Chosen"])
        self.assertEqual({row["item"] for row in manifest}, {"Chosen", "Other"})


if __name__ == "__main__":
    unittest.main()
