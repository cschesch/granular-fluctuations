"""Temporal and identity safeguards for the separate Exhibit 21 overlay."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from applications.welburn.welburn_subsidiary_matching import analyze_subsidiaries


class SubsidiaryMatchingTests(unittest.TestCase):
    def test_dates_parent_collisions_existing_ids_and_self_links(self):
        names = ["UNIQUE", "REMOVED", "STALE", "FUTURE", "AMBIG", "SELF", "UNIQUE"]
        base = pd.DataFrame({
            "gvkey": ["000001"] * 7,
            "srcdate": pd.to_datetime(["2015-06-01"] * 7),
            "customer_key": names, "usable_name": [True] * 7,
            "customer_gvkey": pd.Series([None] * 6 + ["000007"], dtype="string"),
        })
        rows = [
            (2, "000002", "2015-01-01", "UNIQUE"),
            (3, "000003", "2015-01-01", "REMOVED"),
            (3, "000003", "2015-03-01", "OTHER SUB"),
            (4, "000004", "2013-01-01", "STALE"),
            (5, "000005", "2016-01-01", "FUTURE"),
            (6, "000006", "2015-01-01", "AMBIG"),
            (99, "000099", "2015-01-01", "AMBIG"),
            (1, "000001", "2015-01-01", "SELF"),
        ]
        subs = pd.DataFrame(rows, columns=["cik", "gvkey", "fdate", "subsidiary_name"])
        subs["clean_company"] = subs["subsidiary_name"]
        ids = [f"{i:06d}" for i in range(1, 8)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subs.csv"
            subs.to_csv(path, index=False)
            report, result = analyze_subsidiaries(base, path, ids, temp, collision_path=path)
        self.assertEqual(result.loc[0, "customer_gvkey_after_subsidiaries"], "000002")
        self.assertTrue(result.loc[1:5, "customer_gvkey_after_subsidiaries"].isna().all())
        self.assertTrue(result.loc[4, "source2_ambiguous"])
        self.assertTrue(result.loc[5, "source2_self_link"])
        self.assertEqual(result.loc[6, "customer_gvkey_after_subsidiaries"], "000007")
        self.assertTrue(result.loc[6, "source2_conflict"])
        self.assertEqual(report["windows"]["annual_2015_2019"]["unique_added_edges"], 1)
        self.assertTrue(report["global_parent_collision_check"])


if __name__ == "__main__":
    unittest.main()
