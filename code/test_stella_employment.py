#!/usr/bin/env python3
"""Source-rule checks for Stella's missing-employment reconstruction.

These fixtures exercise operations specified in data_building_balanced.do,
including the distinction between means over joint years and annual ratios.
They do not select employment levels to reproduce published paper moments.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.stella.prepare_stella_employment import reconstruct_employment_ratios, SECTORAL


@contextmanager
def annual_source(rows):
    records = [{"gvkey": str(gvkey), "fyear": year, "emp": emp, "fic": "USA", "curcd": "USD",
                "indfmt": "INDL", "datafmt": "STD", "consol": "C"} for gvkey, year, emp in rows]
    with tempfile.TemporaryDirectory(prefix="stella_employment_test_") as folder:
        path = Path(folder) / "annual.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        with patch("wrds_inputs.wrds_path", return_value=path):
            yield


class StellaEmploymentSourceRules(unittest.TestCase):
    def test_joint_year_means_and_zero_missing_exclusion(self):
        mapping = pd.DataFrame({"gvkey": ["1"], "stella_sid": ["1014"]})
        sectoral = pd.DataFrame({"year": list(range(1990, 1996)), "stella_sid": ["1014"] * 6,
                                 "sectoral_emp": [100, 200, 300, 400, 0, np.nan]})
        with annual_source([(1, 1990, 20), (1, 1991, 80), (1, 1992, 0),
                            (1, 1993, np.nan), (1, 1994, 90), (1, 1995, 100)]):
            result, _ = reconstruct_employment_ratios(mapping, sectoral)
        firm = result.loc["000001"]
        # Only 1990 and 1991 are joint, nonzero observations. Source collapse
        # gives 50/150=1/3, while averaging annual ratios would give 0.3.
        self.assertEqual(firm.n_joint_years, 2)
        self.assertAlmostEqual(firm.mean_firm_emp, 50)
        self.assertAlmostEqual(firm.mean_sectoral_emp, 150)
        self.assertAlmostEqual(firm.ratio_emp, 1 / 3)
        self.assertNotAlmostEqual(firm.ratio_emp, 0.3)
        self.assertEqual((firm.first_year, firm.last_year), (1990, 1991))

    def test_only_three_deposited_special_sector_overrides(self):
        sids = ["1013", "1034", "1048", "1050"]
        mapping = pd.DataFrame([{"gvkey": str(i * 2 + j + 1), "stella_sid": sid}
                                for i, sid in enumerate(sids) for j in range(2)])
        rows = [(i * 2 + j + 1, 1990, [20, 60][j]) for i in range(4) for j in range(2)]
        sectoral = pd.DataFrame({"year": [1990] * 4, "stella_sid": sids, "sectoral_emp": [10] * 4})
        with annual_source(rows):
            result, _ = reconstruct_employment_ratios(mapping, sectoral)
        for sid in sids[:3]:
            selected = result.loc[result.stella_sid.eq(sid)]
            np.testing.assert_allclose(selected.ratio_emp, [0.25, 0.75])
            self.assertAlmostEqual(selected.ratio_emp.sum(), 1)
            self.assertTrue(selected.source_special_sector_override.all())
        # Aerospace is deliberately above one: the source never adds a fourth
        # normalization merely because current data show another exceedance.
        selected = result.loc[result.stella_sid.eq("1050")]
        np.testing.assert_allclose(selected.ratio_emp, [2, 6])
        self.assertFalse(selected.source_special_sector_override.any())

    def test_strict_missing_denominator_does_not_use_scope_proxy(self):
        mapping = pd.DataFrame({"gvkey": ["1", "2"], "stella_sid": ["1014", "1001"]})
        sectoral = pd.DataFrame({"year": [1990, 1990], "stella_sid": ["1014", "1001"],
                                 "sectoral_emp": [100, 200], "strict_concordance_sectoral_emp": [100, np.nan]})
        with annual_source([(1, 1990, 20), (2, 1990, 30)]):
            public, _ = reconstruct_employment_ratios(mapping, sectoral, "public_scope")
            strict, _ = reconstruct_employment_ratios(mapping, sectoral, "strict_concordance")
        self.assertEqual(public.ratio_emp.notna().sum(), 2)
        self.assertEqual(strict.ratio_emp.notna().sum(), 1)
        self.assertTrue(np.isnan(strict.loc["000002", "ratio_emp"]))
        self.assertAlmostEqual(public.loc["000001", "ratio_emp"], strict.loc["000001", "ratio_emp"])

    def test_negative_nonzero_observations_follow_literal_source_rule(self):
        mapping = pd.DataFrame({"gvkey": ["1", "2"], "stella_sid": ["1014", "1015"]})
        sectoral = pd.DataFrame({"year": [1990, 1990], "stella_sid": ["1014", "1015"], "sectoral_emp": [100, -100]})
        with annual_source([(1, 1990, -20), (2, 1990, 30)]):
            result, _ = reconstruct_employment_ratios(mapping, sectoral)
        self.assertAlmostEqual(result.loc["000001", "ratio_emp"], -0.2)
        self.assertAlmostEqual(result.loc["000002", "ratio_emp"], -0.3)
        self.assertTrue(result.n_joint_years.eq(1).all())


if __name__ == "__main__":
    unittest.main()
