"""Table identities and source markers remain aligned with application figures."""

import unittest
from unittest.mock import patch

import pandas as pd

from applications.figure_layout import HIERARCHY
import build_applications_vertical as vertical
import build_homoskedastic_applications_table as homoskedastic


class ApplicationTableTests(unittest.TestCase):
    def test_all_table_applications_join_one_figure_item(self):
        items = [homoskedastic.figure_item(application)
                 for _, applications in vertical.SECTIONS for application in applications
                 if application not in homoskedastic.ANNUAL_COMPUSTAT]
        expected = [item for _, items in HIERARCHY for item in items]
        self.assertEqual(set(items), set(expected))
        self.assertEqual(len(items), len(set(items)))

    def test_reordered_sections_keep_application_identity(self):
        pairs = {
            "Metro GDP": "GDP by metro", "County GDP": "GDP by county",
            "State GDP": "GDP by state", "Oil production": "Oil production by country",
            "CO2 emissions by country": "CO2 emissions by country",
            "World GDP": "GDP (PPP) by country", "Country market cap": "Mkt cap by country",
            "Public debt": "Public debt by country", "Private debt": "Private debt by country",
            "Carvalho--Gabaix (2013)": "Carvalho–Gabaix (2013)",
            "Foerster et al. (2011)": "Foerster et al. (2011)",
            "Stella (2015)": "Stella (2015)",
            "Welburn et al. (2020)": "Welburn et al. (2020)",
            "Herskovic et al. (2020)": "Herskovic et al. (2020)",
        }
        self.assertEqual({application: homoskedastic.figure_item(application) for application in pairs}, pairs)

    def test_network_asterisks_match_current_figure_definitions(self):
        with patch.object(vertical.pd, "read_csv", return_value=pd.DataFrame({"item": []})), \
                patch.object(vertical, "summary_path", return_value=None):
            rows = vertical.network_rows()
        starred = {application for application, row in rows.items() if row["source"].endswith("*")}
        self.assertEqual(starred, {"Stella (2015)", "Herskovic et al. (2020)", "Welburn et al. (2020)"})


if __name__ == "__main__":
    unittest.main()
