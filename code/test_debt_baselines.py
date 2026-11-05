"""Baseline debt definitions, timing, and USD weights."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from applications.additional import common, debt_panels


class DebtBaselineTests(unittest.TestCase):
    def test_annual_growth_uses_preceding_stock_and_source_scope(self):
        gmd, gdd = [], []
        for iso, scale in [('AAA', 1.), ('BBB', 2.)]:
            for year in range(1970, 2025):
                gmd.append(dict(ISO3=iso, countryname=iso, year=year, USDfx=2.,
                                forecast_USDfx=0, forecast_gen_govdebt=0, forecast_nGDP=0,
                                gen_govdebt=1000*scale*1.03**(year-1970)))
                gdd.append(dict(ISO3=iso, country=iso, year=year, gg_data=999., cg_data=888.,
                                tot_ls_data=50., ngdp_data=4*scale*1.02**(year-1970)))
        with patch.object(debt_panels, 'read_inputs', return_value=(pd.DataFrame(gmd), pd.DataFrame(gdd))), \
             patch.object(common, 'standardize_panel', side_effect=lambda y: y), \
             patch('pandas.DataFrame.to_csv'):
            public, private = debt_panels.load_public_debt(), debt_panels.load_private_debt()
        for app in (public, private):
            self.assertEqual(app.weight_date, '1970')
            self.assertEqual(app.frequency, 'Annual')
            self.assertEqual(list(app.y.columns), list(range(1971, 2025)))
        np.testing.assert_allclose(public.sizes, [.5, 1.])
        np.testing.assert_allclose(private.sizes, [1., 2.])
        np.testing.assert_allclose(public.y, np.log(1.03))
        np.testing.assert_allclose(private.y, np.log(1.02))
        self.assertIn('Private nonfinancial', private.entity)


if __name__ == '__main__':
    unittest.main()
