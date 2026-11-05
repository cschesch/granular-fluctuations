"""Small numerical checks for the active Welburn network and aggregation."""
import unittest

import numpy as np
import pandas as pd
from scipy.special import logit

from applications.welburn.welburn_network import PROBABILITY_CUTOFF, calibrate_network, completed_edges, recover_shocks
from welburn_population_logit import PopulationLogitModel


class WelburnNetworkTests(unittest.TestCase):
    def test_literal_cutoff_preserves_observed_edges_and_excludes_boundary(self):
        intercepts = np.full((10, 10), logit(.00134))
        intercepts[0, 1] = logit(.00136)
        intercepts[1, 2] = logit(PROBABILITY_CUTOFF)
        ids = np.array(['000001', '000002', '000003'])
        model = PopulationLogitModel(np.zeros(3), intercepts, np.zeros(3),
            np.ones(3), ids, np.array([1., 2., 3.]), np.array([0, 1, 2]), {})
        known = pd.DataFrame({'gvkey': ['000003'], 'customer_gvkey': ['000001']})
        completed = completed_edges(model, known)
        self.assertEqual(set(completed[['gvkey', 'customer_gvkey']].itertuples(index=False, name=None)),
                         {('000003', '000001'), ('000001', '000002')})

    def test_flow_capacities_requirements_orientation_and_aggregate_recovery(self):
        ids = pd.Index(['000001', '000002', '000003'])
        means = pd.DataFrame({'sale': [100., 80., 60.], 'cogs': [40., 70., 60.]}, index=ids)
        edges = pd.DataFrame({'gvkey': ['000001', '000002'],
            'customer_gvkey': ['000002', '000003'], 'link_source': ['disclosed', 'inferred']})
        b, diagnostics = calibrate_network(ids, means, edges)
        sales = means.sale.to_numpy()
        flows = sales[:, None] * b
        a = flows / sales[None, :]
        np.testing.assert_allclose(a @ sales, flows.sum(axis=1))
        self.assertLess(diagnostics['maximum_sales_constraint_violation_millions'], 1e-8)
        self.assertLess(diagnostics['maximum_cogs_constraint_violation_millions'], 1e-8)
        levels = np.column_stack([sales, sales * [1.01, .99, 1.02], sales * [1.03, 1., 1.01]])
        shocks, influences, _ = recover_shocks(levels, sales, a)
        np.testing.assert_allclose(shocks @ influences, np.diff(levels.sum(axis=0)), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
