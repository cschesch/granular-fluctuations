#!/usr/bin/env python3
"""Independent algebra and disclosure checks for the single Herskovic path."""
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from applications.herskovic.run_herskovic_firm_network import (
    check_network, complete_rows, match_sizes, schur_recover,
)


class HerskovicChecks(unittest.TestCase):
    def test_schur_matches_direct_hidden_system(self):
        w = np.array([[0., .2, .8, 0.], [.4, 0., .3, .3],
                      [.1, .2, 0., .7], [.6, .4, 0., 0.]])
        gamma = .918
        observed, hidden = np.array([2, 0]), np.array([1, 3])
        growth = np.array([[.4, -.3], [.1, .7], [-.2, .6]])
        weights = np.array([.3, .7])
        shocks, exposures, _, _ = schur_recover(csr_matrix(w), gamma, observed, growth, weights)
        hidden_growth = np.linalg.solve(np.eye(2) - gamma*w[np.ix_(hidden, hidden)],
                                         gamma*w[np.ix_(hidden, observed)] @ growth.T).T
        full_growth = np.empty((3, 4))
        full_growth[:, observed] = growth
        full_growth[:, hidden] = hidden_growth
        direct_shocks = full_growth @ (np.eye(4)-gamma*w).T
        np.testing.assert_allclose(direct_shocks[:, hidden], 0., atol=1e-14)
        np.testing.assert_allclose(shocks, direct_shocks[:, observed], atol=1e-14)
        np.testing.assert_allclose(shocks @ exposures, growth @ weights, atol=1e-14)
        # Deleting hidden firms loses their feedback even with zero hidden shocks.
        self.assertGreater(np.max(np.abs(shocks-growth@(np.eye(2)-gamma*w[np.ix_(observed, observed)]).T)), .01)

    def test_size_population_and_identity_assignment(self):
        sizes = np.array([1., 10., 11., 100., 1000.])
        sales = pd.Series([10.1, 10.2, 900.], index=[30, 10, 20])
        firms = match_sizes(sales, sizes)
        np.testing.assert_array_equal(np.sort(firms.model_size), sizes)
        self.assertEqual(firms.model_source_node.nunique(), len(sizes))
        self.assertEqual(set(firms.loc[firms.permno.gt(0), 'permno']), set(sales.index))
        self.assertEqual(firms.loc[firms.permno.eq(30), 'model_size'].item(), 10.)

    def test_disclosed_weights_and_screen(self):
        sizes = np.arange(1., 13.)
        known = csr_matrix(([.25], ([0], [1])), shape=(12, 12))
        identified = np.arange(12) < 2
        screened = np.arange(12) == 0
        w, _ = complete_rows(sizes, known, screened, identified, {'Z': .1, 'psi': .184}, 123)
        check_network(w, known, screened, identified)
        self.assertEqual(w[0, 1], .25)
        np.testing.assert_allclose(w.sum(axis=1), 1.)


if __name__ == '__main__':
    unittest.main()
