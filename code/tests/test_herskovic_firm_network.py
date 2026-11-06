#!/usr/bin/env python3
"""Independent algebra and disclosure checks for the single Herskovic path."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import applications.herskovic.run_herskovic_firm_network as herskovic
from applications.herskovic.run_herskovic_firm_network import (
    check_network, complete_rows, match_sizes, schur_recover,
)


class HerskovicChecks(unittest.TestCase):
    def test_generated_sizes_exist_before_cache_fingerprint(self):
        # The first run must use the same source fingerprint as later runs.
        with TemporaryDirectory() as folder:
            size_file = Path(folder) / 'sizes.mat'
            (Path(folder) / 'network_W.npz').touch()
            cached = dict(outcomes=np.ones((2, 1)), shares=np.ones(1),
                          ids=pd.Index([1]), dates=pd.period_range('1980Q1', periods=2, freq='Q'),
                          diagnostics={})

            def prepare_sizes():
                size_file.touch()
                return np.ones(1)

            def fingerprint(prefix, params, sources):
                self.assertIn(size_file, sources)
                self.assertTrue(size_file.is_file())
                return Path(folder) / 'cache.pkl'

            with patch.object(herskovic, 'SIZE_CACHE', size_file), \
                    patch.object(herskovic, 'DERIVED', Path(folder)), \
                    patch.object(herskovic, 'original_sizes', side_effect=prepare_sizes), \
                    patch.object(herskovic, 'clean_panel_cache_path', side_effect=fingerprint), \
                    patch.object(herskovic, 'read_clean_panel_cache', return_value=cached):
                result = herskovic.build_herskovic_inputs()
            np.testing.assert_array_equal(result.outcomes, cached['outcomes'])

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
        gamma = .918
        w, _ = complete_rows(sizes, known, screened, identified, {'Z': .1, 'psi': .184, 'gamma': gamma}, 123)
        check_network(w, known, screened, identified, gamma)
        self.assertAlmostEqual(gamma * w[0, 1], .25)
        np.testing.assert_allclose(w.sum(axis=1), 1.)

    def test_disclosed_sales_cannot_exceed_intermediate_sales(self):
        known = csr_matrix(([.7, .25], ([0, 0], [1, 2])), shape=(3, 3))
        with self.assertRaisesRegex(ValueError, 'exceeding gamma'):
            complete_rows(np.ones(3), known, np.ones(3, bool), np.ones(3, bool),
                          {'Z': .1, 'psi': .184, 'gamma': .918}, 123)

    def test_disclosure_threshold_applies_to_gamma_times_weight(self):
        # The extra public link has W=.11 but a raw sales share below .10.
        w = csr_matrix([[0., .89, .11], [1., 0., 0.], [1., 0., 0.]])
        known = csr_matrix(([.89*.9], ([0], [1])), shape=(3, 3))
        check_network(w, known, np.array([True, False, False]), np.ones(3, bool), .9)
        with self.assertRaises(AssertionError):
            check_network(csr_matrix([[0., .88, .12], [1., 0., 0.], [1., 0., 0.]]),
                          csr_matrix(([.88*.9], ([0], [1])), shape=(3, 3)),
                          np.array([True, False, False]), np.ones(3, bool), .9)


if __name__ == '__main__':
    unittest.main()
