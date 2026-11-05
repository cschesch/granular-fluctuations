"""Failed Monte Carlo solves must stop inference, never become observations."""

from dataclasses import replace
import hashlib
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import inference_mc as mc
import plot_inference_mc as plotting
import run_inference_mc as runner

TEST_GROUP = "numerical"


class MonteCarloSolverFailures(unittest.TestCase):
    def test_directional_failure_and_nonfinite_value_stop_inference(self):
        est = SimpleNamespace(sigma=np.eye(2)[None], share=np.array([0.5]),
                              aggregate_var=np.array([2.0]))
        directions = np.zeros((1, 2, 2, 2))
        cases = (([1., 2.], [True, False]), ([1., np.nan], [True, True]))
        for values, solved in cases:
            with self.subTest(values=values, solved=solved), patch.object(
                    mc, "near_optimal_derivative",
                    return_value=(np.array(values), np.array(solved))):
                with self.assertRaisesRegex(RuntimeError, "Monte Carlo near-optimal"):
                    mc._directional_shares(directions, est, np.array([0]), 0.1)

    def test_point_and_resampled_shares_reject_failed_solves(self):
        failed = SimpleNamespace(value=np.array([1.]),
                                 acceptable=Mock(return_value=np.array([False])))
        x = np.random.default_rng(34).standard_normal((1, 20, 2))
        with patch.object(mc, "mgp_batch", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "Monte Carlo point estimates"):
                mc.point_estimates(x, 1)
            with self.assertRaisesRegex(RuntimeError, "Monte Carlo resampled shares"):
                mc._shares_from_cov(np.eye(2)[None, None])

    def test_nonfinite_share_derivative_stops_inference(self):
        est = SimpleNamespace(sigma=np.eye(2)[None], share=np.array([0.5]),
                              aggregate_var=np.array([1e-309]))
        directions = np.zeros((1, 2, 2, 2))
        with patch.object(mc, "near_optimal_derivative", return_value=(
                np.array([1., 2.]), np.array([True, True]))), \
                np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(RuntimeError, "Monte Carlo share derivatives"):
                mc._directional_shares(directions, est, np.array([0]), 0.1)

    def test_grouped_population_rejects_failed_solve(self):
        failed = SimpleNamespace(value=np.array([1.]),
                                 acceptable=Mock(return_value=np.array([False])))
        design = mc.Design("failed_population", "test", n=8, m=2, r=1)
        with patch.object(mc, "_firm_level_truth", return_value=0.1), \
                patch.object(mc, "mgp_batch", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "Grouped population MGP"):
                mc.build_population(design, use_cache=False)

    def test_full_panel_retries_same_problem_after_unaccepted_batch(self):
        sizes = np.array([0.5, 0.3, 0.2])
        sigma = np.diag([1., 2., 3.])
        failed = SimpleNamespace(value=np.array([999.]),
                                 acceptable=Mock(return_value=np.array([False])))
        with patch.object(mc, "firm_level_mgp", return_value=failed), \
                patch.object(mc, "licensed_mosek", return_value=False):
            value = mc._firm_level_truth(mc.Design("fallback", "test", n=3),
                                         sigma, sizes, use_cache=False)
        failed.acceptable.assert_called_once_with(mc.FIRM_GAP_TOL)
        self.assertAlmostEqual(value, float(sizes**2 @ np.diag(sigma)), places=7)

    def test_old_population_cache_is_retained_but_not_loaded(self):
        design = mc.Design("cache", "test", n=8, m=2, r=1)
        key = mc.population_key(design)
        rebuilt = SimpleNamespace(design=design)
        with TemporaryDirectory() as temporary, \
                patch.object(mc, "POP_CACHE", Path(temporary)), \
                patch.dict(mc._POPULATION_CACHE, clear=True), \
                patch.object(mc, "_build_population", return_value=rebuilt) as build:
            old = Path(temporary) / (mc._key_slug(key).replace(mc.POP_CACHE_VERSION, "v2", 1) + ".pkl")
            old.write_bytes(pickle.dumps("unchecked population"))
            self.assertIs(mc.build_population(design), rebuilt)
            build.assert_called_once_with(design, True)
            self.assertTrue(old.exists())

    def test_old_simulation_checkpoint_is_retained_but_not_loaded(self):
        design = mc.Design("cache", "test", n=8, m=2, r=1)
        name = f"common_r{design.r}_z{design.zeta}_m{design.m}_idio{design.idio}_rho{design.rho}_shock{design.shock}"
        numerical = replace(design, name=name, label="")
        old_fingerprint = hashlib.sha256(repr(
            (numerical, runner.simulation_config(True), "paired-1000-v1")
        ).encode()).hexdigest()[:16]
        result = dict(population=None, row=dict(gamma_bar0=0.03), coverage=[])
        with TemporaryDirectory() as temporary, \
                patch.object(runner, "CACHE", Path(temporary)), \
                patch.object(runner, "regularity", return_value=True), \
                patch.object(mc, "run_design", return_value=result) as run:
            old = Path(temporary) / f"{old_fingerprint}.pkl"
            old.write_bytes(pickle.dumps("unchecked simulation"))
            runner.worker((design, True))
            self.assertEqual(run.call_count, 1)
            self.assertTrue(old.exists())
            self.assertEqual(len(list(Path(temporary).glob("*.pkl"))), 2)

    def test_plotting_rejects_old_bundles_without_removing_them(self):
        current = dict(cache_version=runner.CACHE_VERSION,
                       population_cache_version=mc.POP_CACHE_VERSION)
        cases = ({}, {**current, "cache_version": "paired-1000-v1"},
                 {**current, "population_cache_version": "v2"})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mc_results.pkl"
            for metadata in cases:
                with self.subTest(metadata=metadata):
                    contents = pickle.dumps(dict(meta=metadata))
                    path.write_bytes(contents)
                    with self.assertRaisesRegex(ValueError, "Rerun run_inference_mc.py"):
                        plotting.load_simulations(path)
                    self.assertEqual(path.read_bytes(), contents)
            valid = dict(meta=current, main=[], sweep=[], delta=[])
            path.write_bytes(pickle.dumps(valid))
            self.assertEqual(plotting.load_simulations(path), valid)


if __name__ == "__main__":
    unittest.main()
