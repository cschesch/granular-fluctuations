"""A checkpoint must match both the input sample and resampling stream."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import gradient_bootstrap as gb
from gradient_bootstrap import bootstrap_family


class GradientCheckpointTests(unittest.TestCase):
    def run_bootstrap(self, checkpoint=None):
        x = np.random.default_rng(110).normal(size=(40, 3))
        with contextlib.redirect_stdout(io.StringIO()):
            return bootstrap_family(x, b=4, block_length=4,
                                    rng=np.random.default_rng(120), solver="CLARABEL",
                                    checkpoint_path=checkpoint, checkpoint_every=100)

    def test_failed_derivative_never_reaches_quantiles(self):
        with patch.object(gb, "near_optimal_derivative", return_value=(np.array([123.]), np.array([False]))), \
                patch.object(gb, "_ci") as interval:
            with self.assertRaisesRegex(RuntimeError, "Gradient-bootstrap derivative"):
                self.run_bootstrap()
            interval.assert_not_called()

    def test_nonfinite_share_transformation_is_rejected(self):
        derivative = object.__new__(gb.ShareDirectionalDerivative)
        derivative.a = np.finfo(float).tiny
        derivative.g_hat = 0.0
        with patch.object(derivative, "levels", return_value=np.array([100.])), \
                np.errstate(over="ignore"):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite.*share derivative"):
                derivative.shares(np.zeros((1, 2, 2)))

    def test_unverified_checkpoint_rows_are_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "draws.pkl"
            expected = self.run_bootstrap(checkpoint)
            verified = pd.read_pickle(checkpoint)
            for damage in ("missing_flag", "failed_flag", "old_policy", "nonfinite"):
                with self.subTest(damage=damage):
                    rows = verified.copy()
                    rows["directional_stat"] = 123456.0
                    if damage == "missing_flag":
                        rows = rows.drop(columns=["directional_solved", "solver_policy"])
                    elif damage == "failed_flag":
                        rows["directional_solved"] = False
                    elif damage == "old_policy":
                        rows["solver_policy"] = "unverified"
                    else:
                        rows["directional_stat"] = np.nan
                    rows.to_pickle(checkpoint)
                    actual = self.run_bootstrap(checkpoint)
                    for field in ("gradient_low", "gradient_high", "block_gradient_low", "block_gradient_high"):
                        self.assertAlmostEqual(actual[field], expected[field], places=11)
                    self.assertTrue(pd.read_pickle(checkpoint).directional_solved.all())

    def test_failure_checkpoints_only_verified_draws_and_resumes_same_stream(self):
        original = gb.near_optimal_derivative
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return np.array([987654.0]), np.array([False])
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "draws.pkl"
            with patch.object(gb, "near_optimal_derivative", side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    self.run_bootstrap(checkpoint)
            partial = pd.read_pickle(checkpoint)
            self.assertEqual(partial.draw.tolist(), [1])
            self.assertTrue(partial.directional_solved.all())
            resumed = self.run_bootstrap(checkpoint)
            fresh = self.run_bootstrap()
            for field in ("gradient_low", "gradient_high", "block_gradient_low", "block_gradient_high"):
                self.assertAlmostEqual(resumed[field], fresh[field], places=11)
            self.assertEqual(len(pd.read_pickle(checkpoint)), 8)

    def test_changed_data_or_seed_cannot_reuse_old_draws(self):
        x = np.random.default_rng(110).normal(size=(40, 3))
        changed = x.copy()
        changed[0, 0] += 5

        def run(sample, seed, checkpoint=None):
            with contextlib.redirect_stdout(io.StringIO()):
                return bootstrap_family(
                    sample, b=4, block_length=4,
                    rng=np.random.default_rng(seed), solver='CLARABEL',
                    checkpoint_path=checkpoint,
                )

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'draws.pkl'
            previous = None
            for sample, seed in [(x, 120), (changed, 120), (changed, 121)]:
                cached = run(sample, seed, checkpoint)
                fresh = run(sample, seed)
                for field in ('bootstrap_low', 'bootstrap_high', 'gradient_low',
                              'gradient_high', 'block_gradient_low', 'block_gradient_high'):
                    self.assertAlmostEqual(cached[field], fresh[field], places=11)
                rows = pd.read_pickle(checkpoint)
                self.assertEqual(len(rows), 8)
                signature = rows.estimator_id.unique()
                self.assertEqual(len(signature), 1)
                self.assertNotEqual(signature[0], previous)
                previous = signature[0]

            before = pd.read_pickle(checkpoint)
            run(changed, 121, checkpoint)
            pd.testing.assert_frame_equal(before, pd.read_pickle(checkpoint))


if __name__ == '__main__':
    unittest.main()
