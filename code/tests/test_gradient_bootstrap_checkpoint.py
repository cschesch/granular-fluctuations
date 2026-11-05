"""A checkpoint must match both the input sample and resampling stream."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gradient_bootstrap import bootstrap_family


class GradientCheckpointTests(unittest.TestCase):
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
