"""Equal-size units must have the same LPT assignment on every platform."""

import unittest

import numpy as np
import pandas as pd

from mgp_core import align_sizes_to_panel, lpt_buckets


class GroupingTies(unittest.TestCase):
    def test_alignment_preserves_panel_order_within_size_ties(self):
        names = [f"unit_{i}" for i in range(100)]
        panel = pd.DataFrame({"outcome": range(100)}, index=names)
        sizes = pd.Series([2.0 if i % 2 else 1.0 for i in range(100)], index=names)
        aligned, weights = align_sizes_to_panel(panel, sizes.iloc[::-1])
        expected = names[1::2] + names[::2]
        self.assertEqual(aligned.index.tolist(), expected)
        self.assertEqual(weights.index.tolist(), expected)

    def test_equal_jobs_use_input_order_and_first_lightest_bucket(self):
        buckets, loads = lpt_buckets(np.ones(30), 3)
        self.assertEqual(buckets, [list(range(i, 30, 3)) for i in range(3)])
        np.testing.assert_array_equal(loads, [10, 10, 10])


if __name__ == "__main__":
    unittest.main()
