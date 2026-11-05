"""The rank cache must notice changed observations with unchanged dimensions."""
import unittest

import numpy as np

from check_factor_rank_figure1 import panel_fingerprint


class RankPanelFingerprintTest(unittest.TestCase):
    def test_values_and_order_are_part_of_cache_identity(self):
        x = np.arange(12.0).reshape(4, 3)
        changed = x.copy()
        changed[0, 0] = 1
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(changed))
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(x[::-1]))
        self.assertNotEqual(panel_fingerprint(x), panel_fingerprint(x.reshape(3, 4)))

    def test_array_storage_does_not_change_cache_identity(self):
        x = np.arange(12.0).reshape(4, 3)
        self.assertEqual(panel_fingerprint(x), panel_fingerprint(np.asfortranarray(x)))
        self.assertEqual(panel_fingerprint(x), panel_fingerprint(x.astype(int)))


if __name__ == "__main__":
    unittest.main()
