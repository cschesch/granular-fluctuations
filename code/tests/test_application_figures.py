"""Figure redraws retain row identities and require valid inference caches."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from applications.figure_layout import hierarchy_positions
import plot_expanded_applications as expanded
import plot_m_sweep as msweep


class ApplicationFigureTests(unittest.TestCase):
    def test_cached_coordinates_cannot_move_an_estimate_to_another_row(self):
        row = dict(item="Stocks", y=1000, estimate=.5, low=np.nan, high=np.nan,
                   homoskedastic=np.nan)
        with patch("matplotlib.figure.Figure.savefig"), patch.object(expanded.plt, "close"):
            expanded.plot([row])
            figure = expanded.plt.gcf()
            self.assertEqual(figure.axes[0].lines[0].get_ydata()[0], hierarchy_positions()[0]["Stocks"])
        expanded.plt.close(figure)

    def test_presentation_edits_do_not_invalidate_numerical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("gradient_bootstrap.py", "near_optimal.py", "fast_mgp.py",
                         "mgp_kernels.py", "mgp_core.py", "mosek_face.py"):
                (root / name).write_text("numerical code")
            with patch.object(msweep, "__file__", str(root / "plot_m_sweep.py")):
                before = msweep.inference_fingerprint()
                (root / "plot_m_sweep.py").write_text("changed plot layout")
                self.assertEqual(msweep.inference_fingerprint(), before)
                (root / "mosek_face.py").write_text("changed solver retries")
                self.assertNotEqual(msweep.inference_fingerprint(), before)

    def saved_rows(self):
        return pd.DataFrame([dict(panel="Saved panel", M=5, share=.3, ci_low=.2, ci_high=.4,
                                  bootstrap_B=1000, block_length=5,
                                  inference_fingerprint=msweep.inference_fingerprint())])

    def test_unverified_and_nonfinite_intervals_cannot_be_redrawn(self):
        good = self.saved_rows()
        pd.testing.assert_frame_equal(msweep.checked_results(good), good)
        for bad in (good.drop(columns="inference_fingerprint"),
                    good.assign(inference_fingerprint="old solver"),
                    good.assign(ci_high=np.nan)):
            with self.assertRaisesRegex(ValueError, "unverified, stale or nonfinite"):
                msweep.checked_results(bad)

    def test_partial_run_validates_retained_panels_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(msweep, "RUN_OUTPUT_CACHE", Path(directory)), \
                patch.object(msweep, "PANELS", {}):
            path = Path(directory) / "msweep_shares.csv"
            good = self.saved_rows()
            good.to_csv(path, index=False)
            retained = msweep.compute_sweep_results("CLARABEL", 4, 123, only=["New panel"])
            pd.testing.assert_frame_equal(retained, good)
            good.assign(inference_fingerprint="old solver").to_csv(path, index=False)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "without --from-cache or --only"):
                msweep.compute_sweep_results("CLARABEL", 4, 123, only=["New panel"])
            self.assertEqual(path.read_bytes(), before)

    def test_nonfinite_per_group_checkpoint_is_recomputed(self):
        y = np.random.default_rng(123).normal(size=(6, 20))
        shares = np.full(6, 1/6)
        answer = dict(g_hat=.3, ci_low=.2, ci_high=.4, B=4, block_length=3,
                      common_rank=1, regular=True, fallbacks=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(msweep, "RUN_OUTPUT_CACHE", Path(directory)), \
                patch.object(msweep.GB, "moving_block_gradient_ci", return_value=answer) as fit:
            settings = dict(is_daily=False, bootstrap_b=4, seed=123,
                            start_date="0", last_date="19")
            msweep.sweep(y, shares, [2], "CLARABEL", "Test", **settings)
            path = next((Path(directory) / "msweep").glob("*.pkl"))
            saved = pd.read_pickle(path)
            saved["ci_high"] = np.nan
            pd.to_pickle(saved, path)
            msweep.sweep(y, shares, [2], "CLARABEL", "Test", **settings)
            self.assertEqual(fit.call_count, 2)
            self.assertTrue(np.isfinite(pd.read_pickle(path)["ci_high"]))


if __name__ == "__main__":
    unittest.main()
