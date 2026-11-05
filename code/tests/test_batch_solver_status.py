"""Check barrier counts and rejection/retry of unusable terminal iterates."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import fast_mgp
import mgp_kernels
import plot_inference_tightness as tightness

TEST_GROUP = "numerical"


def test_barrier_counts_and_successful_solutions():
    covariance = np.repeat(np.eye(3)[None], 3, axis=0)
    basis = np.linalg.qr(np.array([[1., 1.], [-1., 1.], [0., -2.]]))[0]
    masks = (False, True, np.array([True, False, True]))
    for compiled in ((False, True) if mgp_kernels.NUMBA_AVAILABLE else (False,)):
        with patch.object(mgp_kernels, "NUMBA_AVAILABLE", compiled):
            for reduced in (None, basis):
                for mask, count in zip(masks, (0, 3, 2)):
                    solution = fast_mgp.solve_batch(covariance, nonneg=mask, basis=reduced, mu_end=1e-8)
                    q = 3 if reduced is None else 2
                    assert np.array_equal(solution.barrier_count, np.full(3, q + count))
                    np.testing.assert_allclose(solution.relative_gap, (q + count) * solution.mu / solution.value)
                    assert solution.acceptable(1e-6).all()
                    np.testing.assert_allclose(solution.value, 3., rtol=1e-6)
    if mgp_kernels.NUMBA_AVAILABLE:
        masks = np.array([[False, False, False], [True, False, True], [True, True, True]])
        solution = fast_mgp.solve_batch(covariance, nonneg=masks,
                                       basis=np.repeat(basis[None], 3, axis=0), mu_end=1e-8)
        assert np.array_equal(solution.barrier_count, [2, 4, 5])
        assert solution.acceptable(1e-6).all()


def _terminal(nu, mu, *, need_dual=True):
    nu = np.asarray(nu, dtype=float)
    k, m = nu.shape
    return fast_mgp._finish_solution(
        np.repeat(np.eye(m)[None], k, axis=0), None, None,
        np.ones(m), 1., np.ones(m, dtype=bool), np.ones(k),
        nu, np.asarray(mu, dtype=float), np.ones(k, dtype=bool), 1, need_dual,
    )


def test_small_gap_does_not_override_failed_status():
    solution = fast_mgp.solve_batch(np.eye(3), mu_end=1e-10, need_dual=False)
    assert solution.relative_gap[0] < 1e-6
    assert solution.acceptable(1e-6)[0]
    assert not replace(solution, converged=np.array([False])).acceptable(1e-6)[0]
    assert not replace(solution, feasible=np.array([False])).acceptable(1e-6)[0]
    assert not replace(solution, value=np.array([np.nan])).acceptable()[0]
    assert not replace(solution, mu=np.array([-1.])).acceptable()[0]
    assert not replace(solution, value=np.array([1e-320])).acceptable()[0]


def test_terminal_feasibility_and_failed_inverse_are_reported():
    solution = _terminal([[.5, .5], [2., 2.], [-.5, -.5], [np.nan, .5], [.5, .5]],
                         [.5, .5, .5, .5, .01])
    assert np.array_equal(solution.feasible, [True, False, False, False, False])
    assert np.array_equal(solution.acceptable(), solution.feasible)
    inverse = np.linalg.inv

    def fail_one(cone):
        if cone.ndim == 3 or cone[0, 0] == .25:
            raise np.linalg.LinAlgError("forced terminal inverse failure")
        return inverse(cone)

    with patch.object(np.linalg, "inv", side_effect=fail_one):
        solution = _terminal([[.5, .5], [.75, .75]], [.5, .25])
    assert np.array_equal(solution.acceptable(), [True, False])
    with patch.object(np.linalg, "inv", side_effect=AssertionError("dual inversion was omitted")):
        assert _terminal([[.5, .5]], [.5], need_dual=False).acceptable()[0]


def test_tightness_retries_the_same_draw_and_uses_conic_backstop():
    calls = []
    solve = fast_mgp.solve_batch

    def failed_batch(sigma, **kwargs):
        calls.append(np.array(sigma, copy=True))
        solution = solve(sigma, **kwargs)
        # Even a tiny diagnostic must not certify this failed solve.
        return replace(solution, mu=np.full(len(sigma), 1e-30),
                       converged=np.zeros(len(sigma), dtype=bool))

    # Two buckets have no balanced one-factor shortcut, so both forced batch
    # failures must reach the independent conic solve of this same covariance.
    with patch.object(tightness, "solve_batch", side_effect=failed_batch), \
            patch.object(tightness, "licensed_mosek", return_value=False):
        rows = tightness.block(12, .75, 2, 1, .03, .5, 0, 1e-9)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], calls[1])
    assert len(rows) == 1 and np.isfinite(rows[0]["looseness"])
    assert rows[0]["max_relative_gap"] < 1e-5


def test_tightness_balanced_certificate_and_successful_run():
    settings = (12, 0., 3, 1, .03, .5, 4, 1e-9)
    with patch.object(tightness, "solve_batch", wraps=fast_mgp.solve_batch) as reference_solver:
        reference = tightness.block(*settings)
    assert reference[0]["looseness"] == reference[0]["max_relative_gap"] == 0.
    calls = []
    solve = fast_mgp.solve_batch

    def fail_balanced_draw(sigma, **kwargs):
        solution = solve(sigma, **kwargs)
        calls.append(sigma)
        if len(calls) == 1:
            return replace(solution, value=np.full(len(sigma), np.nan),
                           converged=np.zeros(len(sigma), dtype=bool))
        return solution

    # This seed satisfies the one-factor balancing inequality. Its exact
    # certificate remains usable even when the numerical solve is rejected.
    with patch.object(tightness, "solve_batch", side_effect=fail_balanced_draw):
        checked = tightness.block(*settings)
    assert len(calls) == reference_solver.call_count
    for checked_sigma, reference_call in zip(calls, reference_solver.call_args_list):
        np.testing.assert_array_equal(checked_sigma, reference_call.args[0])
    assert checked == reference


def test_tightness_old_checkpoint_cannot_bypass_validation():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "example.csv"
        path.write_text("old unchecked checkpoint\n")
        with patch.object(tightness, "RUN_OUTPUT_CACHE", Path(directory)), \
                patch("sys.argv", ["tightness", "--stem", "example", "--from-cache"]):
            try:
                tightness.main()
            except FileNotFoundError as error:
                assert tightness.CACHE_VERSION in str(error)
            else:
                raise AssertionError("An unchecked legacy cache was reused")
        assert path.read_text() == "old unchecked checkpoint\n"
