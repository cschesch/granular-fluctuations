"""Failed directional solves must be retried unchanged or remain unavailable."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import mosek_face
import near_optimal as solver


TEST_GROUP = "numerical"


class NearOptimalFailureTests(unittest.TestCase):
    def setUp(self):
        self.sigma = np.eye(2)
        self.cap = 2.2
        self.directions = np.array([np.diag([1., -1.]), np.diag([-1., 1.])])

    def test_original_invalid_covariance_rejected_by_both_paths(self):
        invalid = (-np.eye(2), np.diag([1., 0.]), np.diag([1., -1.]),
                   np.array([[np.nan, 0.], [0., 1.]]))
        for sigma in invalid:
            with self.subTest(sigma=sigma):
                for compiled in (False, True):
                    with patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", compiled), \
                            patch.object(solver.mgp_kernels, "barrier_nearopt") as kernel:
                        with self.assertRaisesRegex(ValueError, "positive definite"):
                            solver.near_optimal_derivative(self.directions, sigma, self.cap)
                        kernel.assert_not_called()
                with self.assertRaisesRegex(ValueError, "positive definite"):
                    solver.near_optimal_derivative_numpy(self.directions, sigma, self.cap)

    def test_exhausted_barrier_retries_same_problem_with_and_without_numba(self):
        for compiled in (False, True):
            if compiled and not solver.mgp_kernels.NUMBA_AVAILABLE:
                continue
            with self.subTest(compiled=compiled), \
                    patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", compiled), \
                    patch.object(mosek_face, "licensed_mosek", return_value=False):
                values, solved = solver.near_optimal_derivative(
                    self.directions, self.sigma, self.cap, max_iter=0,
                )
                self.assertTrue(solved.all())
                # For Sigma=I, diag(Y)>=1 and trace(Y)<=2.2 imply a minimum -0.2.
                np.testing.assert_allclose(values, [-.2, -.2], atol=1e-7)

    def test_failed_or_nonfinite_iterates_never_escape_with_finite_values(self):
        for values, flags in (([123., 456.], [False, False]),
                              ([np.inf, np.nan], [True, True])):
            with self.subTest(values=values), \
                    patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", False), \
                    patch.object(solver, "near_optimal_derivative_numpy",
                                 return_value=(np.array(values), np.array(flags))), \
                    patch.object(solver, "_open_solver") as retry:
                got, solved = solver.near_optimal_derivative(
                    self.directions, self.sigma, self.cap, backstop=False,
                )
                self.assertTrue(np.isnan(got).all())
                self.assertFalse(solved.any())
                retry.assert_not_called()

    def test_numpy_direct_failure_returns_nan(self):
        values, solved = solver.near_optimal_derivative_numpy(
            self.directions, self.sigma, self.cap, max_iter=0,
        )
        self.assertTrue(np.isnan(values).all())
        self.assertFalse(solved.any())

    def test_numpy_factorization_failure_reaches_general_solver(self):
        with patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", False), \
                patch.object(solver, "near_optimal_derivative_numpy",
                             side_effect=np.linalg.LinAlgError("singular iterate")), \
                patch.object(mosek_face, "licensed_mosek", return_value=False):
            values, solved = solver.near_optimal_derivative(self.directions, self.sigma, self.cap)
        self.assertTrue(solved.all())
        np.testing.assert_allclose(values, [-.2, -.2], atol=1e-7)

    def test_partial_mosek_failure_is_retried_by_clarabel(self):
        mosek = MagicMock()
        mosek.__enter__.return_value = mosek
        mosek.values.return_value = (np.array([5., np.nan]), np.array([False, True]))
        with patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", False), \
                patch.object(mosek_face, "licensed_mosek", return_value=True), \
                patch.object(mosek_face, "NearOptimalSolver", return_value=mosek), \
                patch.object(solver, "_open_solver",
                             return_value=(np.array([7.]), np.array([False]))) as clarabel:
            values, solved = solver.near_optimal_derivative(
                self.directions, self.sigma, self.cap, max_iter=0,
            )
        np.testing.assert_array_equal(values, [5., 7.])
        self.assertTrue(solved.all())
        np.testing.assert_array_equal(clarabel.call_args.args[0], self.directions[1:])
        np.testing.assert_array_equal(clarabel.call_args.args[1], self.sigma)
        self.assertEqual(clarabel.call_args.args[2], self.cap)

    def test_all_retries_failed_leave_nan_and_require_solved_raises(self):
        with patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", False), \
                patch.object(mosek_face, "licensed_mosek", return_value=False), \
                patch.object(solver, "_open_solver",
                             return_value=(np.array([8., 9.]), np.array([True, True]))):
            values, solved = solver.near_optimal_derivative(
                self.directions, self.sigma, self.cap, max_iter=0,
            )
        self.assertTrue(np.isnan(values).all())
        self.assertFalse(solved.any())
        with self.assertRaisesRegex(RuntimeError, "2/2 solves unsuccessful or nonfinite"):
            solver.require_solved(values, solved, "Bootstrap")
        with self.assertRaises(RuntimeError):
            solver.require_solved([np.nan], [True])

    def test_covariance_groups_retry_only_their_own_directions(self):
        directions = np.concatenate([self.directions, self.directions[:1]])
        sigma = np.array([np.eye(2), 2 * np.eye(2)])
        caps = np.array([2.2, 6.])
        owner = np.array([1, 0, 1])
        with patch.object(solver.mgp_kernels, "NUMBA_AVAILABLE", False), \
                patch.object(mosek_face, "licensed_mosek", return_value=False):
            values, solved = solver.near_optimal_derivative(
                directions, sigma, caps, owner, max_iter=0,
            )
        self.assertTrue(solved.all())
        np.testing.assert_allclose(values, [-1., -.2, -1.], atol=1e-7)

    def test_conic_validation_rejects_infeasible_or_suboptimal_outputs(self):
        direction = np.diag([1., -1.])
        valid_y = np.diag([1., 1.2])
        valid_nu, valid_t = np.array([2., 0.]), 1.
        self.assertTrue(solver._conic_solution_valid(
            valid_y, direction, self.sigma, self.cap, valid_nu, valid_t))
        for y, nu, t in ((np.diag([.9, 1.3]), valid_nu, valid_t),
                         (np.diag([1., 1.3]), valid_nu, valid_t),
                         (np.array([[1., 2.], [2., 1.2]]), valid_nu, valid_t),
                         (valid_y, np.array([3., 0.]), valid_t),
                         (np.eye(2), valid_nu, valid_t)):
            self.assertFalse(solver._conic_solution_valid(
                y, direction, self.sigma, self.cap, nu, t))


if __name__ == "__main__":
    unittest.main()
