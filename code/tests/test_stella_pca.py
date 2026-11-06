"""Check standardized PCA and propagation of the full residual covariance."""

import numpy as np

from applications.stella.run_stella_firm_network import one_pc_variance_decomposition


def test_standardized_pc_preserves_original_units_and_residual_covariance():
    shocks = np.array([[-1., -1., -1., -3., -3.], [-1., -1., -1., 3., 3.],
                       [1., 1., 1., -3., -3.], [1., 1., 1., 3., 3.]])
    propagation = np.diag([.1, .1, .1, .5, .5])
    propagation[3, 4] = propagation[4, 3] = .2
    result = one_pc_variance_decomposition(shocks + [5., -2., 7., 4., 1.], propagation)
    # Standardized PCA removes the three correlated low-variance columns.
    # The two remaining columns retain their original variances and covariance.
    expected = 9 * 1.4 / (9 * 1.4 + .3)
    np.testing.assert_allclose(result["one_pc_residual_share"], expected, atol=1e-12)
    np.testing.assert_allclose(sum(result.values()), 1., atol=1e-12)
