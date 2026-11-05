#!/usr/bin/env python3
"""Homoskedastic grouped GIV with known or estimated loading spaces.

This module implements the fixed-dimensional procedure in Hahn, He, Liao and
Zhou (2026, Algorithm 1), adapted to the whitened grouped system used in the
paper.  If

    X_t = phi * p_t * b + B f_t + u_t,       Var(u_t) = sigma_u^2 I_M,

then valid instruments are ``A' X_t``, where the columns of ``A`` span the
orthogonal complement of ``[b, B]``.  Hahn et al. use ``b = 1``; the algebra is
unchanged for a known nonzero direction ``b``.  The aggregate is ``y_t=d'X_t``
and the reference outcome for the demand equation is
``x_b,t = b'X_t/(b'b)``.

The implementation deliberately keeps the oracle and feasible variance paths
separate.  With a known loading space the ordinary GMM moment covariance is
correct.  With an estimated space, the influence function contains Hahn et
al.'s generated-eigenspace term.  The separate ``naive_se`` diagnostic keeps
the feasible point estimate and its corrected optimal weight fixed, but forms
the final sandwich as if the estimated space were known; it is never silently
substituted for the feasible standard error.

Two estimators share one instrument space. ``estimate_joint_giv`` implements
Algorithm 1 with both equations, all residual directions, and cross-equation
optimal weighting. ``scalar_iv`` collapses the space to the single
aggregate-predicting direction ``P d``, where ``P = A A'`` is the projector
onto the valid space and ``d`` carries the aggregate weights, and applies the
ordinary IV ratio; that direction attains the information bound of the paper's
grouping proposition, and with one moment there is no many-moment two-step
bias.  Its standard error still carries the generated-eigenspace term when the
space is estimated: the influence function of ``d'P_hat m_T`` is ``(P d)' xi_t``
with Hahn et al.'s ``xi_t``, because the projector perturbation acts on the
population moment ``E[X_t eps_t]``, which the true space annihilates.

The module is intended for the Monte Carlo appendix, not for arbitrary
diagonal heteroskedasticity: squared-size whitening produces spherical grouped
errors only when the micro errors are homoskedastic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy.stats import chi2


RANK_TOL = 1e-10
INFO_TOL = 1e-10


@dataclass(frozen=True)
class InstrumentSpace:
    """A known or sample-estimated valid-instrument space."""

    basis: np.ndarray
    combined_rank: int
    residual_dimension: int
    information: float
    eigvals: np.ndarray
    noise_variance: float
    eigengap: float
    upsilon: np.ndarray
    bic_values: np.ndarray
    estimated: bool


@dataclass(frozen=True)
class EquationEstimate:
    """One structural estimate and its inference diagnostics."""

    estimate: float
    se: float
    naive_se: float
    preliminary: float
    j_stat: float
    j_pvalue: float
    j_df: int
    derivative_strength: float
    covariance_condition: float
    failed: bool
    failure_reason: str


@dataclass(frozen=True)
class GroupedGIVEstimate:
    """The two structural estimates that share one instrument space."""

    phi: EquationEstimate
    psi: EquationEstimate
    space: InstrumentSpace


def orthogonal_complement(columns: np.ndarray, *, tol: float = RANK_TOL) -> np.ndarray:
    """Return an orthonormal basis for ``col(columns)^perp``.

    ``columns`` has shape ``(m, q)``.  The SVD formulation also handles empty,
    collinear and saturated loading matrices, which is essential for the
    failure diagnostics in the Monte Carlo.
    """

    columns = np.asarray(columns, dtype=float)
    if columns.ndim != 2:
        raise ValueError("columns must be a two-dimensional array")
    m = columns.shape[0]
    if columns.shape[1] == 0:
        return np.eye(m)
    _, singular, vh = np.linalg.svd(columns.T, full_matrices=True)
    scale = singular[0] if singular.size else 1.0
    rank = int(np.sum(singular > tol * max(scale, 1.0)))
    return vh[rank:].T.copy()


def hahn_bic(eigvals: np.ndarray, sample_size: int) -> tuple[int, np.ndarray]:
    """Hahn et al. (2026), equations (47)--(48).

    The returned integer is the combined rank ``rank([b, B])``.  Thus one
    means no heterogeneous factor beyond the known price direction, and two
    means one such factor.
    """

    mu = np.sort(np.asarray(eigvals, dtype=float))
    if mu.ndim != 1 or mu.size < 1:
        raise ValueError("eigvals must contain the M-1 projected eigenvalues")
    if sample_size <= 1:
        raise ValueError("sample_size must exceed one")
    if mu[0] <= 0.0:
        raise ValueError("the projected sample covariance must be positive definite")

    m = mu.size + 1
    values = np.empty(m - 1)
    for j in range(1, m):
        noise = mu[: m - j]
        lack_of_fit = (
            sample_size
            * np.sum((noise - mu[0]) ** 2 / (2.0 * noise**2))
            / (m - j)
        )
        values[j - 1] = lack_of_fit + j * np.log(sample_size)
    return int(np.argmin(values) + 1), values


def known_instrument_space(
    price_direction: np.ndarray,
    loadings: np.ndarray,
    aggregate_direction: np.ndarray,
) -> InstrumentSpace:
    """Construct the finite-sample-exact oracle space."""

    b = np.asarray(price_direction, dtype=float).reshape(-1)
    d = np.asarray(aggregate_direction, dtype=float).reshape(-1)
    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim == 1:
        loadings = loadings[:, None]
    if loadings.shape[0] != b.size or d.size != b.size:
        raise ValueError("price, loading and aggregate dimensions disagree")
    span = np.column_stack((b, loadings))
    basis = orthogonal_complement(span)
    combined_rank = b.size - basis.shape[1]
    information = float(np.sum((basis.T @ d) ** 2))
    return InstrumentSpace(
        basis=basis,
        combined_rank=combined_rank,
        residual_dimension=basis.shape[1],
        information=information,
        eigvals=np.empty(0),
        noise_variance=float("nan"),
        eigengap=float("nan"),
        upsilon=np.zeros((b.size, b.size)),
        bic_values=np.empty(0),
        estimated=False,
    )


def estimated_instrument_space(
    grouped_outcomes: np.ndarray,
    price_direction: np.ndarray,
    aggregate_direction: np.ndarray,
    *,
    combined_rank: int | None = None,
) -> InstrumentSpace:
    """Estimate the loading-space complement using Hahn's eigenspace method.

    When ``combined_rank`` is omitted it is selected by Hahn's BIC.  Supplying
    the true rank isolates eigenspace estimation from rank-selection error.
    """

    x = np.asarray(grouped_outcomes, dtype=float)
    b = np.asarray(price_direction, dtype=float).reshape(-1)
    d = np.asarray(aggregate_direction, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != b.size or d.size != b.size:
        raise ValueError("grouped outcome, price and aggregate dimensions disagree")
    t, m = x.shape
    if not (1 <= m - 1 < t):
        raise ValueError("Hahn's fixed-M covariance step requires T > M-1 >= 1")
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError("non-finite data passed to the loading-space estimator")
    bnorm = float(b @ b)
    if bnorm <= 0.0:
        raise ValueError("the known price direction must be nonzero")

    q_b = orthogonal_complement(b[:, None])
    sigma_hat = x.T @ x / t
    restricted = q_b.T @ sigma_hat @ q_b
    restricted = 0.5 * (restricted + restricted.T)
    eigvals, eigvecs = np.linalg.eigh(restricted)
    selected, bic_values = hahn_bic(eigvals, t)
    rank = selected if combined_rank is None else int(combined_rank)
    if rank < 1 or rank > m:
        raise ValueError("combined_rank must lie between one and M")

    residual_dimension = m - rank
    if residual_dimension <= 0:
        basis = np.empty((m, 0))
        upsilon = np.zeros((m, m))
        gap = float("nan")
        information = 0.0
    else:
        a0 = eigvecs[:, :residual_dimension]
        basis = q_b @ a0
        information = float(np.sum((basis.T @ d) ** 2))
        factor_dimension = rank - 1
        if factor_dimension:
            a0_perp = eigvecs[:, residual_dimension:]
            factor_eigvals = eigvals[residual_dimension:]
            denominators = eigvals[0] - factor_eigvals
            # A zero denominator is a genuine spectral-identification failure.
            if np.any(np.abs(denominators) <= RANK_TOL * max(eigvals[-1], 1.0)):
                upsilon = np.full((m, m), np.nan)
            else:
                upsilon = (
                    q_b
                    @ a0_perp
                    @ np.diag(1.0 / denominators)
                    @ a0_perp.T
                    @ q_b.T
                )
            gap = float(eigvals[residual_dimension] - eigvals[0])
        else:
            upsilon = np.zeros((m, m))
            gap = float("nan")

    return InstrumentSpace(
        basis=basis,
        combined_rank=rank,
        residual_dimension=residual_dimension,
        information=information,
        eigvals=eigvals,
        noise_variance=float(eigvals[0]),
        eigengap=gap,
        upsilon=upsilon,
        bic_values=bic_values,
        estimated=True,
    )


def _failed_equation(reason: str) -> EquationEstimate:
    return EquationEstimate(
        estimate=float("nan"),
        se=float("nan"),
        naive_se=float("nan"),
        preliminary=float("nan"),
        j_stat=float("nan"),
        j_pvalue=float("nan"),
        j_df=0,
        derivative_strength=float("nan"),
        covariance_condition=float("inf"),
        failed=True,
        failure_reason=reason,
    )


def projected_weight(space: InstrumentSpace, aggregate_direction: np.ndarray) -> np.ndarray:
    """The aggregate-predicting instrument direction ``P d`` inside ``space``.

    Its squared norm is the space's information ``||A'd||^2``; with an oracle
    space this is the grouped information bound of the paper's proposition.
    """

    d = np.asarray(aggregate_direction, dtype=float).reshape(-1)
    a = space.basis
    if a.shape[0] != d.size:
        raise ValueError("aggregate direction and instrument space disagree")
    return a @ (a.T @ d)


def scalar_iv(
    instruments: np.ndarray,
    weight: np.ndarray,
    outcome: np.ndarray,
    regressor: np.ndarray,
    *,
    upsilon: np.ndarray | None = None,
) -> EquationEstimate:
    """Just-identified IV with the scalar instrument ``z_t = weight' x_t``.

    ``instruments`` is the ``T x K`` matrix whose rows are combined by
    ``weight``.  With ``upsilon=None`` the weight is treated as known and the
    reported standard error is the ordinary heteroskedasticity-robust one.
    With ``upsilon`` the weight is ``P_hat d`` for an estimated eigenspace and
    the standard error adds Hahn et al.'s generated-space term; ``naive_se``
    then reports the fixed-weight value for comparison.
    """

    x = np.asarray(instruments, dtype=float)
    w = np.asarray(weight, dtype=float).reshape(-1)
    o = np.asarray(outcome, dtype=float).reshape(-1)
    r = np.asarray(regressor, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != w.size or x.shape[0] != o.size or o.size != r.size:
        raise ValueError("instrument, weight, outcome and regressor dimensions disagree")
    t = x.shape[0]
    if upsilon is not None and not np.isfinite(upsilon).all():
        return _failed_equation("unseparated estimated eigenspace")
    if float(w @ w) <= INFO_TOL:
        return _failed_equation("zero residual-space information")

    z = x @ w
    derivative = float(np.mean(z * r))
    numerator = float(np.mean(z * o))
    scale = max(float(np.sqrt(np.mean(z**2) * np.mean(r**2))), 1e-300)
    if abs(derivative) <= INFO_TOL * scale:
        return _failed_equation("weak sample first stage")
    estimate = numerator / derivative
    residual = o - estimate * r
    # Contract the influence function with w before taking its variance.  This
    # is algebraically the same sandwich, without a K x K covariance matrix;
    # it also makes the scalar estimator practical for the full N=1,000 panel.
    oracle_if = z * residual
    oracle_if -= oracle_if.mean()
    oracle_var = float(np.mean(oracle_if**2))
    corrected_var = oracle_var
    if upsilon is not None:
        mean_xb = np.mean(x * residual[:, None], axis=0)
        direction = upsilon @ mean_xb
        correction = z * (x @ direction)
        corrected_if = oracle_if + correction - correction.mean()
        corrected_var = float(np.mean(corrected_if**2))
    variance = corrected_var
    if not np.isfinite(variance) or variance <= 0.0 or oracle_var <= 0.0:
        return _failed_equation("singular scalar moment variance")
    se = float(np.sqrt(variance / t) / abs(derivative))
    naive_se = float(np.sqrt(oracle_var / t) / abs(derivative))
    return EquationEstimate(
        estimate=float(estimate),
        se=se,
        naive_se=naive_se,
        preliminary=float(estimate),
        j_stat=float("nan"),
        j_pvalue=float("nan"),
        j_df=0,
        derivative_strength=float(abs(derivative) * np.sqrt(t / oracle_var)),
        covariance_condition=1.0,
        failed=False,
        failure_reason="",
    )


def joint_covariance_failure(sample_size: int, instrument_count: int) -> str:
    """Algorithm 1 requires an ordinary inverse of a centered 2K covariance."""
    if 2 * instrument_count >= sample_size:
        return "joint moment covariance singular: 2K exceeds T-1"
    return ""


def estimate_joint_giv(
    outcomes: np.ndarray,
    price: np.ndarray,
    aggregate: np.ndarray,
    price_direction: np.ndarray,
    aggregate_direction: np.ndarray,
    space: InstrumentSpace,
    *,
    diagnostics: bool = True,
) -> GroupedGIVEstimate:
    """Hahn et al. (2026), balanced-data Algorithm 1, Steps 2--6.

    Step 1's eigenspace and Upsilon are supplied in ``space``. Both equations
    use all K instruments and their cross-equation covariance. No ridge,
    pseudoinverse, separate-equation weight, or scalar-IV fallback is used.
    For whitened groups, replace 1 by the known direction b and the equal
    average by b'X/(b'b); the rest of the algorithm is unchanged.

    ``diagnostics=False`` omits the condition number and uses equivalent
    symmetric products and early contractions to reduce memory and work.
    """
    x = np.asarray(outcomes, dtype=float)
    p = np.asarray(price, dtype=float).reshape(-1)
    y = np.asarray(aggregate, dtype=float).reshape(-1)
    b = np.asarray(price_direction, dtype=float).reshape(-1)
    d = np.asarray(aggregate_direction, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape != (p.size, b.size) or y.size != p.size or d.size != b.size:
        raise ValueError("joint GMM data dimensions disagree")
    t, _ = x.shape
    k = space.residual_dimension

    def fail(reason: str) -> GroupedGIVEstimate:
        failed = _failed_equation(reason)
        return GroupedGIVEstimate(failed, failed, space)

    if k <= 0:
        return fail("no residual instrument dimension")
    reason = joint_covariance_failure(t, k)
    if reason:
        return fail(reason)
    if not np.isfinite(space.upsilon).all():
        return fail("unseparated estimated eigenspace")

    z = x @ space.basis
    reference = x @ b / float(b @ b)
    derivative = np.zeros((2 * k, 2))
    derivative[:k, 0] = z.T @ p / t
    derivative[k:, 1] = z.T @ y / t
    numerator = np.r_[z.T @ reference / t, z.T @ p / t]
    gram0 = derivative.T @ derivative
    try:
        preliminary = np.linalg.solve(gram0, derivative.T @ numerator)
    except np.linalg.LinAlgError:
        return fail("singular preliminary GMM first stage")

    # A.1 contracted with A before forming A.2 is exactly A' V_hat A.
    residuals = (reference - preliminary[0] * p, p - preliminary[1] * y)
    if diagnostics:
        corrected, naive = [], []
        for residual in residuals:
            raw = z * residual[:, None]
            naive.append(raw - raw.mean(axis=0))
            direction = space.upsilon @ (x.T @ residual / t)
            generated = z * (x @ direction)[:, None]
            influence = raw + generated
            corrected.append(influence - influence.mean(axis=0))
        influence = np.column_stack(corrected if space.estimated else naive)
    else:
        # Store the stacked influence once rather than keeping raw, naive,
        # generated and corrected copies of both T x K blocks.
        influence = np.empty((t, 2 * k))
        for j, residual in enumerate(residuals):
            adjusted = residual
            if space.estimated:
                direction = space.upsilon @ (x.T @ residual / t)
                adjusted = residual + x @ direction
            block = influence[:, j * k:(j + 1) * k]
            np.multiply(z, adjusted[:, None], out=block)
            block -= block.mean(axis=0)
    if diagnostics:
        covariance = influence.T @ influence / t
        covariance = (covariance + covariance.T) / 2
    else:
        # SYRK computes one triangle of the same symmetric cross-product.
        covariance = sla.blas.dsyrk(1.0 / t, np.asfortranarray(influence.T), lower=1)
        covariance += np.tril(covariance, -1).T
    try:
        factor = sla.cho_factor(covariance, lower=True)
        weighted_d = sla.cho_solve(factor, derivative)
        weighted_n = sla.cho_solve(factor, numerator)
        information = derivative.T @ weighted_d
        parameter_covariance = np.linalg.inv(information) / t
        estimate = np.linalg.solve(information, derivative.T @ weighted_n)
    except np.linalg.LinAlgError:
        return fail("singular joint moment covariance or GMM information")
    if not np.isfinite(parameter_covariance).all() or np.any(np.diag(parameter_covariance) <= 0):
        return fail("invalid joint GMM variance")
    gamma = np.linalg.solve(information, weighted_d.T)
    if diagnostics:
        naive_influence = np.column_stack(naive)
        naive_covariance = naive_influence.T @ naive_influence / t
        naive_variance = gamma @ naive_covariance @ gamma.T / t
    else:
        # Contract first: Gamma (Xi'Xi) Gamma' = (Xi Gamma')'(Xi Gamma').
        naive_scores = np.zeros((t, 2))
        for j, residual in enumerate(residuals):
            naive_scores += (z @ gamma[:, j * k:(j + 1) * k].T) * residual[:, None]
        naive_scores -= naive_scores.mean(axis=0)
        naive_variance = naive_scores.T @ naive_scores / t**2
    moments = numerator - derivative @ estimate
    j_stat = float(t * moments @ sla.cho_solve(factor, moments))
    j_df = 2 * k - 2
    j_pvalue = float(chi2.sf(j_stat, j_df)) if j_df > 0 else float("nan")
    # This optional SVD is a reporting diagnostic, unused by the estimator.
    condition = float(np.linalg.cond(covariance)) if diagnostics else float("nan")
    estimates = [
        EquationEstimate(
            estimate=float(estimate[j]), se=float(np.sqrt(parameter_covariance[j, j])),
            naive_se=float(np.sqrt(max(naive_variance[j, j], 0))),
            preliminary=float(preliminary[j]), j_stat=j_stat, j_pvalue=j_pvalue,
            j_df=j_df, derivative_strength=float(np.sqrt(t * information[j, j])),
            covariance_condition=condition, failed=False, failure_reason="",
        ) for j in range(2)
    ]
    return GroupedGIVEstimate(estimates[0], estimates[1], space)
