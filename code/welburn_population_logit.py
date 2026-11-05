"""Exact full-dyad Bernoulli logit for the Welburn equation-(5.1) audit.

This is an *alternative*, explicitly zero-coded experiment: every ordered pair
of distinct firms is an observation; disclosed links are one and all remaining
pairs are zero.  The report's treatment of unknown versus insignificant flows
is not documented sufficiently to establish that this is its estimation sample.
The module does not replace the headline network completion.

Only three standardized revenue regressors and one intercept for each ordered
one-digit SIC pair are fitted.  Blocks accumulate the exact likelihood, score,
and Hessian without storing a full dyad design.  A Schur complement eliminates
the sector intercepts from each Newton step, leaving a three-by-three solve.
No negative subsampling, regularization, pseudo-counts, or link-count target is
used.  Sector pairs with no successes have the extended MLE intercept -inf
(similarly +inf for cells with no failures), with their boundary status exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class PopulationLogitModel:
    coefficients_: np.ndarray
    sector_intercepts_: np.ndarray
    feature_mean_: np.ndarray
    feature_scale_: np.ndarray
    firm_ids_: np.ndarray
    sales_: np.ndarray
    sectors_: np.ndarray
    diagnostics: dict

    def score(self, suppliers, buyers) -> np.ndarray:
        """Linear predictors for aligned supplier/customer identifier arrays."""
        suppliers = np.asarray(suppliers, dtype=str).reshape(-1)
        buyers = np.asarray(buyers, dtype=str).reshape(-1)
        if suppliers.shape != buyers.shape:
            raise ValueError("Supplier and buyer arrays must have the same shape.")
        index = pd.Index(self.firm_ids_)
        left = index.get_indexer(suppliers)
        right = index.get_indexer(buyers)
        if np.any(left < 0) or np.any(right < 0):
            raise ValueError("Scoring pairs contain firms outside the fitted universe.")
        sale_left, sale_right = self.sales_[left], self.sales_[right]
        continuous = np.column_stack((sale_left, sale_right, sale_left * sale_right))
        return (
            ((continuous - self.feature_mean_) / self.feature_scale_) @ self.coefficients_
            + self.sector_intercepts_[self.sectors_[left], self.sectors_[right]]
        )

    def probability(self, suppliers, buyers) -> np.ndarray:
        return expit(self.score(suppliers, buyers))

    def predict_proba(self, suppliers, buyers) -> np.ndarray:
        probability = self.probability(suppliers, buyers)
        return np.column_stack((1.0 - probability, probability))


def _positive_pairs(positives, ids: pd.Index) -> np.ndarray:
    if isinstance(positives, pd.DataFrame):
        names = ["gvkey", "customer_gvkey"]
        array = positives[names].to_numpy(str) if set(names) <= set(positives) else positives.to_numpy(str)
    else:
        array = np.asarray(list(positives), dtype=str)
    if array.size == 0:
        return np.empty((0, 2), dtype=int)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Positive pairs must contain two firm-identifier columns.")
    array = np.unique(array, axis=0)
    locations = np.column_stack((ids.get_indexer(array[:, 0]), ids.get_indexer(array[:, 1])))
    if np.any(locations < 0):
        raise ValueError("Positive pairs contain firms outside the financial universe.")
    if np.any(locations[:, 0] == locations[:, 1]):
        raise ValueError("Self-pairs are excluded from the full-dyad model.")
    return locations


def _standardization(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full ordered off-diagonal moments, computed analytically in O(N)."""
    s = sales.astype(np.longdouble)
    n = len(s)
    total = np.longdouble(n * (n - 1))
    sum1, sum2, sum4 = np.sum(s), np.sum(s**2), np.sum(s**4)
    mean_sale = sum1 / n
    mean_product = (sum1**2 - sum2) / total
    variance_sale = sum2 / n - mean_sale**2
    variance_product = (sum2**2 - sum4) / total - mean_product**2
    mean = np.array([mean_sale, mean_sale, mean_product], dtype=float)
    variance = np.maximum(np.array([variance_sale, variance_sale, variance_product], dtype=float), 0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean, scale


def fit(
    means: pd.DataFrame,
    positives,
    *,
    block_size: int = 128,
    max_iter: int = 100,
    score_tolerance: float = 1e-7,
    verbose: bool = False,
) -> PopulationLogitModel:
    """Fit the exact unpenalized population logit on all N(N-1) ordered pairs.

    ``means`` has a unique firm-ID index and columns ``sale`` and ``sic``;
    ``positives`` is an iterable of (supplier, buyer) IDs or a two-column frame.
    ``score_tolerance`` bounds the largest absolute likelihood-score component.
    The returned diagnostics distinguish a converged finite fit from extended
    sector-cell MLEs and convergence failure.  A failed fit raises rather than
    silently supplying a network with unconverged probabilities.
    """
    start_time = time.monotonic()
    ids = pd.Index(means.index.astype(str))
    if ids.has_duplicates or len(ids) < 2:
        raise ValueError("At least two firms with unique identifiers are required.")
    sales = means["sale"].to_numpy(float)
    sic = means["sic"].to_numpy(float)
    if not np.isfinite(sales).all() or np.any(sales < 0) or not np.isfinite(sic).all():
        raise ValueError("Revenue must be finite/nonnegative and SIC must be finite.")
    if block_size < 1 or max_iter < 1 or score_tolerance <= 0:
        raise ValueError("Block size, maximum iterations and score tolerance must be positive.")
    sectors = np.clip(sic.astype(int) // 1000, 0, 9)
    pairs = _positive_pairs(positives, ids)
    n, observations = len(ids), len(ids) * (len(ids) - 1)
    mean, scale = _standardization(sales)
    standardized_sale = (sales - mean[0]) / scale[0]
    positions = [np.flatnonzero(sectors == k) for k in range(10)]
    sector_n = np.array([len(x) for x in positions], dtype=np.int64)
    cell_n = np.outer(sector_n, sector_n)
    cell_n[np.diag_indices(10)] -= sector_n
    cell_y = np.zeros((10, 10), dtype=np.int64)
    if len(pairs):
        np.add.at(cell_y, (sectors[pairs[:, 0]], sectors[pairs[:, 1]]), 1)
    active = (cell_y > 0) & (cell_y < cell_n)
    active_cells = list(zip(*np.where(active)))
    active_y = cell_y[active].astype(float)
    cell_intercept = np.full((10, 10), -np.inf)
    cell_intercept[(cell_n > 0) & (cell_y == cell_n)] = np.inf
    alpha = np.log(active_y / (cell_n[active] - active_y))
    beta = np.zeros(3)
    positive_active = active[sectors[pairs[:, 0]], sectors[pairs[:, 1]]] if len(pairs) else np.zeros(0, bool)
    fit_pairs = pairs[positive_active]
    if len(fit_pairs):
        sl, sr = sales[fit_pairs[:, 0]], sales[fit_pairs[:, 1]]
        positive_x_sum = ((np.column_stack((sl, sr, sl * sr)) - mean) / scale).sum(axis=0)
    else:
        positive_x_sum = np.zeros(3)

    def evaluate(beta_value, alpha_value, *, hessian=True, threshold=False):
        loss = -float(beta_value @ positive_x_sum + alpha_value @ active_y)
        g_beta = -positive_x_sum.copy()
        g_alpha = -active_y.copy()
        h_beta = np.zeros((3, 3))
        h_cross = np.zeros((len(active_cells), 3))
        h_alpha = np.zeros(len(active_cells))
        threshold_count = int(cell_n[(cell_n > 0) & (cell_y == cell_n)].sum())
        for k, (supplier_sector, buyer_sector) in enumerate(active_cells):
            left, right = positions[supplier_sector], positions[buyer_sector]
            buyer_sales = sales[right]
            z2 = standardized_sale[right]
            for first in range(0, len(left), block_size):
                chosen = left[first:first + block_size]
                z1 = standardized_sale[chosen]
                z3 = (sales[chosen, None] * buyer_sales[None, :] - mean[2]) / scale[2]
                eta = alpha_value[k] + beta_value[0] * z1[:, None] + beta_value[1] * z2[None, :] + beta_value[2] * z3
                if supplier_sector == buyer_sector:
                    eta[np.arange(len(chosen)), np.arange(first, first + len(chosen))] = -np.inf
                loss += float(np.logaddexp(0.0, eta).sum())
                if not hessian and not threshold:
                    continue
                probability = expit(eta)
                if threshold:
                    threshold_count += int(np.count_nonzero(eta > 0))
                if not hessian:
                    continue
                pr, pc = probability.sum(axis=1), probability.sum(axis=0)
                g_beta += [pr @ z1, pc @ z2, np.sum(probability * z3)]
                g_alpha[k] += float(pr.sum())
                q = probability * (1.0 - probability)
                qr, qc = q.sum(axis=1), q.sum(axis=0)
                qz3 = q * z3
                h_beta[0, 0] += float(qr @ z1**2)
                h_beta[1, 1] += float(qc @ z2**2)
                h_beta[2, 2] += float(np.sum(qz3 * z3))
                h_beta[0, 1] += float(z1 @ q @ z2)
                h_beta[0, 2] += float(z1 @ qz3.sum(axis=1))
                h_beta[1, 2] += float(z2 @ qz3.sum(axis=0))
                h_cross[k] += [qr @ z1, qc @ z2, np.sum(qz3)]
                h_alpha[k] += float(qr.sum())
        h_beta[1, 0], h_beta[2, 0], h_beta[2, 1] = h_beta[0, 1], h_beta[0, 2], h_beta[1, 2]
        return loss, g_beta, g_alpha, h_beta, h_cross, h_alpha, threshold_count

    iterations, line_search_steps, converged = 0, 0, False
    history = []
    for iteration in range(max_iter):
        result = evaluate(beta, alpha)
        loss, gb, ga, hb, hc, ha, _ = result
        score_max = float(max(np.max(np.abs(gb)), np.max(np.abs(ga), initial=0.0)))
        history.append({"iteration": iteration, "negative_log_likelihood": loss, "score_max_abs": score_max})
        if verbose:
            print(f"population logit iteration {iteration}: nll={loss:.10g}, max score={score_max:.5g}", flush=True)
        if score_max <= score_tolerance:
            converged = True
            break
        if np.any(ha <= 0):
            raise RuntimeError("Population-logit information vanished; possible continuous-regressor separation.")
        schur = hb - (hc.T / ha) @ hc
        rhs = gb - hc.T @ (ga / ha)
        # Rank-deficient revenue features (e.g. constant sales) have no unique
        # slope representation; the minimum-norm Newton direction still fits
        # the unique identifiable probabilities, without adding a penalty.
        delta_beta = np.linalg.lstsq(schur, rhs, rcond=1e-13)[0]
        delta_alpha = (ga - hc @ delta_beta) / ha
        descent = float(gb @ delta_beta + ga @ delta_alpha)
        if not np.isfinite(descent) or descent <= 0:
            raise RuntimeError(f"Population-logit Newton step is not descending (score={score_max:g}).")
        step = 1.0
        accepted = False
        for _ in range(45):
            candidate_beta = beta - step * delta_beta
            candidate_alpha = alpha - step * delta_alpha
            candidate_loss = evaluate(candidate_beta, candidate_alpha, hessian=False)[0]
            # Permit floating-point noise near the maximum; the following
            # iteration must still pass the explicit score convergence test.
            roundoff = 8 * np.finfo(float).eps * max(1.0, abs(loss))
            if candidate_loss <= loss - 1e-4 * step * descent + roundoff:
                beta, alpha = candidate_beta, candidate_alpha
                accepted = True
                break
            step *= 0.5
            line_search_steps += 1
        if not accepted:
            raise RuntimeError(f"Population-logit line search failed with score={score_max:g}.")
        iterations = iteration + 1
    # Re-evaluate after the final update, including thresholded probabilities.
    loss, gb, ga, hb, hc, ha, above_half = evaluate(beta, alpha, threshold=True)
    score_max = float(max(np.max(np.abs(gb)), np.max(np.abs(ga), initial=0.0)))
    converged = converged or score_max <= score_tolerance
    if not converged:
        raise RuntimeError(f"Population logit failed to converge in {max_iter} iterations: score={score_max:g}.")
    cell_intercept[active] = alpha
    model = PopulationLogitModel(beta, cell_intercept, mean, scale, ids.to_numpy(str), sales, sectors, {})
    positive_probabilities = model.probability(ids.to_numpy(str)[pairs[:, 0]], ids.to_numpy(str)[pairs[:, 1]])
    known_above_half = int(np.count_nonzero(positive_probabilities > 0.5))
    count_residual = float(ga.sum())
    model.diagnostics = {
        "specification": "supplier revenue + buyer revenue + revenue product + full ordered one-digit SIC-pair intercepts",
        "estimation_sample_assumption": "all ordered nonself financial-firm dyads; disclosed=1, all other pairs=0",
        "n_firms": n,
        "n_dyads": observations,
        "n_positive_dyads": int(len(pairs)),
        "n_negative_dyads": observations - int(len(pairs)),
        "n_sector_pair_cells": int(np.count_nonzero(cell_n)),
        "n_interior_sector_pair_cells": len(active_cells),
        "n_zero_success_boundary_cells": int(np.count_nonzero((cell_n > 0) & (cell_y == 0))),
        "n_zero_failure_boundary_cells": int(np.count_nonzero((cell_n > 0) & (cell_y == cell_n))),
        "finite_sector_intercept_mle": bool(np.all((cell_y[cell_n > 0] > 0) & (cell_y[cell_n > 0] < cell_n[cell_n > 0]))),
        "converged": bool(converged),
        "iterations": iterations,
        "line_search_halvings": line_search_steps,
        "negative_log_likelihood": float(loss),
        "score_max_abs": score_max,
        "continuous_score_residual": (-gb).tolist(),
        "sector_score_max_abs": float(np.max(np.abs(ga), initial=0.0)),
        "sum_probability": float(len(pairs) + count_residual),
        "sum_y": int(len(pairs)),
        "sum_probability_minus_sum_y": count_residual,
        "probability_above_half_all_pairs": int(above_half),
        "probability_above_half_known_pairs": known_above_half,
        "probability_above_half_missing_pairs": int(above_half - known_above_half),
        "sum_probability_known_pairs": float(positive_probabilities.sum()),
        "sum_probability_missing_pairs": float(len(pairs) + count_residual - positive_probabilities.sum()),
        "sector_pair_observation_counts": cell_n.tolist(),
        "sector_pair_positive_counts": cell_y.tolist(),
        "elapsed_seconds": time.monotonic() - start_time,
        "optimizer_history": history,
    }
    return model
