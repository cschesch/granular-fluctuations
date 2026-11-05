"""Volatility paths at fixed covariance and changing weights."""

import numpy as np
import pandas as pd

from mgp_core import lpt_buckets, sample_cov, solve_grouped_mgp_dual, solve_primal_mtfa


def grouped_covariance(sigma_unit: np.ndarray, weights: np.ndarray,
                       buckets: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """(Sigma-tilde_t, S-tilde_t) for one weight vector under a fixed partition.

    Sigma-tilde = G Sigma G' with G[m, i] = S_i 1{i in B_m}: the frozen unit
    covariance is grouped directly, so nothing is re-estimated year by year.
    """
    g = np.zeros((len(buckets), weights.size))
    for m, bucket in enumerate(buckets):
        g[m, bucket] = weights[bucket]
    loads = np.array([float(np.sum(weights[b] ** 2)) for b in buckets])
    return g @ sigma_unit @ g.T, loads


def ceiling_of(sigma_grouped: np.ndarray, solver: str) -> tuple[float, float, float]:
    """(level, aggregate variance, share) of the grouped MGP at one Sigma-tilde."""
    one = np.ones(sigma_grouped.shape[0])
    level = float(solve_grouped_mgp_dual(sigma_grouped, solver)["value"])
    variance = float(one @ sigma_grouped @ one)
    return level, variance, level / variance


def normalized_idiosyncratic_loads(sigma_grouped: np.ndarray, loads: np.ndarray,
                                   solver: str) -> np.ndarray:
    """u*_m = E*_m / S-tilde_m at the grouped MGP optimum: the implied idiosyncratic
    variance per unit of squared weight, which is what carries across years."""
    common, _ = solve_primal_mtfa(sigma_grouped, solver)
    diagonal = np.diag(sigma_grouped - 0.5 * (common + common.T))
    return np.clip(diagonal, 0.0, None) / loads


def admissible_scale(sigma_grouped: np.ndarray, idiosyncratic: np.ndarray) -> float:
    """max{lambda : Sigma-tilde - lambda Diag(e) >= 0}, in closed form.

    A frozen decomposition is essentially never admissible at a different weight
    vector, and not by a little: the MGP optimum sits exactly ON the boundary of
    the PSD cone -- C* is rank-deficient -- so any first-order move in Sigma-tilde
    pushes Sigma-tilde - Diag(e) indefinite. Carrying u*(t0) forward therefore
    needs the one scalar that restores feasibility, which is the largest lambda
    with Sigma-tilde - lambda Diag(e) >= 0, i.e. the reciprocal of the largest
    eigenvalue of Diag(sqrt e) Sigma-tilde^{-1} Diag(sqrt e). It is capped at one:
    the year-t0 decomposition is carried forward, not re-optimized.
    """
    root = np.sqrt(np.clip(idiosyncratic, 0.0, None))
    pencil = root[:, None] * np.linalg.solve(sigma_grouped, np.diag(root))
    top = float(np.linalg.eigvalsh(0.5 * (pencil + pencil.T)).max())
    return min(1.0, 1.0 / top) if top > 0 else 1.0


def build_paths(growth, weights, m: int, solver: str, n_references: int = 3) -> pd.DataFrame:
    """The re-optimized ceiling path, the dynamic-LPT check, and one frozen-
    decomposition path per reference year."""
    sigma_unit = sample_cov(growth.T)
    years = weights.index.to_numpy(dtype=int)
    mean_weights = weights.mean(axis=0).to_numpy(dtype=float)
    fixed_buckets, _ = lpt_buckets(mean_weights ** 2, m)

    grouped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, float]] = []
    for year, row in weights.iterrows():
        w = row.to_numpy(dtype=float)
        sigma_grouped, loads = grouped_covariance(sigma_unit, w, fixed_buckets)
        grouped[int(year)] = (sigma_grouped, loads)
        level, variance, share = ceiling_of(sigma_grouped, solver)
        dynamic_buckets, _ = lpt_buckets(w ** 2, m)
        dyn_sigma, _ = grouped_covariance(sigma_unit, w, dynamic_buckets)
        _, _, dyn_share = ceiling_of(dyn_sigma, solver)
        rows.append({
            "year": int(year),
            "ceiling_level": level,
            "ceiling_volatility": float(np.sqrt(max(level, 0.0))),
            "ceiling_share": share,
            "dynamic_lpt_share": dyn_share,
            "aggregate_variance": variance,
            "aggregate_volatility": float(np.sqrt(max(variance, 0.0))),
            # The Carvalho--Gabaix diagonal benchmark at the same weights.
            "diagonal_level": float(np.sum(w ** 2 * np.diag(sigma_unit))),
            "diagonal_volatility": float(np.sqrt(np.sum(w ** 2 * np.diag(sigma_unit)))),
            "weight_sum": float(w.sum()),
            "sqrt_hhi": float(np.sqrt(np.sum((w / w.sum()) ** 2))),
            "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        })
    paths = pd.DataFrame(rows).set_index("year").sort_index()

    positions = np.linspace(0, len(years) - 1, n_references).round().astype(int)
    references = [int(years[p]) for p in dict.fromkeys(positions)]
    one = np.ones(m)
    for reference in references:
        sigma_ref, loads_ref = grouped[reference]
        u = normalized_idiosyncratic_loads(sigma_ref, loads_ref, solver)
        dual = np.asarray(solve_grouped_mgp_dual(sigma_ref, solver)["Y"], dtype=float)
        level, share, scale, dual_share = [], [], [], []
        for year in paths.index:
            sigma_grouped, loads = grouped[int(year)]
            lam = admissible_scale(sigma_grouped, u * loads)
            implied = float(u @ loads)
            variance = float(one @ sigma_grouped @ one)
            level.append(implied)
            share.append(implied / variance)
            scale.append(lam)
            # Weak duality: any Y >= 0 with unit diagonal upper-bounds the ceiling
            # at EVERY weight vector, so the year-t0 dual certificate traces a
            # curve the re-optimized ceiling can never cross.
            dual_share.append(float(np.sum(dual * sigma_grouped)) / variance)
        paths[f"frozen{reference}_level"] = level
        paths[f"frozen{reference}_volatility"] = np.sqrt(np.clip(level, 0.0, None))
        paths[f"frozen{reference}_share"] = share
        paths[f"frozen{reference}_scale"] = scale
        # Shrinking the frozen decomposition by ``scale`` restores admissibility,
        # so this is the largest granular share the year-t covariance supports in
        # the year-t0 direction: the feasible version of the line above.
        paths[f"frozen{reference}_feasible_share"] = np.asarray(share) * np.asarray(scale)
        paths[f"frozen{reference}_dual_share"] = dual_share

    return paths
