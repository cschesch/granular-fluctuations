#!/usr/bin/env python3
"""Matched Foerster--Sarte--Watson decomposition and MGP comparison.

The published Table 7 decomposition and our covariance ceiling are otherwise
easy to compare on different objects.  This script puts them on the same
recovered epsilon innovations, sample windows, propagation map, aggregate
weights, and model-implied variance denominator.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mgp_core import (
    RUN_OUTPUT_CACHE,
    build_lpt_grouped_data,
    choose_solver,
    sample_cov,
    solve_grouped_mgp_dual,
)
from applications.networks import network_variance_matrix, reported_dynamic_share
from applications.foerster.run_foerster_epsilon import (
    DEFAULT_CAP_RHO_CSV,
    DEFAULT_EPSILON_CSV,
    DEFAULT_PII_A_CSV,
    DEFAULT_WEIGHTS_CSV,
    DEFAULT_XI_CSV,
    load_foerster_shocks_and_weights,
    read_network_matrix,
)


from project_paths import ROOT
OUT = ROOT / "output" / "table_foerster_matched_tabular.tex"
WINDOWS = (
    ("1972Q2", "1983Q4", 0.19),
    ("1984Q1", "2007Q4", 0.50),
    ("1972Q2", "2007Q4", None),
)


def export_tables(rows):
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Sample & Foerster et al. & Grouped ceiling & 95\% UCL \\",
        r"\midrule[\heavyrulewidth]",
    ]
    for row in rows:
        label = row["period"]
        if pd.isna(row["published_idiosyncratic_share"]):
            lines.append(r"\midrule[\lightrulewidth]")
            label = "Full sample"
        estimates = [row.get(key) for key in (
            "fsw_idiosyncratic_share", "matched_mgp_ceiling", "moving_block_gradient_ucl95",
        )]
        cells = [f"{100 * value:.1f}" if pd.notna(value) else "" for value in estimates]
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUT.write_text("\n".join(lines))


def two_factor_decomposition(eps: np.ndarray, network_v: np.ndarray) -> dict[str, object]:
    """Reproduce the two-factor covariance construction in factor_io.gss."""
    x = np.asarray(eps, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("Foerster innovations contain a zero-variance sector.")

    # FSW estimate PCs from standardized innovations, then regress the
    # unstandardized innovations on those factors.
    x_std = x / std
    u, _, _ = np.linalg.svd(x_std, full_matrices=False)
    factors = u[:, :2] * np.sqrt(x.shape[0])
    loadings = x.T @ factors @ np.linalg.inv(factors.T @ factors)
    residuals = x - factors @ loadings.T

    factor_cov = loadings @ np.diag(factors.std(axis=0, ddof=1) ** 2) @ loadings.T
    idio_cov = np.diag(residuals.std(axis=0, ddof=1) ** 2)
    factor_var = float(np.sum(network_v * factor_cov))
    idio_var = float(np.sum(network_v * idio_cov))
    total_var = factor_var + idio_var
    return {
        "model_covariance": factor_cov + idio_cov,
        "factor_variance": factor_var,
        "idiosyncratic_variance": idio_var,
        "model_variance": total_var,
        "fsw_idiosyncratic_share": idio_var / total_var,
    }


def matched_ceiling(
    eps: np.ndarray,
    network_v: np.ndarray,
    denominator: float,
    solver: str,
    m: int = 10,
) -> dict[str, float]:
    influences = np.sqrt(np.maximum(np.diag(network_v), 0.0))
    scale = float(influences.sum() ** 2)
    shares = influences / influences.sum()
    grouped, _, _, _ = build_lpt_grouped_data(eps.T, shares, m)
    mgp = solve_grouped_mgp_dual(sample_cov(grouped), solver)
    numerator = scale * float(mgp["value"])
    return {
        "mgp_numerator": numerator,
        "matched_mgp_ceiling": reported_dynamic_share(numerator / denominator),
        "matched_mgp_ceiling_uncapped": numerator / denominator,
    }


def fitted_covariance_ceiling(
    sigma_eps: np.ndarray,
    network_v: np.ndarray,
    denominator: float,
    solver: str,
    m: int = 10,
) -> float:
    """MGP ceiling evaluated mechanically on FSW's fitted factor covariance."""
    influences = np.sqrt(np.maximum(np.diag(network_v), 0.0))
    scale = float(influences.sum() ** 2)
    shares = influences / influences.sum()
    _, _, _, grouping = build_lpt_grouped_data(
        np.zeros((len(shares), 2)), shares, m
    )
    sigma_grouped = grouping @ sigma_eps @ grouping.T
    numerator = scale * float(solve_grouped_mgp_dual(sigma_grouped, solver)["value"])
    return reported_dynamic_share(numerator / denominator)


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    eps, mean_weights = load_foerster_shocks_and_weights(DEFAULT_EPSILON_CSV, DEFAULT_WEIGHTS_CSV)
    n = eps.shape[1]
    pii_a = read_network_matrix(DEFAULT_PII_A_CSV, n, "Pi_a")
    xi = read_network_matrix(DEFAULT_XI_CSV, n, "Xi")
    cap_rho = read_network_matrix(DEFAULT_CAP_RHO_CSV, n, "cap_rho")
    network_v = network_variance_matrix(mean_weights.to_numpy(float), pii_a, xi, cap_rho)
    return eps, network_v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", action="store_true")
    if parser.parse_args().blank:
        export_tables([
            {"period": f"{start}--{end}", "published_idiosyncratic_share": published,
             "fsw_idiosyncratic_share": published}
            for start, end, published in WINDOWS
        ])
        return
    eps, network_v = load_inputs()
    solver = choose_solver("auto")
    rows = []
    for start, end, published in WINDOWS:
        sample = eps.loc[start:end].to_numpy(float)
        dec = two_factor_decomposition(sample, network_v)
        model_covariance = np.asarray(dec.pop("model_covariance"), dtype=float)
        ceiling = matched_ceiling(sample, network_v, dec["model_variance"], solver)
        rows.append(
            {
                "period": f"{start}--{end}",
                "T": sample.shape[0],
                "published_idiosyncratic_share": published,
                **dec,
                **ceiling,
                "fitted_covariance_ceiling": fitted_covariance_ceiling(
                    model_covariance,
                    network_v,
                    float(dec["model_variance"]),
                    solver,
                ),
            }
        )

    for row in rows:
        published = row["published_idiosyncratic_share"]
        if published is not None and round(row["fsw_idiosyncratic_share"], 2) != published:
            raise RuntimeError(
                "Matched FSW reconstruction misses the published idiosyncratic "
                f"share in {row['period']}: "
                f"{row['fsw_idiosyncratic_share']:.4f} versus "
                f"{published:.2f}."
            )
        if abs(
            float(row["fitted_covariance_ceiling"])
            - float(row["fsw_idiosyncratic_share"])
        ) > 1e-6:
            raise RuntimeError(
                "MGP does not recover the diagonal share of FSW's fitted "
                f"factor covariance in {row['period']}."
            )

    result = pd.DataFrame(rows)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    result.to_pickle(RUN_OUTPUT_CACHE / "foerster_matched_comparison.pkl")

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
