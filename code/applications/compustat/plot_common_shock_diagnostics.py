#!/usr/bin/env python3
"""Common-shock diagnostics for quarterly Compustat nominal sales growth.

Builds the PCA diagnostic for the grouped observations that illustrates
factor-loading averaging in the text.
"""

from __future__ import annotations

import argparse

from mgp_core import OUTPUT_DIR, RUN_OUTPUT_CACHE, build_lpt_grouped_data
from wrds_inputs import wrds_path, ANALYSIS_LAST_QUARTER

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from applications.compustat import run_compustat_quarterly as quarterly


def load_sales_panel() -> quarterly.PanelData:
    args = argparse.Namespace(
        compustat_csv=wrds_path("compustat_quarterly_sales"),
        last_date=ANALYSIS_LAST_QUARTER,
        T=100,
        coverage_threshold=0.8,
        use_panel_cache=True,
        refresh_panel_cache=False,
    )
    return quarterly.load_compustat_panel(args)


def pca_diagnostic(
    panel: quarterly.PanelData, m: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA of the normalized LPT bucket averages, plus bucket-level diagnostics."""
    weights = panel.sizes.to_numpy(float)
    weights /= weights.sum()
    firm_growth = panel.wide.to_numpy(float)
    grouped, buckets, _, _ = build_lpt_grouped_data(firm_growth, weights, m)
    bucket_shares = np.array([weights[bucket].sum() for bucket in buckets])

    # The PCA is run on the grouped observations themselves, which are what
    # enter the MGP; normalizing by bucket size would change the covariance.
    x_tilde = grouped - grouped.mean(axis=0, keepdims=True)
    aggregate = x_tilde.sum(axis=1)                      # y_t = 1' x_t

    u, singular_values, vt = np.linalg.svd(x_tilde, full_matrices=False)
    scores = u * singular_values[None, :]
    eigenvalues = singular_values**2 / (x_tilde.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()          # lambda_j / Tr(Sigma)
    correlations = np.array([
        abs(np.corrcoef(scores[:, j], aggregate)[0, 1])
        for j in range(scores.shape[1])
    ])
    # Share of AGGREGATE variance rather than of the trace. Since y_t = 1'x_t,
    # y_t = sum_j (1'v_j) score_tj with scores orthogonal in sample, so
    # 1'Sigma 1 = Var(y) = sum_j lambda_j (1'v_j)^2 exactly.
    ones = np.ones(m)
    aggregate_contribution = eigenvalues * (vt @ ones) ** 2
    aggregate_share = aggregate_contribution / aggregate_contribution.sum()
    assert np.isclose(aggregate_contribution.sum(), aggregate.var(ddof=1))

    # Normalized averages are kept only for the loading-averaging diagnostics
    # below, which is what the law-of-large-numbers argument is about.
    bucket_averages = grouped / bucket_shares[None, :]
    bucket_averages -= bucket_averages.mean(axis=0, keepdims=True)
    components = pd.DataFrame({
        "component": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues,
        "variance_share": explained,
        "cumulative_variance_share": np.cumsum(explained),
        "aggregate_variance_share": aggregate_share,
        "cumulative_aggregate_variance_share": np.cumsum(aggregate_share),
        "abs_correlation_with_sales_weighted_aggregate": correlations,
        "M": m,
        "N": panel.wide.shape[0],
        "T": panel.wide.shape[1],
    })

    # N_eff is the effective number of firms behind a bucket's normalized
    # loading; the text reads the co-movement of bucket averages against it.
    n_eff = np.array([
        bucket_shares[j] ** 2 / (weights[bucket] ** 2).sum()
        for j, bucket in enumerate(buckets)
    ])
    corr = np.corrcoef(bucket_averages.T)
    bucket_table = pd.DataFrame({
        "bucket": np.arange(1, m + 1),
        "n_firms": [len(bucket) for bucket in buckets],
        "sales_share": bucket_shares,
        "N_eff": n_eff,
        "sd": bucket_averages.std(axis=0, ddof=1),
        "abs_correlation_with_aggregate": [
            abs(np.corrcoef(bucket_averages[:, j], aggregate)[0, 1]) for j in range(m)
        ],
        "mean_correlation_with_other_buckets": [
            (corr[j].sum() - 1.0) / (m - 1) for j in range(m)
        ],
    })
    # LPT isolates the largest firms in singleton buckets; the rest are
    # diversified. Splitting the correlation matrix on that line isolates the
    # averaging effect from the residual heterogeneity of the undiversified
    # buckets.
    singleton = n_eff < 2.0
    def block_mean(rows: np.ndarray, cols: np.ndarray, diagonal: bool) -> float:
        i, j = np.flatnonzero(rows), np.flatnonzero(cols)
        block = corr[np.ix_(i, j)]
        if not diagonal:
            return float(block.mean())
        return float((block.sum() - len(i)) / (len(i) * (len(i) - 1)))

    size_totals = bucket_shares / np.linalg.norm(bucket_shares)
    ones_unit = np.ones(m) / np.sqrt(m)
    blocks = pd.DataFrame([{
        "pc1_cosine_with_ones": float(abs(vt[0] @ ones_unit)),
        "pc1_cosine_with_size_totals": float(abs(vt[0] @ size_totals)),
        "n_singleton_buckets": int(singleton.sum()),
        "n_diversified_buckets": int((~singleton).sum()),
        "min_N_eff_diversified": float(n_eff[~singleton].min()),
        "max_N_eff_diversified": float(n_eff[~singleton].max()),
        "mean_corr_within_diversified": block_mean(~singleton, ~singleton, True),
        "mean_corr_within_singleton": block_mean(singleton, singleton, True),
        "mean_corr_singleton_diversified": block_mean(singleton, ~singleton, False),
    }])
    return components, bucket_table, blocks


def plot_pca(d: pd.DataFrame) -> None:
    """Total- and aggregate-variance decompositions on one 0-100% axis."""
    x = d["component"].to_numpy(int)
    total = 100 * d["variance_share"].to_numpy(float)
    aggregate = 100 * d["aggregate_variance_share"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.plot(
        x, total, linestyle="none", marker="o", markersize=6.0,
        color="#1f77b4", zorder=3, label="Share of total variance",
    )
    ax.plot(
        x, aggregate, linestyle="none", marker="D", markersize=6.0,
        color="#e8820c", zorder=4, label="Share of aggregate variance",
    )
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor="0.75", facecolor="white",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sales_factor_averaging.pdf", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_sales_panel()
    diagnostic, buckets, blocks = pca_diagnostic(panel)
    diagnostic.to_csv(RUN_OUTPUT_CACHE / "sales_factor_averaging_data.csv", index=False)
    buckets.to_csv(RUN_OUTPUT_CACHE / "sales_factor_averaging_buckets.csv", index=False)
    blocks.to_csv(RUN_OUTPUT_CACHE / "sales_factor_averaging_blocks.csv", index=False)
    plot_pca(diagnostic)
    print(
        f"[PCA] N={int(diagnostic.loc[0, 'N'])}, T={int(diagnostic.loc[0, 'T'])}, "
        f"M={int(diagnostic.loc[0, 'M'])}, PC1 variance share={diagnostic.loc[0, 'variance_share']:.3f}, "
        "|corr(PC1, aggregate)|="
        f"{diagnostic.loc[0, 'abs_correlation_with_sales_weighted_aggregate']:.3f}",
        flush=True,
    )
    b = blocks.loc[0]
    print(
        f"[PCA] {int(b['n_singleton_buckets'])} singleton and "
        f"{int(b['n_diversified_buckets'])} diversified buckets "
        f"(N_eff {b['min_N_eff_diversified']:.1f}-{b['max_N_eff_diversified']:.1f}); "
        f"mean pairwise correlation {b['mean_corr_within_diversified']:.3f} within "
        f"diversified, {b['mean_corr_within_singleton']:.3f} within singleton, "
        f"{b['mean_corr_singleton_diversified']:.3f} across",
        flush=True,
    )


if __name__ == "__main__":
    main()
