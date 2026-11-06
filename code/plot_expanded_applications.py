#!/usr/bin/env python3
"""Hierarchical overview of selected core and additional applications."""

import numpy as np
import pandas as pd

from compute_applications_coefficients import CACHE, core_estimates, summary_path, summary_rows
from applications.figure_layout import HIERARCHY, hierarchy_positions, format_application_rows, format_share_axis
from applications.registry import PANEL_ENTITIES
from mgp_core import OUTPUT_DIR as OUT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    core = core_estimates()
    core_rows = {
        "Sales": "compustat_qtr",
        "Employment": "compustat_ann_emp",
        "Productivity": "compustat_qtr_tfp",
        "Stocks": "crsp",
        "Forbes Billionaires": "forbes_full",
        "Price by item": "cpi1967",
        "Employment by sector": "bls_ces",
        "Carvalho–Gabaix (2013)": "carvalho_gabaix_bea_bls_historical",
        "Acemoglu et al. (2012)": "acemoglu2012_bea1992",
        "Foerster et al. (2011)": "foerster_net",
        "Atalay (2017)": "atalay_epsilon_network",
    }
    records = {}
    for label, stem in core_rows.items():
        if stem not in core.index:
            continue
        r = core.loc[stem]
        records[label] = dict(estimate=r.estimate, low=r.ci_low, high=r.ci_high,
                              homoskedastic=r.naive_homoscedastic,
                              heteroskedastic=r.naive, inference="block bootstrap")

    network_summaries = {
        "No I/O network or lags": "bea_bls_sector_tfp_no_network",
        "Welburn et al. (2020)": "welburn_firm_network_M10",
        "Stella (2015)": "stella_revtq_firm_network",
        "Herskovic et al. (2020)": "herskovic_firm_network",
    }
    for label, stem in network_summaries.items():
        path = summary_path(stem)
        if path is None:
            continue
        r = pd.read_pickle(path).iloc[0]
        records[label] = dict(
            estimate=r.mgp_share, low=r.moving_block_gradient_bootstrap_p05,
            high=r.moving_block_gradient_bootstrap_p95,
            homoskedastic=r.naive_homoscedastic, heteroskedastic=r.naive,
            inference="block bootstrap",
        )

    elections = {label: label for label in ("Democratic votes by state", "Democratic votes by county")}
    for stem, entities in (("election_applications", elections),
                           ("additional_applications", PANEL_ENTITIES)):
        table = summary_rows(stem)
        for label, entity in entities.items():
            if entity not in table.index:
                continue
            r = table.loc[entity]
            records[label] = dict(estimate=r.MGP_share, low=r.CI90_low, high=r.CI90_high,
                                  homoskedastic=r.Naive_homoscedastic,
                                  heteroskedastic=r.Direct_diagonal,
                                  inference="block bootstrap")

    positions, _, _ = hierarchy_positions()

    plotted = []
    for section, items in HIERARCHY:
        for item in items:
            row = records.get(item)
            plotted.append(dict(section=section, item=item, y=positions[item],
                            **(row or dict(estimate=np.nan, low=np.nan, high=np.nan,
                                               homoskedastic=np.nan, heteroskedastic=np.nan,
                                               inference="missing_input"))))
    pd.DataFrame(plotted).to_csv(CACHE / "applications_coefficients_expanded_data.csv", index=False)

    plot(plotted)


def plot(plotted):
    positions, headings, ymax = hierarchy_positions()
    fig, ax = plt.subplots(figsize=(10, 12.5))
    for row in plotted:
        if not np.isfinite(row["estimate"]):
            continue
        p = positions[row["item"]]; lo = 100*row["low"]; hi = 100*row["high"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(p, 0, max(0, lo), color="#B9D9EE", lw=3.2, zorder=2)
            ax.hlines(p, lo, hi, color="#1f77b4", lw=3.2, zorder=3)
            ax.vlines([lo, hi], p-.09, p+.09, color="#1f77b4", lw=1.2, zorder=3)
        ax.plot(100*row["estimate"], p, "o", color="#1f77b4", ms=5.2, zorder=4)
        # Same benchmark on every row: sigma_x^2 * HHI / sigma_y^2 with the
        # average unit variance, capped at the axis limit.
        if np.isfinite(row["homoskedastic"]):
            ax.plot(min(100, 100*row["homoskedastic"]), p, "D", color="#d62728", ms=5.8,
                    clip_on=False, zorder=5)

    format_application_rows(ax, positions, headings, ymax)
    format_share_axis(ax)
    ax.legend(handles=[
        Line2D([0], [0], marker="D", color="#d62728", lw=0, markersize=5.8,
               label="Naive estimate"),
        Line2D([0], [0], marker="o", color="#1f77b4", lw=3, markersize=5.2,
               label="Estimated ceiling"),
    ], loc="upper right", frameon=True, framealpha=.95, edgecolor="0.75",
       facecolor="white", borderpad=.55, fontsize=11.5)
    fig.subplots_adjust(left=.215, right=.98, top=.988, bottom=.055)
    fig.savefig(OUT / "figure1.pdf", dpi=190, facecolor="white", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
