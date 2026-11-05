#!/usr/bin/env python3
"""Hierarchical overview of selected core and additional applications."""
from functools import partial

import numpy as np
import pandas as pd

from compute_applications_coefficients import CACHE, core_estimates, summary_path, summary_rows
from mgp_core import OUTPUT_DIR as OUT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HIERARCHY = [
    ("Firms", ["Sales", "Employment", "Productivity"]),
    ("Returns", ["Stocks", "Mutual funds", "Forbes Billionaires"]),
    ("Financial flows", ["Bank loans", "Corporate bond issuance", "Mutual fund flows", "Treasury demand by country"]),
    ("Trade", ["Export products", "Import products", "Export destinations", "Import origins"]),
    ("Prices and labor", ["Price by item", "Employment by sector"]),
    ("Innovation", ["Patent count by firm", "Patent value by firm", "Patent count by field", "Patent value by field",
                    "Drug approvals by firm"]),
    ("Geography", ["GDP by metro", "GDP by county", "GDP by state"]),
    ("Elections", ["Democratic votes by state", "Democratic votes by county"]),
    ("Energy and climate", ["Power generation by plant", "CO2 emissions by plant", "Oil production by country", "CO2 emissions by country"]),
    ("World economy", ["GDP (PPP) by country", "Mkt cap by country", "Public debt by country", "Private debt by country"]),
    ("Sector networks", ["No I/O network or lags", "Foerster et al. (2011)",
                       "Acemoglu et al. (2012)", "Carvalho–Gabaix (2013)", "Atalay (2017)"]),
    ("Firm networks", ["Stella (2015)", "Herskovic et al. (2020)",
                       "Welburn et al. (2020)"]),
]

# These applications reconstruct unavailable inputs or empirical networks
# rather than running the published empirical package on its complete inputs.
# Keep the internal keys clean and add the flag only when labels are rendered.
APPROXIMATE_REPLICATIONS = {
    "Acemoglu et al. (2012)",
    "Carvalho–Gabaix (2013)",
    "Stella (2015)",
    "Herskovic et al. (2020)",
    "Welburn et al. (2020)",
}


def available_items():
    results = pd.read_csv(CACHE / "applications_coefficients_expanded_data.csv")
    return set(results.loc[results.estimate.notna(), "item"])


def hierarchy_positions():
    positions, headings = {}, []
    y = 0.0
    for section, items in HIERARCHY:
        headings.append((section, y))
        y += .72
        for item in items:
            positions[item] = y
            y += .72
        y += .48
    return positions, headings, y


def tick_labels(positions):
    return [item + ("*" if item in APPROXIMATE_REPLICATIONS else "") for item in positions]


def additional_loaders():
    from applications.additional import run_additional_applications as add

    return {
        "GDP (PPP) by country": add.load_countries,
        "Mkt cap by country": add.load_country_equity_returns,
        "Bank loans": add.load_banks,
        "GDP by metro": add.load_cities,
        "Export products": partial(add.load_trade, "exports", "hs2"),
        "Import products": partial(add.load_trade, "imports", "hs2"),
        "Export destinations": partial(add.load_trade, "exports", "partners"),
        "Import origins": partial(add.load_trade, "imports", "partners"),
        "GDP by state": add.load_states,
        "GDP by county": add.load_counties,
        "Power generation by plant": partial(add.load_campd, "Gross Load (MWh)", "Power plants", "Gross-load growth"),
        "CO2 emissions by plant": partial(add.load_campd, "CO2 Mass (short tons)", "Power-plant emissions", "CO2-emissions growth"),
        "Patent count by firm": add.load_patents,
        "Oil production by country": add.load_oil,
        "Public debt by country": add.load_public_debt,
        "Private debt by country": add.load_private_debt,
        "Treasury demand by country": add.load_tic,
        "CO2 emissions by country": add.load_country_co2,
        "Corporate bond issuance": add.load_bond_issues,
    }


def innovation_loaders():
    from applications.innovation import run_innovation_applications as innovation

    return {
        "Patent value by firm": innovation.patent_value_by_firm,
        "Patent count by field": innovation.patent_count_by_field,
        "Patent value by field": innovation.patent_value_by_field,
        "Drug approvals by firm": innovation.novel_drugs_by_sponsor,
    }


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

    elections = summary_rows("election_applications")
    for label in ("Democratic votes by state", "Democratic votes by county"):
        if label not in elections.index:
            continue
        r = elections.loc[label]
        records[label] = dict(
            estimate=r.MGP_share, low=r.CI90_low, high=r.CI90_high,
            homoskedastic=r.Naive_homoscedastic,
            heteroskedastic=r.Direct_diagonal,
            inference="block bootstrap",
        )

    add = summary_rows("additional_applications")
    add_rows = {
        "GDP (PPP) by country": "Countries", "Public debt by country": "Public debt by country",
        "Private debt by country": "Private nonfinancial debt by country",
        "Mkt cap by country": "Country equity markets",
        "Bank loans": "Banks", "Corporate bond issuance": "US corporate bond issuers",
        "Treasury demand by country": "Foreign Treasury holders",
        "GDP by metro": "Metropolitan areas", "GDP by state": "US states", "GDP by county": "US counties",
        "Export products": "Exported products", "Import products": "Imported products",
        "Export destinations": "Export destinations", "Import origins": "Import origins",
        "Patent count by firm": "Patent assignees", "Power generation by plant": "Power plants",
        "CO2 emissions by plant": "Power-plant emissions", "Oil production by country": "Oil-producing countries",
        "CO2 emissions by country": "Country CO2 emissions",
        "Patent value by firm": "Patent-value firms",
        "Patent count by field": "Patent-count fields",
        "Patent value by field": "Patent technology fields",
        "Drug approvals by firm": "Novel-drug sponsors",
        "Mutual funds": "Mutual fund portfolios: returns",
        "Mutual fund flows": "Mutual fund portfolios: flows",
    }
    for label, entity in add_rows.items():
        if entity not in add.index:
            continue
        r = add.loc[entity]
        records[label] = dict(estimate=r.MGP_share, low=r.CI90_low, high=r.CI90_high,
                              homoskedastic=r.Naive_homoscedastic,
                              heteroskedastic=r.Direct_diagonal,
                              inference="block bootstrap")

    positions, headings, ymax = hierarchy_positions()

    plotted = []
    for section, items in HIERARCHY:
        for item in items:
            row = records.get(item)
            plotted.append(dict(section=section, item=item, y=positions[item],
                            **(row or dict(estimate=np.nan, low=np.nan, high=np.nan,
                                               homoskedastic=np.nan, heteroskedastic=np.nan,
                                               inference="missing_input"))))
    pd.DataFrame(plotted).to_csv(CACHE / "applications_coefficients_expanded_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 12.5))
    for row in plotted:
        if not np.isfinite(row["estimate"]):
            continue
        p = row["y"]; lo = 100*row["low"]; hi = 100*row["high"]
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

    ax.set_yticks(list(positions.values()), tick_labels(positions), fontsize=12)
    ax.tick_params(axis="y", length=0, pad=4)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for section, yp in headings:
        ax.text(-.012, yp, section, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", fontsize=12.2, clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)], fontsize=11.5)
    ax.set_xticks(range(5, 100, 5), minor=True)
    ax.set_ylim(ymax-.25, -.45)
    ax.set_xlabel("Granular share of aggregate fluctuations", fontsize=12)
    ax.grid(False)
    for x in range(5, 100, 5):
        if x % 20 == 0:
            color, linewidth, linestyle = "0.82", .72, "--"
        elif x % 10 == 0:
            color, linewidth, linestyle = "0.88", .66, "--"
        else:
            color, linewidth, linestyle = "0.91", .62, ":"
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("0.25"); spine.set_linewidth(.8)
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
