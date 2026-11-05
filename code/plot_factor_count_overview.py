#!/usr/bin/env python3
"""Overview of factor lower bounds at the largest admissible M up to 10."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from factor_rank import factor_rank_ledermann, j_test_reportable
from mgp_core import ROOT, RUN_OUTPUT_CACHE
from applications.figure_layout import HIERARCHY, hierarchy_positions, tick_labels


OUT = ROOT / "output"
SUMMARY = RUN_OUTPUT_CACHE / "figure1_factor_rank_summary.pkl"


def load_rows() -> list[dict[str, object]]:
    estimates = pd.read_pickle(SUMMARY).set_index("dataset")
    meta = estimates.reset_index().replace({"item": {
        "Herskovic et al. (2020), Schur shocks": "Herskovic et al. (2020)",
        "Herskovic et al. (2020), reported links": "Herskovic et al. (2020)",
    }}).set_index(["section", "item"])

    rows: list[dict[str, object]] = []
    for section, items in HIERARCHY:
        for item in items:
            if (section, item) not in meta.index:
                rows.append(dict(section=section, item=item, dataset=None, N=np.nan, T=np.nan, M=np.nan,
                                 factor_lower_bound=np.nan, at_test_ceiling=False, all_ranks_rejected=False,
                                 ledermann_bound=np.nan, rank_test_status="missing_input"))
                continue
            m = meta.loc[(section, item)]
            estimate = np.nan
            at_ceiling = False
            all_rejected = False
            status = "not_reportable" if not j_test_reportable(int(m["M"]), int(m["T"])) else "pending"
            if status != "not_reportable" and not estimates.empty and m["dataset"] in estimates.index:
                result = estimates.loc[m["dataset"]]
                raw_status = result.get("rank_test_status", "ok")
                status = "ok" if pd.isna(raw_status) else str(raw_status)
                if status == "ok" and pd.notna(result.get("factor_lb_J", np.nan)):
                    estimate = float(result["factor_lb_J"])
                    at_ceiling = bool(result.get("regular_certified_J", False))
                    all_rejected = bool(result.get("all_ranks_rejected_J", False))
            rows.append({
                "section": section, "item": item, "dataset": m["dataset"],
                "N": int(m["N"]), "T": int(m["T"]), "M": int(m["M"]),
                "factor_lower_bound": estimate, "at_test_ceiling": at_ceiling,
                "all_ranks_rejected": all_rejected,
                "ledermann_bound": int(factor_rank_ledermann(int(m["M"]))),
                "rank_test_status": status,
            })
    return rows


def main() -> None:
    rows = load_rows()
    positions, headings, ymax = hierarchy_positions()
    for row in rows:
        row["y"] = positions[row["item"]]
    pd.DataFrame(rows).to_csv(RUN_OUTPUT_CACHE / "factor_count_overview_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 12.5))

    # Rows we do not test, and rows whose values fall outside the axis, are
    # simply left blank.
    xmax = float(factor_rank_ledermann(10))
    for row in rows:
        y = float(row["y"])
        bound = float(row["ledermann_bound"])
        estimate = float(row["factor_lower_bound"])
        if row["rank_test_status"] != "ok":
            continue
        if max(bound, estimate) > xmax:
            continue
        if np.isfinite(estimate):
            # r_hat is a lower bound, so every value to its right stays
            # compatible with the data; the band runs to the edge of the axis.
            ax.hlines(y, estimate, xmax, color="#B9D9EE", lw=3.2, zorder=2)
        # Draw the blue estimate second, so equality reads as a circle overlaid
        # on the Ledermann diamond.
        ax.plot(bound, y, "D", color="#d62728", ms=6.5, zorder=4)
        if np.isfinite(estimate):
            ax.plot(estimate, y, "o", color="#1f77b4", ms=5.2, zorder=5)

    ax.set_yticks(list(positions.values()), tick_labels(positions), fontsize=11.8)
    ax.tick_params(axis="y", length=0, pad=4)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for section, y in headings:
        ax.text(
            -0.015, y, section, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontweight="bold", fontsize=12.2, clip_on=False,
        )

    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, int(xmax) + 1, 1))
    ax.tick_params(axis="x", labelsize=11.2)
    ax.set_ylim(ymax - 0.25, -0.45)
    ax.set_xlabel("Number of common factors", fontsize=12)
    for x in range(1, int(xmax)):
        ax.axvline(x, color="0.88" if x % 2 == 0 else "0.92",
                   linestyle="--" if x % 2 == 0 else ":", linewidth=0.72, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("0.25"); spine.set_linewidth(0.8)

    handles = [
        Line2D([0], [0], marker="o", color="#1f77b4", lw=0, markersize=5.8,
               label="Factor-rank lower bound"),
        Line2D([0], [0], marker="D", color="#d62728", lw=0, markersize=5.8,
               label="Integer Ledermann bound"),
    ]
    ax.legend(
        handles=handles, loc="upper right", ncol=1,
        frameon=True, framealpha=0.95,
        edgecolor="0.75", facecolor="white", borderpad=0.55, fontsize=10.6,
    )
    fig.subplots_adjust(left=0.255, right=0.985, top=0.96, bottom=0.055)
    fig.savefig(OUT / "factor_count_overview.pdf", dpi=190, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
