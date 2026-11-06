#!/usr/bin/env python3
"""Render the appendix comparison of Stella's inputs with the reconstruction."""

import argparse
import json

import numpy as np
import pandas as pd

from applications.stella.audit_stella_growth_moments import growth_moments
from applications.stella.paths import CALIBRATION, DENSE, UNBALANCED
from applications.stella.prepare_stella_calibration import SMALL
from mgp_core import ROOT, RUN_OUTPUT_CACHE


TABLE = ROOT / "output/table_stella_input_moments.tex"

# Each row gives the label, source key, published value, and display format.
SECTIONS = (
    ("Sample", (
        ("Selected unbalanced firms", "current_vintage_top500_union_after_adf", 1610, ",.0f"),
        ("Balanced firms", "mapped_with_employment", 234, ",.0f"),
        ("Balanced share of sales", "current_vintage_selected_mean_share_of_unbalanced_sales", .45, ".2%"),
        ("SD of aggregate sales growth (pp)", "aggregate_growth_pct", 2.56, ".2f"),
        ("SD of continuing-firm contribution (pp)", "continuing_contribution_pp", 2.21, ".2f"),
        ("SD of entry contribution (pp)", "entry_contribution_pp", 1.02, ".2f"),
        ("SD of exit contribution (pp)", "exit_contribution_pp", .52, ".2f"),
        ("GDP correlation: nominal sales", "top500_nominal_sales_gdp_correlation", .39, ".2f"),
        ("GDP correlation: real sales", "top500_real_sales_gdp_correlation", .39, ".2f"),
    )),
    ("Customer network", (
        ("Suppliers with a matched client", "baseline_firms_with_clients", 58, ",.0f"),
        ("Suppliers with a positive client link", "baseline_firms_with_clients_positive_calibrated_share", 58, ",.0f"),
        ("Mean total input share", "baseline_mean_total_input_share", .02, ".2%"),
        ("Median total input share", "baseline_median_total_input_share", 0, ".2%"),
    )),
    ("Dense sector-based network", (
        ("Firms with counterparties", "dense_firms_with_suppliers", 234, ",.0f"),
        ("Median column degree (suppliers)", "dense_median_suppliers", 215, ",.0f"),
        ("Minimum column degree (suppliers)", "dense_min_suppliers", 190, ",.0f"),
        ("Median row degree (clients)", "dense_median_clients", 215, ",.0f"),
        ("Minimum row degree (clients)", "dense_min_clients", 190, ",.0f"),
        ("Mean total input share", "dense_mean_total_input_share", .09, ".2%"),
        ("Median total input share", "dense_median_total_input_share", .06, ".2%"),
    )),
)

# Stella gives one number for suppliers with clients and one for the GDP correlation.
SHARED_TARGETS = {
    "baseline_firms_with_clients": r"\multirow{2}{*}{58}",
    "baseline_firms_with_clients_positive_calibrated_share": "",
    "top500_nominal_sales_gdp_correlation": r"\multirow{2}{*}{0.39}",
    "top500_real_sales_gdp_correlation": "",
}


def reconstructed_moments():
    moments = json.loads((CALIBRATION / "diagnostics.json").read_text())
    matrix = np.loadtxt(DENSE / "matrix_io.csv", delimiter=",")
    economic = np.where(matrix > SMALL, matrix, 0)
    suppliers = (economic > 0).sum(axis=0)
    clients = (economic > 0).sum(axis=1)
    return {
        **moments,
        "dense_firms_with_suppliers": int((suppliers > 0).sum()),
        "dense_median_suppliers": float(np.median(suppliers)),
        "dense_min_suppliers": int(suppliers.min()),
        "dense_median_clients": float(np.median(clients)),
        "dense_min_clients": int(clients.min()),
        "dense_mean_total_input_share": float(economic.sum(axis=0).mean()),
        "dense_median_total_input_share": float(np.median(economic.sum(axis=0))),
        **growth_moments(UNBALANCED),
    }


def render_table(moments):
    lines = [
        r"\begin{tabular}{@{}p{0.48\linewidth}rr@{}}",
        r"\toprule",
        r"Moment & Stella & Reconstruction\\",
        r"\midrule",
    ]
    rows = []
    for section, specifications in SECTIONS:
        if rows:
            lines.append(r"\addlinespace[0.35em]")
        lines.append(r"\multicolumn{3}{@{}l@{}}{\textit{" + section + r"}}\\")
        for label, key, published, spec in specifications:
            target = SHARED_TARGETS.get(key)
            if target is None:
                target = (
                    rf"$\simeq{published:.0%}$".replace("%", r"\%")
                    if spec.endswith("%") else format(published, spec)
                )
            value = None if moments is None else moments[key]
            displayed = "" if value is None else format(value, spec).replace("%", r"\%")
            target_cell = f" {target}" if target else ""
            lines.append(f"{label} &{target_cell} & {displayed}" + r"\\")
            rows.append(dict(section=section, moment=label, source_key=key,
                             published=published, current_revtq=value))
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", action="store_true", help="Leave reconstruction cells empty.")
    args = parser.parse_args()
    latex, summary = render_table(None if args.blank else reconstructed_moments())
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(latex)
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    summary.to_pickle(RUN_OUTPUT_CACHE / "stella_input_moment_comparison.pkl")


if __name__ == "__main__":
    main()
