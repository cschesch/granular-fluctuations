#!/usr/bin/env python3
"""Build aggregate Welburn comparison artifacts from the density-matched run.

Reads saved private inputs and point estimates only. It never connects to WRDS,
fits a model, or exports firm identifiers, financial observations, or links.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from applications.welburn.welburn_network import (
    COMPLETED, MODEL, OBSERVED, PAPER_FIRMS, PAPER_INFERRED,
    PAPER_OBSERVED, PROBABILITY_CUTOFF,
)

from project_paths import ROOT
RUN_OUTPUTS = ROOT / "code/.cache/run_outputs"
OUTPUT = ROOT / "output/welburn_input_moments"


def graph_moments(edges: pd.DataFrame, ids: pd.Index) -> dict:
    """Count degrees over the entire eligible universe, including isolates."""
    ids = pd.Index(ids).astype(str)
    pairs = edges[["gvkey", "customer_gvkey"]].astype(str)
    assert ids.is_unique and not pairs.duplicated().any()
    assert not pairs.gvkey.eq(pairs.customer_gvkey).any()
    assert pairs.gvkey.isin(ids).all() and pairs.customer_gvkey.isin(ids).all()
    outdegree = pairs.gvkey.value_counts().reindex(ids, fill_value=0).to_numpy()
    indegree = pairs.customer_gvkey.value_counts().reindex(ids, fill_value=0).to_numpy()
    degree = indegree + outdegree
    return dict(
        nodes=len(ids), directed_edges=len(pairs),
        density=len(pairs) / (len(ids) * (len(ids) - 1)),
        mean_total_degree=float(degree.mean()),
        median_total_degree=float(np.median(degree)),
        maximum_total_degree=int(degree.max()),
        minimum_total_degree=int(degree.min()),
        maximum_indegree=int(indegree.max()), maximum_outdegree=int(outdegree.max()),
        isolated_firms=int((degree == 0).sum()),
    )


def build(blank=False) -> pd.DataFrame:
    observed = {}
    inferred = total = density = cutoff = None
    if not blank:
        with MODEL.open("rb") as stream:
            ids = pd.Index(pickle.load(stream)["model"].firm_ids_)
        observed = graph_moments(pd.read_parquet(OBSERVED), ids)
        completed = pd.read_parquet(COMPLETED)
        inferred = int(completed.link_source.eq("inferred").sum())
        total = len(completed)
        density = 100 * total / (len(ids) * (len(ids) - 1))
        cutoff = 100 * PROBABILITY_CUTOFF
    paper_density = 100 * (PAPER_OBSERVED + PAPER_INFERRED) / (PAPER_FIRMS * (PAPER_FIRMS - 1))
    rows = [
        ("Observed network", "Firms", PAPER_FIRMS, observed.get("nodes"), "count", None),
        ("Observed network", "Relationships", PAPER_OBSERVED, observed.get("directed_edges"), "count", None),
        ("Observed network", "Total degree, mean", 7, observed.get("mean_total_degree"), "decimal", None),
        ("Observed network", "Total degree, median", 3, observed.get("median_total_degree"), "count", None),
        ("Observed network", "Total degree, maximum", 1334, observed.get("maximum_total_degree"), "count", None),
        ("Completed network", r"Probability cutoff (\%)", 50, cutoff, "percent3", "50"),
        ("Completed network", "Inferred relationships", PAPER_INFERRED, inferred, "count", r"$\approx95{,}000$"),
        ("Completed network", "Total relationships", PAPER_OBSERVED + PAPER_INFERRED, total, "count", r"$\approx115{,}477$"),
        ("Completed network", r"Directed density (\%)", paper_density, density, "percent3", fr"$\approx{paper_density:.3f}$"),
    ]
    return pd.DataFrame(rows, columns=["panel", "moment", "published", "reconstructed", "units", "published_display"])


def display(value, units, override):
    if pd.notna(override):
        return override
    if value is None:
        return ""
    if units == "count":
        return f"{value:,.0f}"
    return f"{value:.3f}" if units == "percent3" else f"{value:.2f}"


def latex_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
        r"Moment & Welburn et al. & Reconstruction \\", r"\midrule",
    ]
    for panel, rows in frame.groupby("panel", sort=False):
        lines.append(r"\multicolumn{3}{@{}l}{\textit{" + panel + r"}} \\")
        for row in rows.to_dict("records"):
            paper = display(row["published"] if pd.notna(row["published"]) else None,
                            row["units"], row["published_display"])
            current = display(row["reconstructed"] if pd.notna(row["reconstructed"]) else None,
                              row["units"], None)
            lines.append(f"{row['moment']} & {paper} & {current} " + r"\\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", action="store_true")
    args = parser.parse_args()
    frame = build(args.blank)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.with_suffix(".tex").write_text(latex_table(frame))
    if not args.blank:
        RUN_OUTPUTS.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(RUN_OUTPUTS / "welburn_input_moments.pkl")


if __name__ == "__main__":
    main()
