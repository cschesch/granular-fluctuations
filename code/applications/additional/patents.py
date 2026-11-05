"""Patent application counts by firm from the Ma linkage."""

from __future__ import annotations

import pandas as pd

from .common import RAW, Application, prepare_flows


PATENTS = RAW / "patent-firm-link-share" / "firm_year_panel_public.csv"


def load_patents() -> Application:
    d = pd.read_csv(PATENTS, usecols=["gvkey", "year", "n_patents_flow"], dtype={"gvkey": str})
    d = d[d.year.between(1976, 2020)]
    flows, mean_flow = prepare_flows(
        d.pivot(index="gvkey", columns="year", values="n_patents_flow").reindex(columns=range(1976, 2021))
    )
    strict_n = int((flows.ge(10).all(axis=1)).sum())
    return Application("Patent assignees", "Eventually granted US patent applications", "Ma (2025) PatentsView–Compustat linkage",
        "Analysis-ready public firm-year linkage accompanying Ma (2025), built from PatentsView and Compustat/CRSP names.",
        flows, mean_flow, pd.Series(flows.index, index=flows.index),
        "1976–2020 filing years", "Annual", "Full-sample mean patent flow",
        f"Raw n_patents_flow enters in levels without detrending or winsorization; "
        f"unit flows sum to total patent counts; full-sample mean flows determine sizes for grouping and aggregation. "
        f"zeros are retained and post-2020 filing years omitted for grant right-censoring. "
        f"The strict at-least-10-patents-in-every-year screen would retain {strict_n} firms and is not the baseline.",
        aggregation="sum")
