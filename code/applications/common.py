"""Shared panel representation, cleaning, and estimation for empirical applications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import gradient_bootstrap as GB
from mgp_core import (
    align_sizes_to_panel,
    analysis_block_length,
    build_lpt_grouped_data,
    mgp_share,
    regularity_report,
    sample_cov,
    standardize_panel,
)


@dataclass
class Application:
    entity: str
    outcome: str
    source: str
    benchmark: str
    y: pd.DataFrame
    sizes: pd.Series
    labels: pd.Series
    sample: str
    frequency: str
    weight_date: str
    notes: str
    dependence_horizon: int = 1
    aggregation: str = "weighted"

    def __post_init__(self) -> None:
        if self.aggregation not in {"weighted", "sum"}:
            raise ValueError(f"Unknown aggregation rule: {self.aggregation}")
        # Additive flows retain exact accounting, including large observations.
        # Deterministic adjustments, if any, belong in the application loader.
        self.y = self.y.astype(float).copy() if self.aggregation == "sum" else standardize_panel(self.y)

    def relative_outcomes(self, y: pd.DataFrame, sizes: pd.Series) -> pd.DataFrame:
        """Express flows per unit of size before size-weighted aggregation."""
        return y.div(sizes, axis=0) if self.aggregation == "sum" else y


def balanced_log_growth(levels: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    years = list(range(first, last + 1))
    levels = levels.reindex(columns=years).apply(pd.to_numeric, errors="coerce")
    levels = levels.replace([np.inf, -np.inf], np.nan)
    levels = levels.loc[(levels > 0).all(axis=1)]
    return np.log(levels).diff(axis=1).iloc[:, 1:]


def prepare_flows(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep raw additive flows and their full-sample mean sizes.

    Signed flows are allowed; callers
    choose positive grouping sizes separately when mean flow is not a valid size.
    """
    flows = levels.fillna(0.0).astype(float)
    flows = flows.loc[flows.ne(0).any(axis=1)]
    mean_flow = flows.mean(axis=1)
    return flows, mean_flow


def estimate(
    app: Application,
    solver: str,
    bootstrap_b: int,
    seed: int,
    m_override: int | None = None,
) -> dict[str, object]:
    y, sizes = align_sizes_to_panel(app.y, app.sizes)
    shares = (sizes / sizes.sum()).to_numpy(float)
    m = max(2, min(len(y), 30 if "daily" in app.frequency.lower() else 10))
    if m_override is not None:
        m = int(m_override)
    block_length = analysis_block_length(y.shape[1], app.dependence_horizon)
    yv = app.relative_outcomes(y, sizes).to_numpy(float)
    grouped, _, _, _ = build_lpt_grouped_data(yv, shares, m)
    result = mgp_share(grouped, solver)
    reg = regularity_report(sample_cov(grouped), solver)
    rng = np.random.default_rng(np.random.SeedSequence([seed, *app.entity.encode("utf-8")]))
    bootstrap = GB.moving_block_gradient_ci(
        grouped, b=bootstrap_b, block_length=block_length, rng=rng, solver=solver
    )
    # sigma_x: equal-weighted mean unit time-series std; sigma_y: aggregate std.
    # Same definitions as run_multi_source / build_table keep the naive column comparable.
    unit_variance = np.var(yv, axis=1, ddof=1)
    aggregate_variance = float(np.var(shares @ yv, ddof=1))
    sigma_x = float(np.nanmean(np.nanstd(yv, axis=1, ddof=1)))
    sigma_y = float(np.sqrt(aggregate_variance))
    row = {
        "Entity": app.entity, "Outcome": app.outcome, "Source": app.source, "Literature_benchmark": app.benchmark,
        "Sample": app.sample, "Frequency": app.frequency, "Weight_date": app.weight_date,
        "T": y.shape[1], "N": y.shape[0], "M": m, "HHI": float(shares @ shares),
        "sigma_x": sigma_x, "sigma_y": sigma_y,
        "Top1_share": float(shares.max()), "Top10_share": float(np.sort(shares)[-10:].sum()),
        "Direct_diagonal": float((shares**2) @ unit_variance / aggregate_variance),
        "Naive_homoscedastic": float((shares @ shares) * unit_variance.mean() / aggregate_variance),
        "MGP_share": result["mgp_share"], "SE": result["se_share"],
        "CI90_low": bootstrap["ci_low"], "CI90_high": bootstrap["ci_high"],
        "UCL95": bootstrap["ucl95"], "Bootstrap_B": bootstrap_b,
        "Block_length": block_length, "Dependence_horizon": app.dependence_horizon,
        "Common_rank": reg["common_rank"], "Regular": reg["regular"], "Solver": result["solver"],
        "Notes": app.notes,
        "Aggregation": app.aggregation,
    }
    return row


def additive_flow_application(entity: str, outcome: str, source: str, benchmark: str,
                              levels: pd.DataFrame, labels: pd.Series, sample: str,
                              notes: str) -> Application:
    flows, mean = prepare_flows(levels)
    return Application(entity, outcome, source, benchmark, flows, mean,
                       labels.reindex(flows.index), sample, "Annual", "Full-sample mean flow",
                       notes + " Raw flow levels are retained without detrending or winsorization. "
                       "Full-sample mean flows determine sizes for grouping and aggregation; the aggregate is proportional to total flows.",
                       aggregation="sum")
