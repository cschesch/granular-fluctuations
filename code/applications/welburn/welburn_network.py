"""Welburn's active network construction: observed links, fixed cutoff and flow LP.

One specification: the improved 2015Q1--2019Q1 observed network, exact population
logit, a density-matching probability cutoff, and the existing balanced-panel flow LP.
Acquisition is separate: this module never authenticates or accesses WRDS.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import logit

from mgp_core import ROOT, build_lpt_grouped_data
from applications.compustat.run_compustat_quarterly import load_compustat_panel
from applications.panels import load_gdp_deflator
import welburn_population_logit
from wrds_inputs import ANALYSIS_LAST_QUARTER, wrds_path

PRIVATE = ROOT / 'code/.cache/firm_networks/welburn_three_sources'
RAW = ROOT / 'data/welburn2020/wrds_inputs'
OBSERVED = PRIVATE / 'subsidiary_union_edges_paper_2015Q1_2019Q1.parquet'
MODEL = PRIVATE / 'density_completion_population_logit.pkl'
APPLICATION = PRIVATE / 'current_application.pkl'
COMPLETED = PRIVATE / 'density_target_full_edges.parquet'
SUMMARY_NAME = 'welburn_firm_network_M10_summary.pkl'
FUNDA = wrds_path('compustat_annual')
# This cutoff separates the adjacent scores around the density target of 135,094 links.
PROBABILITY_CUTOFF = .00134865075
POSITIVITY_FLOOR = 1e-8
PAPER_FIRMS, PAPER_OBSERVED, PAPER_INFERRED = 5821, 20477, 95000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _firm_means() -> pd.DataFrame:
    """2015--2019 U.S. sales, COGS, and modal SIC for the completion model.

    Compustat North America reports Canadian observations in CAD.  Mixing those
    levels with USD revenue would distort both the gravity regression and the
    flow constraints.  The quarterly outcome panel is U.S./USD throughout, so
    the calibration universe is restricted the same way.  This gives 6,296
    eligible firms before imposing the balanced quarterly panel, close to the
    5,821-firm financial-data universe in Welburn et al. (2020).
    """
    f = pd.read_csv(FUNDA, usecols=["gvkey", "fyear", "sale", "cogs", "sic", "conm", "fic", "curcd"])
    f["gvkey"] = f["gvkey"].astype(str).str.zfill(6)
    f = f.loc[
        f["fyear"].between(2015, 2019)
        & f["fic"].eq("USA")
        & f["curcd"].eq("USD")
    ].copy()

    def modal(values: pd.Series) -> float:
        values = values.dropna()
        return float(values.mode().iloc[0]) if not values.empty else np.nan

    means = f.groupby("gvkey").agg(
        sale=("sale", "mean"), cogs=("cogs", "mean"), sic=("sic", modal),
        conm=("conm", "last"),
    )
    return means.loc[
        means[["sale", "cogs"]].notna().all(axis=1)
        & means[["sale", "cogs"]].gt(0).all(axis=1)
        & means["sic"].notna()
    ]

def quarterly_levels(panel) -> pd.DataFrame:
    """Real sales levels for the T growth dates plus their initial lag."""
    ids = panel.wide.index
    dates = pd.period_range(panel.start_date, panel.last_date, freq="Q")
    levels = panel.panel["saleq"].unstack("datadate").reindex(index=ids, columns=dates)
    initial = panel.panel.loc[panel.start_date, "lagged_saleq"].reindex(ids)
    levels.insert(0, panel.start_date - 1, initial)
    deflator = load_gdp_deflator("quarterly").reindex(levels.columns)
    if deflator.isna().any() or levels.isna().any().any():
        raise ValueError("Quarterly levels or GDP deflator are incomplete.")
    return levels.div(deflator, axis=1)

def calibrate_network(
    ids: pd.Index,
    means: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Solve Welburn equations (5.2)--(5.7) on a supplied adjacency.

    Returns the report's sales-normalized supplier allocation matrix B, where
    B[i,j] = F[i,j]/sales[i].  ``network_matrices`` below converts the same
    calibrated flow matrix into the conventional requirements matrix A.
    """
    pos = {g: i for i, g in enumerate(ids)}
    suppliers = edges["gvkey"].map(pos).to_numpy(int)
    buyers = edges["customer_gvkey"].map(pos).to_numpy(int)
    e = len(edges)
    n = len(ids)
    a = np.zeros((2 * n, e))
    a[suppliers, np.arange(e)] = 1.0
    a[n + buyers, np.arange(e)] = 1.0
    bounds = np.r_[means["sale"].to_numpy(float), means["cogs"].to_numpy(float)]
    # Maximizing matched intermediate flow is equivalent to minimizing the sum
    # of the sales and input residuals in Welburn et al. equation (5.7).
    # Equation (5.5) requires every admitted edge to carry a positive flow.  A
    # tiny share floor enforces that restriction without determining the scale.
    lower = POSITIVITY_FLOOR * means["sale"].to_numpy(float)[suppliers]
    result = linprog(
        -np.ones(e), A_ub=a, b_ub=bounds,
        bounds=list(zip(lower, np.full(e, np.inf), strict=True)), method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    maximum_matched_flow = float(result.x.sum())
    flows = np.maximum(result.x, 0.0)
    w = np.zeros((n, n))
    w[suppliers, buyers] = flows / means["sale"].to_numpy(float)[suppliers]
    outgoing = np.bincount(suppliers, weights=flows, minlength=n)
    incoming = np.bincount(buyers, weights=flows, minlength=n)
    sales = means["sale"].to_numpy(float)
    cogs = means["cogs"].to_numpy(float)
    rho = float(max(abs(np.linalg.eigvals(w))))
    if rho >= 1 - 1e-9:
        raise ValueError(f"Calibrated network is not productive: spectral radius={rho:g}")
    diag = {
        "network_firms": n,
        "candidate_directed_links": e,
        "positive_calibrated_links": int((flows > 0).sum()),
        "links_at_positivity_floor": int(np.isclose(
            flows, lower, rtol=1e-6, atol=1e-12,
        ).sum()),
        "links_above_positivity_floor": int((
            flows > lower * (1 + 1e-6) + 1e-12
        ).sum()),
        "links_above_one_thousand_dollars": int((flows > .001).sum()),
        "links_above_one_million_dollars": int((flows > 1.0).sum()),
        "disclosed_candidate_links": int(edges["link_source"].eq("disclosed").sum()),
        "inferred_candidate_links": int(edges["link_source"].eq("inferred").sum()),
        "matched_flow_millions": float(flows.sum()),
        "maximum_matched_flow_millions": maximum_matched_flow,
        "matched_flow_optimum_gap_millions": float(maximum_matched_flow - flows.sum()),
        "matched_flow_relative_optimum_gap": float(
            (maximum_matched_flow - flows.sum()) / max(maximum_matched_flow, 1e-30)
        ),
        "tie_break_seed": None,
        "sales_residual_share": float(1 - flows.sum() / sales.sum()),
        "cogs_residual_share": float(1 - flows.sum() / cogs.sum()),
        "objective_total_residual_millions": float(
            (sales - outgoing).sum() + (cogs - incoming).sum()
        ),
        "maximum_sales_constraint_violation_millions": float(
            np.maximum(outgoing - sales, 0).max(initial=0)
        ),
        "maximum_cogs_constraint_violation_millions": float(
            np.maximum(incoming - cogs, 0).max(initial=0)
        ),
        "median_supplier_output_utilization": float(np.median(outgoing / sales)),
        "median_buyer_cogs_utilization": float(np.median(incoming / cogs)),
        "spectral_radius": rho,
        "maximum_supplier_intermediate_share": float(w.sum(axis=1).max()),
    }
    return w, diag

def score_blocks(model, known, block_size=128):
    """Yield unknown nonself indices and exact model logits without string grids."""
    n = len(model.firm_ids_)
    index = pd.Index(model.firm_ids_)
    left = index.get_indexer(known.gvkey)
    right = index.get_indexer(known.customer_gvkey)
    assert np.all(left >= 0) and np.all(right >= 0)
    excluded = {}
    for i, j in zip(left, right, strict=True):
        excluded.setdefault(i, []).append(j)
    sales, sectors = model.sales_, model.sectors_
    beta, mean, scale = model.coefficients_, model.feature_mean_, model.feature_scale_
    for first in range(0, n, block_size):
        rows = np.arange(first, min(first + block_size, n))
        scores = (
            beta[0] * ((sales[rows, None] - mean[0]) / scale[0])
            + beta[1] * ((sales[None, :] - mean[1]) / scale[1])
            + beta[2] * ((sales[rows, None] * sales[None, :] - mean[2]) / scale[2])
            + model.sector_intercepts_[sectors[rows, None], sectors[None, :]]
        )
        keep = np.ones(scores.shape, dtype=bool)
        keep[np.arange(len(rows)), rows] = False
        for local, row in enumerate(rows):
            keep[local, excluded.get(row, [])] = False
        i, j = np.nonzero(keep)
        yield scores[keep], (rows[i].astype(np.int64) * n + j)

@dataclass
class WelburnInputs:
    outcomes: np.ndarray  # T x N, fixed-base changes in final demand
    shares: np.ndarray  # normalized propagation influences
    ids: pd.Index
    dates: pd.PeriodIndex
    grouped: np.ndarray
    diagnostics: dict


def prepare_observed_network(*, force=False):
    """Rebuild the two-source matching overlay only when its local inputs change."""
    from applications.welburn.prepare_welburn_links import (
        DEFAULT_CACHE, DEFAULT_DIAGNOSTICS, STELLA_ALIASES, WELBURN_ALIASES, prepare,
    )
    stems = ['wrds_supply_chain_ids', 'wrds_company_subsidiaries', 'wrds_subsidiary_global_aliases']
    files = [wrds_path('compustat_customers'), FUNDA, STELLA_ALIASES, WELBURN_ALIASES]
    for stem in stems:
        found = sorted(RAW.glob(stem + '_*.csv.gz'))
        if not found:
            raise FileNotFoundError(f'Missing licensed local input: {stem}')
        files.append(found[-1])
    files += [Path(__file__).with_name(name) for name in [
        'prepare_welburn_links.py', 'audit_welburn_three_sources.py', 'welburn_subsidiary_matching.py']]
    fingerprint = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    manifest = PRIVATE / 'observed_inputs_manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not force and previous.get('fingerprint') == fingerprint and OBSERVED.exists():
        if previous.get('observed_sha256') == sha256(OBSERVED):
            return
    print('[welburn] preparing observed links and dated subsidiary matches', flush=True)
    prepare(wrds_path('compustat_customers'), FUNDA, DEFAULT_CACHE, DEFAULT_DIAGNOSTICS)
    from applications.welburn.audit_welburn_three_sources import main as prepare_overlay
    prepare_overlay(quiet=True)
    manifest.write_text(json.dumps(dict(fingerprint=fingerprint, observed_sha256=sha256(OBSERVED)), indent=2) + '\n')


def load_link_model(means=None, known=None):
    """Use the existing exact population fit when all model inputs agree."""
    means = _firm_means() if means is None else means
    known = pd.read_parquet(OBSERVED)[['gvkey', 'customer_gvkey']] if known is None else known
    fingerprint = dict(observed_sha256=sha256(OBSERVED),
        financial_sha256=hashlib.sha256(pd.util.hash_pandas_object(
            means[['sale', 'sic']], index=True).to_numpy().tobytes()).hexdigest(),
        fitter_sha256=sha256(welburn_population_logit.__file__))
    cached = pickle.loads(MODEL.read_bytes()) if MODEL.exists() else None
    if cached is not None and cached.get('fingerprint') == fingerprint:
        return cached['model'], fingerprint
    model = welburn_population_logit.fit(means, known, verbose=True)
    MODEL.write_bytes(pickle.dumps(dict(model=model, fingerprint=fingerprint)))
    return model, fingerprint


def completed_edges(model, known):
    """Retain every observed link and add unknown pairs above the fixed cutoff."""
    chosen = [pairs[scores > logit(PROBABILITY_CUTOFF)] for scores, pairs in score_blocks(model, known)]
    flat = np.sort(np.concatenate(chosen))
    n = len(model.firm_ids_)
    inferred = pd.DataFrame({'gvkey': model.firm_ids_[flat // n],
                             'customer_gvkey': model.firm_ids_[flat % n]})
    edges = pd.concat([known.assign(link_source='disclosed'),
                       inferred.assign(link_source='inferred')], ignore_index=True)
    assert not edges.duplicated(['gvkey', 'customer_gvkey']).any()
    assert not edges.gvkey.eq(edges.customer_gvkey).any()
    return edges


def recover_shocks(levels, base, requirements):
    """Requirements inversion and its exact aggregate propagation influences."""
    operator = np.eye(len(base)) - requirements
    primitive_levels = operator @ levels
    shocks = np.diff(primitive_levels, axis=1).T / base[None, :]
    influence = base * np.linalg.solve(operator.T, np.ones(len(base)))
    error = float(np.max(np.abs(shocks @ influence - np.diff(levels.sum(axis=0)))))
    assert error < 1e-8
    return shocks, influence, dict(
        primitive_inversion='requirements',
        primitive_inversion_note='A_ij=F_ij/sales_j; d=(I-A)y',
        exact_aggregate_identity_max_error=error,
        primitive_level_negative_cell_share=float((primitive_levels < 0).mean()),
        primitive_level_firms_ever_negative=int((primitive_levels < 0).any(axis=1).sum()),
        primitive_level_minimum=float(primitive_levels.min()))


def build_welburn_inputs(*, refresh_inputs=False, use_cache=True):
    """The one active specification, shared by estimates, plots and factor tests."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prepare_observed_network(force=refresh_inputs)
    means_full = _firm_means()
    known = pd.read_parquet(OBSERVED)[['gvkey', 'customer_gvkey']].drop_duplicates()
    model, fit_fingerprint = load_link_model(means_full, known)
    panel = load_compustat_panel(argparse.Namespace(
        compustat_csv=wrds_path('compustat_quarterly_sales'), last_date=ANALYSIS_LAST_QUARTER,
        T=100, coverage_threshold=.8, use_panel_cache=True, refresh_panel_cache=False))
    levels = quarterly_levels(panel)
    levels.index = pd.Index(levels.index.astype(str).str.zfill(6))
    ids = levels.index
    means = means_full.reindex(ids)
    assert means[['sale', 'cogs']].notna().all().all()
    digest = hashlib.sha256()
    for frame in [levels, means[['sale', 'cogs']]]:
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    fingerprint = dict(fit=fit_fingerprint, panel_financials_sha256=digest.hexdigest(),
        cutoff=PROBABILITY_CUTOFF, core_sha256=sha256(__file__))
    old = pickle.loads(APPLICATION.read_bytes()) if use_cache and APPLICATION.exists() else None
    if old is not None and old.get('fingerprint') == fingerprint:
        return WelburnInputs(**old['inputs'])
    full_edges = completed_edges(model, known)
    full_edges.to_parquet(COMPLETED, index=False)
    edges = full_edges.loc[full_edges.gvkey.isin(ids) & full_edges.customer_gvkey.isin(ids)]
    edges = edges.sort_values(['gvkey', 'customer_gvkey'], kind='stable').reset_index(drop=True)
    edges['salecs'] = np.nan
    print(f'[welburn] {len(full_edges):,} full links; {len(ids)} balanced firms, {len(edges):,} links', flush=True)
    b, calibration = calibrate_network(ids, means, edges)
    sales = means.sale.to_numpy(float)
    flows = sales[:, None] * b
    a = flows / sales[None, :]
    shocks, influence, inversion = recover_shocks(
        levels.to_numpy(float), levels.iloc[:, 0].to_numpy(float), a)
    shares = influence / influence.sum()
    grouped, groups, loads, _ = build_lpt_grouped_data(shocks.T, shares, 10)
    aggregate = np.diff(levels.to_numpy(float).sum(axis=0)) / influence.sum()
    identity_error = float(np.max(np.abs(grouped.sum(axis=1) - aggregate)))
    assert identity_error < 1e-10
    assert calibration['maximum_sales_constraint_violation_millions'] < 1e-5
    assert calibration['maximum_cogs_constraint_violation_millions'] < 1e-5
    diagnostics = dict(probability_cutoff=PROBABILITY_CUTOFF, N=len(ids), T=len(shocks), M=10,
        start_date=str(panel.start_date), last_date=str(panel.last_date), weight_date=str(panel.weight_date),
        observation_window='2015Q1--2019Q1 source-date approximation',
        financial_window='2015--2019 mean annual accounts', full_universe_firms=len(means_full),
        full_observed_links=len(known), full_inferred_links=len(full_edges)-len(known),
        full_total_links=len(full_edges), panel_observed_links=int(edges.link_source.eq('disclosed').sum()),
        panel_inferred_links=int(edges.link_source.eq('inferred').sum()), panel_total_links=len(edges),
        primitive_inversion='requirements: A_ij=F_ij/sales_j; d=(I-A)y',
        flow_selection='Original maximum-flow LP; default HiGHS optimum; lexicographic edge order',
        calibration_scope='Global adjacency restricted to the balanced panel before the flow LP',
        grouping_identity_error=identity_error, shares_hhi=float(shares @ shares),
        source_fingerprint=fit_fingerprint, calibration=calibration, inversion=inversion)
    inputs = dict(outcomes=shocks, shares=shares, ids=ids, dates=panel.wide.columns,
                  grouped=grouped, diagnostics=diagnostics)
    payload = dict(fingerprint=fingerprint, inputs=inputs, ids=ids, edges=edges, B=b, A=a,
                   F=flows, epsilon=shocks, weights=shares, grouped=grouped, groups=groups,
                   loads=loads, levels=levels, full_edges=full_edges)
    APPLICATION.write_bytes(pickle.dumps(payload))
    return WelburnInputs(**inputs)
