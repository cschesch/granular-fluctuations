#!/usr/bin/env python3
"""One 1980 Herskovic completion, closest-size matching, and zero-hidden-shock Schur inversion.

All deposited parameters and the original MATLAB size population are retained.
Actual average real annual sales identify model slots; they never replace sizes.
Observed 1980 within-panel sales shares are fixed exactly. The residual network
is one calibrated imputation. Numerical identities are asserted, not tabulated.
Licensed firm-level inputs and derived objects must remain private.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import solve
from scipy.sparse import csr_matrix, eye, save_npz
from scipy.sparse.linalg import splu

from mgp_core import (
    ROOT, RUN_OUTPUT_CACHE, ANALYSIS_BOOTSTRAP_DRAWS, analysis_block_length,
    build_lpt_grouped_data, choose_solver, clean_panel_cache_path, make_panel_rng,
    read_clean_panel_cache, write_clean_panel_cache, sample_cov,
)
from project_paths import DERIVED_CACHE, upstream
from applications.foerster.run_foerster_epsilon import (
    cap_dynamic_share_report, network_direct_diagonal_share,
    network_gradient_family, network_observed_stats,
)

REPLICATION_ROOT = upstream("herskovic2020")
SALES_FILE = REPLICATION_ROOT / "data/Compustat Qtrly Sales and GIC July2017.txt"
CPI_FILE = REPLICATION_ROOT / "data/fred_deflators_annual.xlsx"
PACKAGE_ROOT = REPLICATION_ROOT
LINKAGES_FILE = PACKAGE_ROOT / "data/linkages2013.mat"
PARAMETERS_FILE = PACKAGE_ROOT / "code/guess/benchmark_parameters.mat"
SIZE_CACHE = DERIVED_CACHE / "herskovic2020/calibrated_sizes_matlab_seed100.mat"
DERIVED = DERIVED_CACHE / "herskovic2020" / "schur1980"
DIAGNOSTICS_FILE = ROOT / "output/herskovic_firm_network_diagnostics.json"
SEED = 20261103


@dataclass(frozen=True)
class HerskovicInputs:
    outcomes: np.ndarray  # T x N; a common scale makes shares sum to one
    shares: np.ndarray  # normalized propagation exposures
    raw_shocks: np.ndarray
    growth: np.ndarray
    ids: pd.Index
    dates: pd.PeriodIndex
    initial_sales_shares: np.ndarray
    diagnostics: dict


def benchmark_parameters(path: Path = PARAMETERS_FILE) -> dict[str, float]:
    """Read the deposited SMM solution; no downloaded code is executed."""
    values = np.ravel(loadmat(path)["benchmark_parameters"]).astype(float)
    if values.size != 8:
        raise ValueError(f"Unexpected Herskovic parameter vector length: {values.size}")
    names = ("gamma", "sigma_epsilon", "lambda", "Z", "psi", "mu_size", "sigma_size", "N_public")
    return dict(zip(names, values, strict=True))

def load_linkages(path: Path = LINKAGES_FILE) -> pd.DataFrame:
    """Load the six-column official supplier-customer-year MATLAB object."""
    raw = loadmat(path)
    required = ("permno", "cpermno", "dates", "sale", "percsale", "ctotsale")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Missing linkage variables: {missing}")
    out = pd.DataFrame({name: np.ravel(raw[name]) for name in required})
    out[["permno", "cpermno", "dates"]] = out[["permno", "cpermno", "dates"]].astype(int)
    return out

def _cpi_by_year(path: Path = CPI_FILE) -> pd.Series:
    frame = pd.read_excel(path, sheet_name="FRED Graph", header=13)
    out = pd.Series(
        pd.to_numeric(frame["CPIAUCSL_NBD20100101"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["observation_date"]).dt.year.to_numpy(),
        name="cpi",
    ).dropna()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out.astype(float)

def load_sales_panel(*, use_cache=True):
    """Author growth construction, balanced 1980--2012, and mean real annual sizes."""
    cache = clean_panel_cache_path("herskovic_schur_panel", {"version": 1},
                                  [SALES_FILE, CPI_FILE, Path(__file__)])
    cached = read_clean_panel_cache(cache) if use_cache else None
    if cached is not None:
        return cached
    d = pd.read_csv(SALES_FILE, sep="\t", usecols=["GVKEY", "LPERMNO", "datadate", "saleq"])
    for column in d:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d = d.loc[d.LPERMNO.notna() & d.datadate.notna()].copy()
    d["LPERMNO"] = d.LPERMNO.astype(int)
    d["yyyymm"] = (d.datadate // 100).astype(int)
    keys = ["GVKEY", "LPERMNO", "yyyymm"]
    # Annual size construction follows ReplicationFile04's ordinary sum:
    # duplicate rows and missing/nonpositive quarters invalidate that year's sum.
    annual = d.copy()
    annual.loc[annual.saleq.le(0), "saleq"] = np.nan
    annual["sales_clean"] = annual.groupby(keys, sort=False).saleq.transform("mean")
    annual.loc[annual.duplicated(keys), "sales_clean"] = np.nan
    annual["year"] = annual.yyyymm // 100
    totals = annual.groupby(["year", "LPERMNO"]).sales_clean.agg(["sum", "count", "size"])
    totals = totals["sum"].where(totals["size"].ge(4) & totals["count"].eq(totals["size"]))
    real_sales = totals.unstack("LPERMNO").mul(1e6).mul(100 / _cpi_by_year(), axis=0)

    monthly = d.loc[d.saleq.gt(0)].groupby(keys, as_index=False).saleq.mean()
    lagged = monthly.copy().rename(columns={"saleq": "lagged_saleq"})
    lagged["yyyymm"] += 100
    monthly = monthly.merge(lagged, on=keys, how="left", validate="one_to_one")
    monthly["growth"] = np.log(monthly.saleq / monthly.lagged_saleq)
    low, high = np.percentile(monthly.growth.dropna(), [1, 99], method="hazen")
    monthly["growth"] = monthly.growth.clip(low, high)
    monthly["quarter"] = pd.PeriodIndex(pd.to_datetime(monthly.yyyymm.astype(str), format="%Y%m"), freq="Q")
    dates = pd.period_range("1980Q1", "2012Q4", freq="Q")
    growth = monthly.loc[monthly.quarter.between(dates[0], dates[-1])].pivot_table(
        index=["GVKEY", "LPERMNO"], columns="quarter", values="growth", aggfunc="mean"
    ).reindex(columns=dates).dropna()
    stable_ids = growth.index
    ids = pd.Index(stable_ids.get_level_values("LPERMNO"), name="LPERMNO")
    assert not ids.has_duplicates
    growth.index = ids
    initial = monthly.loc[monthly.quarter.dt.year.eq(1979)].groupby(["GVKEY", "LPERMNO"]).saleq.agg(["sum", "count"])
    initial = initial["sum"].where(initial["count"].ge(4)).reindex(stable_ids)
    initial.index = ids
    assert initial.notna().all() and initial.gt(0).all()
    sizes = real_sales.loc[1980:2012].reindex(columns=ids)
    mean_real = sizes.mean(axis=0)
    assert mean_real.notna().all() and mean_real.gt(0).all()
    payload = dict(growth=growth, initial_sales_shares=initial / initial.sum(), mean_real_sales=mean_real,
                   diagnostics={"growth_winsor_p01": float(low), "growth_winsor_p99": float(high),
                                "missing_annual_size_cells": int(sizes.isna().to_numpy().sum())})
    write_clean_panel_cache(cache, payload)
    return payload


def original_sizes():
    if not SIZE_CACHE.exists():
        from .prepare import prepare
        prepare()
    raw = loadmat(SIZE_CACHE, squeeze_me=True)
    params = benchmark_parameters()
    np.testing.assert_array_equal(raw["theta"], np.array(list(params.values())))
    assert raw["N"] == 5000 and raw["size_seed"] == 100
    sizes = np.asarray(raw["S"], dtype=float)
    np.testing.assert_allclose(sizes, np.exp(params["mu_size"] + params["sigma_size"] * raw["z"]), rtol=2e-14)
    return sizes


def match_sizes(real_sales, sizes):
    """Greedy closest remaining log-size pairs, with each model slot used once."""
    real_sales = real_sales.sort_index()
    n, m = len(sizes), len(real_sales)
    if m > n:
        raise ValueError("More observed identities than calibrated model firms")
    distance = np.abs(np.log(real_sales.to_numpy())[:, None] - np.log(sizes)[None, :])
    assignment = np.empty(m, dtype=int)
    for _ in range(m):
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        assignment[i] = j
        distance[i, :] = np.inf
        distance[:, j] = np.inf
    remaining = np.setdiff1d(np.arange(n), assignment)
    remaining = remaining[np.argsort(-sizes[remaining], kind="stable")]
    source = np.r_[assignment, remaining]
    firms = pd.DataFrame({"node": np.arange(n), "model_source_node": source, "model_size": sizes[source],
                          "permno": np.r_[real_sales.index.to_numpy(int), -np.arange(1, n-m+1)],
                          "mean_real_sales": np.r_[real_sales.to_numpy(), np.full(n-m, np.nan)]})
    np.testing.assert_array_equal(np.sort(firms.model_size), np.sort(sizes))
    return firms


def snapshot_links(raw):
    d = raw.loc[raw.dates.eq(1980)].copy()
    valid = (np.isfinite(d.percsale) & d.percsale.between(.1, 1) &
             np.isfinite(d.sale) & d.sale.gt(0) & d.permno.ne(d.cpermno))
    d = d.loc[valid].drop_duplicates(["permno", "cpermno", "percsale"])
    pairs = d.groupby(["permno", "cpermno"], as_index=False).percsale.sum()
    if pairs.groupby("permno").percsale.sum().gt(1 + 1e-12).any():
        raise ValueError("1980 disclosed supplier shares exceed one; refusing to rescale")
    return pairs


def complete_rows(sizes, known: csr_matrix, disclosure_valid, public, params,
                  seed: int, max_attempts: int = 10000):
    """One independent RNG per row; failures are errors, with no silent fallback."""
    n = len(sizes)
    p = np.clip(n ** (1-.87) * sizes / sizes.sum() / params['Z'], 0, 1)
    score = sizes ** params['psi']
    rows, cols, values = [], [], []
    audit = []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, i]))
        start, end = known.indptr[i:i+2]
        fixed_j, fixed_w = known.indices[start:end], known.data[start:end]
        q = float(fixed_w.sum())
        if q > 1 + 1e-12:
            raise ValueError(f'Observed row {i} sums to {q}; refusing to rescale')
        residual = max(0., 1-q)
        candidates = np.ones(n, dtype=bool)
        candidates[i] = False
        candidates[fixed_j] = False
        available = np.flatnonzero(candidates)
        rejected_empty = rejected_disclosure = 0
        missing = np.array([], dtype=int)
        missing_w = np.array([], dtype=float)
        attempts = 0
        if residual > 1e-14:
            for attempts in range(1, max_attempts+1):
                missing = available[rng.random(len(available)) < p[i]]
                if not len(missing):
                    if not disclosure_valid[i]:
                        # Original simulator's mandatory one-customer rule.
                        missing = np.array([rng.choice(available)])
                    else:
                        rejected_empty += 1
                        continue
                missing_w = residual * score[missing] / score[missing].sum()
                if disclosure_valid[i] and np.any(missing_w[public[missing]] >= .1):
                    rejected_disclosure += 1
                    continue
                break
            else:
                raise RuntimeError(f'Row {i} exhausted {max_attempts} attempts; '
                                   'no fallback was applied')
        jj = np.r_[fixed_j, missing]
        ww = np.r_[fixed_w, missing_w]
        rows.extend([i]*len(jj)); cols.extend(jj.tolist()); values.extend(ww.tolist())
        audit.append((i, float(p[i]), len(fixed_j), q, len(missing), attempts,
                      rejected_empty, rejected_disclosure))
    w = csr_matrix((values, (rows, cols)), shape=(n,n))
    diagnostics = pd.DataFrame(audit, columns=[
        'node', 'connection_probability', 'observed_links', 'observed_share',
        'imputed_links', 'attempts', 'rejected_empty', 'rejected_disclosure',
    ])
    return w, diagnostics

def check_network(w, known, disclosure_valid, public) -> dict:
    row_error = float(np.max(np.abs(np.asarray(w.sum(axis=1)).ravel()-1)))
    assert row_error < 1e-12
    assert np.all(w.data >= 0) and np.all(w.diagonal() == 0)
    assert np.all(np.diff(w.indptr) > 0)
    rr, cc = known.nonzero()
    observed_error = float(np.max(np.abs(np.asarray(w[rr,cc]).ravel()-known.data), initial=0))
    assert observed_error == 0
    extra = w-known
    er, ec = extra.nonzero()
    unexpected = disclosure_valid[er] & public[ec] & (extra.data >= .1)
    assert not unexpected.any()
    return {'maximum_row_sum_error': row_error, 'maximum_observed_share_error': observed_error,
            'unexpected_disclosed_public_links': int(unexpected.sum()),
            'all_rows_have_customers': True, 'no_self_links': True}

def schur_recover(w, gamma, positions, growth, weights):
    """L_OO inverse is the Schur complement; hidden firms retain propagation."""
    operator = eye(w.shape[0], format="csc") - gamma * w.tocsc()
    selector = np.zeros((w.shape[0], len(positions)))
    selector[positions, np.arange(len(positions))] = 1
    inverse_rows = splu(operator).solve(selector, trans="T").T
    inverse_error = float(np.max(np.abs(operator.T @ inverse_rows.T - selector)))
    assert inverse_error < 1e-8
    response = inverse_rows[:, positions]
    effective_operator = solve(response, np.eye(len(positions)))
    shocks = growth @ effective_operator.T
    exposures = response.T @ weights
    identity_error = float(np.max(np.abs(shocks @ exposures - growth @ weights)))
    assert identity_error < 1e-10 and np.all(exposures > 0)
    return shocks, exposures, effective_operator, {"selected_inverse_error": inverse_error,
                                                 "aggregate_identity_error": identity_error}


def build_herskovic_inputs(*, use_cache=True):
    sources = [Path(__file__), SALES_FILE, CPI_FILE, LINKAGES_FILE, PARAMETERS_FILE, SIZE_CACHE]
    cache = clean_panel_cache_path("herskovic_schur_inputs", {"version": 1, "seed": SEED}, sources)
    cached = read_clean_panel_cache(cache) if use_cache else None
    if cached is not None and (DERIVED / "network_W.npz").exists():
        return HerskovicInputs(**cached)
    panel = load_sales_panel(use_cache=use_cache)
    growth = panel["growth"]
    ids = growth.index
    params = benchmark_parameters()
    firms = match_sizes(panel["mean_real_sales"], original_sizes())
    n, m = len(firms), len(ids)
    lookup = pd.Series(firms.node.to_numpy(), index=firms.permno)
    pairs = snapshot_links(load_linkages())
    internal = pairs.loc[pairs.permno.isin(ids) & pairs.cpermno.isin(ids)]
    outside = pairs.loc[pairs.permno.isin(ids) & ~pairs.cpermno.isin(ids)]
    known = csr_matrix((internal.percsale, (lookup.reindex(internal.permno), lookup.reindex(internal.cpermno))), shape=(n,n))
    identified = firms.permno.gt(0).to_numpy()
    screened = firms.permno.isin(pairs.permno).to_numpy() & identified
    w, row_diagnostics = complete_rows(firms.model_size.to_numpy(), known, screened, identified, params, SEED)
    checks = check_network(w, known, screened, identified)
    positions = lookup.reindex(ids).to_numpy(int)
    weights = panel["initial_sales_shares"].to_numpy()
    g = growth.to_numpy().T
    shocks, exposures, effective_operator, schur_checks = schur_recover(w, params["gamma"], positions, g, weights)
    scale = float(exposures.sum())
    shares = exposures / scale
    outcomes = shocks * scale
    assert np.max(np.abs(outcomes @ shares - g @ weights)) < 1e-10
    ratio = firms.loc[identified, "model_size"].to_numpy() / firms.loc[identified, "mean_real_sales"].to_numpy()
    diagnostics = {
        "method": "1980 disclosure-preserving calibrated completion; zero hidden shocks; exact Schur reduction",
        "sample": "1980Q1--2012Q4", "snapshot_year": 1980, "N": m, "T": len(g), "population": n,
        "unobserved_firms": n-m, "size_seed": 100, "network_seed": SEED,
        "size_matching": "closest remaining log-size pairs using mean real annual sales, 1980--2012",
        "calibrated_parameters": params, "model_size_population_preserved": True,
        "median_absolute_log_size_error": float(np.median(np.abs(np.log(ratio)))),
        "observed_links_preserved": known.nnz, "observed_suppliers": int(internal.permno.nunique()),
        "disclosure_screened_suppliers": int(screened.sum()),
        "known_links_to_outside_panel_not_embedded": len(outside), "completed_links": w.nnz,
        "median_customers_observed": float(np.median(np.diff(w.indptr)[positions])),
        "maximum_row_attempts": int(row_diagnostics.attempts.max()),
        "initial_weights": "1979 nominal annual sales shares",
        "replication_scope": "One new imputation from the deposited calibration; fixed 1980 network throughout. Only within-balanced-panel disclosed links are fixed. Other firms transmit shocks but have zero primitive shocks. Inference is conditional on this completion.",
        "numerical_checks": {**checks, **schur_checks}, **panel["diagnostics"],
    }
    DERIVED.mkdir(parents=True, exist_ok=True)
    save_npz(DERIVED / "network_W.npz", w)
    firms.to_csv(DERIVED / "firm_mapping.csv", index=False)
    np.savez_compressed(DERIVED / "shocks.npz", primitive_shocks=shocks, exposures=exposures,
                        effective_operator=effective_operator, growth=g, initial_sales_shares=weights,
                        ids=ids.to_numpy(), quarters=growth.columns.astype(str).to_numpy(dtype="U6"))
    payload = dict(outcomes=outcomes, shares=shares, raw_shocks=shocks, growth=g,
                   ids=ids, dates=growth.columns, initial_sales_shares=weights, diagnostics=diagnostics)
    write_clean_panel_cache(cache, payload)
    return HerskovicInputs(**payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--no-panel-cache", action="store_true")
    args = parser.parse_args()
    inputs = build_herskovic_inputs(use_cache=not args.no_panel_cache)
    m, solver = 10, choose_solver(args.solver)
    grouped, _, _, _ = build_lpt_grouped_data(inputs.outcomes.T, inputs.shares, m)
    network_v = np.outer(inputs.shares, inputs.shares)
    observed = network_observed_stats(grouped, inputs.outcomes, network_v, 1., solver)
    sigma = sample_cov(inputs.outcomes)
    direct = network_direct_diagonal_share(sigma, network_v)
    homoskedastic = float(np.mean(np.diag(sigma)) * np.sum(inputs.shares**2) / observed["aggregate_variance"])
    block = analysis_block_length(len(inputs.dates), overlap_horizon=4)
    gradient = network_gradient_family(grouped, inputs.outcomes, network_v, 1.,
        b=args.bootstrap_b, block_length=block, rng=make_panel_rng(SEED, "herskovic_firm_network_gradient"), solver=solver)
    diagnostics = dict(inputs.diagnostics, M=m, block_length=block, bootstrap_draws=args.bootstrap_b,
                       mgp_share=float(observed["mgp_share"]))
    summary = {
        "dataset": "herskovic_firm_network_M10", "table_outcome": "Schur-recovered firm shocks",
        "frequency": "quarterly", "start_date": str(inputs.dates[0]), "last_date": str(inputs.dates[-1]),
        "N": len(inputs.ids), "T": len(inputs.dates), "M": m, "block_length": block,
        "block_bootstrap_b": args.bootstrap_b, "HHI": float(np.sum(inputs.shares**2)),
        "top1_share": float(inputs.shares.max()), "top10_share": float(np.sort(inputs.shares)[-10:].sum()),
        "direct_diagonal": direct, "naive": direct, "naive_homoscedastic": homoskedastic,
        **{k: v for k, v in observed.items() if k not in {"solver", "status"}}, **gradient,
        "notes": diagnostics["replication_scope"],
    }
    cap_dynamic_share_report(summary)
    diagnostics["ucl95"] = summary.get("moving_block_gradient_bootstrap_p95")
    DIAGNOSTICS_FILE.write_text(json.dumps(diagnostics, indent=2) + "\n")
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    destination = RUN_OUTPUT_CACHE / "herskovic_firm_network_M10_summary.pkl"
    pd.DataFrame([summary]).to_pickle(destination)
    print(pd.Series(summary).to_string(), flush=True)
    print(f"[herskovic] wrote {destination} and {DIAGNOSTICS_FILE}", flush=True)


if __name__ == "__main__":
    main()
