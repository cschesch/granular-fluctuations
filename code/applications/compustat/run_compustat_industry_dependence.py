#!/usr/bin/env python3
"""Point estimates allowing unrestricted dependence within SIC/NAICS clusters.

Retains the headline quarterly nominal-sales panel and cleaning. Collapse the
fixed company-code clusters using size weights, then run squared-industry-size
LPT and the ordinary grouped MGP. No bootstrap, manuscript edits, or re-cleaning
of industry averages. Company header codes are not historical classifications.
"""
from __future__ import annotations

from wrds_inputs import wrds_path, ANALYSIS_LAST_QUARTER

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from applications.compustat import run_compustat_quarterly as quarterly
from mgp_core import (RUN_OUTPUT_CACHE, atomic_write_pickle,
                      build_lpt_grouped_data, mgp_share)

PREFIX = "compustat_industry_dependence"
M = 10


def metadata(panel, csv: Path):
    raw = pd.read_csv(csv, usecols=["gvkey", "datacqtr", "sic", "naics", "loc", "fic"],
                      dtype={"gvkey": str, "sic": str, "naics": str})
    raw["gvkey"] = raw.gvkey.str.zfill(6)
    raw["quarter"] = pd.PeriodIndex(raw.datacqtr, freq="Q")
    raw = raw[raw["loc"].eq("USA") & raw.fic.eq("USA")
              & raw.gvkey.isin(panel.wide.index)
              & raw.quarter.between(panel.weight_date, panel.last_date)]
    raw = raw.drop_duplicates(["quarter", "gvkey"], keep="last").copy()
    audit = {"classification_type": "fixed company header codes; not point-in-time",
             "historical_fields_in_extract": [], "classifications": {}}
    for field, width in (("sic", 4), ("naics", 6)):
        raw[field] = raw[field].str.strip().replace("", pd.NA)
        assert raw[field].dropna().str.fullmatch(r"\d+").all()
        audit["classifications"][field] = {
            "missing_rows": int(raw[field].isna().sum()),
            "firms_with_missing": int(raw.loc[raw[field].isna(), "gvkey"].nunique()),
            "firms_with_code_changes": int(raw.groupby("gvkey")[field].nunique().gt(1).sum()),
        }
    fixed = raw[raw.quarter.eq(panel.weight_date)].set_index("gvkey").reindex(panel.wide.index)
    assert fixed[["sic", "naics"]].notna().all().all()
    fixed["sic"] = fixed.sic.str.zfill(4)
    return fixed, audit


def cluster_grouping(y: np.ndarray, shares: np.ndarray, labels: pd.Series):
    """Collapse industries, then LPT on their total squared size shares."""
    codes, names = pd.factorize(labels, sort=True)
    k = len(names)
    assert k >= M and (codes >= 0).all()
    incidence = np.eye(k)[codes].T
    industry_shares = incidence @ shares
    industry_contributions = (incidence * shares) @ y
    industry_growth = industry_contributions / industry_shares[:, None]
    x, buckets, loads, grouping = build_lpt_grouped_data(industry_growth, industry_shares, M)
    # Validate two-stage aggregation against applying the composite map to firms.
    composite = (grouping / industry_shares[None, :]) @ (incidence * shares)
    assert np.allclose(x, y.T @ composite.T, atol=1e-13, rtol=1e-12)
    assert np.allclose(x.sum(axis=1), shares @ y, atol=1e-13, rtol=1e-12)
    assert (np.count_nonzero(composite, axis=0) == 1).all()
    firm_bucket = composite.argmax(axis=0)
    for c in range(k):
        assert np.unique(firm_bucket[codes == c]).size == 1, "industry split across buckets"
    return x, industry_shares, firm_bucket, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()
    csv = wrds_path("compustat_quarterly_sales")
    panel = quarterly.load_compustat_panel(argparse.Namespace(
        compustat_csv=csv, last_date=ANALYSIS_LAST_QUARTER, T=100, coverage_threshold=0.8))
    shares = panel.sizes.to_numpy(float); shares /= shares.sum()
    y = panel.wide.to_numpy(float); y = y - y.mean()
    fixed, audit = metadata(panel, csv)
    assert (fixed.sic.astype(int).to_numpy() ==
            panel.panel.loc[panel.start_date].reindex(panel.wide.index).sic.to_numpy()).all()
    naics_sector = fixed.naics.str[:2].replace(
        {"31": "31-33", "32": "31-33", "33": "31-33",
         "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"})
    specifications = [("naics2", "NAICS, 2-digit sectors", naics_sector)]
    specifications += [(f"naics{digits}", f"NAICS, {digits} digits", fixed.naics.str[:digits])
                       for digits in (3, 4, 5, 6)]
    baseline_x, buckets, _, _ = build_lpt_grouped_data(y, shares, M)
    baseline = mgp_share(baseline_x, args.solver)
    cached = pd.read_pickle(RUN_OUTPUT_CACHE / "compustat_qtr_summary.pkl").iloc[0]
    assert abs(baseline["mgp_share"] - cached.mgp_share) < 1e-7
    rows = [dict(key="firms", label="Independent firms (baseline)", clusters=len(shares),
                 largest_cluster_share=float(shares.max()), **baseline)]
    assignments = {}; grouped_panels = {"firms": baseline_x}
    for key, label, labels in specifications:
        x, cluster_shares, assignment, names = cluster_grouping(y, shares, labels)
        estimate = mgp_share(x, args.solver)
        assert abs(estimate["aggregate_variance"] - baseline["aggregate_variance"]) < 1e-12
        rows.append(dict(key=key, label=label, clusters=len(names),
                         largest_cluster_share=float(cluster_shares.max()), **estimate))
        assignments[key] = pd.DataFrame({"gvkey": panel.wide.index,
                                        "industry": labels.to_numpy(), "bucket": assignment})
        grouped_panels[key] = x
        print(f"{label}: K={len(names)}, ceiling={100*estimate['mgp_share']:.4f}%", flush=True)
    result = pd.DataFrame(rows)
    result["N"] = len(shares); result["T"] = y.shape[1]; result["M"] = M
    result["start_date"] = str(panel.start_date); result["last_date"] = str(panel.last_date)
    result["weight_date"] = str(panel.weight_date)
    for field in ("sic", "naics"):
        special = fixed[field].str.startswith("99")
        audit["classifications"][field]["special_99_prefix_firms"] = int(special.sum())
        audit["classifications"][field]["special_99_prefix_share"] = float(shares[special].sum())
    audit.update(N=len(shares), T=y.shape[1], M=M,
                 weight_date=str(panel.weight_date), dates=[str(panel.start_date), str(panel.last_date)],
                 aggregation="size-weighted industry means; LPT on squared total industry shares",
                 all_industries_unsplit=True, same_aggregate_all_specifications=True,
                 baseline_difference=float(baseline["mgp_share"]-cached.mgp_share))
    atomic_write_pickle(result, RUN_OUTPUT_CACHE / f"{PREFIX}_summary.pkl")
    atomic_write_pickle(assignments, RUN_OUTPUT_CACHE / f"{PREFIX}_assignments.pkl")
    atomic_write_pickle(grouped_panels, RUN_OUTPUT_CACHE / f"{PREFIX}_panels.pkl")
    (RUN_OUTPUT_CACHE / f"{PREFIX}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(result[["label", "clusters", "largest_cluster_share", "mgp_share"]].to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
