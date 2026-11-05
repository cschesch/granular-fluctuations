#!/usr/bin/env python3
"""Reconstruct Stella's (2015) firm input calibration as closely as possible.

The released MATLAB constructor is followed literally. Its deposited 90-sector
1997 BEA use table and value-added vector are used directly; Historical Customer
Segments supplies a binary firm link even when customer sales are unreported;
and the buyer's sector purchase is allocated among its observed same-sector
suppliers in proportion to average Compustat employment.

The archive omits Stella's generated 234-firm ID list, firm/sector crosswalk,
annual firm-employment input, BLS sector employment, and customer crosswalk.
Those pieces are transparently reconstructed from the current WRDS vintage and
auditable public/manual concordances. Diagnostics distinguish input
reconstruction from the exact deposited calibration and solver code.
"""

from __future__ import annotations

from wrds_inputs import wrds_path
from applications.stella.stella_quarterly_inputs import QFILE, REVENUE_FIELD, read_quarterly, input_description, verify_input_provenance

import argparse
import difflib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT
from project_paths import upstream
from applications.stella.stella_provenance import archive_existing, file_fingerprints


PUBLIC = ROOT / "data/stella2015/public_inputs"
BEA = PUBLIC / "bea_1997_summary"
OFFICIAL = upstream("stella2015")
OFFICIAL_IO = OFFICIAL / "filtering/data/matrix_io_input.csv"
OFFICIAL_SURPLUS = OFFICIAL / "filtering/data/valadd_IIcalibration.csv"
OUT = ROOT / "code/.cache/firm_networks/stella_reconstructed"
BASELINE_OUT = OUT.with_name("stella_literal")
LEGACY_SELECTION_ADF = OUT / "selection_adf_pvalues.csv"
SELECTION_ADF = BASELINE_OUT / "selection_adf_pvalues_sax12.csv"
FUNDAFILE = wrds_path("compustat_annual")
MATCHES = ROOT / "code/.cache/firm_networks/customer_name_matches.parquet"
MANUAL_ALIASES = ROOT / "data/stella2015/manual_customer_aliases.csv"
MANUAL_SECTORS = ROOT / "data/stella2015/manual_firm_sectors.csv"
SID_CROSSWALK = ROOT / "data/stella2015/bea_summary_to_stella_sid.csv"

QUARTERS = pd.period_range("1970Q1", "2011Q4", freq="Q")
SMALL = 1e-19


def normalize_gvkey(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def load_quarterly() -> pd.DataFrame:
    columns = ["gvkey", "fyearq", "fqtr", "fic", "curcdq", "saleq", "naics", "sic"]
    d = read_quarterly(columns)
    d = d.loc[(d["fic"] == "USA") & (d["curcdq"] == "USD")].copy()
    d["gvkey"] = normalize_gvkey(d["gvkey"])
    valid_date = d["fyearq"].notna() & d["fqtr"].notna()
    d = d.loc[valid_date].copy()
    d["quarter"] = pd.PeriodIndex(
        d["fyearq"].astype(int).astype(str) + "Q" + d["fqtr"].astype(int).astype(str),
        freq="Q",
    )
    return d


def released_screen_panel(d: pd.DataFrame, *, strict_lag_ranks: bool = True) -> tuple[pd.DataFrame, int]:
    """Reproduce Stella's pre-X-12 consecutive-history/top-500-union screen."""
    x = d.loc[d["quarter"].between(pd.Period("1962Q1"), pd.Period("2011Q4"))].copy()
    has_nonpositive = x["saleq"].le(0).groupby(x["gvkey"]).max()
    x = x.loc[~x["gvkey"].isin(has_nonpositive[has_nonpositive].index) & x["saleq"].notna()]
    x = x.groupby(["gvkey", "quarter"], as_index=False).agg(saleq=("saleq", "mean"))
    x = x.sort_values(["gvkey", "quarter"])
    gap = x.groupby("gvkey")["quarter"].apply(
        lambda q: max((b.ordinal - a.ordinal for a, b in zip(q.iloc[:-1], q.iloc[1:])), default=1)
    )
    x = x.loc[x["gvkey"].isin(gap[gap <= 1].index)].copy()
    tenure = x.groupby("gvkey").size()
    x["lag_sale"] = x.groupby("gvkey")["saleq"].shift()
    selected: set[str] = set()
    for _, quarter_data in x.groupby("quarter"):
        n_at_date = len(quarter_data)
        eligible = quarter_data.loc[quarter_data["gvkey"].map(tenure).ge(41)]
        if n_at_date > 500:
            # egen rank() in the released Stata file leaves missing lagged
            # sales unranked. pandas nlargest otherwise fills spare slots
            # with NaNs, adding firms that never pass the actual screen.
            if strict_lag_ranks:
                eligible = eligible.loc[eligible["lag_sale"].notna()]
            eligible = eligible.nlargest(500, "lag_sale")
        selected.update(eligible["gvkey"])
    return x.loc[x["gvkey"].isin(selected)].copy(), len(selected)


def released_screen_candidates(d: pd.DataFrame, adf_path: Path = SELECTION_ADF) -> tuple[pd.DataFrame, dict]:
    """Apply the released top-500 union, ADF, and 1970--2011 balance screens.

    Stella runs X-12 and a no-lag ADF with an intercept on the entire selected
    history before constructing the balanced panel.  The corresponding X-13
    audit is intentionally a separate, cached calculation because it launches
    one Census process per firm.  Refuse a stale cache rather than silently
    changing the selection rule.
    """
    x, top500_union_n = released_screen_panel(d)
    if not adf_path.exists():
        raise FileNotFoundError(
            f"Missing {adf_path}; run audit_stella_selection_adf.py first."
        )
    provenance_path = adf_path.with_suffix(".provenance.json")
    if provenance_path.exists():
        verify_input_provenance(json.loads(provenance_path.read_text()))
    adf = pd.read_csv(adf_path, dtype={"gvkey": str})
    adf["gvkey"] = normalize_gvkey(adf["gvkey"])
    selected = set(x["gvkey"])
    if not selected.issubset(set(adf["gvkey"])) or adf["gvkey"].duplicated().any():
        raise ValueError("Stella selection ADF cache does not cover the current top-500 union.")
    # The earlier cache tested a superset because of the missing-lag rank bug.
    # Firm histories/ADF statistics are unaffected; intersect rather than
    # falsely counting those cached extra firms as currently selected.
    adf = adf.loc[adf["gvkey"].isin(selected)].copy()
    # Failed current-X-13 cases have missing p-values and are retained here;
    # the audit verifies that none can enter the balanced panel.
    rejected = set(adf.loc[adf["pvalue"].gt(0.01), "gvkey"])
    x = x.loc[~x["gvkey"].isin(rejected)].copy()
    balanced = (
        x.loc[x["quarter"].isin(QUARTERS)]
        .pivot(index="gvkey", columns="quarter", values="saleq")
        .reindex(columns=QUARTERS)
        .dropna()
    )
    return balanced, {
        "current_vintage_top500_union_unique_firms": int(top500_union_n),
        "current_vintage_adf_rejections_before_balance": int(len(rejected)),
        "current_vintage_top500_union_after_adf": int(x["gvkey"].nunique()),
        "selection_adf_threshold": 0.01,
        "selection_adf_cache": str(adf_path),
        "selection_adf_specification": f"{json.loads(provenance_path.read_text()).get('seasonal_adjustment_engine', 'unknown')}-adjusted real q/q log {REVENUE_FIELD} growth; no augmentation lags; intercept; MacKinnon p-value",
        "selection_adf_max_pvalue": float(adf["pvalue"].max()),
        "selection_x13_or_adf_failures_unclassified_and_retained": int(adf["status"].ne("ok").sum()),
    }


def firm_universe(d: pd.DataFrame, sample_mode: str = "literal", adf_path: Path = SELECTION_ADF) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    sample = d.loc[d["quarter"].isin(QUARTERS) & d["saleq"].gt(0)].copy()
    wide = (
        sample.pivot_table(index="gvkey", columns="quarter", values="saleq", aggfunc="mean")
        .reindex(columns=QUARTERS)
        .dropna()
    )
    candidates, screen_diag = released_screen_candidates(d, adf_path)
    # Keep every firm passing the reconstructed source screens. The archive
    # omits the original IDs and intermediate count; forcing its final N would
    # introduce a selection rule absent from the source. Retain that earlier
    # dimension-matched selection only as an explicit sensitivity.
    if sample_mode == "legacy234":
        ids = candidates.mean(axis=1).nlargest(234).index
        selection_note = "Dimension-matching proxy: 234 largest mean-saleq firms within current-vintage candidates surviving the reconstructed top-500-union, ADF, and balance screens; the paper's exact IDs and intermediate count are unavailable"
    elif sample_mode == "literal":
        ids = candidates.index
        selection_note = "All current-vintage firms surviving the reconstructed top-500-union, ADF, and balance screens; no artificial truncation to the paper's final N=234"
    else:
        raise ValueError(f"Unknown sample mode: {sample_mode}")

    def modal(values: pd.Series) -> float:
        clean = values.dropna()
        return clean.mode().iloc[0] if not clean.empty else np.nan

    sic = sample.loc[sample["gvkey"].isin(ids)].groupby("gvkey")["sic"].agg(modal).reindex(ids)
    naics = sample.loc[sample["gvkey"].isin(ids)].groupby("gvkey")["naics"].agg(modal).reindex(ids)
    old_dimension_only_ids = wide.mean(axis=1).nlargest(234).index
    diagnostics = {
        "current_vintage_fully_balanced_positive_1970_2011": int(len(wide)),
        **screen_diag,
        "current_vintage_balanced_before_mapping_or_sector_employment_screen": int(len(candidates)),
        "overlap_with_previous_234_largest_of_all_balanced_rule": int(len(ids.intersection(old_dimension_only_ids))),
        "sample_mode": sample_mode,
        "reconstruction_selection": selection_note,
    }
    return candidates.loc[ids], sic, naics, diagnostics


def census_sic_to_naics() -> dict[str, list[str]]:
    x = pd.read_excel(PUBLIC / "1987_sic_to_1997_naics.xls", header=None, dtype=str)
    out: dict[str, list[str]] = defaultdict(list)
    for sic, naics in zip(x.iloc[:, 0], x.iloc[:, 3]):
        s = "".join(c for c in str(sic) if c.isdigit())
        n = "".join(c for c in str(naics) if c.isdigit())
        if len(s) == 4 and 2 <= len(n) <= 6:
            out[s].append(n)
    return out


def public_bea_summary() -> tuple[dict[str, str], pd.DataFrame, pd.Series, pd.Series]:
    """Return 1997 NAICS-to-summary map and public BEA summary accounts."""
    key = pd.read_excel(BEA / "IO-CodeSummary.xls")
    summary_codes = set(key.iloc[:, 0].dropna().astype(str).str.strip())
    cross = pd.read_excel(BEA / "NAICS-IO.xls", header=None)
    mapping: dict[str, str] = {}
    current = None
    for _, row in cross.iterrows():
        io = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if io in summary_codes:
            current = io
        if current is None:
            continue
        for value in [io, *row.iloc[2:].tolist()]:
            n = "".join(c for c in str(value).replace(".0", "") if c.isdigit())
            if 2 <= len(n) <= 6:
                mapping[n] = current

    use = pd.read_excel(BEA / "NAICSUseSummary.xls", header=None)
    codes = use.iloc[5, 2:].astype(str).str.strip()
    row_codes = use.iloc[:, 0].astype(str).str.strip()
    row_pos = {code: i for i, code in enumerate(row_codes)}
    col_pos = {code: j + 2 for j, code in enumerate(codes)}
    common = [code for code in codes if code in row_pos]
    matrix = pd.DataFrame(
        [[pd.to_numeric(use.iloc[row_pos[r], col_pos[c]], errors="coerce") for c in common] for r in common],
        index=common,
        columns=common,
    ).fillna(0.0)
    output = pd.Series({c: float(pd.to_numeric(use.iloc[145, col_pos[c]], errors="coerce")) for c in common})
    surplus = pd.Series({c: float(pd.to_numeric(use.iloc[144, col_pos[c]], errors="coerce")) for c in common})
    return mapping, matrix, output, surplus


def official_stella_accounts() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = pd.read_csv(OFFICIAL_IO, header=None)
    sids = raw.iloc[0, 1:].astype(int).astype(str).tolist()
    row_sids = raw.iloc[1:-1, 0].astype(int).astype(str).tolist()
    if row_sids != sids:
        raise ValueError("Deposited Stella sector rows and columns are not aligned.")
    matrix = pd.DataFrame(raw.iloc[1:-1, 1:].to_numpy(float), index=sids, columns=sids)
    output = pd.Series(raw.iloc[-1, 1:].to_numpy(float), index=sids)
    surplus_values = pd.read_csv(OFFICIAL_SURPLUS, header=None).iloc[0].to_numpy(float)
    if len(surplus_values) != len(sids):
        raise ValueError("Deposited value-added vector is not aligned with the I-O matrix.")
    surplus = pd.Series(surplus_values, index=sids)
    return matrix, output, surplus


def exact_io_crosswalk_diagnostics(
    public_matrix: pd.DataFrame,
    public_output: pd.Series,
    public_surplus: pd.Series,
    official_matrix: pd.DataFrame,
    official_output: pd.Series,
    official_surplus: pd.Series,
) -> dict:
    cross = pd.read_csv(SID_CROSSWALK, dtype=str)
    groups = cross.groupby("stella_sid")["bea_1997_summary_code"].apply(list).to_dict()
    sids = official_matrix.index.tolist()
    aggregate = pd.DataFrame(0.0, index=sids, columns=sids)
    agg_output = pd.Series(0.0, index=sids)
    agg_surplus = pd.Series(0.0, index=sids)
    for i in sids:
        rows = [code for code in groups[i] if code in public_matrix.index]
        agg_output[i] = public_output.reindex(rows).sum()
        agg_surplus[i] = public_surplus.reindex(rows).sum()
        for j in sids:
            cols = [code for code in groups[j] if code in public_matrix.columns]
            aggregate.loc[i, j] = public_matrix.reindex(index=rows, columns=cols).fillna(0).to_numpy().sum()
    close = np.isclose(aggregate.to_numpy(), official_matrix.to_numpy(), atol=1e-7, rtol=0)
    nonspecial = [sid for sid in sids if sid != "1089"]
    close_nonspecial = np.isclose(
        aggregate.loc[nonspecial, nonspecial].to_numpy(),
        official_matrix.loc[nonspecial, nonspecial].to_numpy(),
        atol=1e-7,
        rtol=0,
    )
    output_ratio = agg_output[nonspecial] / official_output[nonspecial]
    output_relative_gap = (agg_output[nonspecial] - official_output[nonspecial]) / official_output[nonspecial]
    return {
        "deposited_io_dimension": len(sids),
        "public_bea_cells_matching_deposit": int(close.sum()),
        "public_bea_cells_total": int(close.size),
        "non_special_cells_matching_deposit": int(close_nonspecial.sum()),
        "non_special_cells_total": int(close_nonspecial.size),
        "non_special_output_entries_matching": int(np.isclose(agg_output[nonspecial], official_output[nonspecial], atol=1e-7, rtol=0).sum()),
        "public_to_deposited_output_ratio_median": float(output_ratio.median()),
        "public_to_deposited_output_ratio_min": float(output_ratio.min()),
        "public_to_deposited_output_ratio_max": float(output_ratio.max()),
        "public_vs_deposited_output_max_abs_relative_gap": float(output_relative_gap.abs().max()),
        "non_special_surplus_entries_matching": int(np.isclose(agg_surplus[nonspecial], official_surplus[nonspecial], atol=1e-7, rtol=0).sum()),
        "special_sector_note": "SID 1089 is Stella's special government-enterprise aggregate; no reconstructed firm uses it.",
    }


def closest_prefix(value: str, choices: list[str]) -> str | None:
    if not choices:
        return None

    def score(candidate: str) -> tuple[int, int]:
        common = 0
        for a, b in zip(value, candidate):
            if a != b:
                break
            common += 1
        return common, len(candidate)

    return max(choices, key=score)


def map_firms(
    ids: pd.Index,
    sic: pd.Series,
    current_naics: pd.Series,
    sic_map: dict[str, list[str]],
    naics_to_bea: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], set[str]]:
    cross = pd.read_csv(SID_CROSSWALK, dtype=str)
    bea_to_sid = dict(zip(cross["bea_1997_summary_code"], cross["stella_sid"]))
    manual = pd.read_csv(MANUAL_SECTORS, dtype=str).set_index("gvkey")
    firm_sid: dict[str, str] = {}
    firm_bea: dict[str, str] = {}
    firm_naics97: dict[str, str] = {}
    manual_used: set[str] = set()
    for gvkey in ids:
        if gvkey in manual.index:
            row = manual.loc[gvkey]
            firm_sid[gvkey] = row["stella_sid"]
            firm_bea[gvkey] = row["bea_1997_summary_code"]
            firm_naics97[gvkey] = row["bea_1997_summary_code"]
            manual_used.add(gvkey)
            continue
        s = str(int(sic.loc[gvkey])).zfill(4) if pd.notna(sic.loc[gvkey]) else ""
        now = str(int(current_naics.loc[gvkey])) if pd.notna(current_naics.loc[gvkey]) else ""
        n97 = closest_prefix(now, sic_map.get(s, [])) or now
        matches = [
            (len(naics), sector)
            for naics, sector in naics_to_bea.items()
            if n97.startswith(naics) or naics.startswith(n97)
        ]
        if not matches:
            continue
        bea_sector = max(matches)[1]
        if bea_sector not in bea_to_sid:
            continue
        firm_sid[gvkey] = bea_to_sid[bea_sector]
        firm_bea[gvkey] = bea_sector
        firm_naics97[gvkey] = n97
    return firm_sid, firm_bea, firm_naics97, manual_used


@lru_cache(maxsize=1)
def annual_average_employment() -> pd.Series:
    # Stella merges annual Compustat by fiscal year, not datadate calendar year.
    # Use the full annual extract because it retains fyear and the standard
    # industrial/consolidated filters; the narrower employee file does not.
    emp = pd.read_csv(
        FUNDAFILE,
        usecols=["gvkey", "fyear", "emp", "fic", "curcd", "indfmt", "datafmt", "consol"],
    )
    emp["gvkey"] = normalize_gvkey(emp["gvkey"])
    return (
        emp.loc[
            emp["fyear"].between(1970, 2011)
            & emp["emp"].gt(0)
            & emp["fic"].eq("USA")
            & emp["curcd"].eq("USD")
            & emp["indfmt"].eq("INDL")
            & emp["datafmt"].eq("STD")
            & emp["consol"].eq("C")
        ]
        .groupby("gvkey")["emp"]
        .mean()
    )


def average_employment(ids: pd.Index) -> pd.Series:
    return annual_average_employment().reindex(ids).dropna()


def resolve_edges(usable: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Resolve named customers within the reconstructed target universe."""
    from applications.welburn.prepare_welburn_links import normalize_name

    links = pd.read_parquet(MATCHES)
    links["gvkey"] = normalize_gvkey(links["gvkey"])
    links["customer_gvkey"] = links["customer_gvkey"].astype("string").str.zfill(6)
    # The shared cache also contains parent resolutions reviewed for a much
    # later Welburn sample.  Reset those and apply only Stella's period/sample-
    # specific reviewed crosswalk below; exact Compustat name matches remain.
    shared_reviewed = links["match_method"].eq("reviewed_alias_or_parent")
    links.loc[shared_reviewed, "customer_gvkey"] = pd.NA
    aliases: dict[str, set[str]] = defaultdict(set)
    for gvkey, name in links.loc[links["gvkey"].isin(usable), ["gvkey", "conm"]].drop_duplicates().itertuples(index=False):
        key = normalize_name(name)
        if key:
            aliases[key].add(gvkey)
    company_names = pd.read_csv(FUNDAFILE, usecols=["gvkey", "conm"], dtype={"gvkey": str})
    company_names["gvkey"] = normalize_gvkey(company_names["gvkey"])
    for gvkey, name in company_names.loc[company_names["gvkey"].isin(usable)].drop_duplicates().itertuples(index=False):
        key = normalize_name(name)
        if key:
            aliases[key].add(gvkey)
    unique_alias = {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1}
    unresolved = links["customer_gvkey"].isna() & links["customer_key"].ne("")
    links.loc[unresolved, "customer_gvkey"] = links.loc[unresolved, "customer_key"].map(unique_alias)

    alias_keys = list(unique_alias)
    fuzzy_map: dict[str, str] = {}
    relevant = links["gvkey"].isin(usable) & links["year"].between(1970, 2011)
    candidates = links.loc[
        relevant & links["customer_gvkey"].isna() & links["usable_name"],
        "customer_key",
    ].dropna().unique()
    for key in candidates:
        scores = sorted(
            ((difflib.SequenceMatcher(None, key, alias).ratio(), alias) for alias in alias_keys),
            reverse=True,
        )
        if len(scores) < 2:
            continue
        score, alias = scores[0]
        margin = score - scores[1][0]
        prefix = min(len(key), len(alias)) >= 8 and (key.startswith(alias) or alias.startswith(key))
        if (score >= 0.88 and margin >= 0.15) or (prefix and score >= 0.70 and margin >= 0.20):
            fuzzy_map[key] = unique_alias[alias]
    unresolved = links["customer_gvkey"].isna()
    links.loc[unresolved, "customer_gvkey"] = links.loc[unresolved, "customer_key"].map(fuzzy_map)

    reviewed = pd.read_csv(MANUAL_ALIASES, dtype={"customer_gvkey": str})
    reviewed["normalized_key"] = reviewed["customer_key"].map(normalize_name)
    reviewed["customer_gvkey"] = normalize_gvkey(reviewed["customer_gvkey"])
    reviewed_map = reviewed.set_index("normalized_key")["customer_gvkey"].to_dict()
    unresolved = links["customer_gvkey"].isna()
    links.loc[unresolved, "customer_gvkey"] = links.loc[unresolved, "customer_key"].map(reviewed_map)

    # Stella uses customer identities as binary edges; reported customer sales
    # are not an input to matrix_construction_IIcalibration.m and may be missing.
    common = (
        links["year"].between(1970, 2011)
        & links["usable_name"]
        & links["gvkey"].isin(usable)
        & links["customer_gvkey"].isin(usable)
        & links["gvkey"].ne(links["customer_gvkey"])
    )
    edges = links.loc[common, ["gvkey", "customer_gvkey"]].drop_duplicates()
    positive_edges = links.loc[common & links["salecs"].gt(0), ["gvkey", "customer_gvkey"]].drop_duplicates()
    return edges, {
        "high_confidence_fuzzy_customer_aliases": len(fuzzy_map),
        "reviewed_manual_customer_aliases": len(reviewed_map),
        "binary_edges_if_incorrectly_requiring_positive_salecs": int(len(positive_edges)),
        "binary_edge_rule": "All resolved named-company links; salecs is not required, matching the released MATLAB constructor.",
    }


def current_vintage_selection_audit(
    d: pd.DataFrame,
    sic_map: dict[str, list[str]],
    naics_map: dict[str, str],
    output_dir: Path = OUT,
    adf_path: Path = SELECTION_ADF,
) -> dict:
    """Apply the released top-500/balance screens to the current vintage."""
    balanced, screen_diag = released_screen_candidates(d, adf_path)
    ids = set(balanced.index)
    literal_ids = pd.Index(sorted(ids))
    source = d.loc[d["gvkey"].isin(literal_ids) & d["quarter"].isin(QUARTERS)]

    def modal(values: pd.Series) -> float:
        clean = values.dropna()
        return clean.mode().iloc[0] if not clean.empty else np.nan

    literal_sic = source.groupby("gvkey")["sic"].agg(modal).reindex(literal_ids)
    literal_naics = source.groupby("gvkey")["naics"].agg(modal).reindex(literal_ids)
    literal_mapping, _, _, _ = map_firms(
        literal_ids, literal_sic, literal_naics, sic_map, naics_map
    )
    literal_emp = average_employment(literal_ids)
    selected_panel, _ = released_screen_panel(d)
    selected_panel = selected_panel.loc[selected_panel["quarter"].isin(QUARTERS)].copy()
    adf = pd.read_csv(adf_path, dtype={"gvkey": str})
    adf["gvkey"] = normalize_gvkey(adf["gvkey"])
    rejected = set(adf.loc[adf["pvalue"].gt(0.01), "gvkey"])
    pre_adf_total_sales = selected_panel.groupby("quarter")["saleq"].sum().reindex(QUARTERS)
    selected_panel = selected_panel.loc[~selected_panel["gvkey"].isin(rejected)]
    total_sales = selected_panel.groupby("quarter")["saleq"].sum().reindex(QUARTERS)
    target_ids = set(pd.read_csv(output_dir / "firm_mapping.csv", dtype={"gvkey": str})["gvkey"].map(lambda x: str(x).zfill(6)))
    target_sales = (
        selected_panel.loc[selected_panel["gvkey"].isin(target_ids)]
        .groupby("quarter")["saleq"].sum().reindex(QUARTERS)
    )
    coverage = target_sales / total_sales
    coverage_pre_adf = target_sales / pre_adf_total_sales
    candidate_sales = (
        selected_panel.loc[selected_panel["gvkey"].isin(ids)]
        .pivot_table(index="gvkey", columns="quarter", values="saleq", aggfunc="mean")
        .reindex(index=literal_ids, columns=QUARTERS)
    )
    all_candidate_coverage = candidate_sales.sum(axis=0) / total_sales
    legacy_ids = balanced.mean(axis=1).nlargest(234).index
    legacy_sales = candidate_sales.loc[legacy_ids].sum(axis=0)
    legacy_coverage = legacy_sales / total_sales
    legacy_coverage_pre_adf = legacy_sales / pre_adf_total_sales
    smallest_ids = balanced.mean(axis=1).nsmallest(234).index
    smallest_coverage = candidate_sales.loc[smallest_ids].sum(axis=0) / total_sales
    rng = np.random.default_rng(2015)
    values = candidate_sales.to_numpy(float)
    denominator = total_sales.to_numpy(float)
    random_coverage = np.empty(10_000)
    for draw in range(len(random_coverage)):
        chosen = rng.choice(len(values), size=234, replace=False)
        random_coverage[draw] = np.mean(values[chosen].sum(axis=0) / denominator)
    pd.Series(sorted(ids), name="gvkey").to_csv(output_dir / "literal_current_vintage_balanced_ids.csv", index=False)
    return {
        "paper_unbalanced_unique_firms": 1610,
        "released_do_file_unbalanced_tid_loop_upper_bound": 1613,
        **screen_diag,
        "paper_balanced_final_firms": 234,
        "current_vintage_balanced_before_mapping_or_sector_employment_screen": int(len(ids)),
        "current_vintage_balanced_with_manual_public_sector_mapping": int(len(ids & set(literal_mapping))),
        "current_vintage_balanced_with_positive_annual_firm_employment": int(len(ids & set(literal_emp.index))),
        "current_vintage_selected_firms": len(target_ids),
        "current_vintage_selected_mean_share_of_unbalanced_sales": float(coverage.mean()),
        "current_vintage_selected_mean_share_of_pre_adf_top500_union_sales": float(coverage_pre_adf.mean()),
        "current_vintage_selected_min_share_of_unbalanced_sales": float(coverage.min()),
        "current_vintage_selected_max_share_of_unbalanced_sales": float(coverage.max()),
        "current_vintage_all_candidates_mean_share_of_unbalanced_sales": float(all_candidate_coverage.mean()),
        "current_vintage_reconstructed_234_mean_share_of_unbalanced_sales": float(legacy_coverage.mean()),
        "current_vintage_reconstructed_234_mean_share_of_pre_adf_top500_union_sales": float(legacy_coverage_pre_adf.mean()),
        "current_vintage_smallest_234_of_candidates_mean_share_of_unbalanced_sales": float(smallest_coverage.mean()),
        "current_vintage_random_234_of_candidates_sales_share_draws": 10000,
        "current_vintage_random_234_of_candidates_sales_share_mean": float(random_coverage.mean()),
        "current_vintage_random_234_of_candidates_sales_share_p95": float(np.quantile(random_coverage, 0.95)),
        "current_vintage_random_234_of_candidates_sales_share_p99": float(np.quantile(random_coverage, 0.99)),
        "current_vintage_reconstructed_234_min_share_of_unbalanced_sales": float(legacy_coverage.min()),
        "current_vintage_reconstructed_234_max_share_of_unbalanced_sales": float(legacy_coverage.max()),
        "screen_timing_note": f"The paper does not report its corresponding pre-concordance/pre-employment count. The {len(ids)}-versus-234 difference can reflect Compustat vintage, seasonal-adjustment build, firm/sector concordances, annual employment, or BLS coverage; it cannot be assigned to any one screen.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=["legacy234", "literal"], default="literal")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checks-output", type=Path, default=None)
    parser.add_argument("--selection-adf", type=Path, default=SELECTION_ADF)
    parser.add_argument("--employment-source", choices=["public_scope", "strict_concordance", "mean_firm"], default="public_scope",
                        help="source year-matched public BLS ratios, strict identified-only denominators, or archived mean-firm approximation")
    args = parser.parse_args()
    output_dir = args.output_dir or (
        BASELINE_OUT if (args.sample, args.employment_source) == ("literal", "public_scope")
        else BASELINE_OUT.with_name(f"stella_{args.sample}_{args.employment_source}"))
    checks_output = args.checks_output or ROOT / "output" / (
        "stella_reconstruction_sanity_checks.csv" if (args.sample, args.employment_source) == ("literal", "public_scope")
        else f"stella_{args.sample}_{args.employment_source}_reconstruction_sanity_checks.csv")
    quarterly = load_quarterly()
    levels, sic, current_naics, sample_diag = firm_universe(quarterly, args.sample, args.selection_adf)
    ids = levels.index
    sic_map = census_sic_to_naics()
    naics_map, public_io, public_output, public_surplus = public_bea_summary()
    official_io, official_output, official_surplus = official_stella_accounts()
    io_diag = exact_io_crosswalk_diagnostics(
        public_io, public_output, public_surplus, official_io, official_output, official_surplus
    )
    firm_sid, firm_bea, firm_naics97, manual_used = map_firms(ids, sic, current_naics, sic_map, naics_map)
    emp = average_employment(ids)
    usable = pd.Index([gvkey for gvkey in ids if gvkey in firm_sid and gvkey in emp.index])
    employment_diagnostics = {"employment_source": "mean_firm", "note": "Archived whole-1970--2011 mean-firm employment approximation; no BLS denominator"}
    ratio_emp = emp.copy()
    employment_files = []
    employment_ratios = None
    if args.employment_source != "mean_firm":
        from applications.stella.prepare_stella_employment import SECTORAL, reconstruct_employment_ratios
        employment_ratios, employment_diagnostics = reconstruct_employment_ratios(
            pd.DataFrame({"gvkey": usable, "stella_sid": [str(firm_sid[g]) for g in usable]}),
            denominator_mode=args.employment_source)
        employment_diagnostics["employment_source"] = args.employment_source
        ratio_emp = employment_ratios["ratio_emp"]
        usable = usable.intersection(ratio_emp.index[ratio_emp.gt(0)], sort=False)
        emp = employment_ratios["mean_firm_emp"].reindex(usable)
        employment_files = [SECTORAL]
    edges, match_diag = resolve_edges(usable)

    position = {gvkey: i for i, gvkey in enumerate(usable)}
    gma = np.zeros((len(usable), len(usable)))
    for buyer, buyer_edges in edges.groupby("customer_gvkey"):
        supplier_sectors = buyer_edges["gvkey"].map(firm_sid)
        for supplier_sid, sector_edges in buyer_edges.groupby(supplier_sectors):
            suppliers = sector_edges["gvkey"].unique()
            denominator = float(ratio_emp.reindex(suppliers).sum())
            if denominator <= 0:
                continue
            buyer_sid = firm_sid[buyer]
            sector_share = float(official_io.loc[supplier_sid, buyer_sid] / official_output[buyer_sid])
            for supplier in suppliers:
                gma[position[supplier], position[buyer]] = sector_share * float(ratio_emp[supplier] / denominator)

    alpha = np.diag([official_surplus[firm_sid[gvkey]] / official_output[firm_sid[gvkey]] for gvkey in usable])
    dense = np.array(
        [[official_io.loc[firm_sid[i], firm_sid[j]] > 0 for j in usable] for i in usable],
        dtype=bool,
    )
    input_totals = gma.sum(axis=0)
    dense_shares = np.array([
        [official_io.loc[firm_sid[i], firm_sid[j]] / official_output[firm_sid[j]] * ratio_emp[i]
         for j in usable] for i in usable]) if args.employment_source != "mean_firm" else None
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("*.csv"):
        if not previous.name.startswith("selection_"):
            archive_existing(previous)
    archive_existing(output_dir / "diagnostics.json")
    archive_existing(checks_output)
    # Exact solver inputs are headerless, as in Stella's MATLAB/GAUSS pipeline.
    np.savetxt(output_dir / "matrix_io.csv", np.where(gma == 0, SMALL, gma), delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "alpha.csv", alpha, delimiter=",", fmt="%.17g")
    capital = np.eye(len(usable))
    np.savetxt(output_dir / "matrix_capital.csv", np.where(capital == 0, SMALL, capital), delimiter=",", fmt="%.17g")
    pd.DataFrame(gma, index=usable, columns=usable).to_csv(output_dir / "matrix_io_economic_labeled.csv")
    if dense_shares is not None:
        pd.DataFrame(dense_shares, index=usable, columns=usable).to_csv(output_dir / "matrix_io_dense_economic_labeled.csv")
    if employment_ratios is not None:
        employment_ratios.to_csv(output_dir / "firm_employment_ratios.csv")
        (output_dir / "employment_diagnostics.json").write_text(json.dumps(employment_diagnostics, indent=2) + "\n")
    pd.DataFrame({
        "gvkey": usable,
        "sic": sic.reindex(usable).values,
        "naics97_or_reviewed_basis": [firm_naics97[g] for g in usable],
        "bea_1997_summary_code": [firm_bea[g] for g in usable],
        "stella_sid": [firm_sid[g] for g in usable],
        "mean_employment": emp.reindex(usable).values,
        "ratio_emp": ratio_emp.reindex(usable).values,
        "reviewed_manual_sector": [g in manual_used for g in usable],
    }).to_csv(output_dir / "firm_mapping.csv", index=False)
    levels.loc[usable].to_csv(output_dir / "nominal_saleq_levels.csv")

    selection_diag = current_vintage_selection_audit(quarterly, sic_map, naics_map, output_dir, args.selection_adf)
    diagnostics = {
        **sample_diag,
        **selection_diag,
        **io_diag,
        **match_diag,
        **input_description(),
        "target_firms": len(ids),
        "mapped_firms": len(firm_sid),
        "mapped_with_employment": len(usable),
        "candidate_firms_dropped_without_sector_mapping": len(set(ids) - set(firm_sid)),
        "candidate_firms_dropped_without_positive_employment": len(set(ids) - set(emp.index)),
        "candidate_firms_dropped_without_joint_sector_employment": len(set(employment_ratios.index) - set(usable)) if employment_ratios is not None else None,
        "manual_firm_sector_mappings": sorted(manual_used),
        "io_sectors_used": len(set(firm_sid[g] for g in usable)),
        "observed_binary_edges": len(edges),
        "positive_calibrated_edges": int((gma > 0).sum()),
        "max_buyer_intermediate_share": float(input_totals.max()),
        "baseline_firms_with_clients": int(edges["gvkey"].nunique()),
        "baseline_firms_with_clients_positive_calibrated_share": int(((gma > 0).sum(axis=1) > 0).sum()),
        "baseline_firms_with_suppliers": int(edges["customer_gvkey"].nunique()),
        "baseline_firms_with_suppliers_positive_calibrated_share": int(((gma > 0).sum(axis=0) > 0).sum()),
        "baseline_mean_total_input_share": float(input_totals.mean()),
        "baseline_median_total_input_share": float(np.median(input_totals)),
        "dense_firms_with_suppliers": int((dense.sum(axis=0) > 0).sum()),
        "dense_median_suppliers": float(np.median(dense.sum(axis=0))),
        "dense_min_suppliers": int(dense.sum(axis=0).min()),
        "dense_mean_total_input_share": float(dense_shares.sum(axis=0).mean()) if dense_shares is not None else None,
        "dense_median_total_input_share": float(np.median(dense_shares.sum(axis=0))) if dense_shares is not None else None,
        "dense_max_total_input_share": float(dense_shares.sum(axis=0).max()) if dense_shares is not None else None,
        "employment_diagnostics": employment_diagnostics,
        "calibration": "Literal matrix_construction_IIcalibration.m with deposited BEA accounts: binary Historical Segments links and within-sector employment allocation.",
        "employment_approximation": ("Source joint-year mean firm/sector employment ratios reconstructed from public BLS national QCEW and a CES railroad supplement. Historical classification scope proxies are declared separately; original unpublished BLS series/vintage are not identified. The source's three within-sample sector overrides are retained exactly."
                                     if employment_ratios is not None else employment_diagnostics["note"]),
        "remaining_input_gap": "The released archive omits the exact 234 gvkeys and generated concordances. " + (
            "The sample is a dimension-matched current-vintage reconstruction, not an exact sample rerun."
            if args.sample == "legacy234" else "This sample applies the reconstructed screens without forcing the published final count; historical vintage, seasonal-adjustment build, BLS coverage and concordance gaps remain."),
        "input_file_fingerprints": file_fingerprints(
            [QFILE, FUNDAFILE, MATCHES, MANUAL_ALIASES, MANUAL_SECTORS,
             SID_CROSSWALK, args.selection_adf, OFFICIAL_IO, OFFICIAL_SURPLUS] + employment_files
            + sorted(PUBLIC.glob("*.xls")) + sorted(BEA.glob("*.xls")), ROOT),
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    checks = pd.DataFrame([
        ("Balanced firms", 234, len(usable), "count", "Dimension-matched identities reconstructed" if args.sample == "legacy234" else "Literal reconstructed screens; published count not imposed"),
        ("Baseline firms with an in-sample client", 58, diagnostics["baseline_firms_with_clients"], "count", "Paper does not specify whether 58 is counted before or after zero BEA cells are removed; this row counts binary matched links"),
        ("Baseline firms with a positive calibrated client share", 58, diagnostics["baseline_firms_with_clients_positive_calibrated_share"], "count", "Alternative interpretation of the same published 58; not a second independent target"),
        ("Baseline mean total input share", 0.02, diagnostics["baseline_mean_total_input_share"], "share", "Paper reports approximately 0.02"),
        ("Baseline median total input share", 0.0, diagnostics["baseline_median_total_input_share"], "share", "Paper reports close to zero"),
        ("Dense firms with a counterparty", 234, diagnostics["dense_firms_with_suppliers"], "count", "Positive sector-use counterparties"),
        ("Dense median counterparties", 215, diagnostics["dense_median_suppliers"], "count", "Constructor column degree counts suppliers; paper's client label requires direction qualification"),
        ("Dense minimum counterparties", 190, diagnostics["dense_min_suppliers"], "count", "Constructor column degree counts suppliers; paper's client label requires direction qualification"),
        ("Dense mean total input share", 0.09, diagnostics["dense_mean_total_input_share"], "share", "Public BLS reconstruction; original series and vintage unavailable"),
        ("Dense median total input share", 0.06, diagnostics["dense_median_total_input_share"], "share", "Public BLS reconstruction; original series and vintage unavailable"),
    ], columns=["diagnostic", "stella_2015", "reconstruction", "units", "comparison_note"])
    checks_output.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(checks_output, index=False)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
