#!/usr/bin/env python3
"""Reconstruct Stella's missing BLS denominators from historical national QCEW.

The released do-file averages firm and sector employment only over their joint
observed years, then divides these means. Its three within-sample overrides are
implemented literally. Public QCEW denominators span 1990--2011; the railroad
CES supplement spans 1970--2011. Stella's undisclosed original BLS series and
vintage are not inferred from fit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from project_paths import DERIVED_CACHE, ROOT
PUBLIC = ROOT / "data/stella2015/public_inputs"
PUBLIC_CACHE = DERIVED_CACHE / "stella"
SECTORAL = PUBLIC_CACHE / "bls_sectoral_employment.csv"
SPECIAL_SIDS = {"1013", "1034", "1048"}
# These published-scope substitutes are declared from classification content,
# before looking at paper moments. The full concordance does not identify the
# old sector total where a new leaf crosses Stella sectors. Keeping the strict
# column exposes the gap while the primary reconstruction uses the closest
# published BLS industry scopes. Classification alone determines these choices.
PUBLISHED_SCOPE_PROXIES = {
    "1001": ["23"],
    "1007": ["213"],
    "1058": ["561"],
    "1059": {2002: ["516", "518", "519"], 2007: ["518", "519"], 2012: ["518", "519"]},
    "1074": ["5111"],
    "1054": ["5152", "517"],
}


def code(value) -> str:
    return str(value).strip().removesuffix(".0")


def pairs(path: Path, sheet=0) -> list[tuple[str, str]]:
    x = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    return [(code(a), code(b)) for a, b in zip(x.iloc[:, 0], x.iloc[:, 2])
            if re.fullmatch(r"\d{6}", code(a)) and re.fullmatch(r"\d{6}", code(b))]


def vintage_sid_crosswalk() -> tuple[dict[int, dict[str, set[str]]], pd.DataFrame]:
    """Map current-year leaf industries to sets of original Stella sectors.

    A many-to-many Census concordance is retained as a set; a leaf crossing
    Stella sectors is never allocated using arbitrary equal shares.
    """
    cross = pd.read_csv(ROOT / "data/stella2015/bea_summary_to_stella_sid.csv", dtype=str)
    bea_to_sid = cross.set_index("bea_1997_summary_code")["stella_sid"].to_dict()
    table = pd.read_excel(PUBLIC / "bea_1997_summary/NAICS-IO.xls", header=None, dtype=str)
    prefix_sid = {}
    current = None
    for row in table.itertuples(index=False, name=None):
        io_code = code(row[0])
        if io_code in bea_to_sid:
            current = bea_to_sid[io_code]
        if current:
            for value in row[2:]:
                n = code(value).removesuffix("*")
                if re.fullmatch(r"\d{2,6}", n):
                    prefix_sid[n] = current
    paths = PUBLIC / "naics_concordances"
    initial = pairs(paths / "1997_NAICS_to_2002_NAICS.xls", sheet="Concordance 23 US NoD")
    original = {}
    for n97, _ in initial:
        candidates = [p for p in prefix_sid if n97.startswith(p)]
        original[n97] = {prefix_sid[max(candidates, key=len)]} if candidates else {"unmapped"}

    def forward(previous: dict[str, set[str]], concordance: list[tuple[str, str]]) -> dict[str, set[str]]:
        result = defaultdict(set)
        for old, new in concordance:
            result[new].update(previous.get(old, {"unmapped"}))
        return dict(result)

    x2002 = forward(original, initial)
    x2007 = forward(x2002, pairs(paths / "2002_to_2007_NAICS.xls"))
    x2012 = forward(x2007, pairs(paths / "2007_to_2012_NAICS.xls"))
    maps = {2002: x2002, 2007: x2007, 2012: x2012}
    audit = pd.DataFrame([{"naics_vintage": vintage, "naics_code": leaf, "stella_sids": ";".join(sorted(sids)),
                           "unambiguous": len(sids) == 1 and "unmapped" not in sids}
                          for vintage, mapping in maps.items() for leaf, sids in mapping.items()])
    return maps, audit


def build_sectoral() -> tuple[pd.DataFrame, dict]:
    PUBLIC_CACHE.mkdir(parents=True, exist_ok=True)
    maps, audit = vintage_sid_crosswalk()
    audit.to_csv(PUBLIC_CACHE / "bls_qcew_naics_to_stella.csv", index=False)
    rows, component_rows, unresolved_rows = [], [], []
    for year in range(1990, 2012):
        path = PUBLIC / "bls_qcew" / f"{year}_US000_annual.csv"
        vintage = 2002 if year <= 2006 else 2007 if year <= 2010 else 2012
        mapping = maps[vintage]
        raw = pd.read_csv(path, dtype={"area_fips": str, "own_code": str, "industry_code": str, "disclosure_code": str})
        # Private-sector wage and salary employees: Compustat companies are
        # private enterprises. Annual averages are persons, converted below
        # to thousands to match Compustat emp.
        raw = raw.loc[raw["area_fips"].eq("US000") & raw["own_code"].eq("5")].copy()
        raw = raw.loc[raw["industry_code"].str.fullmatch(r"\d{2,6}")].copy()
        raw = raw.drop_duplicates("industry_code").set_index("industry_code")
        available = {}
        for prefix, row in raw.iterrows():
            if pd.notna(row.get("disclosure_code")) and str(row["disclosure_code"]).strip() == "N":
                continue
            employment = pd.to_numeric(row["annual_avg_emplvl"], errors="coerce")
            if pd.isna(employment) or employment <= 0:
                continue
            leaves = {leaf: sids for leaf, sids in mapping.items() if leaf.startswith(prefix)}
            all_sids = set().union(*leaves.values()) if leaves else set()
            if len(all_sids) == 1 and "unmapped" not in all_sids:
                available[prefix] = (next(iter(all_sids)), float(employment))
        # Select the coarsest published exact homogeneous branches, ensuring
        # no employment is counted twice. A coarser branch avoids suppressed
        # fine-industry observations when BLS publishes their aggregate.
        selected = {}
        for prefix in sorted(available, key=lambda n: (len(n), n)):
            if not any(prefix.startswith(p) for p in selected):
                selected[prefix] = available[prefix]
        all_sids = sorted(set().union(*mapping.values()) - {"unmapped"})
        for sid in all_sids:
            expected = {leaf for leaf, sids in mapping.items() if sid in sids}
            pieces = {p: value for p, (s, value) in selected.items() if s == sid}
            uncovered = sorted(leaf for leaf in expected if not any(leaf.startswith(p) for p in pieces))
            ambiguous = sorted(leaf for leaf in expected if mapping[leaf] != {sid})
            # Missing or cross-sector pieces are reported; do not call a
            # partial-sector sum the total sector denominator.
            complete = bool(pieces) and not uncovered
            value = sum(pieces.values()) / 1000 if complete else np.nan
            strict_value = value
            method = "complete_Census_concordance"
            proxy = PUBLISHED_SCOPE_PROXIES.get(sid)
            if proxy is not None:
                prefixes = proxy[vintage] if isinstance(proxy, dict) else proxy
                values = pd.to_numeric(raw.reindex(prefixes).annual_avg_emplvl, errors="coerce")
                disclosures = raw.reindex(prefixes).disclosure_code.fillna("").astype(str).str.strip()
                if values.gt(0).all() and not disclosures.eq("N").any():
                    value = float(values.sum() / 1000)
                    method = "published_scope_proxy_for_cross_sector_NAICS_revision"
                    pieces = {prefix: float(v) for prefix, v in zip(prefixes, values)}
            rows.append({"year": year, "stella_sid": sid, "sectoral_emp": value,
                         "strict_concordance_sectoral_emp": strict_value, "method": method,
                         "source": "BLS QCEW US000 private annual average employment", "naics_vintage": vintage,
                         "complete_concordance_coverage": complete, "n_components": len(pieces),
                         "uncovered_leaf_codes": ";".join(uncovered), "ambiguous_leaf_codes": ";".join(ambiguous)})
            for prefix, value_persons in pieces.items():
                component_rows.append({"year": year, "stella_sid": sid, "naics_code": prefix, "employment_persons": value_persons})
            if uncovered:
                unresolved_rows.append({"year": year, "stella_sid": sid, "uncovered_leaf_codes": uncovered, "ambiguous_leaf_codes": ambiguous})
    sectoral = pd.DataFrame(rows)
    rail_rows = []
    for start, end in ((1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2011)):
        path = PUBLIC / "bls_ces_supplement" / f"CEU4348200001_{start}_{end}.json"
        body = json.loads(path.read_text())
        rail_rows.extend(body["Results"]["series"][0]["data"])
    if not rail_rows:
        raise FileNotFoundError("Run fetch_stella_public_inputs.py to obtain the CES railroad supplement; QCEW railroad counts exclude most employees")
    rail = pd.DataFrame(rail_rows)
    rail = rail.loc[rail.period.isin([f"M{m:02d}" for m in range(1, 13)])].copy()
    rail["year"] = rail.year.astype(int)
    rail["value"] = pd.to_numeric(rail.value)
    if rail.duplicated(["year", "period"]).any():
        raise ValueError("CES railroad supplement contains duplicate months")
    annual_rail = rail.groupby("year").value.agg(["mean", "count"])
    if (annual_rail["count"] != 12).any():
        raise ValueError("CES railroad supplement has an incomplete year")
    sectoral = sectoral.loc[sectoral.stella_sid.ne("1019")].copy()
    sectoral = pd.concat([sectoral, pd.DataFrame([
        {"year": int(year), "stella_sid": "1019", "sectoral_emp": float(row["mean"]),
         "strict_concordance_sectoral_emp": float(row["mean"]), "method": "CES_rail_supplement_for_QCEW_coverage_exclusion",
         "source": "BLS CES CEU4348200001 annual mean of 12 NSA monthly employment observations", "naics_vintage": 2022,
         "complete_concordance_coverage": True, "n_components": 1, "uncovered_leaf_codes": "", "ambiguous_leaf_codes": ""}
        for year, row in annual_rail.iterrows()])], ignore_index=True).sort_values(["year", "stella_sid"])
    component_rows = [row for row in component_rows if row["stella_sid"] != "1019"]
    component_rows.extend({"year": int(year), "stella_sid": "1019", "naics_code": "482 (CES CEU4348200001)",
                           "employment_persons": float(row["mean"] * 1000)} for year, row in annual_rail.iterrows())
    sectoral.to_csv(SECTORAL, index=False)
    pd.DataFrame(component_rows).to_csv(PUBLIC_CACHE / "bls_sectoral_employment_components.csv", index=False)
    diagnostics = {"source": "https://www.bls.gov/cew/downloadable-data-files.htm",
                   "classification_source": "https://www.bls.gov/cew/classifications/industry/industry-titles.htm",
                   "years": sorted(sectoral.year.unique().astype(int).tolist()),
                   "first_year": int(sectoral.year.min()), "last_year": int(sectoral.year.max()),
                   "public_units": "thousands of private wage and salary employees (annual averages)",
                   "not_original_input": "Stella identifies BLS but omits sectoral_emp_raw.dta, series IDs, series scope and vintage. QCEW is a documented public-source reconstruction, not verified original series.",
                   "pre_1990_note": "No sector-level NAICS QCEW employment is available before 1990; joint firm-sector averaging uses only available years, as in released code. We do not extrapolate 1990 values back to 1970.",
                   "crosswalk_method": "Official Census 1997->2002->2007->2012 six-digit concordances, aggregated to original BEA/Stella SID. Coarsest published homogeneous industry branches, no double counting. Cross-sector ambiguous or missing leaves make a SID-year denominator missing, not a partial total.",
                   "published_scope_proxy_definitions": PUBLISHED_SCOPE_PROXIES,
                   "published_scope_proxy_note": "These published-industry groups approximate old definitions where Census many-to-many splits cross original Stella sectors. Their levels are observed BLS totals; no allocation weights or levels are tuned to paper moments. strict_concordance_sectoral_emp preserves the identified-only alternative.",
                   "railroad_coverage_source": "https://www.bls.gov/cew/overview.htm",
                   "railroad_supplement": "QCEW excludes railroad UI employees. SID1019 uses CES CEU4348200001 annual average NSA all-employees levels, 1970--2011. Railroad CES counts are never replaced by small QCEW covered-worker totals.",
                   "total_sid_years": len(sectoral), "usable_sid_years": int(sectoral.sectoral_emp.notna().sum()),
                   "complete_concordance_sid_years": int(sectoral.strict_concordance_sectoral_emp.notna().sum()),
                   "incomplete_sector_years": unresolved_rows}
    (PUBLIC_CACHE / "bls_sectoral_employment_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return sectoral, diagnostics


def reconstruct_employment_ratios(mapping: pd.DataFrame, sectoral: pd.DataFrame | None = None,
                                  denominator_mode: str = "public_scope") -> tuple[pd.DataFrame, dict]:
    """Return per-gvkey source-style ratios and coverage diagnostics.

    mapping must contain columns gvkey and stella_sid. Returned DataFrame is
    indexed by normalized gvkey; ratio_emp is NaN for firms lacking joint data.
    Special-sector normalizations use only the supplied sample's joint-data
    survivors, exactly as in the source. No excluded firm is silently retained.
    """
    from wrds_inputs import wrds_path
    sectoral = pd.read_csv(SECTORAL, dtype={"stella_sid": str}) if sectoral is None else sectoral.copy()
    sectoral["stella_sid"] = sectoral["stella_sid"].astype(str)
    if denominator_mode == "strict_concordance":
        sectoral["sectoral_emp"] = sectoral["strict_concordance_sectoral_emp"]
    elif denominator_mode != "public_scope":
        raise ValueError("denominator_mode must be public_scope or strict_concordance")
    if sectoral.duplicated(["year", "stella_sid"]).any():
        raise ValueError("Duplicate public employment sector-year")
    firms = mapping[["gvkey", "stella_sid"]].copy()
    firms["gvkey"] = firms.gvkey.map(lambda x: code(x).zfill(6))
    firms["stella_sid"] = firms.stella_sid.astype(str)
    if firms.gvkey.duplicated().any():
        raise ValueError("Firm mapping must contain one row per gvkey")
    annual_path = wrds_path("compustat_annual")
    annual = pd.read_csv(annual_path, usecols=["gvkey", "fyear", "emp", "fic", "curcd", "indfmt", "datafmt", "consol"], dtype={"gvkey": str})
    annual["gvkey"] = annual.gvkey.map(lambda x: code(x).zfill(6))
    # data_building_balanced.do keeps emp!=. & emp!=0. Do not silently
    # strengthen this to positivity: the deposited rule retains negatives.
    annual = annual.loc[annual.fyear.between(1970, 2011) & annual.emp.notna() & annual.emp.ne(0)
                        & annual.fic.eq("USA") & annual.curcd.eq("USD") & annual.indfmt.eq("INDL")
                        & annual.datafmt.eq("STD") & annual.consol.eq("C")].copy()
    if annual.duplicated(["gvkey", "fyear"]).any():
        raise ValueError("Duplicate firm employment years after Compustat filters; source merge cannot identify these")
    annual = annual.rename(columns={"fyear": "year", "emp": "firm_emp"})
    joint = firms.merge(annual[["gvkey", "year", "firm_emp"]], on="gvkey", how="left").merge(
        sectoral[["year", "stella_sid", "sectoral_emp"]], on=["year", "stella_sid"], how="left")
    # The source drops zero sector employment before its merge, then keeps
    # nonmissing joint observations. Retain exactly those semantics.
    joint = joint.loc[joint.firm_emp.notna() & joint.sectoral_emp.notna() & joint.sectoral_emp.ne(0)].copy()
    result = joint.groupby("gvkey").agg(mean_firm_emp=("firm_emp", "mean"), mean_sectoral_emp=("sectoral_emp", "mean"),
                                         first_year=("year", "min"), last_year=("year", "max"), n_joint_years=("year", "nunique"))
    result = firms.set_index("gvkey").join(result)
    result["ratio_emp"] = result.mean_firm_emp / result.mean_sectoral_emp
    special = result.stella_sid.isin(SPECIAL_SIDS)
    within_sample_sum = result.groupby("stella_sid").mean_firm_emp.transform("sum")
    result.loc[special, "ratio_emp"] = result.loc[special, "mean_firm_emp"] / within_sample_sum[special]
    result["source_special_sector_override"] = special
    result["first_time"] = (result.first_year - 1970) * 4 + 1
    result["last_time"] = (result.last_year - 1970) * 4 + 4
    sector_sums = result.groupby("stella_sid").ratio_emp.sum()
    proxy_sids = sectoral.loc[sectoral.get("method", pd.Series("", index=sectoral.index)).eq(
        "published_scope_proxy_for_cross_sector_NAICS_revision"), "stella_sid"].unique()
    proxy_firms = result.loc[result.stella_sid.isin(proxy_sids)]
    records = {"employment_method": __doc__, "public_sectoral_file": str(SECTORAL),
               "denominator_mode": denominator_mode,
               "public_sectoral_sha256": hashlib.sha256(SECTORAL.read_bytes()).hexdigest(),
               "sample_firms": len(result), "firms_with_joint_employment": int(result.ratio_emp.notna().sum()),
               "firms_without_joint_employment": result.index[result.ratio_emp.isna()].tolist(),
               "sids_without_joint_employment": sorted(result.loc[result.ratio_emp.isna(), "stella_sid"].unique().tolist()),
               "source_special_sids": sorted(SPECIAL_SIDS), "mean_joint_years": float(result.n_joint_years.mean()),
               "earliest_joint_year": int(result.first_year.min()) if result.first_year.notna().any() else None,
               "latest_joint_year": int(result.last_year.max()) if result.last_year.notna().any() else None,
               "special_sector_ratio_sums": result.loc[special].groupby("stella_sid").ratio_emp.sum().to_dict()}
    records["published_scope_proxy_firms"] = int(len(proxy_firms)) if denominator_mode == "public_scope" else 0
    records["published_scope_proxy_sids_and_firm_counts"] = proxy_firms.stella_sid.value_counts().to_dict() if denominator_mode == "public_scope" else {}
    records["sector_ratio_sums_above_one_outside_source_overrides"] = sector_sums.loc[
        (sector_sums > 1) & ~sector_sums.index.isin(SPECIAL_SIDS)].to_dict()
    records["no_extra_sector_caps_note"] = "Only the three source overrides are applied. Global Compustat firm employment can exceed US BLS sector employment; additional exceedances are reported without adding new caps."
    return result, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path)
    p.add_argument("--rebuild-public", action="store_true")
    p.add_argument("--denominator-mode", choices=["public_scope", "strict_concordance"], default="public_scope")
    args = p.parse_args()
    if args.rebuild_public or not SECTORAL.exists():
        _, diagnostic = build_sectoral()
        print(json.dumps({k: v for k, v in diagnostic.items() if k != "incomplete_sector_years"}, indent=2))
    if args.calibration_dir:
        mapping = pd.read_csv(args.calibration_dir / "firm_mapping.csv", dtype={"gvkey": str, "stella_sid": str})
        result, diagnostic = reconstruct_employment_ratios(mapping, denominator_mode=args.denominator_mode)
        suffix = "" if args.denominator_mode == "public_scope" else "_strict_concordance"
        result.to_csv(args.calibration_dir / f"firm_employment_ratios{suffix}.csv")
        (args.calibration_dir / f"employment_diagnostics{suffix}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
