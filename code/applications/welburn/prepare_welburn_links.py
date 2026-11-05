#!/usr/bin/env python3
"""Prepare observed Compustat major-customer links for a Welburn-style network.

The binary relationship is the input to Welburn et al.'s link model and flow
calibration.  It must therefore retain an identified named customer even when
Compustat does not report ``salecs`` for that relationship.  A separate file of
positive reported customer sales is retained for validation and other uses.
Licensed row-level outputs stay under .cache.
"""

from __future__ import annotations

from wrds_inputs import wrds_path

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT

DEFAULT_CUSTOMERS = wrds_path("compustat_customers")
DEFAULT_FUNDA = wrds_path("compustat_annual")
DEFAULT_CACHE = ROOT / "code/.cache/firm_networks"
DEFAULT_DIAGNOSTICS = ROOT / "output/welburn_observed_link_diagnostics.json"
STELLA_ALIASES = ROOT / "data/stella2015/manual_customer_aliases.csv"
WELBURN_ALIASES = ROOT / "data/compustat/welburn_customer_aliases.csv"

LEGAL_SUFFIXES = {
    "AG", "BV", "CO", "COMPANY", "CORP", "CORPORATION", "GMBH", "GROUP",
    "HOLDING", "HOLDINGS", "INC", "INCORPORATED", "LIMITED", "LLC", "LLP",
    "ADR", "LP", "LTD", "NV", "PLC", "SA", "SPA",
}
NON_NAMES = re.compile(
    r"(?:^|\s)(?:CUSTOMERS?|DISTRIBUTORS?|DEALERS?|RETAILERS?|NOT REPORTED|"
    r"UNNAMED|VARIOUS|MULTIPLE|OTHER)(?:\s|$)"
)


def normalize_name(value: object) -> str:
    """Conservative ASCII company-name key, removing only terminal legal forms."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    text = text.upper().replace("&", " AND ")
    words = re.sub(r"[^A-Z0-9]+", " ", text).split()
    while words and words[0] == "THE":
        words = words[1:]
    # SEC/Compustat names commonly end in ``INC (THE)`` or ``CO, THE``.
    # Removing THE and legal forms in the same loop makes these resolve to the
    # same key as ``THE HOME DEPOT, INC.`` without changing interior words.
    while words and (words[-1] in LEGAL_SUFFIXES or words[-1] == "THE"):
        words.pop()
    return " ".join(words)


def reviewed_alias_map(paths: list[Path]) -> dict[str, str]:
    """Load auditable historical-name and parent-company resolutions."""
    out: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        aliases = pd.read_csv(path, dtype={"customer_gvkey": "string"})
        required = {"customer_key", "customer_gvkey"}
        if not required.issubset(aliases.columns):
            raise ValueError(f"Alias file {path} must contain {sorted(required)}")
        for key, gvkey in aliases[["customer_key", "customer_gvkey"]].itertuples(index=False, name=None):
            normalized = normalize_name(key)
            if normalized:
                out[normalized] = str(gvkey).zfill(6)
    return out


def alias_maps(funda: Path, customers: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Build unique name and compact-name maps from Compustat company aliases."""
    aliases: dict[str, set[str]] = defaultdict(set)
    a = pd.read_csv(funda, usecols=["gvkey", "conm"], dtype={"gvkey": "string"})
    for gvkey, name in a.drop_duplicates().itertuples(index=False, name=None):
        key = normalize_name(name)
        if key:
            aliases[key].add(str(gvkey))

    # Segment headers preserve additional historical spellings of supplier names.
    for chunk in pd.read_csv(
        customers, usecols=["gvkey", "conm"], dtype={"gvkey": "string"},
        chunksize=200_000,
    ):
        for gvkey, name in chunk.drop_duplicates().itertuples(index=False, name=None):
            key = normalize_name(name)
            if key:
                aliases[key].add(str(gvkey))

    unique = {key: next(iter(ids)) for key, ids in aliases.items() if len(ids) == 1}
    compact_sets: dict[str, set[str]] = defaultdict(set)
    for key, gvkey in unique.items():
        compact_sets[key.replace(" ", "")].add(gvkey)
    compact = {key: next(iter(ids)) for key, ids in compact_sets.items() if len(ids) == 1}
    return unique, compact


def prepare(
    customers: Path,
    funda: Path,
    cache_dir: Path,
    diagnostics_path: Path,
    alias_paths: list[Path] | None = None,
) -> dict:
    unique, compact = alias_maps(funda, customers)
    reviewed = reviewed_alias_map(
        alias_paths if alias_paths is not None else [STELLA_ALIASES, WELBURN_ALIASES]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    raw_company_rows = 0

    usecols = ["srcdate", "gvkey", "conm", "cid", "cnms", "ctype", "salecs", "sid", "stype"]
    for chunk in pd.read_csv(customers, usecols=usecols, dtype={"gvkey": "string"}, chunksize=200_000):
        chunk = chunk.loc[chunk["ctype"].eq("COMPANY")].copy()
        raw_company_rows += len(chunk)
        chunk["customer_key"] = chunk["cnms"].map(normalize_name)
        chunk["year"] = pd.to_datetime(chunk["srcdate"], errors="coerce").dt.year
        chunk["salecs"] = pd.to_numeric(chunk["salecs"], errors="coerce")
        chunk["customer_gvkey"] = chunk["customer_key"].map(unique)
        missing = chunk["customer_gvkey"].isna() & chunk["customer_key"].ne("")
        chunk.loc[missing, "customer_gvkey"] = (
            chunk.loc[missing, "customer_key"].str.replace(" ", "", regex=False).map(compact)
        )
        chunk["match_method"] = np.where(
            chunk["customer_gvkey"].notna(), "unique_exact_or_compact_name", "unmatched"
        )
        missing = chunk["customer_gvkey"].isna() & chunk["customer_key"].ne("")
        manual = chunk.loc[missing, "customer_key"].map(reviewed)
        chunk.loc[missing, "customer_gvkey"] = manual
        chunk.loc[missing & manual.notna(), "match_method"] = "reviewed_alias_or_parent"
        chunk["usable_name"] = chunk["customer_key"].ne("") & ~chunk["customer_key"].str.contains(NON_NAMES, na=False)
        pieces.append(chunk)

    links = pd.concat(pieces, ignore_index=True)
    valid = links[
        links["year"].notna() & links["usable_name"] & links["customer_gvkey"].notna()
        & links["gvkey"].ne(links["customer_gvkey"])
    ].copy()
    valid["year"] = valid["year"].astype(int)
    binary = valid.groupby(["year", "gvkey", "customer_gvkey"], as_index=False).agg(
        salecs=("salecs", lambda x: x.sum(min_count=1)),
        source_rows=("customer_key", "size"),
    )
    flows = binary.loc[binary["salecs"].gt(0), ["year", "gvkey", "customer_gvkey", "salecs"]]

    links.to_parquet(cache_dir / "customer_name_matches.parquet", index=False)
    binary.to_parquet(cache_dir / "observed_supplier_customer_links.parquet", index=False)
    flows.to_parquet(cache_dir / "observed_supplier_customer_flows.parquet", index=False)

    positive = links["salecs"].gt(0)
    matched = links["customer_gvkey"].notna()
    total_sales = float(links.loc[positive, "salecs"].sum())
    matched_sales = float(links.loc[positive & matched, "salecs"].sum())
    diagnostics = {
        "raw_company_rows": raw_company_rows,
        "rows_with_usable_names": int(links["usable_name"].sum()),
        # ``matched`` includes both unambiguous name matches and reviewed
        # aliases/parents; keep the diagnostic labels explicit about that.
        "rows_resolved_total": int(matched.sum()),
        "rows_resolved_by_unique_exact_or_compact_name": int(
            links["match_method"].eq("unique_exact_or_compact_name").sum()
        ),
        "rows_matched_by_reviewed_alias_or_parent": int(
            links["match_method"].eq("reviewed_alias_or_parent").sum()
        ),
        "reviewed_aliases_available": int(len(reviewed)),
        "positive_sales_rows": int(positive.sum()),
        "positive_sales_rows_matched": int((positive & matched).sum()),
        "resolved_share_positive_rows": float((positive & matched).sum() / max(positive.sum(), 1)),
        "resolved_share_positive_sales": matched_sales / max(total_sales, 1.0),
        "observed_directed_firm_year_binary_links": int(len(binary)),
        "observed_directed_firm_year_links_with_positive_sales": int(len(flows)),
        # Backward-compatible name: this used to be the positive-sales count.
        "observed_directed_firm_year_links": int(len(binary)),
        "unique_suppliers_in_binary_links": int(binary["gvkey"].nunique()),
        "unique_customers_in_binary_links": int(binary["customer_gvkey"].nunique()),
        "unique_suppliers_in_matched_flows": int(flows["gvkey"].nunique()),
        "unique_customers_in_matched_flows": int(flows["customer_gvkey"].nunique()),
        "first_year": int(binary["year"].min()),
        "last_year": int(binary["year"].max()),
        "row_level_outputs": "Licensed; stored only in code/.cache/firm_networks.",
        "network_input_note": (
            "Welburn's binary adjacency uses every resolved named relationship; positive salecs "
            "is not required because edge flows are assigned in the later sales/COGS program."
        ),
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS)
    parser.add_argument("--funda", type=Path, default=DEFAULT_FUNDA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--aliases", type=Path, action="append", default=None)
    args = parser.parse_args()
    result = prepare(args.customers, args.funda, args.cache_dir, args.diagnostics, args.aliases)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
