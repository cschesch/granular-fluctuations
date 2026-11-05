"""Offline, dated Exhibit 21 aliases for a separate Welburn input overlay.

This module does not connect to WRDS and never changes the shared customer
matching cache.  A name must match an extracted subsidiary name exactly after
the existing name normalization, identify one eligible financial parent, and
appear in that parent's most recent observed Exhibit 21 filing at the customer
record date.  An alias expires at the next observed parent filing or after
550 days.  Filing dates establish availability, not exact ownership dates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from applications.welburn.prepare_welburn_links import normalize_name


MAX_FILING_AGE_DAYS = 550


def _ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _edges(frame: pd.DataFrame, mask: pd.Series, column: str, ids: set[str]) -> pd.DataFrame:
    out = frame.loc[
        mask & frame["usable_name"].fillna(False)
        & frame["gvkey"].isin(ids) & frame[column].isin(ids)
        & frame["gvkey"].ne(frame[column]),
        ["gvkey", column],
    ].rename(columns={column: "customer_gvkey"})
    return out.drop_duplicates().sort_values(["gvkey", "customer_gvkey"]).reset_index(drop=True)


def _topology(edges: pd.DataFrame, ids: set[str]) -> dict:
    index = pd.Index(sorted(ids))
    outgoing = edges.groupby("gvkey").size().reindex(index, fill_value=0)
    incoming = edges.groupby("customer_gvkey").size().reindex(index, fill_value=0)
    total = outgoing + incoming
    return {
        "firms": len(ids), "directed_edges": len(edges),
        "mean_total_degree": float(total.mean()),
        "median_total_degree": float(total.median()),
        "max_total_degree": int(total.max()),
        "max_indegree": int(incoming.max()), "max_outdegree": int(outgoing.max()),
        "isolated_firms": int(total.eq(0).sum()),
    }


def _alias_filings(path, target_names):
    """Read thin aliases in chunks, retaining every parent's filing timeline."""
    aliases, filing_pieces = [], []
    normalization_cache = {}
    rows = invalid_dates = missing_parent_ids = 0
    parent_ids = set()
    for chunk in pd.read_csv(path, dtype={"gvkey": "string", "cik": "string"}, chunksize=200_000):
        required = {"gvkey", "cik", "fdate", "subsidiary_name", "clean_company"}
        if not required.issubset(chunk):
            raise ValueError(f"Subsidiary input must contain {sorted(required)}")
        rows += len(chunk)
        chunk["parent_gvkey"] = _ids(chunk["gvkey"])
        cik = chunk["cik"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
        # CIK, including parents without an eligible GVKEY, defines collision
        # identity and filing history.  Missing CIK falls back to GVKEY.
        chunk["parent_key"] = "CIK:" + cik
        missing_cik = cik.isna() | cik.eq("0000000000")
        chunk.loc[missing_cik, "parent_key"] = "GVKEY:" + chunk.loc[missing_cik, "parent_gvkey"]
        chunk["filing_date"] = pd.to_datetime(chunk["fdate"], errors="coerce")
        invalid_dates += int(chunk["filing_date"].isna().sum())
        missing_parent_ids += int(chunk["parent_key"].isna().sum())
        chunk = chunk.loc[chunk["filing_date"].notna() & chunk["parent_key"].notna()]
        parent_ids.update(chunk["parent_gvkey"].dropna())
        filing_pieces.append(chunk[["parent_key", "filing_date"]].drop_duplicates())
        for column in ["subsidiary_name", "clean_company"]:
            for value in chunk[column].dropna().drop_duplicates():
                if value not in normalization_cache:
                    normalization_cache[value] = normalize_name(value)
            keys = chunk[column].map(normalization_cache)
            part = chunk.loc[keys.isin(target_names), ["parent_key", "parent_gvkey", "filing_date"]].copy()
            part["customer_key"] = keys.loc[part.index]
            part["alias_source"] = column
            aliases.append(part.drop_duplicates())
    filings = pd.concat(filing_pieces, ignore_index=True).drop_duplicates().sort_values(
        ["parent_key", "filing_date"],
    )
    filings["next_filing_date"] = filings.groupby("parent_key")["filing_date"].shift(-1)
    aliases = pd.concat(aliases, ignore_index=True).drop_duplicates().merge(
        filings, on=["parent_key", "filing_date"], how="left", validate="many_to_one",
    )
    return aliases, filings, {
        "lookup_rows": rows, "invalid_filing_date_rows": invalid_dates,
        "missing_parent_identifier_rows": missing_parent_ids,
        "distinct_nonmissing_parent_gvkeys": len(parent_ids),
    }


def analyze_subsidiaries(base_or_matched, subsidiary_path, universe, private_dir, collision_path=None):
    """Return aggregate diagnostics and the full, separately matched row frame.

    ``base_or_matched`` is a DataFrame or parquet path.  The source-1 field
    ``customer_gvkey_after_ids`` takes precedence when present; otherwise the
    original ``customer_gvkey`` is used.  Only unresolved usable names in source
    years 2015--2019 can receive new IDs.  Existing matches are preserved even
    when they disagree with an Exhibit 21 candidate.  Row-level evidence and
    before/after edge unions are written exclusively to ``private_dir``.  Supply
    a thin all-parent ``collision_path`` to require uniqueness beyond the
    financial-universe extract, including filing parents without a GVKEY.
    """
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    frame = (
        base_or_matched.copy() if isinstance(base_or_matched, pd.DataFrame)
        else pd.read_parquet(base_or_matched)
    ).reset_index(drop=True)
    base_column = (
        "customer_gvkey_after_ids" if "customer_gvkey_after_ids" in frame
        else "customer_gvkey"
    )
    frame["gvkey"] = _ids(frame["gvkey"])
    frame[base_column] = _ids(frame[base_column])
    ids = set(_ids(pd.Series(list(universe))).dropna())
    dates = pd.to_datetime(frame["srcdate"], errors="coerce")
    annual = dates.dt.year.between(2015, 2019)
    paper = dates.between("2015-01-01", "2019-03-31")
    usable = frame["usable_name"].fillna(False).astype(bool)
    considered = annual & usable
    queries = pd.DataFrame({
        "source_row": np.flatnonzero(considered),
        "srcdate": dates.loc[considered].to_numpy(),
        "customer_key": frame.loc[considered, "customer_key"].to_numpy(),
    })

    target_names = set(queries["customer_key"].dropna())
    lookup_path = collision_path if collision_path is not None else subsidiary_path
    aliases, filings, lookup_report = _alias_filings(lookup_path, target_names)
    candidate = queries.merge(aliases, on="customer_key", how="inner", validate="many_to_many")
    candidate["filing_age_days"] = (candidate["srcdate"] - candidate["filing_date"]).dt.days
    not_future = candidate["filing_date"].le(candidate["srcdate"])
    before_next = candidate["next_filing_date"].isna() | candidate["srcdate"].lt(candidate["next_filing_date"])
    not_stale = candidate["filing_age_days"].le(MAX_FILING_AGE_DAYS)
    valid = candidate.loc[not_future & before_next & not_stale].copy()
    parent_counts = valid.groupby("source_row")["parent_key"].nunique()
    mapped_parent_counts = valid.groupby("source_row")["parent_gvkey"].nunique()
    first_parent = valid.drop_duplicates(["source_row", "parent_gvkey"]).groupby("source_row")["parent_gvkey"].first()
    frame["subsidiary_candidate_parent_count"] = frame.index.to_series().map(parent_counts).fillna(0).astype(int)
    frame["subsidiary_candidate_gvkey_count"] = frame.index.to_series().map(mapped_parent_counts).fillna(0).astype(int)
    frame["subsidiary_candidate_parent"] = frame.index.to_series().map(first_parent).astype("string")
    unique = frame["subsidiary_candidate_parent_count"].eq(1) & frame["subsidiary_candidate_gvkey_count"].eq(1)
    frame.loc[~unique, "subsidiary_candidate_parent"] = pd.NA
    eligible_parent = frame["subsidiary_candidate_parent"].isin(ids)
    unresolved = frame[base_column].isna()
    frame["source2_ambiguous"] = considered & frame["subsidiary_candidate_parent_count"].gt(1)
    frame["source2_self_link"] = considered & unique & frame["gvkey"].eq(frame["subsidiary_candidate_parent"]).fillna(False)
    frame["source2_conflict"] = (
        considered & unique & ~unresolved
        & frame[base_column].ne(frame["subsidiary_candidate_parent"]).fillna(False)
    )
    frame["source2_rescued"] = considered & unresolved & unique & eligible_parent & ~frame["source2_self_link"]
    frame["customer_gvkey_after_subsidiaries"] = frame[base_column].copy()
    frame.loc[frame["source2_rescued"], "customer_gvkey_after_subsidiaries"] = frame.loc[
        frame["source2_rescued"], "subsidiary_candidate_parent",
    ]
    pd.testing.assert_series_equal(
        frame.loc[~unresolved, "customer_gvkey_after_subsidiaries"],
        frame.loc[~unresolved, base_column], check_names=False,
    )
    assert not valid["filing_date"].gt(valid["srcdate"]).any()
    assert valid["filing_age_days"].between(0, MAX_FILING_AGE_DAYS).all()

    paths = {}
    for label, data in [("matched_rows", frame), ("valid_candidates", valid), ("matched_alias_filings", aliases)]:
        path = private_dir / f"subsidiary_{label}.parquet"
        data.to_parquet(path, index=False)
        paths[label] = str(path)
    report = {
        "source": str(subsidiary_path), "collision_lookup": str(lookup_path),
        "global_parent_collision_check": collision_path is not None,
        "base_id_column": base_column,
        "normalization": "prepare_welburn_links.normalize_name; exact subsidiary_name or clean_company",
        "rule": "Only unresolved usable 2015--2019 records; unique eligible parent in latest observed Exhibit 21 filing at or before srcdate; expiry at next parent filing or 550 days; existing IDs preserved; selflinks rejected.",
        "max_filing_age_days": MAX_FILING_AGE_DAYS,
        **lookup_report,
        "distinct_parent_filing_dates": len(filings),
        "parents_with_filings": int(filings["parent_key"].nunique()),
        "query_alias_candidate_rows_before_date_filter": len(candidate),
        "future_candidate_rows_rejected": int((~not_future).sum()),
        "superseded_candidate_rows_rejected": int((not_future & ~before_next).sum()),
        "stale_current_candidate_rows_rejected": int((not_future & before_next & ~not_stale).sum()),
        "valid_candidate_rows": len(valid),
        "private_outputs": paths,
        "limitations": [
            "Filing dates define information availability; ownership acquisition/disposal dates are not identified.",
            "Intervals end at next observed parent Exhibit21 filing; an unobserved or empty exhibit cannot be detected.",
            "Same-date parent filings are unioned; exact name collisions across parents remain unresolved.",
            (
                "Collision uniqueness is checked across all parents in the supplied SEC Exhibit21 extraction, including nonfinancial parents; omitted or unextracted filings remain unobserved."
                if collision_path is not None else
                "PROVISIONAL: the alias table contains only eligible financial parents, so uniqueness is within that export, not global."
            ),
        ],
        "windows": {},
    }
    for label, mask in [("annual_2015_2019", annual), ("paper_2015Q1_2019Q1", paper)]:
        before = _edges(frame, mask, base_column, ids)
        after = _edges(frame, mask, "customer_gvkey_after_subsidiaries", ids)
        delta = after.merge(before, on=["gvkey", "customer_gvkey"], how="left", indicator=True)
        delta = delta.loc[delta["_merge"].eq("left_only"), ["gvkey", "customer_gvkey"]]
        edge_path = private_dir / f"subsidiary_union_edges_{label}.parquet"
        added_path = private_dir / f"subsidiary_added_edges_{label}.parquet"
        after.to_parquet(edge_path, index=False)
        delta.to_parquet(added_path, index=False)
        rescue = mask & frame["source2_rescued"]
        unresolved_before = mask & usable & unresolved
        report["windows"][label] = {
            "source_company_rows": int(mask.sum()),
            "usable_named_rows": int((mask & usable).sum()),
            "unresolved_usable_rows_before": int(unresolved_before.sum()),
            "unresolved_usable_names_before": int(frame.loc[unresolved_before, "customer_key"].nunique()),
            "newly_resolved_rows": int(rescue.sum()),
            "newly_resolved_names": int(frame.loc[rescue, "customer_key"].nunique()),
            "unresolved_usable_rows_after": int((unresolved_before & ~rescue).sum()),
            "ambiguous_rows_all": int((mask & frame["source2_ambiguous"]).sum()),
            "ambiguous_rows_previously_unresolved": int((unresolved_before & frame["source2_ambiguous"]).sum()),
            "parent_gvkey_ambiguity_rows": int((mask & frame["subsidiary_candidate_parent_count"].eq(1) & frame["subsidiary_candidate_gvkey_count"].gt(1)).sum()),
            "unique_outside_or_unmapped_parent_rows": int((mask & frame["subsidiary_candidate_parent_count"].eq(1) & ~eligible_parent).sum()),
            "unique_self_candidate_rows_all": int((mask & frame["source2_self_link"]).sum()),
            "unique_self_candidate_rows_previously_unresolved": int((unresolved_before & frame["source2_self_link"]).sum()),
            "existing_match_conflict_rows": int((mask & frame["source2_conflict"]).sum()),
            "existing_match_agreement_rows": int((mask & ~unresolved & unique & ~frame["source2_conflict"]).sum()),
            "newly_resolved_rows_with_eligible_supplier": int((rescue & frame["gvkey"].isin(ids)).sum()),
            "unique_added_edges": len(delta), "before": _topology(before, ids),
            "after": _topology(after, ids),
            "private_union_edges": str(edge_path), "private_added_edges": str(added_path),
        }
    return report, frame
