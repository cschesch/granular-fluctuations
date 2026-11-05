"""Balanced BLS CES employment panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mgp_core import ROOT, standardize_panel
from project_paths import data_file

from applications.panels import (
    SourcePanel,
    WEIGHT_TIMING,
    load_source_panel_cache,
    save_source_panel_cache,
    source_panel_cache_path,
)


def clean_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d.columns = d.columns.str.strip()
    return d


def arborescence(dces_wide: pd.DataFrame, tree_structure: pd.DataFrame, tolerance_pct: float = 1.0) -> pd.DataFrame:
    df_data = dces_wide.copy()
    df_data = df_data.loc[~df_data.index.duplicated(keep="first")]
    df_data.index = df_data.index.astype(str)

    df_tree = tree_structure.copy()
    df_tree["industry_code"] = df_tree["industry_code"].astype(str)
    df_tree["parent_code"] = df_tree["parent_code"].astype(str)
    df_tree["is_residual"] = df_tree["is_residual"].astype(int)

    valid_tree_codes = set(df_tree["industry_code"])
    df_data = df_data.loc[df_data.index.isin(valid_tree_codes)]
    residual_defs = df_tree[df_tree["is_residual"] == 1].copy()
    reported_edges = df_tree[df_tree["is_residual"] == 0].copy()

    child_to_parent = reported_edges.set_index("industry_code")["parent_code"]
    present_reported_children = child_to_parent.index.intersection(df_data.index)
    child_sum = df_data.loc[present_reported_children].groupby(
        child_to_parent.reindex(present_reported_children), sort=False
    ).sum()

    res_parent = residual_defs.set_index("industry_code")["parent_code"]
    present_parents = [p for p in res_parent.unique() if p in df_data.index]
    resid_by_parent = df_data.loc[present_parents].subtract(
        child_sum.reindex(present_parents).fillna(0.0),
        fill_value=0.0,
    )
    resid_rows = resid_by_parent.reindex(res_parent.values).copy()
    resid_rows.index = res_parent.index
    resid_rows = resid_rows.loc[~resid_rows.index.isin(df_data.index)]
    df_data = pd.concat([df_data, resid_rows], axis=0)

    all_child_to_parent = df_tree.set_index("industry_code")["parent_code"]
    present_children = all_child_to_parent.index.intersection(df_data.index)
    kids_sum_all = df_data.loc[present_children].groupby(
        all_child_to_parent.reindex(present_children), sort=False
    ).sum()
    parents_to_check = kids_sum_all.index.intersection(df_data.index)
    parents_to_check = parents_to_check[parents_to_check != "0"]
    err_pct = (
        (df_data.loc[parents_to_check] - kids_sum_all.loc[parents_to_check]).abs()
        / df_data.loc[parents_to_check].abs().clip(lower=1e-9)
    ) * 100.0
    failing = err_pct.max(axis=1).loc[lambda s: s > tolerance_pct].sort_values(ascending=False)
    if len(failing):
        raise ValueError(f"CES arborescence check failed for {len(failing)} parents; max={failing.iloc[0]:.3f}%.")

    current_codes = pd.Index(df_data.index.astype(str))
    active_parents = kids_sum_all.index.intersection(current_codes)
    active_parents = active_parents[active_parents != "0"]
    df_leaves = df_data.loc[~df_data.index.isin(active_parents)].copy()

    name_map = df_tree.set_index("industry_code")["industry_name"].to_dict()
    parent_map = df_tree.set_index("industry_code")["parent_code"].to_dict()
    for residual_code, parent_code in res_parent.items():
        name_map[residual_code] = f"{name_map.get(parent_code, parent_code)}_other"

    def path_of(code: str) -> list[str]:
        out = []
        cur = code
        while cur != "0" and cur is not None:
            out.append(name_map.get(cur, cur))
            cur = parent_map.get(cur, "0")
        return list(reversed(out)) if out else [name_map.get(code, code)]

    paths = [path_of(code) for code in df_leaves.index.astype(str)]
    max_depth = max(map(len, paths)) if paths else 1
    padded = [p + [p[-1]] * (max_depth - len(p)) for p in paths]
    df_leaves.index = pd.MultiIndex.from_tuples(
        [tuple(x) for x in padded],
        names=[f"industry{i + 1}" for i in range(max_depth)],
    )
    print(f"[bls_ces] arborescence leaves: {len(df_data)} nodes -> {len(df_leaves)} leaves", flush=True)
    return df_leaves


def load_bls_ces_panel(use_cache: bool = True, refresh_cache: bool = False) -> SourcePanel:
    data_path = data_file(ROOT / "data/bls_ces/ce.data.0.AllCESSeries.txt")
    industry_path = ROOT / "data/bls_ces/ce.industry.txt"
    tree_path = ROOT / "data/bls_ces/tree.csv"
    cache_path = source_panel_cache_path(
        "bls_ces_employment", {"weight_timing": WEIGHT_TIMING}, [data_path, industry_path, tree_path]
    )
    cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
    if cached is not None:
        return cached

    dces = clean_columns(pd.read_csv(data_path, sep="\t"))
    dces = dces[dces["period"] != "M13"]
    dces = dces[dces["footnote_codes"].isnull()]
    dces["series_id"] = dces["series_id"].astype(str).str.strip()
    dces = dces[dces["series_id"].str.startswith("CES") & dces["series_id"].str.endswith("01")]
    dces["month"] = dces["period"].str.lstrip("M").astype(int)
    dces["date"] = pd.to_datetime(dces[["year", "month"]].assign(day=1)).dt.to_period("M")
    dces["value"] = pd.to_numeric(dces["value"], errors="coerce")
    dces["industry_code"] = dces["series_id"].str[3:-2]
    dces = dces.dropna(subset=["value"])
    dces_wide = (
        dces[["industry_code", "date", "value"]]
        .set_index(["industry_code", "date"])
        .sort_index()
        .squeeze()
        .unstack()
        .loc[:, "1972-01":]
        .dropna(axis=0)
    )

    dcesi = clean_columns(pd.read_csv(industry_path, sep="\t"))
    dcesi["industry_code"] = dcesi["industry_code"].astype(str).str.zfill(8)
    dtree = pd.read_csv(tree_path)
    dtree["industry_code"] = dtree["industry_code"].astype(str)
    check = dtree.merge(dcesi[["industry_code", "industry_name"]], on="industry_code", how="left", suffixes=("", "_ces"))
    non_resid = check["is_residual"].eq(0) & check["industry_name_ces"].notna()
    if not (check.loc[non_resid, "industry_name"] == check.loc[non_resid, "industry_name_ces"]).all():
        raise ValueError("CES tree names do not match ce.industry names.")

    leaves = arborescence(dces_wide, dtree)
    leaves = leaves.loc[:, leaves.columns.sort_values()]
    sizes = leaves.iloc[:, 0].astype(float)
    y = np.log(leaves).diff(axis=1).iloc[:, 1:]
    y = standardize_panel(y)
    y = y.sub(y.mean().mean())
    labels = pd.Series([" / ".join(map(str, idx)) for idx in leaves.index], index=leaves.index)
    panel = SourcePanel(
        name="bls_ces_employment",
        y=y,
        sizes=sizes,
        labels=labels,
        start_date=y.columns[0],
        last_date=y.columns[-1],
        frequency="monthly",
        size_name="employment at 1972-01 (t-1)",
        value_name="log employment growth",
        coverage=1.0,
        notes="CES leaves after synthetic residual nodes; January 1972 (t-1) employment weights February 1972 onward growth; log growth winsorized at 1st/99th pct.",
        weight_timing=WEIGHT_TIMING,
        weight_date=leaves.columns[0],
    )
    save_source_panel_cache(cache_path, panel, use_cache)
    return panel
