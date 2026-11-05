#!/usr/bin/env python3
"""Build CPI arborescence panels for the MGP, mirroring the CES employment build.

Each panel is a partition of All items into leaf units, with one fixed size
share S_i per leaf and a monthly log-inflation series y_{i,t}:
Prices use BLS seasonally adjusted CUSR0000 series. Incomplete categories are
absorbed into broader/residual components, without a second seasonal adjustment.

  * 1967 panel -- the deepest partition from the actual 1967 historical CPI
    hierarchy whose leaves have continuous modern price series back to 1967.
    The historical table is parsed by contiguous weight partitions rather than
    raw indentation alone. Branches where a child lacks a usable price series
    are collected into residual leaves at the nearest priced ancestor. Residual
    inflation is backed out from the parent index after subtracting the actual
    fixed-weight child-subtree aggregate, so the size-weighted leaves telescope
    to All-items inflation.

Outputs (one leaves table + one wide monthly log-inflation matrix per panel) in
``.cache/derived/bls_cpi/panels/``. Residuals make split branches internally additive at
the fixed December 1967 weights.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DERIVED_CACHE, ROOT


BCPI = ROOT / "data" / "bls_cpi"
DERIVED = DERIVED_CACHE / "bls_cpi"
OUT = DERIVED / "panels"


# Some 1967 historical labels pre-date the current BLS item taxonomy. These
# mappings are only used when an exact current CPI item-name match is absent;
# completeness over the 1967 sample is still checked before any node is split.
HISTORICAL_CODE_ALIASES = {
    "apparel and upkeep": "SAA",
    "men s and boys": "SAA1",
    "women s and girl s": "SAA2",
    "men s": "SEAA",
    "boys": "SEAB",
    "women s": "SEAC",
    "girls": "SEAD",
    "rent": "SEHA",
    "hotels and motels": "SEHB",
    "fuel and utilities": "SAH2",
    "fuel oil and coal": "SEHE",
    "fuel oil 2": "SEHE01",
    "gas": "SEHF02",
    "residential water and sewerage services": "SEHG01",
    "household furnishings and operation": "SAH3",
    "furniture and bedding": "SEHJ",
    "bedroom furniture chest and dresser": "SEHJ01",
    "sofas upholstered": "SEHJ02",
    "housekeeping supplies": "SEHN",
    "domestic services": "SEHP01",
    "postal charges": "SEEC01",
    "appliances": "SEHK",
    "refrigerators freezer": "SEHK01",
    "dry cleaning": "SEGD03",
    "bread white": "SS02011",
    "milk fresh": "SS09011",
    "frying chicken": "SS06011",
    "bacon sliced": "SS04011",
    "coffee can": "SEFP01",
    "auto purchase": "SETA",
    "automobiles new": "SETA01",
    "automobiles used": "SETA02",
    "gasoline and motor oil": "SETB",
    "gasoline regular and premium": "SETB01",
    "motor oil premium": "SS47021",
    "auto parts": "SETC",
    "private": "SAT1",
    "public": "SETG",
    "auto repairs and maintenance": "SETD",
    "auto insurance rates": "SETE",
    "auto registration": "SETF01",
    "parking fees private and municipal": "SETF03",
    "local transit fares": "SETG03",
    "railroad fares coach": "SETG02",
    "airplane fares chiefly coach": "SETG01",
    "bus fares intercity": "SETG02",
    "drugs and prescriptions": "SEMF",
    "prescriptions": "SEMF01",
    "general physician house visit": "SEMC01",
    "general physician office visit": "SEMC01",
    "eyeglasses including examination": "SEMC03",
    "dentists fees": "SEMC02",
    "claims": "SEMD",
    "hospital services": "SEMD01",
    "toilet goods": "SEGB",
    "men s haircuts": "SEGC01",
    "beauty shop services": "SEGC01",
    "recreational goods": "SARC",
    "recreational services": "SARS",
    "indoor movie admissions": "SS62031",
    "newspapers street sale and delivery": "SERG01",
    "college tuition": "SEEB01",
    "cigarettes": "SEGA01",
    "beer at home": "SEFW01",
    "whiskey and wine": "SEFW03",
    "away from home beer": "SS20051",
    "funeral services adult": "SEGD02",
    "bank services charges checking account": "SS68021",
    "legal services short form will": "SEGD01",
}


# These are reusable words in old CPI tables whose current BLS labels are broad
# special aggregates, not the intended 1967 expenditure node.
HISTORICAL_NAME_DENYLIST = {
    "commodities",
    "services",
    "private",
    "public",
    "other priced items",
    "overhead",
}


def norm(s: str) -> str:
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Prices: wide monthly index level per item code, and log-growth.
# --------------------------------------------------------------------------
def load_price_wide() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "cpi_price_long.parquet")
    wide = p.pivot(index="date", columns="item_code", values="value").sort_index()
    return wide


def load_item_code_by_name() -> dict[str, str]:
    """Current BLS item-name normalisation -> item code, keeping only unique names."""
    items = pd.read_csv(BCPI / "cu.item", sep="\t")
    items["item_name_norm"] = items["item_name"].map(norm)
    n_codes = items.groupby("item_name_norm")["item_code"].nunique()
    unique_names = set(n_codes[n_codes.eq(1)].index)
    return (
        items[items["item_name_norm"].isin(unique_names)]
        .drop_duplicates("item_name_norm")
        .set_index("item_name_norm")["item_code"]
        .to_dict()
    )


def historical_item_code(item_name: str, code_by_name: dict[str, str]) -> str | None:
    item_key = norm(item_name)
    if item_key in HISTORICAL_CODE_ALIASES:
        return HISTORICAL_CODE_ALIASES[item_key]
    if item_key in HISTORICAL_NAME_DENYLIST:
        return None
    return code_by_name.get(item_key)


def load_1967_historical_tree():
    """Use the actual 1967 historical item hierarchy rather than a modern tree.

    The 1967 Excel indentation is irregular, so source rows are reconstructed as
    contiguous child partitions whose weights add back to the parent. The parser
    tries detailed subtrees first and falls back to terminal leaves only when the
    enclosing partition requires it.
    """
    weights = pd.read_csv(DERIVED / "cpi_weight_series_1967_onward.csv")
    exp = weights[(weights["year"].eq(1967)) & (weights["section"].eq("expenditure"))].copy()
    items = (
        exp[exp["source_block"].eq("individual_items")]
        .dropna(subset=["source_row", "level", "cpi_u"])
        .sort_values("source_row")
        .copy()
    )
    if items.empty:
        raise ValueError("No 1967 historical CPI individual-items hierarchy found.")

    code_by_name = load_item_code_by_name()
    root = "hist:root"
    top_names = set(
        exp[
            exp["source_block"].eq("major_groups")
            & exp["level"].eq(exp.loc[exp["source_block"].eq("major_groups"), "level"].min())
        ]["item_name"].map(norm)
    )
    # The historical individual-item block has a 0.38 pp all-items residual
    # labelled Miscellaneous; it completes the top-level partition to 100.
    top_names.add("miscellaneous")

    weight_tolerance_cents = 1

    item_records = items.to_dict("records")

    def row_cents(pos: int) -> int:
        return int(round(100 * float(item_records[pos]["cpi_u"])))

    @lru_cache(maxsize=None)
    def subtree_options(pos: int, end: int):
        child_end, grandchildren = parse_children(pos, pos + 1, end)
        out = []
        if grandchildren:
            out.append((child_end, grandchildren))
        out.append((pos + 1, ()))
        return tuple(out)

    @lru_cache(maxsize=None)
    def parse_children(parent_pos: int, start: int, end: int):
        target = row_cents(parent_pos)

        @lru_cache(maxsize=None)
        def search(pos: int, total: int):
            if abs(total - target) <= weight_tolerance_cents:
                return pos, ()
            if pos >= end or total > target + weight_tolerance_cents:
                return None
            child_w = row_cents(pos)
            if total + child_w > target + weight_tolerance_cents:
                return None
            for child_end, grandchildren in subtree_options(pos, end):
                found = search(child_end, total + child_w)
                if found is not None:
                    final_end, rest = found
                    return final_end, ((pos, grandchildren),) + rest
            return None

        found = search(start, 0)
        return found if found is not None else (start, ())

    def add_node(pos: int, parent: str, rows: dict[str, dict[str, object]], children: dict[str, list[str]]) -> str:
        r = item_records[pos]
        source_row = int(r["source_row"])
        item_name = str(r["item_name"])
        node_id = f"hist:{source_row}"
        parent_label = str(rows[parent]["label"])
        label = item_name if parent == root else f"{parent_label} / {item_name}"
        rows[node_id] = {
            "item_name": item_name,
            "label": label,
            "code": historical_item_code(item_name, code_by_name),
            "cpi_u": float(r["cpi_u"]),
        }
        children.setdefault(node_id, [])
        children[parent].append(node_id)
        return node_id

    rows: dict[str, dict[str, object]] = {
        root: {
            "item_name": "all items",
            "label": "all items",
            "code": code_by_name.get("all items", "SA0"),
            "cpi_u": 100.0,
        }
    }
    children: dict[str, list[str]] = {root: []}

    top_positions = [i for i, r in enumerate(item_records) if norm(r["item_name"]) in top_names]
    top_weight_cents = sum(row_cents(i) for i in top_positions)
    if abs(top_weight_cents - 10000) > weight_tolerance_cents:
        top_weight = top_weight_cents / 100
        raise ValueError(f"1967 historical top-level CPI weights sum to {top_weight:.3f}, not 100.")

    parsed_positions: set[int] = set()

    def attach(parent: str, pos: int, parsed_children) -> None:
        parsed_positions.add(pos)
        node_id = add_node(pos, parent, rows, children)
        for child_pos, grandchildren in parsed_children:
            attach(node_id, child_pos, grandchildren)

    for n, pos in enumerate(top_positions):
        end = top_positions[n + 1] if n + 1 < len(top_positions) else len(item_records)
        parsed_end, parsed_children = subtree_options(pos, end)[0]
        if parsed_end != end:
            unparsed = item_records[parsed_end:end][:3]
            sample = ", ".join(str(r["item_name"]) for r in unparsed)
            raise ValueError(
                f"1967 historical CPI parser stopped inside {item_records[pos]['item_name']!r}; "
                f"next unparsed rows: {sample}"
            )
        attach(root, pos, parsed_children)

    if len(parsed_positions) != len(item_records):
        raise ValueError(
            f"1967 historical CPI parser attached {len(parsed_positions)} of {len(item_records)} rows."
        )

    tree = pd.DataFrame.from_dict(rows, orient="index")
    return tree, children, root


def last_contiguous_period(series: pd.Series, baseline: str) -> pd.Period:
    """Last valid period before the first gap after the baseline observation."""
    base = pd.Period(baseline, freq="M")
    s = series.loc[base:]
    if s.empty or pd.isna(s.iloc[0]):
        raise ValueError(f"Baseline {baseline} is missing for the root CPI series.")
    missing = s.index[s.isna()]
    if len(missing):
        return missing[0] - 1
    return s.index[-1]


def build_arborescence(
    baseline: str,
    last_period: str | pd.Period | None,
    tree: pd.DataFrame,
    children: dict[str, list[str]],
    root: str,
    price_wide: pd.DataFrame,
    *,
    residual_min_weight: float = 0.005,
    overfull_tolerance: float = 0.05,
):
    """Return (leaves_df, infl_df). Descends into a child if it has a complete
    price series, or if its usable descendants exhaust the child's weight.
    Unexplained parent weight becomes a residual leaf with backed-out inflation."""
    baseline_period = pd.Period(baseline, freq="M")

    def code(nm):
        c = tree.loc[nm, "code"]
        return c if isinstance(c, str) else None

    def display(nm):
        if "label" in tree.columns and isinstance(tree.loc[nm, "label"], str):
            return tree.loc[nm, "label"]
        return nm

    def source_item(nm):
        v = tree.loc[nm, "item_name"]
        return v if isinstance(v, str) else display(nm)

    def wt(nm):
        v = tree.loc[nm, "cpi_u"]
        return float(v) if pd.notna(v) else None

    root_code = code(root)
    if root_code is None or root_code not in price_wide.columns:
        raise ValueError(f"Root {root!r} has no CPI price series.")
    if last_period is None:
        last_period = last_contiguous_period(price_wide[root_code], baseline)
    else:
        last_period = pd.Period(last_period, freq="M")
    level_index = pd.period_range(baseline_period, last_period, freq="M")
    growth_index = level_index[1:]
    if len(growth_index) == 0:
        raise ValueError("CPI panel needs at least one post-baseline month.")

    def complete(nm):
        c = code(nm)
        if c is None or c not in price_wide.columns:
            return False
        return not price_wide[c].reindex(level_index).isna().any()

    @lru_cache(maxsize=None)
    def can_use_subtree(nm):
        if complete(nm):
            return True
        node_w = wt(nm)
        if node_w is None:
            return False
        kids = [k for k in children.get(nm, []) if can_use_subtree(k)]
        if not kids:
            return False
        child_weights = [wt(k) for k in kids]
        if any(w is None for w in child_weights):
            return False
        return abs(sum(float(w) for w in child_weights) - node_w) <= overfull_tolerance

    leaves = []   # (label, code, size, kind, source_item_name)
    infl = {}     # label -> log-growth series

    def node_growth(nm):
        levels = price_wide[code(nm)].reindex(level_index)
        return np.log(levels).diff().loc[growth_index]

    pruned: list[tuple[str, str, float]] = []

    def prune_overfull_children(nm, kids, node_w):
        """Drop clear indentation/crosswalk artifacts whose weights overfill a node."""
        child_w = {k: wt(k) for k in kids}
        total = sum(child_w.values())
        if total <= node_w + overfull_tolerance:
            return kids
        overage = total - node_w
        candidates = [
            k
            for k, w in child_w.items()
            if w >= overage - overfull_tolerance and total - w <= node_w + overfull_tolerance
        ]
        if candidates:
            dropped = [min(candidates, key=lambda k: (abs(child_w[k] - overage), child_w[k]))]
        else:
            dropped = []
            remaining = total
            for k in sorted(kids, key=lambda k: child_w[k]):
                dropped.append(k)
                remaining -= child_w[k]
                if remaining <= node_w + overfull_tolerance:
                    break
        pruned.extend((nm, k, child_w[k]) for k in dropped)
        dropped = set(dropped)
        return [k for k in kids if k not in dropped]

    def visit(nm, size):
        node_w = wt(nm)
        node_w = node_w if node_w is not None else size
        kids = [k for k in children.get(nm, []) if can_use_subtree(k)]
        kids = prune_overfull_children(nm, kids, node_w)
        lbl = display(nm)
        if not kids:
            if not complete(nm):
                raise ValueError(f"CPI node {display(nm)!r} has no complete price series or usable children.")
            leaves.append((lbl, code(nm), size, "leaf", source_item(nm)))
            g = node_growth(nm)
            infl[lbl] = g
            return g
        # Children use their own period weights; any unexplained mass becomes residual.
        child_w = {k: wt(k) for k in kids}
        child_growth = {}
        for k in kids:
            child_growth[k] = visit(k, child_w[k])
        resid_w = node_w - sum(child_w.values())
        if resid_w < -overfull_tolerance:
            raise ValueError(
                f"CPI tree overfull at {nm}: children exceed parent by {-resid_w:.3f} pp"
            )
        if resid_w > residual_min_weight:
            if not complete(nm):
                raise ValueError(
                    f"CPI node {display(nm)!r} needs a residual but has no complete parent price series."
                )
            lbl = f"{lbl} (residual)"
            g_par = node_growth(nm)
            g_kids = sum(child_w[k] * child_growth[k] for k in kids)
            g_res = (node_w * g_par - g_kids) / resid_w
            leaves.append((lbl, "RESIDUAL", resid_w, "residual", source_item(nm)))
            infl[lbl] = g_res
            return g_par
        return sum(child_w[k] * child_growth[k] for k in kids) / node_w

    root_w = wt(root)
    visit(root, root_w if root_w is not None else 100.0)

    lf = pd.DataFrame(leaves, columns=["label", "item_code", "weight", "kind", "source_item_name"])
    lf["size_share"] = lf["weight"] / lf["weight"].sum()
    infl_df = pd.DataFrame(infl)
    if infl_df.isna().any(axis=None):
        bad = infl_df.columns[infl_df.isna().any()].tolist()[:10]
        raise ValueError(f"CPI panel has missing inflation after construction: {bad}")
    if pruned:
        sample = "; ".join(
            f"{display(parent)} -> dropped {display(child)} ({w:.3f})"
            for parent, child, w in pruned[:5]
        )
        print(f"[cpi-panel] pruned {len(pruned)} overfull child link(s): {sample}")
    return lf, infl_df[lf["label"].tolist()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    price = load_price_wide()

    tree, children, root = load_1967_historical_tree()
    leaves, inflation = build_arborescence("1967-12", "2025-09", tree, children, root, price)
    leaves.to_csv(OUT / "cpi_1967_leaves.csv", index=False)
    inflation.to_parquet(OUT / "cpi_1967_inflation.parquet")
    print(f"[panel 1967] {len(leaves)} leaves, {len(inflation)} months; "
          f"{inflation.index.min()}..{inflation.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
