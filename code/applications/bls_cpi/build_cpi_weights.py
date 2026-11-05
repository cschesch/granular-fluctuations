#!/usr/bin/env python3
"""Parse BLS CPI relative-importance files into one clean weights panel.

Inputs (already downloaded into ``data/bls_cpi/relative_importance_raw/``):
  - ri-archive-1990-1999.zip : yearly ``.txt`` tables.
      * 1990-1996 carry an explicit BLS *item code* column (old, pre-1998
        item structure) in a flat ``code  name  CPI-U  CPI-W`` layout.
      * 1997-1999 are the indented-name layout on the *new* (1998) structure.
  - ri-archive-2000-2009.zip, ri-archive-2010-2019.zip : indented-name ``.txt``.
  - 2020.xlsx ... 2025.xlsx, cpi-relative-importance-current.xlsx :
      an ``Indent Level / Item and Group / CPI-U / CPI-W`` workbook layout.

Each yearly table reports the December relative importance (percent of all
items) of every CPI component, in two populations (CPI-U, CPI-W), arranged as a
hierarchy.  The leaves of the "Expenditure category" section partition All
items and so are additive; a second "Special aggregate indexes" section lists
overlapping aggregates and is tagged but kept separate.

Outputs (``data/bls_cpi/``):
  - ``cpi_relative_importance.csv`` : tidy long panel, one row per
    (year, section, item), with hierarchy level, parent, leaf flag, raw and
    normalised name, item code where the source provides it, and both weights.
  - ``cpi_weights_keyed.csv``       : the new-structure (1997+) expenditure
    rows joined to current BLS item codes / series ids via a name crosswalk,
    restricted to codes that actually have a price series in
    ``cu.data.0.Current.txt``.
  - ``cpi_item_code_map.csv``       : the normalised-name -> item-code crosswalk
    used, with its provenance, for auditing.

The series-id join uses the authoritative BLS ``cu.item`` name<->code map
(``data/bls_cpi/cu.item``), cross-checked against the codes that actually have
a price series in ``cu.data.0.Current.txt``.  Item rows whose name does not
match ``cu.item`` are still emitted in the main panel, just without a code.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from project_paths import DERIVED_CACHE, ROOT


BCPI = ROOT / "data" / "bls_cpi"
RAW = BCPI / "relative_importance_raw"
OUT = DERIVED_CACHE / "bls_cpi"
PRICE_FILE = ROOT / "data" / "bls_cpi" / "cu.data.0.Current.txt"

# Two CPI-U/CPI-W float columns, e.g. "100.000", "  .547", "0.075".
FLOAT = re.compile(r"\d*\.\d+")
SECTION_HEADERS = {
    "expenditure category": "expenditure",
    "special aggregate indexes": "special_aggregate",
    "commodity and service group": "commodity_service_group",
}
# Lines inside a section that are structural, not data.
SKIP_TOKENS = (
    "item and group",
    "u.s. city average",
    "u. s. city average",
    "cpi-u",
    "cpi-w",
    "percent of all items",
    "ri us",
    "relative importance",
    "consumer price",
    "purchasing power",
    "table 1",
    "this is",
    "this file",
    "in usri",
)


def _norm(name: str) -> str:
    """Normalise an item label for crosswalking: lowercase, drop punctuation,
    collapse whitespace.  Keeps alphanumerics and spaces only."""
    s = name.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_name(raw: str) -> str:
    """Strip dot leaders and trailing/leading filler from a display name."""
    s = raw.replace("\t", " ")
    s = re.sub(r"[.\s]+$", "", s)  # trailing dot leader + spaces
    s = s.strip()
    return s


def _levels_from_indent(indents: list[int]) -> list[int]:
    """Map raw leading-space counts to dense hierarchy levels 0..K, preserving
    order (each distinct indent that is larger than all smaller seen-so-far gets
    the next level).  We rank by sorted unique indents within a section."""
    uniq = sorted(set(indents))
    rank = {v: i for i, v in enumerate(uniq)}
    return [rank[v] for v in indents]


# ---------------------------------------------------------------------------
# Indented-name tables (1997-2019)
# ---------------------------------------------------------------------------
def parse_indented_txt(text: str, year: int) -> pd.DataFrame:
    rows: list[dict] = []
    section: str | None = None
    pending: str | None = None
    pending_indent = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()

        # Section switches.
        hit = None
        for key, sec in SECTION_HEADERS.items():
            if low.startswith(key):
                hit = sec
                break
        if hit is not None:
            section = hit
            pending = None
            continue
        if any(tok in low for tok in SKIP_TOKENS):
            pending = None
            continue
        if section is None:
            continue

        floats = FLOAT.findall(line)
        indent = len(line) - len(line.lstrip(" "))

        if len(floats) >= 2:
            cpi_u, cpi_w = float(floats[-2]), float(floats[-1])
            # Name = everything before the value block.
            name_part = line[: line.index(floats[-2])] if floats[-2] in line else line
            name = _clean_name(name_part)
            if pending:
                name = f"{pending} {name}".strip()
                indent = pending_indent
                pending = None
            if not name:
                continue
            rows.append(
                {
                    "year": year,
                    "section": section,
                    "indent": indent,
                    "item_name_raw": name,
                    "cpi_u": cpi_u,
                    "cpi_w": cpi_w,
                }
            )
        else:
            # No values: a wrapped (continued) item name. Buffer it.
            frag = _clean_name(line)
            if frag:
                if pending:
                    pending = f"{pending} {frag}"
                else:
                    pending = frag
                    pending_indent = indent

    return _finalize_section_levels(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Coded flat tables (1990-1996, old item structure)
# ---------------------------------------------------------------------------
CODED = re.compile(r"^(?P<code>[A-Z][A-Z0-9]+)\s+(?P<name>.+?)\s+(?P<u>\d+\.\d+)\s+(?P<w>\d+\.\d+)\s*$")


def parse_coded_txt(text: str, year: int) -> pd.DataFrame:
    rows: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = CODED.match(line)
        if not m:
            continue
        code = m.group("code")
        # Section by code prefix: SA*=special aggregate, SE*/SS*=expenditure.
        section = "special_aggregate" if code.startswith("SA") else "expenditure"
        rows.append(
            {
                "year": year,
                "section": section,
                "indent": 0,
                "item_code": code,
                "item_name_raw": m.group("name").strip(),
                "cpi_u": float(m.group("u")),
                "cpi_w": float(m.group("w")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["level"] = 0
    df["is_leaf"] = pd.NA  # hierarchy not reconstructed for the old flat layout
    df["parent_name"] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Workbook tables (2020+)
# ---------------------------------------------------------------------------
def parse_xlsx(path: Path, year: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Table 1", header=None)
    rows: list[dict] = []
    section: str | None = None
    for _, r in raw.iterrows():
        lvl, name, u, w = r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3]
        if isinstance(name, str):
            low = name.strip().lower()
            hit = next((sec for key, sec in SECTION_HEADERS.items() if low.startswith(key)), None)
            if hit is not None:
                section = hit
                continue
            if any(tok in low for tok in SKIP_TOKENS):
                continue
        if section is None or not isinstance(name, str) or not name.strip():
            continue
        if pd.isna(u) and pd.isna(w):
            continue
        rows.append(
            {
                "year": year,
                "section": section,
                "indent": int(lvl) if pd.notna(lvl) else 0,
                "item_name_raw": name.strip(),
                "cpi_u": float(u) if pd.notna(u) else float("nan"),
                "cpi_w": float(w) if pd.notna(w) else float("nan"),
            }
        )
    return _finalize_section_levels(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Hierarchy: dense levels + parent + leaf flag, per (year, section).
# ---------------------------------------------------------------------------
def _finalize_section_levels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    for (year, section), g in df.groupby(["year", "section"], sort=False):
        g = g.reset_index(drop=True)
        g["level"] = _levels_from_indent(g["indent"].tolist())
        # parent = nearest preceding row with a strictly smaller level.
        parents: list[object] = []
        stack: list[tuple[int, str]] = []  # (level, name)
        for i, row in g.iterrows():
            lvl = row["level"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            parents.append(stack[-1][1] if stack else pd.NA)
            stack.append((lvl, row["item_name_raw"]))
        g["parent_name"] = parents
        # leaf = no later row has this row as its parent.
        child_parents = set(p for p in parents if p is not pd.NA)
        g["is_leaf"] = ~g["item_name_raw"].isin(child_parents)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res


# Relative-importance presentation names that BLS has since renamed in cu.item.
# Maps a normalised RI leaf name -> the normalised current cu.item name so the
# leaf keys to its (unchanged) item code.  Keep this list short and audited.
NAME_ALIASES = {
    "housing at school excluding board": "lodging while at school",          # SEHB01
    "technical and business school tuition and fees":
        "technical and vocational school tuition and fixed fees",            # SEEB04
}


def load_valid_item_codes() -> set[str]:
    """Item codes (area 0000) that have a price series in the CPI data file."""
    ids = pd.read_csv(PRICE_FILE, sep="\t")
    ids.columns = [c.strip() for c in ids.columns]
    sid = ids["series_id"].astype(str).str.strip().drop_duplicates()
    return {s[8:] for s in sid if s[4:8] == "0000"}


def build_crosswalk(valid_codes: set[str]) -> pd.DataFrame:
    """Authoritative name->code map from the local BLS ``cu.item`` file, with a
    flag for whether the code has a price series in ``cu.data.0.Current.txt``."""
    it = pd.read_csv(BCPI / "cu.item", sep="\t")
    it.columns = [c.strip() for c in it.columns]
    it["item_code"] = it["item_code"].astype(str).str.strip()
    it["name_norm"] = it["item_name"].map(_norm)
    it["in_price_file"] = it["item_code"].isin(valid_codes)
    it["source"] = "bls_cu.item"
    # Drop normalised names that map to more than one code (unkeyable by name).
    it["ambiguous"] = it["name_norm"].duplicated(keep=False)
    return it[["item_code", "item_name", "name_norm", "display_level",
               "in_price_file", "ambiguous", "source"]]


def main() -> int:
    frames: list[pd.DataFrame] = []

    # 1990-1999 archive.
    with zipfile.ZipFile(RAW / "ri-archive-1990-1999.zip") as z:
        for n in z.namelist():
            m = re.search(r"/(\d{4})\.txt$", n)
            if not m:
                continue
            year = int(m.group(1))
            text = z.read(n).decode("latin-1")
            if year <= 1996:
                df = parse_coded_txt(text, year)
                df["structure_era"] = "pre1998"
            else:
                df = parse_indented_txt(text, year)
                df["structure_era"] = "post1998"
            df["source_file"] = Path(n).name
            frames.append(df)

    # 2000-2019 archives (all indented, new structure).
    for zip_name in ("ri-archive-2000-2009.zip", "ri-archive-2010-2019.zip"):
        with zipfile.ZipFile(RAW / zip_name) as z:
            for n in z.namelist():
                m = re.search(r"/(\d{4})\.txt$", n)
                if not m or "old-weights" in n or "r-cpi-e" in n:
                    continue
                year = int(m.group(1))
                df = parse_indented_txt(z.read(n).decode("latin-1"), year)
                df["structure_era"] = "post1998"
                df["source_file"] = Path(n).name
                frames.append(df)

    # 2020+ workbooks.
    for path in sorted(RAW.glob("20*.xlsx")):
        year = int(path.stem)
        df = parse_xlsx(path, year)
        df["structure_era"] = "post1998"
        df["source_file"] = path.name
        frames.append(df)
    cur = RAW / "cpi-relative-importance-current.xlsx"
    if cur.exists():
        # Title row encodes the December reference year.
        head = pd.read_excel(cur, sheet_name="Table 1", header=None, nrows=1).iloc[0, 1]
        ym = re.search(r"December (\d{4})", str(head))
        year = int(ym.group(1)) if ym else 9999
        if year not in {int(p.stem) for p in RAW.glob("20*.xlsx")}:
            df = parse_xlsx(cur, year)
            df["structure_era"] = "post1998"
            df["source_file"] = cur.name
            frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["item_name"] = panel["item_name_raw"].map(_norm)
    if "item_code" not in panel.columns:
        panel["item_code"] = pd.NA
    panel = panel[
        [
            "year", "structure_era", "section", "level", "parent_name",
            "is_leaf", "item_name_raw", "item_name", "item_code",
            "cpi_u", "cpi_w", "source_file",
        ]
    ].sort_values(["year", "section", "level", "item_name_raw"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "cpi_relative_importance.csv", index=False)

    # --- Series-id crosswalk (new-structure expenditure rows only). ---
    valid = load_valid_item_codes()
    xwalk = build_crosswalk(valid)
    xwalk.to_csv(OUT / "cpi_item_code_map.csv", index=False)

    new = panel[(panel["structure_era"] == "post1998")].copy()
    # Key by normalised name: codes that have a price series and an unambiguous name.
    keyable = xwalk[xwalk["in_price_file"] & ~xwalk["ambiguous"]][["item_code", "name_norm"]]
    name_to_code = keyable.drop_duplicates("name_norm").set_index("name_norm")["item_code"]
    # Apply audited rename aliases before matching.
    new["match_name"] = new["item_name"].replace(NAME_ALIASES)
    new["item_code_x"] = new["match_name"].map(name_to_code)
    keyed = new[new["item_code_x"].notna()].copy()
    keyed["item_code"] = keyed["item_code_x"]
    keyed["series_id_nsa"] = "CUUR0000" + keyed["item_code"]
    keyed["series_id_sa"] = "CUSR0000" + keyed["item_code"]
    keyed = keyed[
        [
            "year", "section", "level", "is_leaf", "item_name_raw", "item_name",
            "item_code", "series_id_nsa", "series_id_sa", "cpi_u", "cpi_w",
        ]
    ].sort_values(["year", "item_code"]).reset_index(drop=True)
    keyed.to_csv(OUT / "cpi_weights_keyed.csv", index=False)

    # --- Diagnostics. ---
    years = sorted(panel["year"].unique())
    print(f"[cpi] parsed {len(panel)} rows across years {years[0]}-{years[-1]} "
          f"({len(years)} tables)")
    exp = panel[(panel["section"] == "expenditure") & (panel["structure_era"] == "post1998")]
    for y in (years[0] if years else None, 2010, 2019, 2024):
        sub = exp[(exp["year"] == y) & (exp["is_leaf"] == True)]  # noqa: E712
        if len(sub):
            print(f"      {y}: expenditure leaves={len(sub)} "
                  f"CPI-U leaf sum={sub['cpi_u'].sum():.2f} "
                  f"CPI-W leaf sum={sub['cpi_w'].sum():.2f}")
    n_codes = keyed["item_code"].nunique()
    latest = keyed["year"].max()
    lk = keyed[keyed["year"] == latest]
    print(f"[cpi] crosswalk: {len(xwalk)} mapped names, "
          f"{xwalk['in_price_file'].sum()} present in price file")
    lk_leaf = lk[lk["is_leaf"] == True]  # noqa: E712
    print(f"[cpi] keyed weights: {len(keyed)} rows, {n_codes} distinct item codes")
    print(f"      {latest}: keyed expenditure leaves={len(lk_leaf[lk_leaf['section']=='expenditure'])}, "
          f"leaf CPI-U sum={lk_leaf[lk_leaf['section']=='expenditure']['cpi_u'].sum():.2f} "
          f"(partition should approach 100)")
    # Unmatched leaf names in the latest year, to show what the crosswalk misses.
    miss = exp[(exp["year"] == latest) & (exp["is_leaf"] == True)].copy()  # noqa: E712
    miss_names = miss["item_name"].replace(NAME_ALIASES)
    miss = miss[~miss_names.isin(name_to_code.index)]
    if len(miss):
        print(f"[cpi] {len(miss)} unmatched {latest} expenditure leaves (sample):")
        for nm in miss["item_name_raw"].head(15):
            print(f"        - {nm}")
    print(f"[cpi] wrote: cpi_relative_importance.csv, cpi_weights_keyed.csv, "
          f"cpi_item_code_map.csv -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
