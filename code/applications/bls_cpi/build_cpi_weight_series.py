#!/usr/bin/env python3
"""Build an annual CPI relative-importance weight series from 1967 onward.

The standard ``build_cpi_weights.py`` parser starts at the BLS 1990 archive.
This script stitches the longer historical sources into a single audit-friendly
panel:

* 1967-1986 from ``historical-relative-importance-1947-1986.xlsx``;
* 1987-1989 from the downloaded BLS RI bridge text files;
* 1990-1996 from the same old-code flat archive format;
* 1997+ from ``cpi_relative_importance.csv`` plus the existing item-code
  crosswalk where available.

Outputs:

* ``.cache/derived/bls_cpi/cpi_weight_series_1967_onward.csv`` -- long, one annual
  relative-importance row per raw BLS item/category;
* ``.cache/derived/bls_cpi/cpi_weight_series_1967_onward_wide.parquet`` -- wide CPI-U
  weights, indexed by December year and keyed by raw item code where available.
"""

from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

from applications.bls_cpi.build_cpi_weights import _clean_name, _levels_from_indent, _norm

from project_paths import DERIVED_CACHE, ROOT


BCPI = ROOT / "data" / "bls_cpi"
OUT = DERIVED_CACHE / "bls_cpi"
RAW = BCPI / "relative_importance_raw"
HIST_XLSX = RAW / "historical-relative-importance-1947-1986.xlsx"

LONG_OUT = OUT / "cpi_weight_series_1967_onward.csv"
WIDE_OUT = OUT / "cpi_weight_series_1967_onward_wide.parquet"

CODED = re.compile(r"^(?P<code>[A-Z][A-Z0-9]+)\s+(?P<name>.+?)\s+(?P<u>\d+\.\d+)\s+(?P<w>\d+\.\d+)\s*$")


def section_marker(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    s = _norm(value)
    s = s.replace("lndex", "index").replace("speclal", "special")
    if s == "major groups":
        return "expenditure", "major_groups"
    if s == "individual items":
        return "expenditure", "individual_items"
    if s == "expenditure category":
        return "expenditure", "expenditure_category"
    if s == "commodity and service group":
        return "commodity_service_group", "commodity_service_group"
    if s in {"special groups", "special group", "special indexes", "special index"}:
        return "special_aggregate", "special_indexes"
    return None


def historical_column_specs(rows: list[tuple[object, ...]]) -> list[tuple[int, int, str]]:
    """Return (column_index, year, population) specs for a historical sheet."""
    specs: list[tuple[int, int, str]] = []
    if not rows:
        return specs
    max_cols = max(len(r) for r in rows)
    header_rows = rows[:4]
    for col in range(1, max_cols):
        text = " ".join(
            str(row[col])
            for row in header_rows
            if col < len(row) and row[col] is not None
        )
        if not text.strip():
            continue
        low = text.lower()
        if "old series" in low or "unrevised" in low:
            continue
        years = re.findall(r"(19\d{2})", text)
        if not years:
            continue
        year = int(years[-1])
        if year < 1967:
            continue
        if "wage" in low or "cpi-w" in low:
            population = "cpi_w"
        else:
            # The pre-1978 historical columns are single-population published
            # relative importances. Store them in cpi_u so users have one
            # continuous all-urban-style series.
            population = "cpi_u"
        specs.append((col, year, population))
    return specs


def finalize_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    for (_, _, section, source_block), g in df.groupby(["year", "source_sheet", "section", "source_block"], sort=False):
        g = g.sort_values("source_row").copy()
        g["level"] = _levels_from_indent(g["indent"].tolist())
        parents: list[object] = []
        stack: list[tuple[int, int, str]] = []
        child_parent_ids: set[int] = set()
        for row_id, row in g.iterrows():
            level = int(row["level"])
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                parents.append(stack[-1][2])
                child_parent_ids.add(stack[-1][1])
            else:
                parents.append(pd.NA)
            stack.append((level, row_id, row["item_name_raw"]))
        g["parent_name"] = parents
        g["is_leaf"] = [idx not in child_parent_ids for idx in g.index]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def parse_historical_workbook() -> pd.DataFrame:
    wb = openpyxl.load_workbook(HIST_XLSX, read_only=True, data_only=True)
    records: dict[tuple[int, str, int, str], dict[str, object]] = {}

    for sheet in wb.sheetnames:
        if sheet in {"Item Names", "User notes"}:
            continue
        ws = wb[sheet]
        rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
        specs = historical_column_specs(rows)
        if not specs:
            continue

        section = "expenditure"
        source_block = "expenditure"
        seen_expenditure_root = False
        for source_row, row in enumerate(rows, start=1):
            raw_name = row[0] if row else None
            marker = section_marker(raw_name)
            row_values = [
                row[col]
                for col, _, _ in specs
                if col < len(row) and isinstance(row[col], (int, float))
            ]
            if marker is not None and not row_values:
                section, source_block = marker
                continue
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            if _norm(raw_name) in {"item and group"} or raw_name.lower().startswith(("table ", "consumer price index")):
                continue
            if not row_values:
                continue

            indent = len(raw_name) - len(raw_name.lstrip(" "))
            item_name_raw = _clean_name(raw_name)
            if (
                _norm(item_name_raw) == "all items"
                and section == "expenditure"
                and source_block in {"expenditure", "expenditure_category"}
            ):
                if seen_expenditure_root and source_row > 10:
                    section = "commodity_service_group"
                    source_block = "commodity_service_group"
                else:
                    seen_expenditure_root = True
            for col, year, population in specs:
                if col >= len(row) or not isinstance(row[col], (int, float)):
                    continue
                key = (year, section, source_row, item_name_raw)
                rec = records.setdefault(
                    key,
                    {
                        "year": year,
                        "structure_era": "pre1978_historical" if year <= 1976 else "1978_revision_historical",
                        "section": section,
                        "source_block": source_block,
                        "indent": indent,
                        "item_name_raw": item_name_raw,
                        "item_name": _norm(item_name_raw),
                        "item_code": pd.NA,
                        "cpi_u": pd.NA,
                        "cpi_w": pd.NA,
                        "source_file": HIST_XLSX.name,
                        "source_sheet": sheet,
                        "source_row": source_row,
                    },
                )
                rec[population] = float(row[col])

    df = pd.DataFrame(records.values())
    return finalize_hierarchy(df)


def classify_old_code(code: str) -> str:
    if code == "SA0" or re.match(r"^SA[1-7]\d*$", code) or code.startswith(("SE", "SS")):
        return "expenditure"
    if code.startswith(("SAC", "SAD", "SAN", "SAS")):
        return "commodity_service_group"
    return "special_aggregate"


def parse_old_code_text(text: str, year: int, source_file: str) -> pd.DataFrame:
    rows = []
    for source_row, line in enumerate(text.splitlines(), start=1):
        m = CODED.match(line.rstrip())
        if not m:
            continue
        code = m.group("code").strip()
        name = _clean_name(m.group("name"))
        rows.append(
            {
                "year": year,
                "structure_era": "pre1998_old_codes",
                "section": classify_old_code(code),
                "source_block": classify_old_code(code),
                "level": pd.NA,
                "parent_name": pd.NA,
                "is_leaf": pd.NA,
                "item_name_raw": name,
                "item_name": _norm(name),
                "item_code": code,
                "cpi_u": float(m.group("u")),
                "cpi_w": float(m.group("w")),
                "source_file": source_file,
                "source_sheet": pd.NA,
                "source_row": source_row,
            }
        )
    return pd.DataFrame(rows)


def parse_old_code_archives() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "1987-1989_RI_archive").glob("*.txt")):
        year = int(path.stem)
        if year >= 1967:
            frames.append(parse_old_code_text(path.read_text(errors="replace"), year, path.name))

    with zipfile.ZipFile(RAW / "ri-archive-1990-1999.zip") as z:
        for name in z.namelist():
            m = re.search(r"/(\d{4})\.txt$", name)
            if not m:
                continue
            year = int(m.group(1))
            if 1990 <= year <= 1996:
                frames.append(parse_old_code_text(z.read(name).decode("latin-1"), year, Path(name).name))
    return pd.concat(frames, ignore_index=True)


def load_modern_post1998() -> pd.DataFrame:
    modern = pd.read_csv(OUT / "cpi_relative_importance.csv")
    modern = modern[modern["year"] >= 1997].copy()
    keyed_path = OUT / "cpi_weights_keyed.csv"
    if keyed_path.exists():
        keyed = pd.read_csv(keyed_path)
        code_map = keyed[["year", "section", "item_name", "item_code"]].drop_duplicates(
            ["year", "section", "item_name"]
        )
        modern = modern.drop(columns=["item_code"], errors="ignore").merge(
            code_map, on=["year", "section", "item_name"], how="left"
        )
    else:
        modern["item_code"] = pd.NA
    modern["source_sheet"] = pd.NA
    modern["source_block"] = modern["section"]
    modern["source_row"] = pd.NA
    modern["indent"] = pd.NA
    return modern


def add_item_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    code = df["item_code"].astype("string")
    has_code = code.notna() & (code.str.len() > 0)
    level = df["level"].map(lambda x: "flat" if pd.isna(x) else str(int(float(x))))
    parent = df["parent_name"].fillna("root").astype(str).map(_norm)
    context = df["section"].astype(str) + ":l" + level + ":" + parent
    df["item_key"] = "name:" + context + ":" + df["item_name"].astype(str)
    has_historical_slot = df["source_sheet"].notna() & df["source_row"].notna()
    slot = df.loc[has_historical_slot, "source_row"].astype(int).astype(str)
    df.loc[has_historical_slot, "item_key"] = (
        "hist:"
        + df.loc[has_historical_slot, "source_sheet"].astype(str)
        + ":r"
        + slot
        + ":"
        + df.loc[has_historical_slot, "item_name"].astype(str)
    )
    df.loc[has_code, "item_key"] = "code:" + context[has_code] + ":" + code[has_code]
    return df


def main() -> int:
    hist = parse_historical_workbook()
    old = parse_old_code_archives()
    modern = load_modern_post1998()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        panel = pd.concat([d for d in (hist, old, modern) if not d.empty], ignore_index=True, sort=False)
    panel = panel[panel["year"] >= 1967].copy()
    panel["item_name"] = panel["item_name_raw"].map(_norm)
    panel["cpi_u"] = pd.to_numeric(panel["cpi_u"], errors="coerce")
    panel["cpi_w"] = pd.to_numeric(panel["cpi_w"], errors="coerce")
    panel = add_item_key(panel)

    columns = [
        "year",
        "structure_era",
        "section",
        "source_block",
        "level",
        "parent_name",
        "is_leaf",
        "item_key",
        "item_code",
        "item_name_raw",
        "item_name",
        "cpi_u",
        "cpi_w",
        "source_file",
        "source_sheet",
        "source_row",
    ]
    panel = panel[columns].sort_values(
        ["year", "section", "source_sheet", "source_row", "item_key"],
        na_position="last",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(LONG_OUT, index=False)
    wide = panel.pivot_table(index="year", columns="item_key", values="cpi_u", aggfunc="first")
    wide.sort_index(axis=1).to_parquet(WIDE_OUT)

    years = sorted(panel["year"].dropna().unique())
    by_year = panel.groupby("year").agg(rows=("item_key", "size"), cpi_u_nonmissing=("cpi_u", "count"))
    exp_roots = panel[(panel["section"] == "expenditure") & (_norm_series(panel["item_name_raw"]) == "all items")]
    print(f"[cpi-weight-series] wrote {len(panel):,} rows, years {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"[cpi-weight-series] long: {LONG_OUT}")
    print(f"[cpi-weight-series] wide: {WIDE_OUT} shape={wide.shape}")
    print("[cpi-weight-series] sample row counts:")
    print(by_year.loc[[y for y in (1967, 1977, 1987, 1997, 2025) if y in by_year.index]].to_string())
    if not exp_roots.empty:
        bad_roots = exp_roots[exp_roots["cpi_u"].round(6) != 100.0]
        if len(bad_roots):
            print(f"[cpi-weight-series] warning: {len(bad_roots)} All-items CPI-U roots differ from 100.")
    return 0


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_norm)


if __name__ == "__main__":
    raise SystemExit(main())
