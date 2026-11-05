#!/usr/bin/env python3
"""Fetch and normalize Forbes billionaire wealth panels.

Outputs under ``data/forbes_billionaires/``:
  - ``raw/annual_YEAR.json``: merged paged Forbes API response for each year.
  - ``raw/realtime_latest.json``: merged current real-time billionaire response.
  - ``annual_panel.csv``: tidy annual panel, one row per listed person-year.
  - ``realtime_latest.csv``: tidy current cross-section.
  - ``fetch_summary.csv``: source URL, reported counts, rows written, status.

Forbes reports ``finalWorth`` in millions of current U.S. dollars. The builder
keeps that field and adds billion-dollar and dollar-scaled versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"

ANNUAL_BASE = "https://www.forbes.com/forbesapi/person/billionaires/{year}/position/true.json"
REALTIME_BASE = "https://www.forbes.com/forbesapi/person/rtb/0/position/true.json"
USER_AGENT = "Mozilla/5.0"

CSV_COLUMNS = [
    "snapshot_kind",
    "snapshot_id",
    "snapshot_date",
    "list_year",
    "list_month",
    "rank",
    "position",
    "person_uri",
    "natural_id",
    "person_name",
    "final_worth_million_usd",
    "final_worth_billion_usd",
    "net_worth_usd",
    "est_worth_prev_million_usd",
    "private_assets_worth_million_usd",
    "forbes_change_million_usd",
    "age",
    "birth_date",
    "gender",
    "country",
    "state",
    "city",
    "country_of_citizenship",
    "source",
    "organization",
    "title",
    "employment_name",
    "employment_title",
    "employment_uri",
    "category",
    "industries",
    "self_made",
    "self_made_rank",
    "status",
    "family_list",
    "wealth_list",
    "philanthropy_score",
    "premium_profile",
    "image_exists",
    "forbes_timestamp_utc",
    "forbes_date_utc",
    "source_url",
    "fetched_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between API page requests.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--refresh", action="store_true", help="Refetch raw JSON even when cached files exist.")
    parser.add_argument("--include-fragments", action="store_true", help="Also request 1997-2000 API fragments.")
    parser.add_argument("--skip-realtime", action="store_true", help="Do not fetch the current real-time list.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ms_to_iso(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        ts = float(value) / 1000.0
        return datetime.fromtimestamp(ts, timezone.utc).replace(microsecond=0).isoformat()
    except (OSError, OverflowError, TypeError, ValueError):
        return ""


def ms_to_date(value: Any) -> str:
    iso = ms_to_iso(value)
    return iso[:10] if iso else ""


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def fmt_dollars_from_millions(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1_000_000:.0f}"


def make_url(base: str, *, start: int, limit: int) -> str:
    return base + "?" + urllib.parse.urlencode({"limit": limit, "start": start})


def fetch_json(url: str, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45) as response:
                body = response.read().decode("utf-8")
            if body.lstrip().startswith("<"):
                raise RuntimeError(f"HTML response from API: {body[:100]!r}")
            return json.loads(body)
        except Exception as exc:  # noqa: BLE001 - print useful retry context.
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def extract_person_list(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    person_list = payload.get("personList", {})
    count = int(person_list.get("count") or 0)
    rows = person_list.get("personsLists") or []
    if not isinstance(rows, list):
        rows = []
    return count, rows


def fetch_paged(base: str, page_size: int, sleep: float, retries: int) -> tuple[dict[str, Any], dict[str, Any]]:
    first_url = make_url(base, start=0, limit=1)
    first_payload = fetch_json(first_url, retries)
    count, _ = extract_person_list(first_payload)

    rows: list[dict[str, Any]] = []
    page_count = 0
    for start in range(0, count, page_size):
        url = make_url(base, start=start, limit=page_size)
        payload = fetch_json(url, retries)
        reported_count, page_rows = extract_person_list(payload)
        if reported_count != count:
            raise RuntimeError(f"Count changed while fetching {base}: {count} -> {reported_count}")
        rows.extend(page_rows)
        page_count += 1
        print(f"  fetched {len(rows):>5,}/{count:>5,} rows from {base} (start={start})", flush=True)
        if sleep:
            time.sleep(sleep)

    merged = {
        "personList": {
            "count": count,
            "personsLists": rows,
        }
    }
    meta = {
        "base_url": base,
        "reported_count": count,
        "rows": len(rows),
        "pages": page_count,
    }
    return merged, meta


def load_or_fetch(
    raw_path: Path,
    base: str,
    *,
    page_size: int,
    sleep: float,
    retries: int,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw_path.exists() and not refresh:
        payload = json.loads(raw_path.read_text())
        count, rows = extract_person_list(payload)
        return payload, {
            "base_url": base,
            "reported_count": count,
            "rows": len(rows),
            "pages": "",
            "cached": True,
        }

    payload, meta = fetch_paged(base, page_size=page_size, sleep=sleep, retries=retries)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    meta["cached"] = False
    return payload, meta


def flatten_row(
    row: dict[str, Any],
    *,
    snapshot_kind: str,
    snapshot_id: str,
    source_url: str,
    fetched_at_utc: str,
) -> dict[str, Any]:
    final_worth = numeric(row.get("finalWorth"))
    timestamp_utc = ms_to_iso(row.get("timestamp"))
    forbes_date_utc = ms_to_iso(row.get("date"))

    if snapshot_kind in {"realtime", "archive_realtime"}:
        snapshot_date = timestamp_utc[:10] if timestamp_utc else fetched_at_utc[:10]
    else:
        snapshot_date = forbes_date_utc[:10] if forbes_date_utc else str(row.get("year") or "")

    employment = row.get("employment") if isinstance(row.get("employment"), dict) else {}
    industries = row.get("industries")
    if isinstance(industries, list):
        industries_text = "|".join(str(x) for x in industries)
    else:
        industries_text = str(industries) if industries not in (None, "") else ""

    return {
        "snapshot_kind": snapshot_kind,
        "snapshot_id": snapshot_id,
        "snapshot_date": snapshot_date,
        "list_year": row.get("year") or "",
        "list_month": row.get("month") or "",
        "rank": row.get("rank") or "",
        "position": row.get("position") or "",
        "person_uri": row.get("uri") or "",
        "natural_id": row.get("naturalId") or "",
        "person_name": row.get("personName") or row.get("name") or "",
        "final_worth_million_usd": fmt_number(final_worth, 6),
        "final_worth_billion_usd": fmt_number(final_worth / 1000 if final_worth is not None else None, 6),
        "net_worth_usd": fmt_dollars_from_millions(final_worth),
        "est_worth_prev_million_usd": fmt_number(numeric(row.get("estWorthPrev")), 6),
        "private_assets_worth_million_usd": fmt_number(numeric(row.get("privateAssetsWorth")), 6),
        "forbes_change_million_usd": fmt_number(numeric(row.get("change")), 6),
        "age": row.get("age") or "",
        "birth_date": ms_to_date(row.get("birthDate")),
        "gender": row.get("gender") or "",
        "country": row.get("country") or "",
        "state": row.get("state") or "",
        "city": row.get("city") or "",
        "country_of_citizenship": row.get("countryOfCitizenship") or "",
        "source": row.get("source") or "",
        "organization": row.get("organization") or "",
        "title": row.get("title") or "",
        "employment_name": employment.get("name") or "",
        "employment_title": employment.get("title") or "",
        "employment_uri": employment.get("uri") or "",
        "category": row.get("category") or "",
        "industries": industries_text,
        "self_made": row.get("selfMade") if row.get("selfMade") is not None else "",
        "self_made_rank": row.get("selfMadeRank") or "",
        "status": row.get("status") or "",
        "family_list": row.get("familyList") if row.get("familyList") is not None else "",
        "wealth_list": row.get("wealthList") if row.get("wealthList") is not None else "",
        "philanthropy_score": row.get("philanthropyScore") or "",
        "premium_profile": row.get("premiumProfile") if row.get("premiumProfile") is not None else "",
        "image_exists": row.get("imageExists") if row.get("imageExists") is not None else "",
        "forbes_timestamp_utc": timestamp_utc,
        "forbes_date_utc": forbes_date_utc,
        "source_url": source_url,
        "fetched_at_utc": fetched_at_utc,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dedupe_panel_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop duplicate person-snapshot records using Forbes natural IDs."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        person_key = row["natural_id"]
        if not person_key:
            raise ValueError("Forbes record is missing natural_id.")
        key = (
            row["snapshot_id"],
            person_key,
            row["rank"],
            row["position"],
            row["person_name"],
            row["final_worth_million_usd"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out, len(rows) - len(out)


def main() -> None:
    args = parse_args()
    if args.include_fragments:
        args.start_year = min(args.start_year, 1997)

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    fetched_at = utc_now()
    annual_rows: list[dict[str, Any]] = []
    realtime_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for year in range(args.start_year, args.end_year + 1):
        base = ANNUAL_BASE.format(year=year)
        raw_path = RAW / f"annual_{year}.json"
        snapshot_id = f"annual_{year}"
        print(f"[annual {year}] {base}", flush=True)
        try:
            payload, meta = load_or_fetch(
                raw_path,
                base,
                page_size=args.page_size,
                sleep=args.sleep,
                retries=args.retries,
                refresh=args.refresh,
            )
            _, rows = extract_person_list(payload)
            flat_rows = [
                flatten_row(
                    row,
                    snapshot_kind="annual",
                    snapshot_id=snapshot_id,
                    source_url=base,
                    fetched_at_utc=fetched_at,
                )
                for row in rows
            ]
            flat_rows, duplicate_rows_dropped = dedupe_panel_rows(flat_rows)
            annual_rows.extend(flat_rows)
            meta["rows_written"] = len(flat_rows)
            meta["duplicate_rows_dropped"] = duplicate_rows_dropped
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - keep going across years.
            print(f"  ERROR: {exc}", flush=True)
            meta = {
                "base_url": base,
                "reported_count": "",
                "rows": "",
                "rows_written": "",
                "duplicate_rows_dropped": "",
                "pages": "",
                "cached": "",
            }
            status = f"error: {exc}"

        summary_rows.append(
            {
                "snapshot_kind": "annual",
                "snapshot_id": snapshot_id,
                "source_url": base,
                "reported_count": meta.get("reported_count", ""),
                "raw_rows": meta.get("rows", ""),
                "rows_written": meta.get("rows_written", meta.get("rows", "")),
                "duplicate_rows_dropped": meta.get("duplicate_rows_dropped", 0),
                "pages": meta.get("pages", ""),
                "cached": meta.get("cached", ""),
                "status": status,
                "fetched_at_utc": fetched_at,
            }
        )

    annual_rows.sort(key=lambda r: (int(r["list_year"] or 0), int(r["rank"] or r["position"] or 0), r["person_name"]))
    write_csv(OUT / "annual_panel.csv", annual_rows, CSV_COLUMNS)

    if not args.skip_realtime:
        base = REALTIME_BASE
        raw_path = RAW / "realtime_latest.json"
        print(f"[realtime] {base}", flush=True)
        try:
            payload, meta = load_or_fetch(
                raw_path,
                base,
                page_size=args.page_size,
                sleep=args.sleep,
                retries=args.retries,
                refresh=args.refresh,
            )
            _, rows = extract_person_list(payload)
            snapshot_date = fetched_at[:10]
            if rows:
                first_ts = ms_to_iso(rows[0].get("timestamp"))
                snapshot_date = first_ts[:10] if first_ts else snapshot_date
            snapshot_id = f"realtime_{snapshot_date}"
            realtime_rows = [
                flatten_row(
                    row,
                    snapshot_kind="realtime",
                    snapshot_id=snapshot_id,
                    source_url=base,
                    fetched_at_utc=fetched_at,
                )
                for row in rows
            ]
            realtime_rows, duplicate_rows_dropped = dedupe_panel_rows(realtime_rows)
            meta["rows_written"] = len(realtime_rows)
            meta["duplicate_rows_dropped"] = duplicate_rows_dropped
            realtime_rows.sort(key=lambda r: (int(r["rank"] or r["position"] or 0), r["person_name"]))
            write_csv(OUT / "realtime_latest.csv", realtime_rows, CSV_COLUMNS)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", flush=True)
            snapshot_id = "realtime_latest"
            meta = {
                "base_url": base,
                "reported_count": "",
                "rows": "",
                "rows_written": "",
                "duplicate_rows_dropped": "",
                "pages": "",
                "cached": "",
            }
            status = f"error: {exc}"

        summary_rows.append(
            {
                "snapshot_kind": "realtime",
                "snapshot_id": snapshot_id,
                "source_url": base,
                "reported_count": meta.get("reported_count", ""),
                "raw_rows": meta.get("rows", ""),
                "rows_written": meta.get("rows_written", meta.get("rows", "")),
                "duplicate_rows_dropped": meta.get("duplicate_rows_dropped", 0),
                "pages": meta.get("pages", ""),
                "cached": meta.get("cached", ""),
                "status": status,
                "fetched_at_utc": fetched_at,
            }
        )

    write_csv(
        OUT / "fetch_summary.csv",
        summary_rows,
        [
            "snapshot_kind",
            "snapshot_id",
            "source_url",
            "reported_count",
            "raw_rows",
            "rows_written",
            "duplicate_rows_dropped",
            "pages",
            "cached",
            "status",
            "fetched_at_utc",
        ],
    )
    print(f"Wrote {len(annual_rows):,} annual rows to {OUT / 'annual_panel.csv'}", flush=True)
    if realtime_rows:
        print(f"Wrote {len(realtime_rows):,} realtime rows to {OUT / 'realtime_latest.csv'}", flush=True)


if __name__ == "__main__":
    main()
