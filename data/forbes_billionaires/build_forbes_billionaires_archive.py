#!/usr/bin/env python3
"""Backfill higher-frequency Forbes real-time billionaire snapshots from the
Internet Archive.

The Forbes real-time endpoint itself only exposes the current cross-section.
The Internet Archive has daily-level captures for that endpoint from 2019-08
onward. This script writes a CDX manifest and optionally fetches a sampled
archive panel (monthly by default, using the last capture in each period).

Examples:
  python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --manifest-only
  python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency monthly --no-raw
  python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency daily --tail-snapshots 43 --no-raw
  python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency daily --no-raw --resume
  python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency weekly --start-date 2024-01-01
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from build_forbes_billionaires import (
    CSV_COLUMNS,
    OUT,
    dedupe_panel_rows,
    flatten_row,
    ms_to_iso,
    utc_now,
    write_csv,
)

ARCHIVE_ENDPOINT = "https://web.archive.org/cdx"
FORBES_RTB_URL = "https://www.forbes.com/forbesapi/person/rtb/0/position/true.json"
USER_AGENT = "Mozilla/5.0"
MANIFEST = OUT / "archive_cdx_daily.csv"
ARCHIVE_RAW = OUT / "raw_archive"

ARCHIVE_COLUMNS = CSV_COLUMNS + [
    "archive_timestamp",
    "archive_capture_date",
    "archive_original_url",
    "archive_digest",
    "archive_length",
    "archive_replay_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly", "quarterly"],
        default="monthly",
        help="Sampling frequency from the daily CDX manifest.",
    )
    parser.add_argument("--start-date", default="2019-08-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between archive replay requests.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-raw", action="store_true", help="Do not save replayed archive JSON payloads.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted temporary build for the same selected snapshots.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Replace final CSVs even if some archive captures fail. Default leaves temp files for resume.",
    )
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument(
        "--tail-snapshots",
        type=int,
        default=None,
        help="After date/frequency sampling, keep only the last N snapshots.",
    )
    return parser.parse_args()


def fetch_url_bytes(url: str, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry with context.
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def archive_replay_url(timestamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{original}"


def decode_json_bytes(body: bytes) -> dict[str, Any]:
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1")
    if text.lstrip().startswith("<"):
        raise RuntimeError(f"Archive replay returned HTML: {text[:100]!r}")
    return json.loads(text)


def fetch_manifest(refresh: bool, retries: int) -> list[dict[str, str]]:
    if MANIFEST.exists() and not refresh:
        with MANIFEST.open(newline="") as f:
            return list(csv.DictReader(f))

    params = {
        "url": FORBES_RTB_URL,
        "matchType": "exact",
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest,length",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
    }
    url = ARCHIVE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    data = json.loads(fetch_url_bytes(url, retries).decode("utf-8"))
    header, records = data[0], data[1:]
    rows = [dict(zip(header, record)) for record in records]
    rows.sort(key=lambda r: r["timestamp"])
    for row in rows:
        row["capture_date"] = f"{row['timestamp'][:4]}-{row['timestamp'][4:6]}-{row['timestamp'][6:8]}"

    write_csv(
        MANIFEST,
        rows,
        ["timestamp", "capture_date", "original", "statuscode", "mimetype", "digest", "length"],
    )
    return rows


def period_key(capture_date: str, frequency: str) -> str:
    d = datetime.strptime(capture_date, "%Y-%m-%d").date()
    if frequency == "daily":
        return d.isoformat()
    if frequency == "weekly":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if frequency == "monthly":
        return d.strftime("%Y-%m")
    if frequency == "quarterly":
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    raise ValueError(f"Unknown frequency: {frequency}")


def select_snapshots(
    rows: list[dict[str, str]],
    *,
    frequency: str,
    start_date: str,
    end_date: str,
    max_snapshots: int | None,
    tail_snapshots: int | None,
) -> list[dict[str, str]]:
    if max_snapshots is not None and tail_snapshots is not None:
        raise ValueError("Use either --max-snapshots or --tail-snapshots, not both.")
    filtered = [r for r in rows if start_date <= r["capture_date"] <= end_date]
    selected_by_period: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in filtered:
        selected_by_period[period_key(row["capture_date"], frequency)] = row
    selected = list(selected_by_period.values())
    if tail_snapshots is not None:
        selected = selected[-tail_snapshots:]
    elif max_snapshots is not None:
        selected = selected[:max_snapshots]
    return selected


def raw_path_for(frequency: str, timestamp: str) -> Path:
    return ARCHIVE_RAW / frequency / f"{timestamp}.json.gz"


def load_or_fetch_replay(
    row: dict[str, str],
    *,
    frequency: str,
    retries: int,
    refresh_raw: bool,
    save_raw: bool,
) -> tuple[dict[str, Any], str]:
    replay_url = archive_replay_url(row["timestamp"], row["original"])
    raw_path = raw_path_for(frequency, row["timestamp"])
    if raw_path.exists() and not refresh_raw:
        body = gzip.decompress(raw_path.read_bytes())
        return json.loads(body.decode("utf-8")), replay_url

    body = fetch_url_bytes(replay_url, retries)
    if body[:2] == b"\x1f\x8b":
        decoded = gzip.decompress(body)
    else:
        decoded = body
    payload = decode_json_bytes(body)
    if save_raw:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(gzip.compress(decoded))
    return payload, replay_url


def open_append_writer(path: Path, columns: list[str], *, append: bool) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    handle = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    if not append or not file_exists or path.stat().st_size == 0:
        writer.writeheader()
    return handle, writer


def completed_timestamps(summary_tmp: Path) -> set[str]:
    if not summary_tmp.exists():
        return set()
    with summary_tmp.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["archive_timestamp"]
        for row in rows
        if row.get("status", "").startswith("ok") and row.get("archive_timestamp")
    }


def latest_summary_rows(summary_tmp: Path, selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not summary_tmp.exists():
        return []
    selected_order = {row["timestamp"]: i for i, row in enumerate(selected)}
    latest: dict[str, dict[str, Any]] = {}
    latest_ok: dict[str, dict[str, Any]] = {}
    with summary_tmp.open(newline="") as f:
        for row in csv.DictReader(f):
            ts = row.get("archive_timestamp", "")
            if ts in selected_order:
                latest[ts] = row
                if row.get("status", "").startswith("ok"):
                    latest_ok[ts] = row
    return [
        latest_ok.get(ts, latest[ts])
        for ts, _ in sorted(selected_order.items(), key=lambda item: item[1])
        if ts in latest
    ]


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fetched_at = utc_now()

    manifest_rows = fetch_manifest(refresh=args.refresh_manifest, retries=args.retries)
    selected = select_snapshots(
        manifest_rows,
        frequency=args.frequency,
        start_date=args.start_date,
        end_date=args.end_date,
        max_snapshots=args.max_snapshots,
        tail_snapshots=args.tail_snapshots,
    )
    print(f"Manifest rows: {len(manifest_rows):,}; selected {len(selected):,} {args.frequency} snapshots.")
    if args.manifest_only:
        return

    out_csv = OUT / f"archive_realtime_{args.frequency}.csv"
    summary_csv = OUT / f"archive_fetch_summary_{args.frequency}.csv"
    out_tmp = out_csv.with_name(out_csv.name + ".tmp")
    summary_tmp = summary_csv.with_name(summary_csv.name + ".tmp")

    if args.resume and out_csv.exists() and summary_csv.exists() and not out_tmp.exists() and not summary_tmp.exists():
        latest_rows = latest_summary_rows(summary_csv, selected)
        missing = len(selected) - len(latest_rows)
        errors = [row for row in latest_rows if not str(row.get("status", "")).startswith("ok")]
        if not missing and not errors:
            rows_written = sum(int(row.get("rows_written") or 0) for row in latest_rows)
            print(
                f"Existing {args.frequency} archive is complete: "
                f"{len(selected):,} snapshots, {rows_written:,} rows. Nothing to resume.",
                flush=True,
            )
            return

    append = args.resume and out_tmp.exists() and summary_tmp.exists()
    completed = completed_timestamps(summary_tmp) if append else set()
    if completed:
        print(f"Resuming temp build: {len(completed):,} completed snapshots already in {summary_tmp}.", flush=True)

    panel_handle, panel_writer = open_append_writer(out_tmp, ARCHIVE_COLUMNS, append=append)
    summary_columns = [
        "frequency",
        "archive_timestamp",
        "archive_capture_date",
        "archive_original_url",
        "archive_replay_url",
        "reported_count",
        "raw_rows",
        "rows_written",
        "duplicate_rows_dropped",
        "status",
        "fetched_at_utc",
    ]
    summary_handle, summary_writer = open_append_writer(summary_tmp, summary_columns, append=append)

    for i, capture in enumerate(selected, start=1):
        if capture["timestamp"] in completed:
            print(f"[{i:>4}/{len(selected):>4}] {capture['capture_date']} already complete; skipping", flush=True)
            continue
        replay_url = archive_replay_url(capture["timestamp"], capture["original"])
        print(f"[{i:>4}/{len(selected):>4}] {capture['capture_date']} {replay_url}", flush=True)
        payload: dict[str, Any] = {}
        try:
            payload, replay_url = load_or_fetch_replay(
                capture,
                frequency=args.frequency,
                retries=args.retries,
                refresh_raw=args.refresh_raw,
                save_raw=not args.no_raw,
            )
            person_list = payload.get("personList", {})
            rows = person_list.get("personsLists") or []
            if not isinstance(rows, list):
                rows = []
            snapshot_id = f"archive_realtime_{capture['capture_date']}"
            flat_rows = []
            for source_row in rows:
                flat = flatten_row(
                    source_row,
                    snapshot_kind="archive_realtime",
                    snapshot_id=snapshot_id,
                    source_url=replay_url,
                    fetched_at_utc=fetched_at,
                )
                # Row-level payload timestamps can be stale/mixed inside an
                # archived Forbes response. For the archive panel, the Wayback
                # capture date is the unique observation date.
                flat["snapshot_date"] = capture["capture_date"]
                flat.update(
                    {
                        "archive_timestamp": capture["timestamp"],
                        "archive_capture_date": capture["capture_date"],
                        "archive_original_url": capture["original"],
                        "archive_digest": capture["digest"],
                        "archive_length": capture["length"],
                        "archive_replay_url": replay_url,
                    }
                )
                flat_rows.append(flat)
            flat_rows, duplicate_rows_dropped = dedupe_panel_rows(flat_rows)
            flat_rows.sort(key=lambda r: (int(r["rank"] or r["position"] or 0), r["person_name"]))
            panel_writer.writerows(flat_rows)
            panel_handle.flush()
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - keep going across archive captures.
            print(f"  ERROR: {exc}", flush=True)
            rows = []
            flat_rows = []
            duplicate_rows_dropped = ""
            status = f"error: {exc}"

        summary_writer.writerow(
            {
                "frequency": args.frequency,
                "archive_timestamp": capture["timestamp"],
                "archive_capture_date": capture["capture_date"],
                "archive_original_url": capture["original"],
                "archive_replay_url": replay_url,
                "reported_count": payload.get("personList", {}).get("count", "") if status == "ok" else "",
                "raw_rows": len(rows),
                "rows_written": len(flat_rows),
                "duplicate_rows_dropped": duplicate_rows_dropped,
                "status": status,
                "fetched_at_utc": fetched_at,
            }
        )
        summary_handle.flush()
        if args.sleep:
            time.sleep(args.sleep)

    panel_handle.close()
    summary_handle.close()

    latest_rows = latest_summary_rows(summary_tmp, selected)
    missing = len(selected) - len(latest_rows)
    errors = [row for row in latest_rows if not str(row.get("status", "")).startswith("ok")]
    if (missing or errors) and not args.allow_errors:
        raise RuntimeError(
            f"Archive build incomplete: missing={missing}, errors={len(errors)}. "
            f"Temp files kept for resume: {out_tmp}, {summary_tmp}"
        )

    write_csv(summary_tmp, latest_rows, summary_columns)
    out_tmp.replace(out_csv)
    summary_tmp.replace(summary_csv)
    rows_written = sum(int(row.get("rows_written") or 0) for row in latest_rows)
    print(f"Wrote {rows_written:,} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
