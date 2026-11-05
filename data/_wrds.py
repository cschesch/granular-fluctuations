"""Shared WRDS login and CSV export; extraction choices live in each downloader."""
import argparse
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(script, dataset, sql, start_date=None):
    """Save raw rows beside the calling script, with SQL and hashes in a sidecar."""
    script = Path(script).resolve()
    parser = argparse.ArgumentParser(description=f"Download {dataset} from WRDS.")
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME", ""))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--print-sql", action="store_true")
    if start_date:
        parser.add_argument("--start-date", type=date.fromisoformat,
                            default=date.fromisoformat(start_date))
    args = parser.parse_args()
    if args.print_sql:
        print(sql)
        return
    params = {"end_date": args.end_date.isoformat(), "end_year": args.end_date.year,
              "end_quarter": f"{args.end_date.year}Q{(args.end_date.month - 1) // 3 + 1}"}
    if start_date:
        if args.start_date > args.end_date:
            parser.error("start-date must precede end-date")
        params["start_date"] = args.start_date.isoformat()

    import wrds

    # The official client uses saved credentials or prompts in the terminal.
    db = wrds.Connection(wrds_username=args.username,
                         wrds_connect_args={"sslmode": "require", "connect_timeout": 180})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%fZ")
    stem = f"{dataset}_wrds_{stamp}"
    target = script.parent / f"{stem}.csv.gz"
    partial = target.with_name(target.name + ".partial")
    try:
        wrapped = db.connection.connection
        raw = getattr(wrapped, "driver_connection", None)
        if raw is None:
            raw = wrapped.connection
        with raw.cursor() as cursor:
            query = cursor.mogrify(sql.strip().rstrip(";"), params).decode()
            print(f"Downloading {dataset}...", flush=True)
            # COPY streams unmodified rows; gzip only reduces storage size.
            with gzip.open(partial, "xb", compresslevel=1) as output:
                cursor.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", output)
            rows = cursor.rowcount
        partial.rename(target)
    finally:
        db.close()
    metadata = {"dataset": dataset, "downloaded_utc": stamp, "file": target.name,
                "sql": sql, "parameters": params, "rows": rows if rows >= 0 else None,
                "bytes": target.stat().st_size, "sha256": sha256(target),
                "script": script.name, "script_sha256": sha256(script),
                "helper_sha256": sha256(Path(__file__)),
                "storage": "gzip-compressed CSV; unmodified WRDS query rows"}
    (script.parent / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {target.name} ({rows:,} rows).", flush=True)
