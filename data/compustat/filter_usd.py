#!/usr/bin/env python3
"""Apply the downloaders' USD restriction to existing WRDS snapshots, without a query.

Preserve every retained CSV record verbatim and archive the unfiltered snapshots.
Sidecars record the local filter and original download provenance.
"""
import ast
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _wrds import sha256

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA.parent


def filter_snapshot(record, column, downloader):
    source = ROOT / record["file"]
    old_metadata = json.loads((ROOT / record["metadata"]).read_text())
    if old_metadata.get("local_filter") == f"{column} = 'USD'":
        return record
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%fZ")
    target = source.parent / f"{record['dataset']}_wrds_{stamp}.csv.gz"
    pending = []
    count = 0
    dates = []
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as raw, gzip.open(
        target, "xt", encoding="utf-8", newline="", compresslevel=1
    ) as output:
        def lines():
            for line in raw:
                pending.append(line)
                yield line
        reader = csv.reader(lines())
        header = next(reader)
        output.write("".join(pending)); pending.clear()
        currency = header.index(column)
        date_col = header.index("datadate")
        for row in reader:
            if row[currency] == "USD":
                output.write("".join(pending))
                count += 1
                dates.append(row[date_col])
            pending.clear()
    tree = ast.parse(downloader.read_text())
    sql = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SQL" for t in n.targets))
    archive = source.parent / "archive"
    archive.mkdir(exist_ok=True)
    source_hash = sha256(source)
    assert source_hash == old_metadata["sha256"]
    metadata = {
        "dataset": record["dataset"], "file": target.name, "created_utc": stamp,
        "creation_method": "Local row subset; no WRDS query rerun",
        "local_filter": f"{column} = 'USD'", "rows": count,
        "sql": sql, "parameters": old_metadata["parameters"],
        "source_file": str((archive / source.name).relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_downloaded_utc": old_metadata["downloaded_utc"],
        "bytes": target.stat().st_size, "sha256": sha256(target),
        "script_sha256": sha256(downloader), "filter_script_sha256": sha256(Path(__file__)),
        "storage": "CSV.gz; original retained CSV records preserved verbatim",
    }
    sidecar = target.with_suffix("").with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    for path in source.parent.glob(source.name.split(".csv")[0] + ".*"):
        if (archive / path.name).exists():
            raise FileExistsError(archive / path.name)
        path.rename(archive / path.name)
    updated = {**record, "file": str(target.relative_to(ROOT)),
               "metadata": str(sidecar.relative_to(ROOT)), "rows": count,
               "bytes": metadata["bytes"], "sha256": metadata["sha256"], "sql": sql,
               "source_coverage": {**(record.get("source_coverage") or {}),
                                   "first": min(dates), "last": max(dates)}}
    print(f"{record['dataset']}: {record['rows']:,} -> {count:,} USD records", flush=True)
    return updated


if __name__ == "__main__":
    inventory_path = DATA / "wrds_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    for i, record in enumerate(inventory["datasets"]):
        column = {"compustat_quarterly_sales": "curcdq", "compustat_annual": "curcd"}.get(record["dataset"])
        if column:
            inventory["datasets"][i] = filter_snapshot(record, column, ROOT / record["download_script"])
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
