#!/usr/bin/env python3
"""Fetch official BEA accounts, historical BLS employment, and Census concordances.

Only the US national member is downloaded from each QCEW by-area ZIP, using
HTTP byte ranges and its central directory. No private data or credentials are
sent. Each downloaded member is checked against the ZIP CRC and SHA256-pinned
in a provenance sidecar. QCEW is an explicit reconstruction: Stella identifies
BLS as the source but does not deposit the original series or vintage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import urllib.request
import zipfile
import zlib

from project_paths import ROOT
PUBLIC = ROOT / "data/stella2015/public_inputs"
HEADERS = {"User-Agent": "Mozilla/5.0 (public economic research replication)"}
BEA_URL = "https://apps.bea.gov/industry/zip/ndn0305.zip"
CONCORDANCES = ["1997_NAICS_to_2002_NAICS.xls", "2002_to_2007_NAICS.xls", "2007_to_2012_NAICS.xls"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str, byte_range: str | None = None) -> tuple[bytes, dict]:
    headers = dict(HEADERS)
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if byte_range and r.status != 206:
            raise ValueError(f"Server did not honor byte range: {url}")
        return r.read(), {"url": r.url, "last_modified": r.headers.get("Last-Modified"), "etag": r.headers.get("ETag")}


def save(path: Path, data: bytes, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    record = {**metadata, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256(data), "bytes": len(data)}
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def national_qcew(year: int, refresh: bool = False) -> dict:
    path = PUBLIC / "bls_qcew" / f"{year}_US000_annual.csv"
    if path.exists() and not refresh:
        record = json.loads(path.with_suffix(".csv.provenance.json").read_text())
        if sha256(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Cached public input checksum mismatch: {path}")
        return record
    url = f"https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_by_area.zip"
    tail, meta = get(url, "-65536")
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise ValueError("ZIP end-of-central-directory not found")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment = struct.unpack("<4s4H2LH", tail[pos:pos + 22])
    if disk or cd_disk or n_disk != n_total:
        raise ValueError("Unsupported multipart ZIP")
    central, _ = get(url, f"{cd_offset}-{cd_offset + cd_size - 1}")
    cursor = 0
    selected = []
    for _ in range(n_total):
        header = central[cursor:cursor + 46]
        fields = struct.unpack("<4s6H3L5H2L", header)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central directory")
        compression, crc, compressed_size, size = fields[4], fields[7], fields[8], fields[9]
        name_len, extra_len, comment_len, offset = fields[10], fields[11], fields[12], fields[16]
        name = central[cursor + 46:cursor + 46 + name_len].decode("utf-8")
        if re.search(r"\bUS000\b", name) and name.endswith(".csv"):
            selected.append((name, compression, crc, compressed_size, size, offset))
        cursor += 46 + name_len + extra_len + comment_len
    if len(selected) != 1:
        raise ValueError(f"Expected one US000 member, found {len(selected)} for {year}")
    name, compression, crc, compressed_size, size, offset = selected[0]
    local_header, _ = get(url, f"{offset}-{offset + 29}")
    fields = struct.unpack("<4s5H3L2H", local_header)
    start = offset + 30 + fields[-2] + fields[-1]
    compressed, _ = get(url, f"{start}-{start + compressed_size - 1}")
    data = zlib.decompress(compressed, -15) if compression == 8 else compressed
    if len(data) != size or zlib.crc32(data) != crc:
        raise ValueError(f"ZIP size/CRC mismatch: {name}")
    record = save(path, data, {**meta, "archive_member": name, "archive_member_crc32": f"{crc:08x}", "year": year,
                              "source": "BLS QCEW annual averages, US total; all ownership categories retained in raw input",
                              "naics_vintage": 2002 if year <= 2006 else 2007 if year <= 2010 else 2012})
    print(f"QCEW {year}: {len(data):,} bytes, CRC verified", flush=True)
    return record


def other_inputs(refresh: bool = False) -> list[dict]:
    records = []
    bea = PUBLIC / "bea_1997_summary.zip"
    data, meta = get(BEA_URL)
    # Prevent an HTML redirect from silently replacing the research input.
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        required = {"NAICSUseSummary.xls", "NAICS-IO.xls", "IO-CodeSummary.xls"}
        if not required.issubset(set(z.namelist())):
            raise ValueError("BEA archive does not contain the 1997 summary inputs")
        old_hash = sha256(bea.read_bytes()) if bea.exists() else None
        if old_hash is not None and old_hash != sha256(data):
            raise ValueError("Downloaded BEA archive differs from existing input; review vintage before replacing")
        records.append(save(bea, data, {**meta, "source_landing_page": "https://www.bea.gov/industry/historical-benchmark-input-output-tables"}))
        # All names in this official archive are simple local filenames.
        for name in z.namelist():
            if Path(name).name != name:
                raise ValueError("Unexpected archive path")
            destination = PUBLIC / "bea_1997_summary" / name
            member = z.read(name)
            if destination.exists():
                if destination.read_bytes() != member:
                    raise ValueError(f"Existing BEA extracted member differs: {name}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(member)
    for name in CONCORDANCES:
        path = PUBLIC / "naics_concordances" / name
        if path.exists() and not refresh:
            records.append(json.loads(path.with_suffix(path.suffix + ".provenance.json").read_text()))
            continue
        url = "https://www.census.gov/naics/concordances/" + name
        data, meta = get(url)
        if not data.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
            raise ValueError(f"Expected an Excel XLS concordance: {url}")
        records.append(save(path, data, {**meta, "source": "US Census official NAICS concordance"}))
    return records


def railroad_ces(refresh: bool = False) -> list[dict]:
    """Railroad UI workers are excluded from QCEW; use the CES total instead."""
    records = []
    for start in range(1970, 2012, 10):
        end = min(start + 9, 2011)
        path = PUBLIC / "bls_ces_supplement" / f"CEU4348200001_{start}_{end}.json"
        if path.exists() and not refresh:
            records.append(json.loads(path.with_suffix(".json.provenance.json").read_text()))
            continue
        url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/CEU4348200001?startyear={start}&endyear={end}"
        data, meta = get(url)
        body = json.loads(data)
        series = body.get("Results", {}).get("series", [])
        if body.get("status") != "REQUEST_SUCCEEDED" or len(series) != 1 or not series[0].get("data"):
            raise ValueError(f"BLS CES railroad retrieval failed: {body.get('message')}")
        records.append(save(path, data, {**meta, "series": "CEU4348200001", "source": "BLS CES rail transportation, all employees, thousands, not seasonally adjusted", "coverage_reason": "QCEW excludes railroad workers under the railroad unemployment insurance system; CES includes these jobs."}))
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2011)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--qcew-only", action="store_true")
    args = p.parse_args()
    records = [] if args.qcew_only else other_inputs(args.refresh)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records.extend(pool.map(lambda y: national_qcew(y, args.refresh), range(args.start_year, args.end_year + 1)))
    records.extend(railroad_ces(args.refresh))
    # Preserve provenance of already acquired sources on --qcew-only resumes.
    indexed = {r["url"] + "#" + r.get("archive_member", ""): r for r in records}
    for path in PUBLIC.rglob("*.provenance.json"):
        record = json.loads(path.read_text())
        if "url" in record:
            indexed.setdefault(record["url"] + "#" + record.get("archive_member", ""), record)
    manifest = {"method": __doc__, "sources": list(indexed.values()),
                "remaining_identification_gap": "The original sectoral_emp_raw.dta and series identifiers/vintage were not deposited; public QCEW is a documented BLS reconstruction, not proof of the original denominator series."}
    (PUBLIC / "public_input_acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
