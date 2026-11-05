#!/usr/bin/env python3
"""Acquire Stella-specific raw WRDS fields without replacing project snapshots.

Run with code/.cache/wrds_venv/bin/python. Saved credentials are used by libpq;
no password is read into this program or written to metadata. --probe-only saves
the server's actual quarterly dictionary, distinguishing an omitted export field
from a field unavailable in the current WRDS subscription. All raw observations
remain licensed local data. This does not recover the author's historical vintage.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

HOST = "wrds-pgdata.wharton.upenn.edu"
PORT = 9737
HERE = Path(__file__).resolve().parent


def saved_username() -> str | None:
    supplied = os.environ.get("WRDS_USERNAME") or os.environ.get("PGUSER")
    if supplied:
        return supplied
    path = Path(os.environ.get("PGPASSFILE", str(Path.home() / ".pgpass")))
    if not path.is_file():
        return None
    users = set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = re.split(r"(?<!\\):", line)
        if (len(fields) == 5 and fields[0] in (HOST, "*")
                and fields[1] in (str(PORT), "*")
                and fields[2] in ("wrds", "*") and fields[3] != "*"):
            users.add(fields[3])
    return next(iter(users)) if len(users) == 1 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import psycopg2
    from psycopg2 import sql

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=saved_username())
    parser.add_argument("--schema", default="comp")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--wrds-cloud", action="store_true",
                        help="Use the official client's authenticated WRDS Cloud connection")
    parser.add_argument("--connect-timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=HERE / "wrds_inputs")
    args = parser.parse_args()
    if not args.username and not args.wrds_cloud:
        parser.error("Use --username or a matching saved WRDS login.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%fZ")
    metadata = {"retrieved_utc": stamp, "schema": args.schema,
                "historical_vintage": "Current WRDS; not the author's 2015 snapshot",
                "script_sha256": sha256(Path(__file__)), "licensed_private": True}
    status = args.output_dir / "latest_acquisition_status.json"
    print("Connecting to WRDS using saved credentials; complete any requested MFA.", flush=True)
    wrapper = None
    try:
        if args.wrds_cloud:
            import wrds
            wrapper = wrds.Connection()
            wrapped = wrapper.connection.connection
            connection = getattr(wrapped, "driver_connection", None)
            if connection is None:
                connection = wrapped.connection
        else:
            connection = psycopg2.connect(host=HOST, port=PORT, dbname="wrds",
                user=args.username, sslmode="require", connect_timeout=args.connect_timeout,
                application_name="stella2015_replication_inputs")
    except psycopg2.Error as error:
        metadata.update(status="connection_failed", error_type=type(error).__name__,
            next_step="Restore WRDS/VPN connectivity or run this script inside WRDS Cloud; retry acquisition.")
        status.write_text(json.dumps(metadata, indent=2) + "\n")
        raise SystemExit("WRDS connection failed before a schema result was obtained. "
                         "Revenue availability remains unverified; see acquisition status.") from None
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '180000'")
            cursor.execute("""SELECT column_name,data_type FROM information_schema.columns
                WHERE table_schema=%s AND table_name='fundq' ORDER BY ordinal_position""",
                (args.schema,))
            columns = dict(cursor.fetchall())
            if not columns:
                raise RuntimeError("No accessible quarterly table found in the selected schema.")
            metadata.update(status="schema_verified", quarterly_columns=columns,
                revtq_available="revtq" in columns,
                saleq_available="saleq" in columns)
            dictionary = args.output_dir / f"fundq_dictionary_{stamp}.json"
            dictionary.write_text(json.dumps(metadata, indent=2) + "\n")
            print("WRDS verified: revtq available =", "revtq" in columns, flush=True)
            if args.probe_only:
                status.write_text(json.dumps(metadata, indent=2) + "\n")
                return
            selected = [name for name in ["gvkey", "datadate", "datacqtr", "datafqtr",
                "fyearq", "fqtr", "fyr", "rdq", "indfmt", "consol", "popsrc",
                "datafmt", "curcdq", "saleq", "revtq", "saley", "sich", "naicsh"]
                if name in columns]
            query = sql.SQL("""SELECT {fields},c.conm,c.conml,c.costat,c.fic,c.loc,c.sic,c.naics
                FROM {schema}.fundq q LEFT JOIN {schema}.company c ON q.gvkey=c.gvkey
                WHERE q.indfmt='INDL' AND q.consol='C' AND q.datafmt='STD'
                AND q.popsrc='D' AND q.curcdq='USD'
                AND q.fyearq BETWEEN 1962 AND 2011 ORDER BY q.gvkey,q.datadate""").format(
                    fields=sql.SQL(",").join(sql.Identifier("q", c) for c in selected),
                    schema=sql.Identifier(args.schema))
            query_text = query.as_string(connection)
            path = args.output_dir / f"stella_quarterly_wrds_{stamp}.csv.gz"
            partial = path.with_name(path.name + ".partial")
            with gzip.open(partial, "xb", compresslevel=1) as output:
                cursor.copy_expert("COPY (" + query_text + ") TO STDOUT WITH CSV HEADER", output)
            rows = cursor.rowcount
            partial.rename(path)
            metadata.update(status="downloaded", file=path.name, sql=query_text,
                sha256=sha256(path), bytes=path.stat().st_size,
                rows=rows if rows >= 0 else None,
                outcome_field="revtq" if "revtq" in columns else "saleq",
                outcome_substitution="revtq" not in columns)
            path.with_name(path.name.removesuffix(".csv.gz") + ".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            status.write_text(json.dumps(metadata, indent=2) + "\n")
            print("Saved private quarterly input:", path.name, flush=True)
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            connection.close()


if __name__ == "__main__":
    main()
