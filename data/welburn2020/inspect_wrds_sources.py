"""Download linkage inputs through an explicitly supplied, already open connection.

This module never authenticates, reconnects, or closes the supplied connection.
Run its functions in the retained WRDS Python session. Licensed rows remain
under this directory; aggregate matching diagnostics are produced separately.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/welburn2020/wrds_inputs'
RAW.mkdir(parents=True, exist_ok=True)


def download_existing(connection, name, sql, params=None):
    """Stream a read-only SQL query through an existing database connection.

    ``name`` sets the output filename prefix; ``params`` binds query parameters.
    Save the extract with its SQL, parameters, and checksum, and return its path.
    """
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S%fZ')
    target = RAW / f'{name}_{stamp}.csv.gz'
    partial = target.with_suffix(target.suffix + '.partial')
    try:
        with connection.cursor() as cursor:
            query = cursor.mogrify(sql.strip().rstrip(';'), params).decode()
            with gzip.open(partial, 'xb', compresslevel=1) as stream:
                cursor.copy_expert(f'COPY ({query}) TO STDOUT WITH CSV HEADER', stream)
            rowcount = cursor.rowcount
        partial.rename(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    metadata = dict(dataset=name, downloaded_utc=stamp, file=target.name,
                    sql=sql, parameters=params, rows=rowcount,
                    bytes=target.stat().st_size, sha256=digest.hexdigest(),
                    connection='Retained programmatic WRDS connection; no login in downloader',
                    privacy='WRDS-licensed raw rows; private research use',
                    downloader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(json.dumps(dict(dataset=name, rows=rowcount, bytes=metadata['bytes'],
                          file=str(target.relative_to(ROOT)))), flush=True)
    return target


def save_catalog(rows, columns):
    target = RAW / 'catalog_2026-11-03.json'
    target.write_text(json.dumps(dict(tables=rows, columns=columns), indent=2, default=str) + '\n')
    return target
