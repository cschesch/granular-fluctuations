"""Select the paper's WRDS inputs, or explicitly request newer local exports."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_YEAR = 2025
ANALYSIS_LAST_DAY = "2025-12-31"
ANALYSIS_LAST_QUARTER = "2025Q4"

_SPECIAL_SOURCES = {
    "stella_quarterly": "stella2015",
    "wrds_supply_chain_ids": "welburn2020",
    "wrds_company_subsidiaries": "welburn2020",
    "wrds_subsidiary_global_aliases": "welburn2020",
}
_SELECTED = {}
_VERIFIED = {}


def _record(record):
    """Record selections made inside analysis subprocesses for the run report."""
    if log := os.environ.get("REPLICATION_INPUT_LOG"):
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"type": "input", **record}) + "\n")


def wrds_path(dataset: str, *, mode: str | None = None, path: Path | None = None) -> Path:
    """Resolve a pinned filename; newer exports require WRDS_INPUTS=latest."""
    mode = mode or os.environ.get("WRDS_INPUTS", "paper")
    if mode not in {"paper", "latest"}:
        raise ValueError("WRDS_INPUTS must be paper or latest")
    source = _SPECIAL_SOURCES.get(dataset)
    if source is None:
        source = "mutual_funds" if dataset.startswith("mutual_fund_") else dataset.split("_", 1)[0]
    directory = ROOT / "data" / source
    metadata = directory / "paper_inputs.json"
    if metadata.is_file():
        pin = json.loads(metadata.read_text())["datasets"][dataset]
    else:
        # A data-free checkout must still report missing applications.
        subdirectory = "wrds_inputs/" if dataset in _SPECIAL_SOURCES else ""
        pin = {"file": f"{subdirectory}{dataset}.csv.gz", "sha256": None,
               "analysis_cutoff": None, "extracted_utc": None}
    paper = directory / pin["file"]
    if path is not None:
        selected, mode = Path(path).expanduser().resolve(), "override"
    elif mode == "latest":
        exports = sorted(paper.parent.glob(f"{dataset}_*.csv.gz"), reverse=True)
        selected = exports[0] if exports else paper.parent / f"{dataset}.csv.gz"
    else:
        selected = paper
    key = (dataset, mode, selected)
    if key not in _SELECTED:
        name = str(selected.relative_to(ROOT)) if selected.is_relative_to(ROOT) else str(selected)
        exported = {}
        for sidecar in (selected.with_suffix("").with_suffix(".metadata.json"),
                        selected.with_suffix(".metadata.json")):
            if sidecar.is_file():
                exported = json.loads(sidecar.read_text())
                break
        extracted = next((exported[key] for key in (
            "source_downloaded_utc", "downloaded_utc", "retrieved_utc", "created_utc"
        ) if exported.get(key)), pin["extracted_utc"] if selected == paper else None)
        record = {
            "dataset": dataset, "file": name, "mode": mode,
            "exists": selected.is_file(), "paper_sha256": pin["sha256"],
            "sha256": None, "analysis_cutoff": pin["analysis_cutoff"],
            "paper_extracted_utc": pin["extracted_utc"],
            "extracted_utc": extracted,
        }
        _SELECTED[key] = record
        print(f"[input] {dataset}: {name} ({mode}{'' if record['exists'] else '; missing'})", flush=True)
    _SELECTED[key]["exists"] = selected.is_file()
    _record(_SELECTED[key])
    if os.environ.get("REPLICATION_VERIFY_INPUTS") == "1":
        verify_inputs([selected])
    return selected


def selected_inputs() -> list[dict]:
    """Return the selected filenames and any checksums verified in this process."""
    return [dict(record) for record in _SELECTED.values()]


def verify_inputs(paths=None) -> None:
    """Hash selected inputs before running; reject changed paper snapshots."""
    requested = None if paths is None else {Path(path).resolve() for path in paths}
    for (_, mode, path), record in _SELECTED.items():
        if (requested is not None and path.resolve() not in requested) or not path.is_file():
            continue
        record["exists"] = True
        if mode == "paper" and record["paper_sha256"] is None:
            raise ValueError(f"Missing paper_inputs.json checksum for {record['dataset']}")
        stat = path.stat()
        key = (path.resolve(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in _VERIFIED:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            _VERIFIED[key] = digest.hexdigest()
        record["sha256"] = _VERIFIED[key]
        record["matches_paper"] = record["sha256"] == record["paper_sha256"]
        _record(record)
        if mode == "paper" and not record["matches_paper"]:
            raise ValueError(
                f"Paper input checksum differs: {record['file']}. "
                "Use WRDS_INPUTS=latest for an explicitly different local export."
            )
