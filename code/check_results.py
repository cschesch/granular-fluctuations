"""Compare newly generated public-input estimates with fixed paper reference values."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd

from project_paths import CODE_DIR, ROOT


def check_results(cache: Path, reference: Path) -> dict:
    checked_at = datetime.now(timezone.utc).isoformat()
    reference_bytes = reference.read_bytes()
    specification = json.loads(reference_bytes)
    if not specification.get("checks"):
        raise ValueError("The numerical reference contains no checks")
    reports, frames, result_hashes = [], {}, {}
    for check in specification["checks"]:
        filename = check["file"]
        if Path(filename).name != filename:
            raise ValueError(f"Expected a result filename, got {filename!r}")
        path = cache / filename
        entry = {"entity": check["entity"], "file": filename, "status": "unavailable"}
        if not path.is_file():
            entry["reason"] = "Result file is absent"
            reports.append(entry)
            continue
        if filename not in frames:
            contents = path.read_bytes()
            result_hashes[filename] = hashlib.sha256(contents).hexdigest()
            frames[filename] = pd.read_pickle(io.BytesIO(contents))
        frame = frames[filename]
        if len(frame.index) == 0:
            entry["reason"] = "Generated summary contains no applications"
            reports.append(entry)
            continue
        required = {"Entity", *check["values"]}
        if missing := required - set(frame.columns):
            raise ValueError(f"Invalid result file {path}: missing columns {sorted(missing)}")
        rows = frame.loc[frame.Entity.eq(check["entity"])]
        if rows.empty:
            entry["reason"] = "Application is absent from the generated summary"
        elif len(rows) != 1:
            raise ValueError(f"Duplicate result rows for {check['entity']}")
        else:
            actual = rows.iloc[0]
            absent = [field for field in check["values"] if pd.isna(actual[field])]
            if absent:
                entry["reason"] = f"Unavailable fields: {', '.join(absent)}"
            else:
                comparisons = {}
                for field, expected in check["values"].items():
                    tolerance = float(check.get("absolute_tolerances", {}).get(field, 0.0))
                    value = float(actual[field])
                    if tolerance < 0 or not math.isfinite(tolerance) or not math.isfinite(float(expected)):
                        raise ValueError(f"Invalid reference value or tolerance for {field}")
                    matches = math.isfinite(value) and abs(value - expected) <= tolerance
                    comparisons[field] = {"expected": expected, "actual": value if math.isfinite(value) else str(value),
                                          "absolute_tolerance": tolerance, "matches": matches}
                entry["fields"] = comparisons
                entry["status"] = "passed" if all(c["matches"] for c in comparisons.values()) else "failed"
        reports.append(entry)
    counts = {status: sum(row["status"] == status for row in reports)
              for status in ("passed", "failed", "unavailable")}
    status = ("failed" if counts["failed"] else "not_checked" if not counts["passed"]
              else "partial" if counts["unavailable"] else "passed")
    return {"status": status, "checked_at": checked_at,
            "reference": str(reference), "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "cache": str(cache), "result_sha256": result_hashes,
            "counts": counts, "checks": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CODE_DIR / ".cache/run_outputs")
    parser.add_argument("--reference", type=Path, default=CODE_DIR / "reference_results.json")
    parser.add_argument("--report", type=Path, default=ROOT / "output/result_checks.json")
    parser.add_argument("--strict", action="store_true", help="Fail if any reference application is unavailable.")
    args = parser.parse_args()
    report = check_results(args.cache, args.reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for check in report["checks"]:
        reason = f": {check['reason']}" if "reason" in check else ""
        print(f"[{check['status']}] {check['entity']}{reason}")
    counts = report["counts"]
    print(f"Numerical checks: {report['status']}; {counts['passed']} passed, "
          f"{counts['failed']} failed, {counts['unavailable']} unavailable. Report: {args.report}")
    return int(bool(counts["failed"] or (args.strict and counts["unavailable"])))


if __name__ == "__main__":
    raise SystemExit(main())
