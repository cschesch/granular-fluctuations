"""Rebuild three public-data point estimates from empty caches and check fixed references.

Uses 12 bootstrap draws to exercise inference; this is not the paper inference run.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from run_tests import CODE, copy_code
from verify_release import manifest_path, sha256


PUBLIC_INPUTS = {
    "data/maddison/maddison_project_gdp_2023_owid.csv",
    "data/fdic/fdic_quarterly_financials_1999q4_2025q4.csv",
    "data/french/F-F_International_Countries.zip",
    "data/french/F-F_Research_Data_Factors_CSV.zip",
    "data/french/world_bank_CM.MKT.LCAP.CD_20_markets_1960_2025.json",
}


def copy_inputs(source, destination):
    """Copy only the reference inputs and verify their hashes before estimation."""
    checks = json.loads((source / "code/reference_results.json").read_text())["checks"]
    expected = {name: digest for check in checks for name, digest in check["inputs"].items()}
    if expected.keys() != PUBLIC_INPUTS:
        raise ValueError("Public references must use exactly the five allowlisted inputs")
    for name, digest in expected.items():
        original = manifest_path(source, name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha256(target) != digest:
            raise ValueError(f"Public input checksum mismatch: {name}")
    return expected


def public_run(report, report_path):
    if (CODE / ".cache").exists():
        raise ValueError("Public verification must start without caches")
    report["started_without_caches"] = True
    sys.modules["mosek"] = None
    import pandas as pd
    from mosek_face import MOSEK_AVAILABLE
    from applications.french.prepare import main as prepare_french
    from applications.maddison.panel import load_countries
    from applications.fdic.panel import load_banks
    from applications.french.panel import load_country_equity_returns
    from applications.common import estimate
    from mgp_core import RUN_OUTPUT_CACHE

    if MOSEK_AVAILABLE:
        raise RuntimeError("Public verification must run without MOSEK")
    report["mosek_import_blocked"] = True
    report["software_versions"] = {name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "numba", "statsmodels")}
    started = time.perf_counter()
    prepare_french()
    report["french_preparation_seconds"] = time.perf_counter() - started
    rows, timings = [], {}
    for load in (load_countries, load_banks, load_country_equity_returns):
        started = time.perf_counter()
        app = load()
        rows.append(estimate(app, solver="CLARABEL", bootstrap_b=12, seed=20260907))
        timings[app.entity] = time.perf_counter() - started
    report["application_seconds"] = timings
    pd.DataFrame(rows).to_pickle(RUN_OUTPUT_CACHE / "additional_applications_summary.pkl")
    checked = report_path.with_name("result_checks.json")
    result = subprocess.run([sys.executable, str(CODE / "check_results.py"),
                             "--strict", "--report", str(checked)], cwd=CODE)
    report["result_checks"] = json.loads(checked.read_text())
    if result.returncode:
        raise RuntimeError("Regenerated public estimates did not pass strict reference checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=CODE / ".cache/public_report.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {"purpose": "public-data point estimates; reduced bootstrap, not paper inference",
              "status": "failed", "solver": "CLARABEL", "bootstrap_draws": 12,
              "seed": 20260907, "python": platform.python_version(), "platform": platform.platform()}
    try:
        if args.worker:
            public_run(report, args.report)
        else:
            with tempfile.TemporaryDirectory(prefix="granularity-public-") as temporary:
                root = Path(temporary)
                copy_code(root / "code")
                report["input_hashes"] = copy_inputs(CODE.parent, root)
                worker_report = root / "public_report.json"
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="2",
                           MPLCONFIGDIR=str(root / ".matplotlib"), NUMBA_CACHE_DIR=str(root / "code/.cache/numba"),
                           PYTHONDONTWRITEBYTECODE="1")
                result = subprocess.run([sys.executable, str(root / "code/check_public_results.py"),
                                         "--worker", "--report", str(worker_report)], cwd=root / "code", env=env)
                if worker_report.exists():
                    report.update(json.loads(worker_report.read_text()))
                if result.returncode:
                    raise RuntimeError(report.get("error", f"Public estimator exited with status {result.returncode}"))
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"Public verification failed: {error}", file=sys.stderr)
    report["seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Public point-estimate checks: {report['status']}. Report: {args.report}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
