#!/usr/bin/env python3
"""Solve and validate the dynamic inverse used by Stella (2015).

The deposited Stella archive calls a proprietary GAUSS routine.  This wrapper
uses the open-source Sims ``gensys.m`` shipped with Dynare, then benchmarks the
port against the exact 117-sector matrices deposited by Foerster et al. (2011),
whose model code Stella reuses.  The benchmark is deliberately a required gate
before solving reconstructed firm inputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from mgp_core import ROOT
from applications.stella.stella_provenance import archive_existing, file_fingerprints, verify_fingerprints


CODE = ROOT / "code"
SOLVER = CODE / "stella_dynamic_solver.m"
from applications.foerster.prepare import MODEL_INPUTS as FOERSTER_INPUTS, EXPORTS as FOERSTER_EXPORTS
STELLA_INPUT = ROOT / "code/.cache/firm_networks/stella_literal"
STELLA_OUTPUT = STELLA_INPUT / "dynamic_solution"
AUDIT_JSON = ROOT / "output/stella_dynamic_solver_benchmark.json"
AUDIT_CSV = ROOT / "output/stella_dynamic_solver_benchmark.csv"
OUTPUT_MATRICES = ("pii_a.csv", "xi.csv", "cap_rho.csv")

def find_gensys() -> Path:
    import os
    if configured := os.environ.get("GENSYS_DIR"):
        path = Path(configured)
        if (path / "gensys.m").is_file():
            return path
    candidates = sorted(
        glob.glob("/usr/local/Cellar/dynare/*/lib/dynare/contrib/ms-sbvar/TZcode/MatlabFiles"),
        reverse=True,
    )
    for candidate in candidates:
        if (Path(candidate) / "gensys.m").exists():
            return Path(candidate)
    raise FileNotFoundError("Dynare's gensys.m was not found under /usr/local/Cellar/dynare.")


def run_solver(input_dir: Path, output_dir: Path, octave: str, gensys_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [octave, "-qf", str(SOLVER), str(input_dir), str(output_dir), str(gensys_dir)]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    diag = np.loadtxt(output_dir / "solver_diagnostics.csv", delimiter=",").reshape(-1)
    keys = [
        "N", "eu_existence", "eu_uniqueness", "condition_k1", "max_ywt",
        "max_contemporaneous_capital_impact", "max_productivity_impact_minus_identity",
    ]
    out = {key: float(value) for key, value in zip(keys, diag, strict=True)}
    # Octave may emit a harmless one-line shutdown message on macOS.  Keep a
    # bounded record without allowing compatibility warnings to swamp JSON.
    out["octave_stdout"] = completed.stdout.strip()[-2000:]
    out["octave_stderr"] = completed.stderr.strip()[-2000:]
    return out


def benchmark(octave: str, gensys_dir: Path, tolerance: float) -> tuple[list[dict[str, float | str]], dict]:
    with tempfile.TemporaryDirectory(prefix="stella_solver_benchmark_") as tmp:
        tmpdir = Path(tmp)
        inputs = tmpdir / "inputs"
        outputs = tmpdir / "outputs"
        inputs.mkdir()
        for name in ("matrix_io", "alpha", "matrix_capital"):
            shutil.copy2(FOERSTER_INPUTS / f"{name}.csv", inputs / f"{name}.csv")
        solver_diag = run_solver(inputs, outputs, octave, gensys_dir)
        rows: list[dict[str, float | str]] = []
        for name in ("pii_a", "xi", "cap_rho"):
            csv_name = f"{name}.csv"
            actual = np.loadtxt(outputs / csv_name, delimiter=",")
            expected = np.loadtxt(FOERSTER_EXPORTS / f"{name}_98_benchmark.csv", delimiter=",")
            difference = actual - expected
            max_abs = float(np.max(np.abs(difference)))
            rmse = float(np.sqrt(np.mean(difference**2)))
            relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(expected))
            correlation = float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
            rows.append({
                "matrix": csv_name.removesuffix(".csv"),
                "dimension": int(actual.shape[0]),
                "max_abs_difference": max_abs,
                "rmse": rmse,
                "relative_l2_difference": relative_l2,
                "correlation": correlation,
                "tolerance": tolerance,
                "passes": bool(max_abs <= tolerance),
            })
    payload = {
        "benchmark": "Foerster-Sarte-Watson 1998-NAICS benchmark solution",
        "reference_format": "Foerster benchmark matrices regenerated from the original raw XLS inputs",
        "solver": "GNU Octave plus Dynare/Sims gensys.m",
        "gensys_directory": str(gensys_dir),
        "max_abs_tolerance": tolerance,
        "all_pass": all(bool(row["passes"]) for row in rows),
        "matrices": rows,
        "solver_diagnostics": solver_diag,
    }
    return rows, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=STELLA_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to INPUT_DIR/dynamic_solution")
    parser.add_argument("--octave", default=shutil.which("octave") or "/usr/local/bin/octave")
    parser.add_argument("--gensys-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--promote-from", type=Path,
                        help="install an already solved, benchmarked staging directory after verifying its input hashes")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "dynamic_solution"
    if args.promote_from:
        source = args.promote_from.resolve()
        payload = json.loads((source / "benchmark_and_solver_diagnostics.json").read_text())
        if not payload["all_pass"]:
            raise ValueError("Cannot promote a failed dynamic-solver benchmark.")
        verify_fingerprints(payload["input_file_fingerprints"], ROOT)
        n = int(payload["stella_solver_diagnostics"]["N"])
        for name in OUTPUT_MATRICES:
            matrix = np.loadtxt(source / name, delimiter=",")
            if matrix.shape != (n, n) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid staged solver matrix: {name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in [*OUTPUT_MATRICES, "solver_diagnostics.csv", "benchmark_and_solver_diagnostics.json"]:
            if (source / name).resolve() != (output_dir / name).resolve():
                archive_existing(output_dir / name)
                shutil.copy2(source / name, output_dir / name)
        print(json.dumps({"promoted_from": str(source), "promoted_to": str(output_dir), "N": n}, indent=2))
        return
    gensys_dir = args.gensys_dir or find_gensys()

    rows, payload = benchmark(args.octave, gensys_dir, args.tolerance)
    AUDIT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not payload["all_pass"]:
        raise RuntimeError(f"Dynamic solver failed Foerster benchmark; see {AUDIT_JSON}")
    if not args.benchmark_only:
        input_fingerprints = file_fingerprints(
            [args.input_dir / f"{name}.csv" for name in ("matrix_io", "alpha", "matrix_capital")], ROOT)
        payload["stella_solver_diagnostics"] = run_solver(
            args.input_dir, output_dir, args.octave, gensys_dir
        )
        verify_fingerprints(input_fingerprints, ROOT)
        payload["input_file_fingerprints"] = input_fingerprints
        (output_dir / "benchmark_and_solver_diagnostics.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
