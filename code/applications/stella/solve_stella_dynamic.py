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
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mgp_core import ROOT
from project_paths import find_program
from network_solver import find_gensys, run_solver
from applications.foerster.prepare import MODEL_INPUTS as FOERSTER_INPUTS, EXPORTS as FOERSTER_EXPORTS
from applications.stella.paths import CALIBRATION
from applications.stella.stella_provenance import file_fingerprints, verify_fingerprints


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
    parser.add_argument("--input-dir", type=Path, default=CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to INPUT_DIR/dynamic_solution")
    parser.add_argument("--octave")
    parser.add_argument("--gensys-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    octave = find_program("octave", args.octave)
    if octave is None:
        parser.error("Octave is required; set OCTAVE or pass --octave.")
    output_dir = args.output_dir or args.input_dir / "dynamic_solution"
    gensys_dir = args.gensys_dir or find_gensys()
    _, payload = benchmark(octave, gensys_dir, args.tolerance)
    if not payload["all_pass"]:
        raise RuntimeError(f"Dynamic solver failed Foerster benchmark:\n{json.dumps(payload, indent=2)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = output_dir / "benchmark_and_solver_diagnostics.json"
    input_fingerprints = file_fingerprints(
        [args.input_dir / f"{name}.csv" for name in ("matrix_io", "alpha", "matrix_capital")], ROOT)
    payload["stella_solver_diagnostics"] = run_solver(args.input_dir, output_dir, octave, gensys_dir)
    verify_fingerprints(input_fingerprints, ROOT)
    payload["input_file_fingerprints"] = input_fingerprints
    diagnostics.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
