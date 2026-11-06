#!/usr/bin/env python3
"""Run the unmodified deposited network constructor on reconstructed inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT
from project_paths import find_program
from applications.stella.prepare_stella_calibration import OFFICIAL, OFFICIAL_IO, OFFICIAL_SURPLUS, resolve_edges
from applications.stella.paths import CALIBRATION
from applications.stella.stella_provenance import archive_existing, file_fingerprints, source_path


def validate(directory: Path, octave: str, *, export_inputs: bool = False) -> dict:
    directory = directory.resolve()
    calibration = json.loads((directory / "diagnostics.json").read_text()).get("network_calibration", "customer")
    mapping = pd.read_csv(directory / "firm_mapping.csv", dtype={"gvkey": str, "stella_sid": str})
    mapping["gvkey"] = mapping["gvkey"].str.zfill(6)
    ratios = pd.read_csv(directory / "firm_employment_ratios.csv", dtype={"gvkey": str}).set_index("gvkey")
    ratios.index = ratios.index.str.zfill(6)
    ratios = ratios.reindex(mapping.gvkey)
    n = len(mapping)
    comp_data = None
    if calibration != "dense":
        position = {gvkey: index + 1 for index, gvkey in enumerate(mapping.gvkey)}
        sid = mapping.set_index("gvkey")["stella_sid"]
        edges, _ = resolve_edges(pd.Index(mapping.gvkey))
        comp_data = np.array([
            [position[supplier], position[buyer], int(sid[supplier]), int(sid[buyer])]
            for supplier, buyer in edges.itertuples(index=False, name=None)
        ])
    tid = np.column_stack((np.arange(1, n + 1), ratios.first_time,
                           ratios.last_time, mapping.stella_sid.astype(int), ratios.ratio_emp))
    results = []
    filename = "matrix_construction_Icalibration.m" if calibration == "dense" else "matrix_construction_IIcalibration.m"
    source = OFFICIAL / "filtering" / filename
    with tempfile.TemporaryDirectory(prefix="stella_original_constructor_") as temporary:
        work = Path(temporary)
        shutil.copyfile(OFFICIAL_IO, work / "matrix_io_input.csv")
        shutil.copyfile(OFFICIAL_SURPLUS, work / "valadd.csv")
        np.savetxt(work / "tid.csv", tid, delimiter=",", fmt="%.17g")
        if comp_data is not None:
            np.savetxt(work / "comp_data.csv", comp_data, delimiter=",", fmt="%.17g")
        if export_inputs:
            destination = directory / "source_constructor_inputs"
            destination.mkdir(parents=True, exist_ok=True)
            for path in work.glob("*.csv"):
                shutil.copyfile(path, destination / path.name)
        subprocess.run([octave, "-qf", str(source)], cwd=work, check=True,
                       text=True, capture_output=True, timeout=300)
        for matrix in ("matrix_io", "alpha", "matrix_capital"):
            expected = np.loadtxt(work / f"{matrix}.csv", delimiter=",")
            actual = np.loadtxt(directory / f"{matrix}.csv", delimiter=",")
            results.append({"calibration": calibration, "matrix": matrix,
                            "N": n, "max_abs_difference": float(np.max(np.abs(expected - actual)))})
    payload = {"claim": "Unmodified deposited MATLAB network constructor executed under Octave on reconstructed inputs",
               "calibration_directory": source_path(directory, ROOT), "N": n,
               "source_hashes": file_fingerprints([source], ROOT),
               "input_hashes": file_fingerprints([directory / name for name in
                    ("firm_mapping.csv", "firm_employment_ratios.csv", "matrix_io.csv")], ROOT),
               "matrices": results, "all_pass": all(row["max_abs_difference"] < 1e-12 for row in results),
               "scope": "Validates constructor execution, not the identity or vintage of reconstructed firm/customer/BLS inputs"}
    if not payload["all_pass"]:
        raise AssertionError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION)
    parser.add_argument("--octave")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-inputs", action="store_true",
                        help="save private headerless inputs for the deposited constructor")
    args = parser.parse_args()
    octave = find_program("octave", args.octave)
    if octave is None:
        parser.error("Octave is required; set OCTAVE or pass --octave.")
    payload = validate(args.calibration_dir, octave, export_inputs=args.export_inputs)
    path = args.output or args.calibration_dir / "original_constructor_validation.json"
    archive_existing(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
