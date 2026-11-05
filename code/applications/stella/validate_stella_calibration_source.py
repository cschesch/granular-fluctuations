#!/usr/bin/env python3
"""Run the unmodified deposited constructors on our reconstructed inputs."""

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
from applications.stella.prepare_stella_calibration import OFFICIAL, OFFICIAL_IO, OFFICIAL_SURPLUS, resolve_edges
from applications.stella.run_stella_firm_network import CALIBRATION
from applications.stella.stella_provenance import archive_existing, file_fingerprints


def validate(directory: Path, octave: str, *, export_inputs: bool = False) -> dict:
    directory = directory.resolve()
    mapping = pd.read_csv(directory / "firm_mapping.csv", dtype={"gvkey": str, "stella_sid": str})
    mapping["gvkey"] = mapping["gvkey"].str.zfill(6)
    ratios = pd.read_csv(directory / "firm_employment_ratios.csv", dtype={"gvkey": str}).set_index("gvkey")
    ratios.index = ratios.index.str.zfill(6)
    ratios = ratios.reindex(mapping.gvkey)
    n = len(mapping)
    position = {gvkey: index + 1 for index, gvkey in enumerate(mapping.gvkey)}
    sid = mapping.set_index("gvkey")["stella_sid"]
    edges, _ = resolve_edges(pd.Index(mapping.gvkey))
    comp_data = np.array([
        [position[supplier], position[buyer], int(sid[supplier]), int(sid[buyer])]
        for supplier, buyer in edges.itertuples(index=False, name=None)
    ])
    tid = np.column_stack((np.arange(1, n + 1), ratios.first_time,
                           ratios.last_time, mapping.stella_sid.astype(int), ratios.ratio_emp))
    if export_inputs:
        destination = directory / "source_constructor_inputs"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OFFICIAL_IO, destination / "matrix_io_input.csv")
        shutil.copyfile(OFFICIAL_SURPLUS, destination / "valadd.csv")
        np.savetxt(destination / "tid.csv", tid, delimiter=",", fmt="%.17g")
        np.savetxt(destination / "comp_data.csv", comp_data, delimiter=",", fmt="%d")
    source_files = []
    results = []
    with tempfile.TemporaryDirectory(prefix="stella_original_constructor_") as temporary:
        work = Path(temporary)
        shutil.copyfile(OFFICIAL_IO, work / "matrix_io_input.csv")
        shutil.copyfile(OFFICIAL_SURPLUS, work / "valadd.csv")
        np.savetxt(work / "tid.csv", tid, delimiter=",", fmt="%.17g")
        np.savetxt(work / "comp_data.csv", comp_data, delimiter=",", fmt="%.17g")
        for calibration, filename in (("sparse", "matrix_construction_IIcalibration.m"),
                                      ("dense", "matrix_construction_Icalibration.m")):
            source = OFFICIAL / "filtering" / filename
            source_files.append(source)
            subprocess.run([octave, "-qf", str(source)], cwd=work, check=True,
                           text=True, capture_output=True, timeout=300)
            expected = np.loadtxt(work / "matrix_io.csv", delimiter=",")
            if calibration == "sparse":
                actual = np.loadtxt(directory / "matrix_io.csv", delimiter=",")
            else:
                actual = pd.read_csv(directory / "matrix_io_dense_economic_labeled.csv", index_col=0).to_numpy(float)
                actual = np.where(actual == 0, 1e-19, actual)
            results.append({"calibration": calibration, "matrix": "matrix_io",
                            "N": n, "max_abs_difference": float(np.max(np.abs(expected - actual)))})
            for matrix in ("alpha", "matrix_capital"):
                expected = np.loadtxt(work / f"{matrix}.csv", delimiter=",")
                actual = np.loadtxt(directory / f"{matrix}.csv", delimiter=",")
                results.append({"calibration": calibration, "matrix": matrix,
                                "N": n, "max_abs_difference": float(np.max(np.abs(expected - actual)))})
    payload = {"claim": "Unmodified deposited MATLAB sparse and dense constructors executed under Octave on reconstructed inputs",
               "calibration_directory": str(directory.relative_to(ROOT)), "N": n,
               "source_hashes": file_fingerprints(source_files, ROOT),
               "input_hashes": file_fingerprints([directory / name for name in
                    ("firm_mapping.csv", "firm_employment_ratios.csv", "matrix_io.csv",
                     "matrix_io_dense_economic_labeled.csv")], ROOT),
               "matrices": results, "all_pass": all(row["max_abs_difference"] < 1e-12 for row in results),
               "scope": "Validates constructor execution, not the identity or vintage of reconstructed firm/customer/BLS inputs"}
    if not payload["all_pass"]:
        raise AssertionError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION)
    parser.add_argument("--octave", default=shutil.which("octave") or "/usr/local/bin/octave")
    parser.add_argument("--output", type=Path, default=ROOT / "output/stella_original_constructor_validation.json")
    parser.add_argument("--export-inputs", action="store_true",
                        help="save private headerless inputs for the untouched deposited constructors")
    args = parser.parse_args()
    payload = validate(args.calibration_dir, args.octave, export_inputs=args.export_inputs)
    path = args.output
    archive_existing(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
