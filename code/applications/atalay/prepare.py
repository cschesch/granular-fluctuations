"""Rebuild Atalay's public filter inputs from the original publication ZIP.

The original archive stays untouched. Stata and MATLAB run in a working copy
under .cache. The sole Stata adaptation lets an installed hprescott satisfy
its SSC dependency without trying to install it again.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

from project_paths import CODE_DIR, ROOT, find_program, matlab_string


ARCHIVE = ROOT / "data/atalay2017/replication_elasticities.zip"
ARCHIVE_SHA256 = "bea450fd194343a2c13ef48212ff63c9898727dde51ccaac618f95daa77a2ce7"
DERIVED = CODE_DIR / ".cache/derived/atalay2017"
WORK = DERIVED / "replication_package"
DRIVERS = (CODE_DIR / "export_atalay_elasticity_profile.m", CODE_DIR / "export_atalay_elasticity_surface.m")
STATA_FILES = ("dataKJ.csv", "dataMJ.csv", "dataOJ.csv", "ioJ_data_dt.csv")


def run_matlab(program, statements):
    prefix = f"ATALAY_REPO_ROOT='{matlab_string(ROOT)}'; "
    subprocess.run([program, "-batch", prefix + statements], cwd=WORK, check=True)


def prepare_stata(program, fingerprint):
    marker = DERIVED / "source.json"
    if marker.is_file() and json.loads(marker.read_text())["fingerprint"] == fingerprint:
        if all((WORK / name).is_file() for name in STATA_FILES):
            return
    if DERIVED.exists():
        shutil.rmtree(DERIVED)
    WORK.mkdir(parents=True)
    with ZipFile(ARCHIVE) as archive:
        archive.extractall(WORK)
    do_file = WORK / "io_read.do"
    source = do_file.read_bytes()
    old = b"ssc install hprescott"
    if source.count(old) != 1:
        raise ValueError("Unexpected SSC dependency line in the original io_read.do")
    adapted = source.replace(old, b"capture which hprescott\r\nif _rc ssc install hprescott")
    do_file.write_bytes(adapted)
    subprocess.run([program, "-b", "do", "io_read.do"], cwd=WORK, check=True)
    missing = [name for name in STATA_FILES if not (WORK / name).is_file()]
    if missing:
        raise RuntimeError(f"Atalay data preparation failed to produce {missing}. See {WORK / 'io_read.log'}")
    marker.write_text(json.dumps({"fingerprint": fingerprint, "original_archive_sha256": ARCHIVE_SHA256}, indent=2) + "\n")


def prepare_benchmark(program):
    outputs = [DERIVED / f"atalay_benchmark_{stem}.csv" for stem in ("productivity_shocks", "network_variance_matrix")]
    if all(path.is_file() for path in outputs):
        return
    run_matlab(program, (
        "ATALAY_SWEEP='eps_M'; ATALAY_PROFILE_GRID=0.1; "
        f"ATALAY_PROFILE_OUT='{matlab_string(DERIVED / 'benchmark')}'; "
        f"run('{matlab_string(DRIVERS[0])}');"
    ))
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat

    benchmark = loadmat(DERIVED / "benchmark/atalay_eps_M_001.mat", squeeze_me=True)
    frame = pd.DataFrame(benchmark["shocks"], columns=[f"shock_{i:02d}" for i in range(1, 31)])
    frame.insert(0, "year", np.asarray(benchmark["years"], dtype=int))
    frame.to_csv(outputs[0], index=False, float_format="%.15g")
    np.savetxt(outputs[1], benchmark["V"], delimiter=",", fmt="%.15g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("benchmark", "profile", "surface", "all"), default="all")
    parser.add_argument("--available-only", action="store_true", help="Skip if the original source or external programs are unavailable.")
    parser.add_argument("--matlab")
    parser.add_argument("--stata")
    args = parser.parse_args()
    matlab = find_program("matlab", args.matlab)
    stata = find_program("stata", args.stata)
    if not ARCHIVE.is_file() or not matlab or not stata:
        message = "Atalay preprocessing requires its original ZIP, Stata with hprescott, and MATLAB with Statistics and Machine Learning Toolbox."
        if args.available_only:
            print(f"[skip] {message}", flush=True)
            return
        raise RuntimeError(message)
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"Atalay original archive hash differs: {archive_hash}")
    digest = hashlib.sha256(archive_hash.encode())
    for path in (*DRIVERS, Path(__file__)):
        digest.update(path.read_bytes())
    prepare_stata(stata, digest.hexdigest())
    prepare_benchmark(matlab)
    if args.section in ("profile", "all"):
        for sweep in ("eps_M", "eps_Q"):
            if not (DERIVED / "elasticity_profile" / sweep / "grid_index.csv").is_file():
                run_matlab(matlab, f"ATALAY_SWEEP='{sweep}'; run('{matlab_string(DRIVERS[0])}');")
    if args.section in ("surface", "all"):
        expected = [DERIVED / "elasticity_surface" / f"atalay_surface_epsD_{value}.mat" for value in (250, 500, 750, 900, 999, 1250)]
        from scipy.io import loadmat

        complete = all(path.is_file() for path in expected)
        if complete:
            for path in expected:
                state = loadmat(path, variable_names=("rows_done", "eps_M_grid"), squeeze_me=True)
                if int(state["rows_done"]) != len(state["eps_M_grid"]):
                    complete = False
                    break
        if not complete:
            run_matlab(matlab, f"run('{matlab_string(DRIVERS[1])}');")


if __name__ == "__main__":
    main()
