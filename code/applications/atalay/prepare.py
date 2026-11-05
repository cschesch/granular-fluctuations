"""Prepare Atalay's benchmark, profiles and calibration comparison from the source ZIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat

from project_paths import CODE_DIR, ROOT, find_program, matlab_string


ARCHIVE = ROOT / "data/atalay2017/replication_elasticities.zip"
ARCHIVE_SHA256 = "bea450fd194343a2c13ef48212ff63c9898727dde51ccaac618f95daa77a2ce7"
DERIVED = CODE_DIR / ".cache/derived/atalay2017"
WORK = DERIVED / "replication_package"
HERE = Path(__file__).resolve().parent
STATA_FILES = ("dataKJ.csv", "dataMJ.csv", "dataOJ.csv", "ioJ_data_dt.csv")
SOURCE_VERSION = 1  # Increment when the Stata adaptation changes, not for plotting edits.
SURFACE_VALUES = ((667, "2/3"), (999, "0.999"), (1333, "4/3"))


def adapted_stata_source():
    with ZipFile(ARCHIVE) as archive:
        source = archive.read("io_read.do")
    old = b"ssc install hprescott"
    if source.count(old) != 1:
        raise ValueError("Unexpected SSC dependency line in the original io_read.do")
    return source.replace(old, b"capture which hprescott\r\nif _rc ssc install hprescott")


def source_record():
    return {"source_version": SOURCE_VERSION, "original_archive_sha256": ARCHIVE_SHA256,
            "stata_files": {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest()
                            for name in STATA_FILES}}


def source_ready():
    """Verify the prepared source; adopt the earlier cache only after checking its adaptation."""
    marker = DERIVED / "source.json"
    if not marker.is_file():
        return False
    recorded = json.loads(marker.read_text())
    if recorded.get("original_archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Atalay prepared inputs refer to a different source archive; archive them before rebuilding.")
    if recorded.get("source_version", SOURCE_VERSION) != SOURCE_VERSION:
        raise ValueError("Atalay source preparation version changed; archive the old prepared inputs before rebuilding.")
    if not all((WORK / name).is_file() for name in (*STATA_FILES, "io_read.do")):
        return False
    if (WORK / "io_read.do").read_bytes() != adapted_stata_source():
        raise ValueError("Atalay io_read.do differs from the reviewed source adaptation.")
    current = source_record()
    if "stata_files" in recorded and recorded["stata_files"] != current["stata_files"]:
        raise ValueError("Atalay prepared Stata inputs changed; archive the affected derivatives before rebuilding.")
    if recorded != current:
        marker.write_text(json.dumps(current, indent=2) + "\n")
    return True


def prepare_stata(program):
    """Replace only the source working copy, after a successful staged Stata run."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stata-", dir=DERIVED) as temporary:
        staging = Path(temporary) / "new"
        staging.mkdir()
        with ZipFile(ARCHIVE) as archive:
            archive.extractall(staging)
        (staging / "io_read.do").write_bytes(adapted_stata_source())
        try:
            subprocess.run([program, "-b", "do", "io_read.do"], cwd=staging, check=True)
        finally:
            if (staging / "io_read.log").is_file():
                log = DERIVED / "stata_preparation.log"
                log.write_bytes((staging / "io_read.log").read_bytes())
                print(f"Atalay Stata log: {log}", flush=True)
        missing = [name for name in STATA_FILES if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"Atalay Stata preparation did not produce {missing}")
        backup = Path(temporary) / "previous"
        if WORK.exists():
            WORK.rename(backup)
        try:
            staging.rename(WORK)
        except BaseException:
            if backup.exists():
                backup.rename(WORK)
            raise
    (DERIVED / "source.json").write_text(json.dumps(source_record(), indent=2) + "\n")


def run_matlab(program, statements):
    prefix = f"ATALAY_REPO_ROOT='{matlab_string(ROOT)}'; "
    subprocess.run([program, "-singleCompThread", "-batch", prefix + statements], cwd=WORK, check=True)


def benchmark_ready():
    return all((DERIVED / f"atalay_benchmark_{stem}.csv").is_file()
               for stem in ("productivity_shocks", "network_variance_matrix"))


def prepare_benchmark(program):
    run_matlab(program, (
        "ATALAY_SWEEP='eps_M'; ATALAY_PROFILE_GRID=0.1; "
        f"ATALAY_PROFILE_OUT='{matlab_string(DERIVED / 'benchmark')}'; "
        f"run('{matlab_string(HERE / 'export_atalay_elasticity_profile.m')}');"
    ))
    benchmark = loadmat(DERIVED / "benchmark/atalay_eps_M_001.mat", squeeze_me=True)
    frame = pd.DataFrame(benchmark["shocks"], columns=[f"shock_{i:02d}" for i in range(1, 31)])
    frame.insert(0, "year", np.asarray(benchmark["years"], dtype=int))
    frame.to_csv(DERIVED / "atalay_benchmark_productivity_shocks.csv", index=False, float_format="%.15g")
    np.savetxt(DERIVED / "atalay_benchmark_network_variance_matrix.csv", benchmark["V"], delimiter=",", fmt="%.15g")


def profile_ready(sweep):
    directory = DERIVED / "elasticity_profile" / sweep
    index = directory / "grid_index.csv"
    if not index.is_file():
        return False
    grid = pd.read_csv(index)
    return (len(grid) == {"eps_M": 75, "eps_Q": 31}[sweep]
            and grid["sweep"].eq(sweep).all() and grid["value"].is_unique
            and grid["status"].eq("ok").all()
            and all((directory / name).is_file() for name in grid["file"]))


def surface_ready(tag):
    path = DERIVED / "elasticity_surface" / f"atalay_surface_epsD_{tag}.mat"
    if not path.is_file():
        return False
    state = loadmat(path, variable_names=("eps_D", "eps_M_grid", "eps_Q_grid", "rows_done", "ok"), squeeze_me=True)
    expected_d = {667: 2 / 3, 999: .999, 1333: 4 / 3}[tag]
    expected_m = np.round(np.r_[.005, .01, .02, .03, .04 + .03 * np.arange(35)], 6)
    expected_m[np.isclose(expected_m, 1, rtol=0, atol=1e-9)] = .999
    expected_q = np.sort(np.r_[.03 * np.arange(1, 51), .999])
    complete = (float(state["eps_D"]) == expected_d
            and np.asarray(state["eps_M_grid"]).shape == expected_m.shape
            and np.allclose(state["eps_M_grid"], expected_m, rtol=0, atol=1e-14)
            and np.asarray(state["eps_Q_grid"]).shape == expected_q.shape
            and np.allclose(state["eps_Q_grid"], expected_q, rtol=0, atol=1e-14)
            and int(state["rows_done"]) == len(expected_m))
    if complete and not np.asarray(state["ok"]).all():
        raise RuntimeError(f"Atalay surface contains failed cells: {path}; inspect its messages before rebuilding.")
    return complete


def fits_ready(tag):
    directory = DERIVED / "elasticity_surface"
    path = directory / f"one_factor_{tag}.csv"
    if not path.is_file() or not surface_ready(tag):
        return False
    state = loadmat(directory / f"atalay_surface_epsD_{tag}.mat",
                    variable_names=("eps_D", "eps_M_grid", "eps_Q_grid", "ok"), squeeze_me=True)
    table = pd.read_csv(path)
    m, q = np.meshgrid(state["eps_M_grid"], state["eps_Q_grid"], indexing="ij")
    expected = np.c_[m[state["ok"].astype(bool)], q[state["ok"].astype(bool)]]
    actual = table[["eps_M", "eps_Q"]].to_numpy()
    return (actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=1e-10)
            and np.allclose(table["eps_D"], float(state["eps_D"]), rtol=0, atol=1e-14)
            and np.isfinite(table["one_factor"]).all() and table["fit_limit"].isin((0, 1)).all())


def prepare_surface(program, tag, value):
    if not surface_ready(tag):
        run_matlab(program, f"ATALAY_SURFACE_EPSD={value}; run('{matlab_string(HERE / 'export_atalay_elasticity_surface.m')}');")
    if not fits_ready(tag):
        run_matlab(program, f"addpath('{matlab_string(HERE)}'); fit_surface(ATALAY_REPO_ROOT, '{tag}');")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("benchmark", "profile", "surface", "all"), default="all")
    parser.add_argument("--available-only", action="store_true", help="Skip when required inputs or programs are unavailable.")
    parser.add_argument("--matlab")
    parser.add_argument("--stata")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3, help="Concurrent surface slabs.")
    args = parser.parse_args()
    if not ARCHIVE.is_file():
        if args.available_only:
            print(f"[skip] Atalay: missing {ARCHIVE}", flush=True)
            return
        raise FileNotFoundError(ARCHIVE)
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"Atalay original archive hash differs: {archive_hash}")
    prepared_source = source_ready()
    benchmark = args.section in ("benchmark", "all") and not benchmark_ready()
    profiles = [s for s in ("eps_M", "eps_Q") if args.section in ("profile", "all") and not profile_ready(s)]
    surfaces = [(tag, value) for tag, value in SURFACE_VALUES
                if args.section in ("surface", "all") and not fits_ready(tag)]
    if not benchmark and not profiles and not surfaces:
        print(f"Atalay {args.section}: prepared inputs are complete.", flush=True)
        return
    matlab = find_program("matlab", args.matlab)
    stata = find_program("stata", args.stata) if not prepared_source else None
    if not matlab or (not prepared_source and not stata):
        message = "Atalay preparation needs MATLAB with Statistics and Machine Learning Toolbox"
        if not prepared_source:
            message += " and Stata with hprescott"
        if args.available_only:
            print(f"[skip] {message}.", flush=True)
            return
        raise RuntimeError(message)
    if not prepared_source:
        prepare_stata(stata)
    if benchmark:
        prepare_benchmark(matlab)
    for sweep in profiles:
        run_matlab(matlab, f"ATALAY_SWEEP='{sweep}'; run('{matlab_string(HERE / 'export_atalay_elasticity_profile.m')}');")
        if not profile_ready(sweep):
            raise RuntimeError(f"Atalay {sweep} profile is incomplete; inspect its grid_index.csv.")
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        list(workers.map(lambda item: prepare_surface(matlab, *item), surfaces))


if __name__ == "__main__":
    main()
