"""Recreate the authors' seeded MATLAB model-size population in the cache."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from project_paths import CODE_DIR, DERIVED_CACHE, ROOT, upstream


DESTINATION = DERIVED_CACHE / "herskovic2020/calibrated_sizes_matlab_seed100.mat"


def prepare() -> Path:
    if DESTINATION.is_file():
        return DESTINATION
    matlab = os.environ.get("MATLAB") or shutil.which("matlab")
    if matlab is None:
        installed = sorted(Path("/Applications").glob("MATLAB_*.app/bin/matlab"), reverse=True)
        matlab = str(installed[0]) if installed else None
    if matlab is None:
        raise FileNotFoundError("MATLAB is required to regenerate the Herskovic model-size draw")
    quote = lambda path: str(path).replace("'", "''")
    expression = (
        f"addpath('{quote(CODE_DIR)}'); "
        f"export_herskovic_calibrated_sizes('{quote(ROOT)}','{quote(upstream('herskovic2020'))}')"
    )
    subprocess.run([matlab, "-batch", expression], cwd=CODE_DIR, check=True)
    return DESTINATION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare()
    except FileNotFoundError as error:
        if not args.available_only:
            raise
        print(f"[skip] Herskovic preprocessing: {error}", flush=True)
