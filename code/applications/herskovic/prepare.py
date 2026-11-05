"""Recreate the authors' seeded MATLAB model-size population in the cache."""

import argparse
from pathlib import Path
import subprocess

from project_paths import CODE_DIR, DERIVED_CACHE, ROOT, upstream, find_program, matlab_string


DESTINATION = DERIVED_CACHE / "herskovic2020/calibrated_sizes_matlab_seed100.mat"


def prepare() -> Path:
    if DESTINATION.is_file():
        return DESTINATION
    matlab = find_program("matlab")
    if matlab is None:
        raise FileNotFoundError("MATLAB is required to regenerate the Herskovic model-size draw")
    expression = (
        f"addpath('{matlab_string(Path(__file__).parent)}'); "
        f"export_herskovic_calibrated_sizes('{matlab_string(ROOT)}','{matlab_string(upstream('herskovic2020'))}')"
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
