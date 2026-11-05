"""Solve the dynamic network model shared by Foerster and Stella."""

from __future__ import annotations

import glob
from pathlib import Path
import subprocess

import numpy as np

from project_paths import CODE_DIR

SOLVER = CODE_DIR / "network_dynamic_solver.m"


def find_gensys() -> Path:
    import os
    if configured := os.environ.get("GENSYS_DIR"):
        path = Path(configured)
        if (path / "gensys.m").is_file():
            return path
        raise FileNotFoundError(f"GENSYS_DIR does not contain gensys.m: {path}")
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
