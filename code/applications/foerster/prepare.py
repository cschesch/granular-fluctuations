"""Rebuild Foerster inputs from the preserved author XLS files.

The original programs are in source.zip. The reads and normalization below
follow IPload_98.gss (level 4, quarterly averages), gam_alpha_98.gss (level 4),
getTheta98.gss (level 4), and factor_io.gss (unadjusted benchmark recurrence).
The existing Octave/Sims solver implements Models_98.gss and Resolkw_98.gss.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd

from project_paths import CODE_DIR, DERIVED_CACHE, upstream


CACHE = DERIVED_CACHE / "foerster2011"
MODEL_INPUTS = CACHE / "model_inputs"
EXPORTS = CACHE / "exports"
STEM = "eps_q_98_0_1972_2007benchmark"
OUTPUTS = (
    f"{STEM}_wide.csv", f"{STEM}_columns.csv", "l4wq_q_98_0_1972_2007.csv",
    "pii_a_98_benchmark.csv", "xi_98_benchmark.csv", "cap_rho_98_benchmark.csv",
)


def structural_inputs(source: Path):
    # GAUSS uses Excel B2:DN118 and W4:Y120 of sheet 10 (one-based).
    io = pd.read_excel(source / "Data/IO98.xls", sheet_name=4, header=None).iloc[1:118, 1:118].to_numpy(float)
    accounts = pd.read_excel(source / "Data/IO98.xls", sheet_name=9, header=None).iloc[3:120, 22:25].to_numpy(float)
    material = io.sum(axis=0)
    total = material + accounts[:, 0] + accounts[:, 1]
    gamma = io / material[None, :] * (material / total)[None, :]
    gamma[gamma == 0] = 1e-19
    alpha = np.diag(accounts[:, 1] / total)
    capital = pd.read_excel(source / "Data/CapFlow98.xls", sheet_name=4, header=None).iloc[1:118, 1:118].to_numpy(float)
    theta = capital / capital.sum(axis=0)[None, :]
    theta[theta == 0] = 1e-19
    return gamma, alpha, theta


def industrial_production(source: Path):
    # IPload_98.gss reads L4 monthly levels from D7 and shares from sheet 12.
    book = source / "Data/IP98.xls"
    levels = pd.read_excel(book, sheet_name=4, header=None)
    weights = pd.read_excel(book, sheet_name=11, header=None).iloc[6:, 3:120].apply(pd.to_numeric, errors="coerce")
    dates = levels.iloc[6:, :2].apply(pd.to_numeric, errors="coerce")
    values = levels.iloc[6:, 3:120].apply(pd.to_numeric, errors="coerce")
    count = min(len(weights.dropna(how="any")), len(values.dropna(how="any")))
    dates, values, weights = (frame.iloc[:count] for frame in (dates, values, weights))
    keep = dates.iloc[:, 0].lt(2008).to_numpy()
    dates, values, weights = (frame.loc[keep].to_numpy(float) for frame in (dates, values, weights))
    if len(values) % 3:
        raise ValueError("The original monthly IP sample does not contain complete quarters")
    # mtoq(...,ag_code=0) averages each consecutive block of three months.
    levels = values.reshape(-1, 3, 117).mean(axis=1)
    weights = weights.reshape(-1, 3, 117).mean(axis=1) / 100
    months = dates[2::3]
    calendar = months[:, 0] + ((months[:, 1] / 3) - 1) / 4
    labels = pd.read_excel(book, sheet_name=4, header=None, nrows=6, dtype=str).iloc[:, 3:120].T.reset_index(drop=True)
    labels.columns = ("level4_series_number", "level3_series_number", "level2_series_number",
                      "level1_series_number", "haver_code", "description")
    columns = [f"eps_{i:03d}" for i in range(1, 118)]
    labels.insert(0, "sector_order", range(1, 118))
    labels.insert(0, "epsilon_column", columns)
    return calendar, levels, weights, labels


def prepare() -> bool:
    source = upstream("foerster2011")
    if not (source / "Data/IP98.xls").is_file():
        print("[skip] Foerster preparation: original publication archive is absent", flush=True)
        return False
    octave = os.environ.get("OCTAVE") or shutil.which("octave")
    if octave is None and Path("/usr/local/bin/octave").is_file():
        octave = "/usr/local/bin/octave"
    if octave is None:
        print("[skip] Foerster preparation: Octave is required to solve the deposited model", flush=True)
        return False
    from applications.stella.solve_stella_dynamic import find_gensys, run_solver

    try:
        gensys = find_gensys()
    except FileNotFoundError:
        print("[skip] Foerster preparation: install Dynare or set GENSYS_DIR to the directory containing gensys.m", flush=True)
        return False
    fingerprint = dict(source=str(source), files={
        str(path.relative_to(CODE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), CODE_DIR / "stella_dynamic_solver.m")
    }, gensys_sha256=hashlib.sha256((gensys / "gensys.m").read_bytes()).hexdigest())
    stamp = CACHE / "preparation.json"
    if stamp.is_file() and json.loads(stamp.read_text()) == fingerprint and all((EXPORTS / name).is_file() for name in OUTPUTS):
        return True
    MODEL_INPUTS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    for name, matrix in zip(("matrix_io", "alpha", "matrix_capital"), structural_inputs(source), strict=True):
        np.savetxt(MODEL_INPUTS / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")
    run_solver(MODEL_INPUTS, EXPORTS, octave, gensys)
    calendar, levels, weights, labels = industrial_production(source)
    pii = np.loadtxt(EXPORTS / "pii_a.csv", delimiter=",")
    xi = np.loadtxt(EXPORTS / "xi.csv", delimiter=",")
    rho = np.loadtxt(EXPORTS / "cap_rho.csv", delimiter=",")
    inverse = np.linalg.inv(pii)
    growth = np.zeros_like(levels)
    growth[1:] = 400 * np.log(levels[1:] / levels[:-1])
    forcing = np.zeros_like(levels)
    forcing[2:] = growth[2:] @ inverse.T + growth[1:-1] @ (-inverse @ rho).T
    autoregression = -inverse @ xi
    shocks = np.zeros_like(levels)
    for t in range(1, len(shocks)):
        shocks[t] = forcing[t] + shocks[t - 1] @ autoregression.T
    pd.DataFrame(np.column_stack([calendar, shocks]), columns=["cal", *labels.epsilon_column]).to_csv(EXPORTS / f"{STEM}_wide.csv", index=False)
    labels.to_csv(EXPORTS / f"{STEM}_columns.csv", index=False)
    np.savetxt(EXPORTS / "l4wq_q_98_0_1972_2007.csv", np.column_stack([calendar, weights]), delimiter=",", fmt="%.15g")
    for name, matrix in (("pii_a", pii), ("xi", xi), ("cap_rho", rho)):
        np.savetxt(EXPORTS / f"{name}_98_benchmark.csv", matrix, delimiter=",", fmt="%.15g")
    stamp.write_text(json.dumps(fingerprint, indent=2) + "\n")
    print(f"[foerster] rebuilt {len(shocks)} quarters and {shocks.shape[1]} sectors", flush=True)
    return True


if __name__ == "__main__":
    prepare()
