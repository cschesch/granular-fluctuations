"""Rebuild Foerster inputs from the preserved author XLS files.

The original programs are in source.zip. The reads and normalization below
follow IPload_98.gss (level 4, quarterly averages), gam_alpha_98.gss (level 4),
getTheta98.gss (level 4), and factor_io.gss (unadjusted benchmark recurrence).
The existing Octave/Sims solver implements Models_98.gss and Resolkw_98.gss.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import ast
import hashlib
import json

import numpy as np
import pandas as pd

from project_paths import CODE_DIR, DERIVED_CACHE, upstream, find_program


CACHE = DERIVED_CACHE / "foerster2011"
MODEL_INPUTS = CACHE / "model_inputs"
EXPORTS = CACHE / "exports"
STEM = "eps_q_98_0_1972_2007benchmark"
OUTPUTS = (
    f"{STEM}_wide.csv", f"{STEM}_columns.csv", "l4wq_q_98_0_1972_2007.csv",
    "pii_a_98_benchmark.csv", "xi_98_benchmark.csv", "cap_rho_98_benchmark.csv",
)
SOURCE_INPUTS = ("Data/IO98.xls", "Data/CapFlow98.xls", "Data/IP98.xls")
MODEL_FILES = ("matrix_io.csv", "alpha.csv", "matrix_capital.csv")
GENSYS_FILES = ("gensys.m", "qzdiv.m", "qzswitch.m")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(path: Path) -> str:
    """Ignore Python comments and docstrings when checking calculation code."""
    if path.suffix != ".py":
        return file_hash(path)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def input_fingerprint(source: Path) -> dict:
    return {
        "inputs": {name: file_hash(source / name) for name in SOURCE_INPUTS},
        "code": {str(path.relative_to(CODE_DIR)): code_hash(path)
                 for path in (Path(__file__), CODE_DIR / "network_solver.py",
                              CODE_DIR / "network_dynamic_solver.m")},
    }


def output_fingerprint(model_inputs: Path, exports: Path) -> dict:
    return {
        **{f"model_inputs/{name}": file_hash(model_inputs / name) for name in MODEL_FILES},
        **{f"exports/{name}": file_hash(exports / name) for name in OUTPUTS},
    }


def gensys_fingerprint(directory: Path) -> dict:
    return {name: file_hash(directory / name) for name in GENSYS_FILES}


def prepared_ready(source: Path | None = None) -> bool:
    """Validate prepared inputs and exports without requiring Octave or Dynare."""
    try:
        recorded = json.loads((CACHE / "preparation.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if (not isinstance(recorded, dict) or not isinstance(recorded.get("gensys"), dict)
            or set(recorded["gensys"]) != set(GENSYS_FILES)):
        return False
    try:
        fingerprint = input_fingerprint(source if source is not None else upstream("foerster2011"))
        outputs = output_fingerprint(MODEL_INPUTS, EXPORTS)
    except FileNotFoundError:
        return False
    if (any(recorded.get(key) != value for key, value in fingerprint.items())
            or recorded.get("outputs") != outputs):
        return False
    from network_solver import find_gensys

    try:
        gensys = find_gensys()
    except FileNotFoundError:
        return True
    return all(recorded["gensys"][name] == file_hash(gensys / name)
               for name in GENSYS_FILES if (gensys / name).is_file())


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
    if prepared_ready():
        print("[foerster] reused validated preparation", flush=True)
        return True
    source = upstream("foerster2011")
    if not all((source / name).is_file() for name in SOURCE_INPUTS):
        print("[skip] Foerster preparation: original publication archive is absent", flush=True)
        return False
    octave = find_program("octave")
    if octave is None:
        print("[skip] Foerster preparation: Octave is required to solve the deposited model", flush=True)
        return False
    from network_solver import find_gensys

    try:
        gensys = find_gensys()
        gensys_files = gensys_fingerprint(gensys)
    except FileNotFoundError:
        print("[skip] Foerster preparation: install Dynare or set GENSYS_DIR to the directory containing gensys.m, qzdiv.m and qzswitch.m", flush=True)
        return False
    fingerprint = input_fingerprint(source)
    fingerprint["gensys"] = gensys_files
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / "preparation.json"
    with TemporaryDirectory(prefix=".prepare-", dir=CACHE) as temporary:
        staging = Path(temporary)
        model_inputs, exports = staging / "model_inputs", staging / "exports"
        model_inputs.mkdir()
        exports.mkdir()
        quarters, sectors = write_preparation(source, model_inputs, exports, octave, gensys)
        if (input_fingerprint(source) != {key: fingerprint[key] for key in ("inputs", "code")}
                or gensys_fingerprint(gensys) != fingerprint["gensys"]):
            raise RuntimeError("Foerster source or calculation code changed during preparation")
        fingerprint["outputs"] = output_fingerprint(model_inputs, exports)
        staged_stamp = staging / "preparation.json"
        staged_stamp.write_text(json.dumps(fingerprint, indent=2) + "\n")
        MODEL_INPUTS.mkdir(parents=True, exist_ok=True)
        EXPORTS.mkdir(parents=True, exist_ok=True)
        # No valid stamp may survive a partially completed promotion.
        stamp.unlink(missing_ok=True)
        for staged, destination in ((model_inputs, MODEL_INPUTS), (exports, EXPORTS)):
            for path in staged.iterdir():
                path.replace(destination / path.name)
        staged_stamp.replace(stamp)
    print(f"[foerster] prepared {quarters} quarters and {sectors} sectors", flush=True)
    return True


def write_preparation(source: Path, model_inputs: Path, exports: Path,
                      octave: str, gensys: Path) -> tuple[int, int]:
    from network_solver import run_solver

    for name, matrix in zip(("matrix_io", "alpha", "matrix_capital"), structural_inputs(source), strict=True):
        np.savetxt(model_inputs / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")
    run_solver(model_inputs, exports, octave, gensys)
    calendar, levels, weights, labels = industrial_production(source)
    pii = np.loadtxt(exports / "pii_a.csv", delimiter=",")
    xi = np.loadtxt(exports / "xi.csv", delimiter=",")
    rho = np.loadtxt(exports / "cap_rho.csv", delimiter=",")
    inverse = np.linalg.inv(pii)
    growth = np.zeros_like(levels)
    growth[1:] = 400 * np.log(levels[1:] / levels[:-1])
    forcing = np.zeros_like(levels)
    forcing[2:] = growth[2:] @ inverse.T + growth[1:-1] @ (-inverse @ rho).T
    autoregression = -inverse @ xi
    shocks = np.zeros_like(levels)
    for t in range(1, len(shocks)):
        shocks[t] = forcing[t] + shocks[t - 1] @ autoregression.T
    pd.DataFrame(np.column_stack([calendar, shocks]), columns=["cal", *labels.epsilon_column]).to_csv(exports / f"{STEM}_wide.csv", index=False)
    labels.to_csv(exports / f"{STEM}_columns.csv", index=False)
    np.savetxt(exports / "l4wq_q_98_0_1972_2007.csv", np.column_stack([calendar, weights]), delimiter=",", fmt="%.15g")
    for name, matrix in (("pii_a", pii), ("xi", xi), ("cap_rho", rho)):
        np.savetxt(exports / f"{name}_98_benchmark.csv", matrix, delimiter=",", fmt="%.15g")
    return shocks.shape


if __name__ == "__main__":
    prepare()
