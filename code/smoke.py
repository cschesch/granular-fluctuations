"""Exercise the estimators on synthetic data with settings smaller than the paper's."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

from run_tests import CODE, copy_code


def synthetic_run(report):
    import numpy as np
    import pandas as pd
    import inference_mc as mc
    from grouped_giv_mc import METHODS, build_population, simulate_cell, summarize
    from mgp_core import build_lpt_grouped_data, choose_solver, mgp_share

    started = time.perf_counter()
    rng = np.random.default_rng(1729)
    n, t, m = 60, 120, 5
    shares = rng.uniform(1, 3, n)
    shares /= shares.sum()
    loading = rng.normal(size=(n, 2))
    outcomes = loading @ rng.normal(size=(2, t)) + rng.normal(size=(n, t))
    grouped, _, _, _ = build_lpt_grouped_data(outcomes, shares, m)
    np.testing.assert_allclose(grouped.sum(axis=1), shares @ outcomes, atol=1e-12)
    estimate = mgp_share(grouped, choose_solver("auto"))
    assert 0 <= estimate["mgp_share"] <= 1

    design = mc.Design("synthetic_smoke", "Synthetic smoke", n=60, m=5, r=2, t=60, seed=1729)
    config = mc.McConfig(reps=4, n_boot=9, n_subsample=4, seed=1729)
    simulation = mc.run_design(design, config)
    assert len(simulation["share"]) == config.reps
    assert np.isfinite(simulation["share"]).all()

    draws = pd.DataFrame(simulate_cell(build_population(1000, 100, "core_weak"), 2, rng))
    assert set(draws.method) == set(METHODS) and len(draws) == 2 * len(METHODS)
    summary = summarize(draws, 2)
    table = report.parent / "synthetic_giv_summary.csv"
    summary.to_csv(table, index=False)
    assert table.stat().st_size > 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.write_text(json.dumps({
        "purpose": "synthetic smoke test; not paper estimates", "status": "passed",
        "python": platform.python_version(), "platform": platform.platform(),
        "seconds": time.perf_counter() - started,
        "peak_memory_mb": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "panel": {"N": n, "T": t, "M": m, "estimate": estimate["mgp_share"]},
        "inference_mc": {"N": 60, "T": 60, "M": 5, "replications": 4, "bootstrap_draws": 9},
        "giv_mc": {"N": 1000, "T": 100, "replications": 2, "methods": list(METHODS)},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=CODE / ".cache/smoke_report.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        synthetic_run(args.report)
        return
    with tempfile.TemporaryDirectory(prefix="granularity-smoke-") as directory:
        root = Path(directory)
        copy_code(root / "code")
        report = root / "smoke_report.json"
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="2",
                   MPLCONFIGDIR=str(root / ".matplotlib"), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, str(root / "code/smoke.py"), "--worker", "--report", str(report)],
                       cwd=root / "code", env=env, check=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(report.read_bytes())
    print(f"Synthetic smoke test passed. Reduced settings are recorded in {args.report}.")


if __name__ == "__main__":
    main()
