"""Estimate the independent panel applications used in the paper."""

import argparse

import pandas as pd

from applications.common import estimate
from applications.registry import INNOVATION, PANELS
from mgp_core import ANALYSIS_BOOTSTRAP_DRAWS, RUN_OUTPUT_CACHE, choose_solver
from project_paths import load_available


def estimate_panels(specs, solver, bootstrap_b, seed, m):
    rows = []
    for app in load_available((spec.entity, spec.load) for spec in specs):
        row = estimate(app, solver, bootstrap_b, seed, m)
        rows.append(row)
        print(f"{app.entity}: N={row['N']}, T={row['T']}, M={row['M']}, "
              f"share={row['MGP_share']:.4f}, UCL={row['UCL95']:.4f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--solver", default="auto")
    args = parser.parse_args()
    solver = choose_solver(args.solver)
    innovation = estimate_panels(INNOVATION, solver, args.bootstrap_b, args.seed, args.M)
    rows = estimate_panels(PANELS, solver, args.bootstrap_b, args.seed, args.M)
    mutual = RUN_OUTPUT_CACHE / "mutual_fund_applications_summary.pkl"
    if mutual.is_file():
        rows.extend(pd.read_pickle(mutual).to_dict("records"))
    rows.extend(innovation)
    pd.DataFrame(rows).to_pickle(RUN_OUTPUT_CACHE / "additional_applications_summary.pkl")


if __name__ == "__main__":
    main()
