"""Estimate patent and drug-approval applications."""

import pandas as pd

from mgp_core import ANALYSIS_BOOTSTRAP_DRAWS, ROOT, choose_solver
from project_paths import load_available
from applications.common import estimate
from applications.kpss.panel import patent_value_by_field, patent_count_by_field, patent_value_by_firm
from applications.fda.panel import novel_drugs_by_sponsor


def main() -> None:
    solver = choose_solver("auto")
    rows = []
    sources = (
        ("Patent-value firms", patent_value_by_firm),
        ("Patent-count fields", patent_count_by_field),
        ("Patent technology fields", patent_value_by_field),
        ("Novel-drug sponsors", novel_drugs_by_sponsor),
    )
    for app in load_available(sources):
        row = estimate(app, solver, bootstrap_b=ANALYSIS_BOOTSTRAP_DRAWS, seed=20260907)
        rows.append(row)
        print(f"{app.entity}: N={row['N']}, T={row['T']}, M={row['M']}, "
              f"share={row['MGP_share']:.4f}, UCL={row['UCL95']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    cache = ROOT / "code/.cache/run_outputs/innovation_applications_summary.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache)


if __name__ == "__main__":
    main()
