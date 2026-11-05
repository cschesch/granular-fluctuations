#!/usr/bin/env python3
"""Estimate the source-specific panels used in the paper."""

import argparse
from functools import partial

import pandas as pd

from mgp_core import ANALYSIS_BOOTSTRAP_DRAWS, RUN_OUTPUT_CACHE, choose_solver
from project_paths import load_available
from applications.common import estimate
from applications.debt.panel import load_private_debt, load_public_debt
from applications.fdic.panel import load_banks
from applications.sdc.panel import load_bond_issues
from applications.treasury.panel import load_tic
from applications.granular_cities.panel import load_cities
from applications.bea_regional.panel import load_counties, load_states
from applications.patent_firms.panel import load_patents
from applications.epa.panel import load_campd
from applications.carbon_emissions.panel import load_country_co2
from applications.owid_energy.panel import load_oil
from applications.comtrade.panel import load_trade
from applications.maddison.panel import load_countries
from applications.french.panel import load_country_equity_returns


APPLICATIONS = (
    ("Countries", load_countries),
    ("Country equity markets", load_country_equity_returns),
    ("Banks", load_banks),
    ("Metropolitan areas", load_cities),
    ("Exported products", partial(load_trade, "exports", "hs2")),
    ("Imported products", partial(load_trade, "imports", "hs2")),
    ("Export destinations", partial(load_trade, "exports", "partners")),
    ("Import origins", partial(load_trade, "imports", "partners")),
    ("US states", load_states),
    ("US counties", load_counties),
    ("Power plants", partial(load_campd, "Gross Load (MWh)", "Power plants", "Gross-load growth")),
    ("Power-plant emissions", partial(load_campd, "CO2 Mass (short tons)", "Power-plant emissions", "CO2-emissions growth")),
    ("Patent assignees", load_patents),
    ("Oil-producing countries", load_oil),
    ("Public debt by country", load_public_debt),
    ("Private nonfinancial debt by country", load_private_debt),
    ("Foreign Treasury holders", load_tic),
    ("Country CO2 emissions", load_country_co2),
    ("US corporate bond issuers", load_bond_issues),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int)
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--solver", default="auto")
    args = parser.parse_args()
    solver = choose_solver(args.solver)
    rows = []
    for app in load_available(APPLICATIONS):
        row = estimate(app, solver, args.bootstrap_b, args.seed, args.M)
        rows.append(row)
        print(f"{app.entity}: N={row['N']}, T={row['T']}, M={row['M']}, "
              f"share={row['MGP_share']:.4f}, UCL={row['UCL95']:.4f}", flush=True)
    for name in ("mutual_fund_applications", "innovation_applications"):
        path = RUN_OUTPUT_CACHE / f"{name}_summary.pkl"
        if path.is_file():
            rows.extend(pd.read_pickle(path).to_dict("records"))
    pd.DataFrame(rows).to_pickle(RUN_OUTPUT_CACHE / "additional_applications_summary.pkl")


if __name__ == "__main__":
    main()
