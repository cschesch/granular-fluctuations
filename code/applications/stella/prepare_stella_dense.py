"""Build Stella's sector-based network on the prepared customer-network panel."""

import json
import shutil

import numpy as np
import pandas as pd

from applications.stella.paths import CALIBRATION, DENSE
from applications.stella.prepare_stella_calibration import (
    OFFICIAL_IO, OFFICIAL_SURPLUS, SMALL, official_stella_accounts,
)
from applications.stella.stella_provenance import file_fingerprints
from mgp_core import ROOT


def main():
    mapping = pd.read_csv(CALIBRATION / "firm_mapping.csv", dtype={"gvkey": str, "stella_sid": str})
    ratios = pd.read_csv(CALIBRATION / "firm_employment_ratios.csv", dtype={"gvkey": str}).set_index("gvkey")
    ratios = ratios.reindex(mapping.gvkey)
    io, output, _ = official_stella_accounts()
    sectors = mapping.stella_sid
    # This is matrix_construction_Icalibration.m, including its self-links.
    shares = (io.loc[sectors, sectors].to_numpy(float)
              / output.loc[sectors].to_numpy(float)[None, :]
              * ratios.ratio_emp.to_numpy(float)[:, None])
    DENSE.mkdir(parents=True, exist_ok=True)
    common = ("firm_mapping.csv", "firm_employment_ratios.csv", "alpha.csv",
              "matrix_capital.csv", "nominal_saleq_levels.csv", "sales_growth_sax12.csv",
              "sales_preparation_sax12_diagnostics.json")
    for name in common:
        shutil.copyfile(CALIBRATION / name, DENSE / name)
    np.savetxt(DENSE / "matrix_io.csv", np.where(shares == 0, SMALL, shares), delimiter=",", fmt="%.17g")
    diagnostic = json.loads((CALIBRATION / "diagnostics.json").read_text())
    diagnostic.update({
        "network_calibration": "dense",
        "calibration": "Deposited matrix_construction_Icalibration.m: sector links scaled by supplier employment share.",
        "positive_calibrated_edges": int((shares > 0).sum()),
        "baseline_firms_with_clients_positive_calibrated_share": int((shares.sum(axis=1) > 0).sum()),
        "baseline_firms_with_suppliers_positive_calibrated_share": int((shares.sum(axis=0) > 0).sum()),
        "baseline_mean_total_input_share": float(shares.sum(axis=0).mean()),
        "baseline_median_total_input_share": float(np.median(shares.sum(axis=0))),
        "max_buyer_intermediate_share": float(shares.sum(axis=0).max()),
        "median_supplier_degree": float(np.median((shares > 0).sum(axis=0))),
        "median_client_degree": float(np.median((shares > 0).sum(axis=1))),
        "input_file_fingerprints": file_fingerprints(
            [CALIBRATION / name for name in common] + [OFFICIAL_IO, OFFICIAL_SURPLUS], ROOT),
    })
    (DENSE / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
