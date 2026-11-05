#!/usr/bin/env python3
"""Source-SPC equivalence, missing-rank selection, and reconstructed-input checks."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT
from applications.stella.prepare_stella_calibration import released_screen_panel
from applications.stella.prepare_stella_sales import locate_x13, parse_d11, x13_spec
from applications.stella.prepare_stella_unbalanced import top500_membership
from applications.stella.audit_stella_growth_moments import growth_accounting


def main() -> None:
    # Real source behavior: missing lag sales cannot occupy a top-500 rank.
    dates = pd.period_range("1962Q1", periods=41, freq="Q")
    synthetic = pd.DataFrame({
        "gvkey": np.repeat([f"{i:06d}" for i in range(1, 502)], len(dates)),
        "quarter": np.tile(dates, 501),
        "saleq": np.repeat(np.arange(1, 502), len(dates)),
    })
    strict, strict_n = released_screen_panel(synthetic)
    legacy, legacy_n = released_screen_panel(synthetic, strict_lag_ranks=False)
    assert strict_n == 500 and legacy_n == 501
    assert "000001" not in set(strict["gvkey"]) and "000001" in set(legacy["gvkey"])
    membership = top500_membership(synthetic)
    assert set(membership["gvkey"]) == set(strict["gvkey"])
    assert dates[0] not in set(membership["quarter"])
    account = growth_accounting(pd.DataFrame([[100., 110., np.nan], [np.nan, 20., 25.],
                                             [50., 50., 50.]]))
    np.testing.assert_allclose(account.iloc[0], [20., 100*10/150, 100*20/150, 0.])
    np.testing.assert_allclose(account.iloc[1], [-100*105/180, 100*5/180, 0., -100*110/180])

    # Independent numerical check against an SPC emitted by the unmodified
    # published Stata sax12.ado, including its sliding-span/history diagnostics.
    public = ROOT / "data/stella2015/public_inputs/sax12"
    provenance = json.loads((public / "provenance.json").read_text())
    assert hashlib.sha256((public / "sax12.ado").read_bytes()).hexdigest() == provenance["sha256"]
    direct = (public / "synthetic_spec_generated_by_stata.spc").read_text()
    assert not re.search(r"\b(?:automdl|arima|pickmdl)\s*\{", direct)
    source_data = (public / "synthetic_revenues.dat").resolve()
    direct = re.sub(r'file\s*=\s*"[^"]+"', 'file = "data.dat"', direct)
    values = np.loadtxt(source_data)[:, 2]
    translated = x13_spec(values, "synthetic", pd.Period("1990Q1"), "sax12")
    assert "automdl" not in translated and "types=(all)" in translated and "method=addone" in translated
    results = []
    with tempfile.TemporaryDirectory(prefix="stella_sax12_test_") as tmp:
        shutil.copyfile(source_data, Path(tmp)/"data.dat")
        for name, spec in [("source", direct), ("translated", translated)]:
            path = Path(tmp) / name
            path.with_suffix(".spc").write_text(spec)
            subprocess.run([str(locate_x13()), name], cwd=tmp, capture_output=True, check=True)
            results.append(parse_d11(path.with_suffix(".d11"), len(values)))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9)

    # Private balanced and unbalanced preparations must share the same adjusted
    # levels, while demeaning growth over different estimation windows.
    base = ROOT / "code/.cache/firm_networks"
    level = pd.read_csv(base / "stella_literal/nominal_saleq_sax12_adjusted.csv", index_col=0)
    full = pd.read_csv(base / "stella_unbalanced/nominal_saleq_sax12_adjusted.csv", index_col=0)
    np.testing.assert_allclose(full.loc[level.index, level.columns], level, rtol=0, atol=1e-9)
    y = pd.read_csv(base / "stella_unbalanced/sales_growth_sax12.csv", index_col=0)
    index = pd.read_csv(base / "stella_unbalanced/top500_index.csv", index_col=0)
    unbalanced = json.loads((base / "stella_unbalanced/diagnostics.json").read_text())
    assert y.shape == (199, unbalanced["N"]) and index.shape == (unbalanced["N"], 199)
    assert y.index[0] == "1962Q2" and y.index[-1] == "2011Q4"
    assert index.isin([0, 1]).all().all() and index.sum(axis=0).max() <= 500
    assert np.max(abs(y.mean(axis=0))) < 1e-14
    for col in y:
        observed = np.flatnonzero(y[col].notna())
        assert len(observed) >= 40 and np.all(np.diff(observed) == 1)
    print("Stella preprocessing: direct Stata-SPC equivalence, missing-rank screen and private-panel checks passed.")


if __name__ == "__main__":
    main()
