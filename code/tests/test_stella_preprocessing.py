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
from applications.stella.paths import CALIBRATION, UNBALANCED
from applications.stella.prepare_stella_sales import locate_x13, parse_d11, x13_spec


TEST_GROUP = "integration"
TEST_CACHE_INPUTS = (
    ".cache/tools/x12a_v03/source/x12a", ".cache/tools/x12a_v03/build_manifest.json",
    ".cache/firm_networks/stella_revtq/nominal_saleq_sax12_adjusted.csv",
    ".cache/firm_networks/stella_revtq_unbalanced/*.csv",
    ".cache/firm_networks/stella_revtq_unbalanced/diagnostics.json",
)

from tests.support import require_files
from unittest import SkipTest


def test_source_seasonal_specification() -> None:
    # Independent numerical check against an SPC emitted by the unmodified
    # published Stata sax12.ado, including its sliding-span/history diagnostics.
    public = ROOT / "data/stella2015/public_inputs/sax12"
    require_files(*(public / name for name in ("provenance.json", "sax12.ado", "synthetic_spec_generated_by_stata.spc", "synthetic_revenues.dat")))
    try:
        executable = locate_x13()
    except FileNotFoundError as error:
        raise SkipTest(str(error)) from error
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
            subprocess.run([str(executable), name], cwd=tmp, capture_output=True, check=True)
            results.append(parse_d11(path.with_suffix(".d11"), len(values)))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9)



def test_private_balanced_and_unbalanced_panels():
    require_files(CALIBRATION / "nominal_saleq_sax12_adjusted.csv",
                  *(UNBALANCED / name for name in ("nominal_saleq_sax12_adjusted.csv", "sales_growth_sax12.csv", "top500_index.csv", "diagnostics.json")))
    # Private balanced and unbalanced preparations must share the same adjusted
    # levels, while demeaning growth over different estimation windows.
    level = pd.read_csv(CALIBRATION / "nominal_saleq_sax12_adjusted.csv", index_col=0)
    full = pd.read_csv(UNBALANCED / "nominal_saleq_sax12_adjusted.csv", index_col=0)
    np.testing.assert_allclose(full.loc[level.index, level.columns], level, rtol=0, atol=1e-9)
    y = pd.read_csv(UNBALANCED / "sales_growth_sax12.csv", index_col=0)
    index = pd.read_csv(UNBALANCED / "top500_index.csv", index_col=0)
    unbalanced = json.loads((UNBALANCED / "diagnostics.json").read_text())
    assert y.shape == (199, unbalanced["N"]) and index.shape == (unbalanced["N"], 199)
    assert y.index[0] == "1962Q2" and y.index[-1] == "2011Q4"
    assert index.isin([0, 1]).all().all() and index.sum(axis=0).max() <= 500
    assert np.max(abs(y.mean(axis=0))) < 1e-14
    for col in y:
        observed = np.flatnonzero(y[col].notna())
        assert len(observed) >= 40 and np.all(np.diff(observed) == 1)
