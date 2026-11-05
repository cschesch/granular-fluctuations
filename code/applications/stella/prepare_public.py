"""Rebuild Stella's public employment denominators and deposited GDP series."""

import argparse
import re

import numpy as np
import pandas as pd

from applications.stella.prepare_stella_employment import PUBLIC, PUBLIC_CACHE, build_sectoral
from project_paths import ROOT, available, upstream


GDP_GROWTH = PUBLIC_CACHE / "deposited_real_gdp_growth.csv"
PUBLIC_INPUTS = (
    ROOT / "data/stella2015/source.zip",
    ROOT / "data/stella2015/source.json",
    ROOT / "data/stella2015/bea_summary_to_stella_sid.csv",
    PUBLIC / "bea_1997_summary/NAICS-IO.xls",
    *(PUBLIC / "naics_concordances" / name for name in (
        "1997_NAICS_to_2002_NAICS.xls", "2002_to_2007_NAICS.xls", "2007_to_2012_NAICS.xls",
    )),
    *(PUBLIC / "bls_qcew" / f"{year}_US000_annual.csv" for year in range(1990, 2012)),
    *(PUBLIC / "bls_ces_supplement" / f"CEU4348200001_{start}_{end}.json"
      for start, end in ((1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2011))),
)


def build_gdp() -> pd.Series:
    """Recover the identical GDP literals in the author's original MATLAB script."""
    source = upstream("stella2015") / "statistical analysis/summary_results_statistical.m"
    vectors = re.findall(r"real_gdp_growth\s*=\s*\[([^\]]+)\]\s*'\s*;", source.read_text())
    arrays = [np.fromstring(vector.replace("\n", " ").replace("\r", " "), sep=" ") for vector in vectors]
    if len(arrays) != 3 or arrays[0].size != 199:
        raise ValueError("Unexpected deposited GDP literal count or length")
    for array in arrays[1:]:
        np.testing.assert_array_equal(array, arrays[0])
    series = pd.Series(
        arrays[0], index=pd.period_range("1962Q2", "2011Q4", freq="Q"),
        name="real_gdp_log_growth_deposited",
    )
    PUBLIC_CACHE.mkdir(parents=True, exist_ok=True)
    series.to_csv(GDP_GROWTH, index_label="quarter")
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    if args.available_only and not available("Stella public inputs", *PUBLIC_INPUTS):
        return
    sectoral, _ = build_sectoral()
    gdp = build_gdp()
    print(f"Stella public inputs rebuilt: {len(sectoral)} employment observations, {len(gdp)} GDP quarters")


if __name__ == "__main__":
    main()
