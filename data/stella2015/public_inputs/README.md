# Stella public inputs

These bundled raw inputs reconstruct the public part of Stella's calibration:

- `bea_1997_summary/`: original workbooks from the
  [BEA 1997 benchmark](https://www.bea.gov/industry/historical-benchmark-input-output-tables).
- `bls_qcew/`: US annual totals from the
  [BLS QCEW archives](https://www.bls.gov/cew/downloadable-data-files.htm).
- `naics_concordances/` and `1987_sic_to_1997_naics.xls`: official Census
  industry concordances.
- `bls_ces_supplement/`: BLS CES railroad employment, which is excluded from
  QCEW coverage.
- `sax12/`: published Stata seasonal-adjustment code and a source-generated
  specification used to verify our translation.
- `omegav03src.tar.gz`: preserved Census X-12 v0.3 source, used by
  `code/applications/stella/build_stella_x12.py`.

Source URLs and hashes remain in the accompanying provenance files.
`python -m applications.stella.prepare_public`, run from `code/`, reconstructs
employment denominators and the deposited GDP vector in the cache.

The original package names BLS as the source of sectoral employment but does
not identify the series or data vintage. This repository reconstructs those
inputs from public sources. The employment-building code records the industry
mappings and three denominator overrides.
