# BLS consumer prices

Raw CPI-U price indexes and December expenditure weights from the
[BLS CPI database](https://www.bls.gov/cpi/data.htm).

- `cu.data.0.Current.txt` and `history/` contain the price series; `cu.item`
  supplies the item-code dictionary.
- `relative_importance_raw/` contains the historical weights workbook,
  1987–1989 bridge text files, 1990–2019 archives and 2020–2025 workbooks.

`make -C code applications` runs the four builders in
`code/applications/bls_cpi/`. They combine the source histories, match item
codes, and construct an additive hierarchy using residual components where
necessary. The paper panel uses December 1967 weights and seasonally adjusted
monthly prices from January 1968 through September 2025.

All parsed tables and panels are written to `code/.cache/derived/bls_cpi/`.
