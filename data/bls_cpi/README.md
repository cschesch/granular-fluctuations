# BLS CPI weights (relative importance)

December relative-importance weights for every CPI component, CPI-U and CPI-W,
parsed from the official BLS files into a tidy panel and keyed to BLS item codes
/ series ids. Built by [`code/applications/bls_cpi/build_cpi_weights.py`](../../code/applications/bls_cpi/build_cpi_weights.py);
re-run that script to regenerate the CSVs.

## Sources (raw, in `relative_importance_raw/`)
- `ri-archive-1990-1999.zip`, `...-2000-2009.zip`, `...-2010-2019.zip` — yearly
  relative-importance tables. 1990–1996 are flat tables that carry the **old
  (pre-1998) item-code** column; 1997–2019 are the indented-name layout on the
  **current (1998) item structure**.
- `2020.xlsx` … `2025.xlsx`, `cpi-relative-importance-current.xlsx` — workbook
  layout with an explicit indent-level column.
- `cu.item`, `cu.series` — BLS item / series dictionaries from
  `download.bls.gov/pub/time.series/cu/` (the authoritative name↔code map).
- `cu.data.0.Current.txt` — CPI index values, used to confirm which item codes
  actually have a price series.

## Outputs
The CPI builders write the files below to `code/.cache/derived/bls_cpi/`.

- **`cpi_relative_importance.csv`** — tidy long panel, one row per
  (`year`, `section`, item). Columns: `structure_era` (pre/post-1998), `section`
  (`expenditure` = additive partition; `special_aggregate` = overlapping
  aggregates), `level` / `parent_name` / `is_leaf` (hierarchy from indentation),
  `item_name_raw`, `item_name` (normalised), `item_code` (only where the source
  prints it, i.e. 1990–1996), `cpi_u`, `cpi_w`. Validation: the post-1998
  `expenditure` **leaves sum to 100.00** every year (204 leaves).
- **`cpi_weights_keyed.csv`** — post-1998 rows joined to current item codes and
  series ids (`series_id_nsa` = `CUUR0000`+code, `series_id_sa` = `CUSR0000`+code),
  restricted to codes with a price series. Filter to
  `section=='expenditure' & is_leaf` for the additive unit set.
- **`cpi_item_code_map.csv`** — the name→code crosswalk used (from `cu.item`),
  with `in_price_file` and `ambiguous` flags, for auditing.

## Price history (`history/`, `cpi_price_long.parquet`)
`cu.data.0.Current.txt` only holds 1997+. The full historical US-city-average
series live in the BLS group files saved under `history/` (`cu.data.1.AllItems`,
`cu.data.2.Summaries`, `cu.data.11`–`18` major groups, `cu.data.20` special
aggregates). [`code/applications/bls_cpi/build_cpi_prices.py`](../../code/applications/bls_cpi/build_cpi_prices.py) unions
them into **`cpi_price_long.parquet`** — tidy (`item_code`, `date`, `value`) for
the **CUSR0000 (US, seasonally adjusted)** monthly series. The application
panels end in September 2025. With December 1967 weights and a January 1968
start, the baseline contains **32 categories and 693 months**, including
synthetic residuals. Categories without complete adjusted histories are
absorbed into broader/residual components; no additional seasonal adjustment
is applied. Earlier NSA coverage counts below describe the pre-switch inventory.

History trades off against granularity (weights from the latest year):

| panel starts | finest = keyed expenditure leaves | CPI-U covered |
|---|---|---|
| 1997 / 2000 | 176 units | 96% |
| 1990 | 89 units | 72% |
| 1980 | 74 units | 43% |
| 1970 | 33 units | 32% |

Coarse end: the **8 major groups** sum to 100% and reach back to 1993 (all
eight), 1967 (six of them; Recreation and Education & communication only begin
1993). So: ~28 years monthly at fine resolution, or ~60 years at major-group
resolution.

## Coverage notes
- Keying is by **normalised item name** against `cu.item`. The 204-leaf
  expenditure structure is stable, but BLS **renames** items over time, so
  matching older-year names against current `cu.item` names loses some coverage:
  keyed expenditure-leaf CPI-U sum runs ~88% (1997) rising to **~98% (2025)**.
  For an MGP cross-section, anchor the size weights to a recent year.
- The unmatched tail is almost entirely `Unsampled …` residual nodes that have
  **no separate price series** by construction (plus one discontinued item,
  "Care of invalids and elderly at home"). Treat this residual mass with a
  synthetic residual node, as the CES loader does.
- A handful of confirmed BLS renames are bridged by an audited `NAME_ALIASES`
  map in the build script (e.g. SEHB01 "Housing at school, excluding board" →
  "Lodging while at school").
- `series_id_nsa` (CUUR) exists for **all** keyed leaf codes; `series_id_sa`
  (CUSR) exists for fewer of the detailed leaves. The estimation pipeline uses
  SA indexes and coarsens incomplete branches to preserve expenditure coverage.

`cu.item` / `cu.series` were retrieved via the macOS `URLSession` stack
(`download.bls.gov` blocks curl/python user-agents but serves the WebKit TLS
fingerprint).

## Weight series (`cpi_weight_series_1967_onward.*`)
[`code/applications/bls_cpi/build_cpi_weight_series.py`](../../code/applications/bls_cpi/build_cpi_weight_series.py) stitches
the historical relative-importance workbook, the 1987-1989 bridge archive, and
the 1990+ parsed RI tables into annual December weights from 1967 onward.

- `cpi_weight_series_1967_onward.csv`: long audit table, 18,400 rows across
  1967-2025.
- `cpi_weight_series_1967_onward_wide.parquet`: wide CPI-U matrix, 59 years by
  3,318 source-slot/category keys.
- `item_key` is collision-resistant: BLS-coded rows include section/hierarchy
  context; no-code historical rows use the workbook sheet and row slot. Use
  `item_code` separately when you want to join coded post-1987 categories.
- Some 1977/1982/1983 rows are CPI-W-only historical shelter/homeownership rows;
  this is why a small number of long-table rows have missing `cpi_u`.

## Paper panel
[`code/applications/bls_cpi/build_cpi_panels.py`](../../code/applications/bls_cpi/build_cpi_panels.py) builds fixed-weight
panel files under `code/.cache/derived/bls_cpi/panels/`. The paper uses
`cpi_1967_leaves.csv` and `cpi_1967_inflation.parquet`, with December 1967
CPI-U weights and monthly observations from January 1968 through September 2025.
