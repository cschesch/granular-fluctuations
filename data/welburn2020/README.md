# Welburn et al. (2020)

The application uses Compustat annual, quarterly, and customer data, together
with WRDS supply-chain identifiers and subsidiary-name matches. These licensed
observations are not included.

`paper_inputs.json` records the exact supplementary WRDS files used for the
paper. Compustat pins are in `data/compustat/paper_inputs.json`.
See [WRDS](../README_WRDS.md) for input selection. `inspect_wrds_sources.py`
provides `download_existing(connection, name, sql, params)` for use with an
already-open programmatic WRDS connection.

Analysis and matching code are in `code/applications/welburn/`. Prepared links,
networks, and estimates are written under `code/.cache/`.
