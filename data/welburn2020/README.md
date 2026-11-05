# Welburn et al. (2020)

The application uses Compustat annual, quarterly, and customer data, together
with WRDS supply-chain identifiers and subsidiary-name matches. These licensed
observations are not included.

`paper_inputs.json` records the exact supplementary WRDS files used for the
paper. The Compustat files used for the paper are listed in
`data/compustat/paper_inputs.json`. See [WRDS](../README_WRDS.md) for input
selection.

Additional WRDS query code is in [inspect_wrds_sources.py](inspect_wrds_sources.py).
It requires an existing WRDS connection and is not a standalone download command.

Analysis and matching code are in `code/applications/welburn/`. Prepared links,
networks, and estimates are written under `code/.cache/`.
