# Stella (2015)

`source.zip` contains the original author release, identified in `source.json`.
`public_inputs/` contains BEA, Census, and BLS inputs and their provenance.

The full reconstruction also requires the Compustat quarterly revenue extract
pinned in `paper_inputs.json`, plus annual and customer files pinned under
`data/compustat/`. Licensed observations are not included. The analysis uses
quarterly `revtq`; it does not fill missing revenue with `saleq`.

## Requirements

Octave and Dynare's `gensys.m` solve the model and validate its constructors.
Set `OCTAVE` and `GENSYS_DIR` when they are not found automatically.

Seasonal adjustment uses X-12. Build the bundled source with GNU Fortran,
or set `X12PATH` to an existing executable. From the repository root:

```sh
cd code
python -m applications.stella.build_stella_x12
python -m applications.stella.prepare_public
```

Public preparation reconstructs employment denominators and the deposited GDP
series in `code/.cache/derived/stella/`. The full application runs through
`run_replication.py`; it reports unavailable licensed inputs in the run report.

See [public inputs](public_inputs/README.md) for source conventions and
[WRDS](../README_WRDS.md) for obtaining and selecting the additional inputs.

## Sector-based network

The alternative calibration follows `matrix_construction_Icalibration.m`.
Every firm can buy from firms in supplying sectors identified by the BEA
accounts. A supplier's weight equals its firm-to-sector employment ratio
times the buyer sector's input expenditure share. The calibration retains
the source's self-links and its three special employment-share overrides.

`applications.stella.prepare_stella_dense` builds this matrix on the same
firm panel as the customer-network application, reusing its sales growth,
fixed initial sales weights, employment ratios, and capital parameters.
The original MATLAB constructor validates the resulting matrices under
Octave. The dynamic solver and grouped bound are then rerun for the new
network. Separate inputs are written to
`code/.cache/firm_networks/stella_revtq_dense/` and results use the prefix
`stella_revtq_dense_firm_network`. The `Stella dense network` stage in
`code/run_replication.py` runs these steps after the customer calibration.

For each network, the one-PC comparison estimates a single principal component
from demeaned shocks divided by their firm-specific sample standard deviations.
It regresses the shocks in their original units on that component and propagates
the full empirical covariance of the residuals. The denominator is the same
model-implied aggregate variance used for the grouped ceiling. The retained
comparison is recorded in each network's summary alongside the grouped ceiling.

Generated diagnostics and summaries stay in `code/.cache/`. Distinct prior
Stella results are preserved under `archive/stella_results/`, with their source
paths and content hashes.
