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
