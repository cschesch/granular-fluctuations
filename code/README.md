# Granularity in High Dimensions

`make all` builds the paper exhibits in `../output/`. Use `make PYTHON=python3 all`
with an activated environment. `make test` checks the estimators and loaders.

`python paper_applications.py --plan` lists the commands. Run one section with
`make applications`, `make variations`, `make simulations`, or `make empirical`.
Missing source data leave blank estimates and are logged as skipped. Intermediate
data and resumable checkpoints live under `.cache/`. Source-specific analyses
live in `applications/`, with shared estimators and exhibit builders in `code/`.

Each application folder contains its source-specific loader or analysis.
`applications/common.py` holds shared panel and estimation routines; `tests/`
contains the checks. Raw inputs live in the corresponding `../data/` source
folders. Original publication packages stay intact; preprocessing writes only
to `.cache/`.

Run `make syncdraft` from the complete research checkout to refresh Overleaf's
exhibits.
