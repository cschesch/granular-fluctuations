# How Granular Are Aggregate Fluctuations?

Replication code and data for Constantin Schesch's **“How Granular Are Aggregate
Fluctuations?”**, including empirical applications, simulations, figures and tables.

- `code/`: estimation, data preparation, simulations and tests.
- `data/`: bundled public inputs and instructions for additional data.
- `output/`: generated figures, tables and run reports.

Some applications require separately obtained WRDS extracts or Forbes snapshots.
See [Data](data/README.md).

## Install

Use Python 3.11 and GNU Make. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

Some network applications also require MATLAB, Stata, Octave or X-12. Their
requirements are listed in [Data](data/README.md).

## Run

```sh
make -C code PYTHON=python all
```

The default runs everything available. Missing inputs leave unavailable entries
in tables and labelled placeholder figures. **Check `output/run_report.json`: a
run with skipped analyses is a partial replication.** To fail when a requested
analysis is unavailable, use `make -C code PYTHON=python strict`.

List commands without running them, or try a short synthetic check:

```sh
python code/run_replication.py --plan
make -C code PYTHON=python smoke
```

See [Code](code/README.md) for sections, tests and verification;
[figures and tables](code/EXHIBITS.md) for the output map; and
[tested environment and timings](code/REPRODUCIBILITY.md) for measured checks.

## Citation and reuse

Please cite Constantin Schesch (2026), *How Granular Are Aggregate Fluctuations?*
Repository citation metadata are in [CITATION.cff](CITATION.cff).

A license for the project's own code has not yet been selected. Third-party
code and data retain their original terms; inclusion here does not change them.
