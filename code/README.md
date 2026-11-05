# Code

See the [main README](../README.md) for installation and the full replication.
Run commands below from the repository root with the environment activated.

## Sections

| Command | Analyses |
| --- | --- |
| `make -C code PYTHON=python applications` | Empirical applications, application figures and tables |
| `make -C code PYTHON=python variations` | Volatility and Atalay elasticity exercises |
| `make -C code PYTHON=python simulations` | Inference and GIV Monte Carlo experiments |
| `make -C code PYTHON=python empirical` | Grouping comparison and Compustat appendix results |

`empirical` refreshes the quarterly Compustat summary and subsampling draws first
when the raw data are present, so it cannot silently use an older export. The `all`
target runs sections in the required order. To inspect a section:

```sh
python code/run_replication.py simulations --plan
```

Each run records completed, skipped and failed analyses, reasons, selected input
files, command logs and timings in `output/run_report.json`. Use `--strict` to
fail on unavailable analyses. Missing table entries are dashes or empty cells,
not estimated zeros. A completed run has not necessarily reproduced reference
numbers; check those separately below.

Paper WRDS filenames are pinned beside each source. Explicitly select newer
local exports with `python code/run_replication.py --inputs latest`; their paths
and checksums are recorded. Such a run is not a replication of the pinned inputs.

## Modules

`run_replication.py` runs the analyses. Source preparation and analysis live in
`applications/`; `applications/registry.py` lists independent panel applications.
`applications/common.py` and `applications/networks.py` contain shared estimators.

`mgp_core.py` provides panel preparation, grouping and estimation helpers;
`gradient_bootstrap.py` provides inference. `fast_mgp.py` and `mgp_kernels.py`
implement batched numerical solvers. `network_solver.py` wraps the shared Octave
model solver. Source-specific MATLAB exporters sit beside their applications.

`run_inference_mc.py` and `run_grouped_giv_mc.py` run the simulations.
`plot_*.py` and `build_*.py` produce figures and tables; `plot_style.py` contains
plotting helpers. See [the output map](EXHIBITS.md).

## Tests

```sh
make -C code PYTHON=python test              # all discovered test modules
make -C code PYTHON=python test-fast         # no licensed data or external tools
make -C code PYTHON=python test-numerical    # numerical solver cross-checks
make -C code PYTHON=python test-integration  # source-specific checks
make -C code PYTHON=python smoke            # small synthetic pipeline
```

Tests are discovered from `tests/test_*.py` and run in temporary copies. Missing
integration inputs produce explicit skips. `python code/run_tests.py --list`
lists the groups. Reports are saved under `code/.cache/`. Smoke settings are
deliberately smaller than the paper settings and never replace paper estimates.

## Three separate checks

- **File integrity:** `make -C code PYTHON=python verify` checks the bundled
  files listed in `SHA256SUMS`. Additional user-supplied data are allowed.
- **Run completeness:** inspect `output/run_report.json`; `make -C code
  PYTHON=python check` checks that the required figure/table files exist and
  reports completeness. `strict` fails if requested analyses are unavailable.
- **Numerical agreement:** `make -C code PYTHON=python check-results` compares
  selected public-data estimates with fixed references and stated tolerances.
  `python code/check_results.py --strict` also fails if any reference result is
  unavailable. References are in `reference_results.json`, separate from new runs.

## Generated files

Figures, tables and run reports go to `output/`. Prepared data, clean panels,
estimates, logs and checkpoints go to `code/.cache/`; writers create directories
as needed. Both locations are ignored by Git. Clean-panel keys use complete
input-file hashes and explicit cleaning versions. Older partial-hash caches
remain on disk but are rebuilt once under the corrected keys.
