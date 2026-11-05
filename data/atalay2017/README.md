# Atalay (2017)

`replication_elasticities.zip` is the original publication package from
[the author](https://enghinatalay.github.io/), corresponding to
[AEA deposit E114149V1](https://doi.org/10.3886/E114149V1). Our preparation and
analysis code is in `code/applications/atalay/`.

## Run

Stata with `hprescott` prepares the source data; MATLAB with Statistics and
Machine Learning Toolbox recovers shocks and fits Atalay's one-factor model.
Set `STATA` and `MATLAB` if their executables are not found automatically.
Completed preparation stages can be reused without these programs.

Run these commands from `code/`:

```sh
python -m applications.atalay.prepare
python -m applications.atalay.run_atalay_epsilon
python -m applications.atalay.run_atalay_elasticity_profile
python -m applications.atalay.run_atalay_elasticity_surface
```

To prepare only one part, use the corresponding command:

```sh
python -m applications.atalay.prepare --section benchmark
python -m applications.atalay.prepare --section profile
python -m applications.atalay.prepare --section surface
```

The two plotting commands accept `--from-cache` to redraw saved estimates.

## Three analyses

- **Benchmark:** the original direct-recursion calibration, used in the main
  applications table and shared application figures.
- **Profiles:** vary material-input or capital/labor–materials substitution,
  holding the other elasticities at unity. Plot both direct-recursion and
  Kalman-filter estimates, using Atalay's original stability screens.
- **Calibration surface:** compare Atalay's one-factor share with our M=10
  ceiling on the same Kalman shocks. The three final-consumption elasticities
  are 2/3, 1 and 4/3; the package represents unity by 0.999. Each row uses a
  39×51 grid in material-input and capital/labor–materials substitution.

The surface uses the deposited `factoran_` routine with one factor,
`Nobs=T-1`, `Delta=1e-6`, tolerances `1e-14` and `MaxIter=5000`, used for the
paper's calibration comparison. Optimizer-limit flags remain in the saved results;
flagged estimates are retained and reported by the runner.

## Files

The original ZIP is unchanged. Its working copy and all generated filter
inputs live in `code/.cache/derived/atalay2017/`, separately for the benchmark,
profiles and surface. The surface stores one file of shocks and propagation
matrices for each final-consumption elasticity, plus three one-factor fit CSVs.
Estimation writes summaries to
`code/.cache/run_outputs/` and these figures to `output/`:

- `atalay_elasticity_profile.pdf`
- `atalay_surface_comparison.pdf`

Preparation records inputs, MATLAB code and settings separately for each stage
under `code/.cache/derived/atalay2017/stages/`. Matching checkpoints can resume;
plotting and README edits do not invalidate them. Older grids without these
records remain untouched until preparation runs again, when they are preserved
under `superseded/` and rebuilt. Changed inputs or computation trigger the same
process; surface changes also invalidate their one-factor fits. Python calculation
changes require bumping the affected entry in `PREPARATION_VERSIONS` in `prepare.py`.
Generated inputs, estimates and checkpoints are local caches and are not bundled
in the public repository.
