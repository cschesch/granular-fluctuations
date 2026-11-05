# Tested environment and measured checks

Checks below were measured on 5 November 2026, on macOS 26.7.1 arm64 with
Python 3.11.14. A fresh virtual environment installed `requirements.txt`
successfully and passed `pip check`. Exact installed Python packages are listed
in [tested_environment.txt](tested_environment.txt). The checks used one
BLAS/OpenMP thread and two Numba threads. CI repeats the clean installation and
fast/numerical checks on Ubuntu with Python 3.11.

| Measured check | Settings | Elapsed time | Peak process RSS |
| --- | --- | ---: | ---: |
| Fast suite | 14 initial modules, 45 cases; no licensed data | 150 s | 423 MiB |
| Added input/status checks | 2 modules, 14 cases | 1.8 s | 232 MiB |
| Release/numerical-reference checks | 1 module, 7 cases | 0.7 s | 101 MiB |
| Numerical suite | 20 cases; optional MOSEK unavailable | 36 s | 1,013 MiB |
| Source integration checks | 6 modules passed; 2 Atalay grid modules skipped | 24 s | 364 MiB |
| Synthetic smoke | 4 inference replications, 9 bootstrap draws; 2 GIV datasets and all 10 methods | 14.5 s | 528 MiB |
| Fresh public point estimates | Maddison, FDIC, French; CLARABEL, no existing caches or licensed solver | 7.5 s | 691 MiB |

These are measured checks, **not a full-paper runtime or memory benchmark**.
Peak RSS is the maximum recorded for a test worker, not total system memory.
Test discovery may add cases after these measurements. An absent MOSEK license
skips its optional cross-check; the open-source solver checks still run.

The full Atalay profile/surface checks require prepared grids and were explicitly
skipped because those grids were absent. MATLAB/Stata and a complete paper run
were not benchmarked in this cleanup. Full simulation settings, especially the
GIV cells with 1,000 units and up to 10,000 periods, are much larger than smoke
settings. No measured full-run memory requirement is claimed.

## Results and cache provenance

`reference_results.json` fixes selected public-input point estimates, dimensions,
input hashes and absolute tolerances. `check_results.py` compares generated
summaries against those values without rewriting the reference. File checksums,
run completeness and numerical agreement are separate checks.

A fresh run using only bundled public inputs reproduced all 15 reference fields
exactly: two point estimates and N/T/M for countries, banks and country equity
markets. All five raw-source hashes matched the reference. This check used 12
bootstrap draws to exercise the estimation path; it does not validate the
paper's full inference run or its confidence intervals.

Clean-panel keys now hash complete input files. Old partial-hash cache files are
retained but are not treated as verified inputs to the new cache keys. Simulation
checkpoints use numerical parameters plus an explicit version, so moving the
checkout or renaming an entry point does not change the experiment's identity.
