# Granularity in High Dimensions

Use Python 3.11 in a virtual environment.

```sh
python -m pip install -r code/requirements.txt
make -C code all
python code/verify_release.py
```

Public input snapshots are bundled; no data download step is needed.
`output/` holds the paper exhibits. Missing licensed data or optional source tools leave
estimates blank while retaining the exhibit layout. See [data sources](data/README.md).
Foerster preprocessing uses Octave and Dynare's `gensys.m` (`GENSYS_DIR`). Atalay
preprocessing uses Stata with `hprescott` and MATLAB with its Statistics toolbox.
Set `OCTAVE`, `MATLAB` or `STATA` to select executables.

The maintained code is byte-identical to the research pipeline at release.
`SHA256SUMS` verifies bundled code and inputs. Runtime caches start empty and stay
ignored by Git. Use `make -C code test` for the estimator and loader checks.
