# Atalay (2017)

`replication_elasticities.zip` contains the original replication package from
[the author's website](https://enghinatalay.github.io/), corresponding to
[AEA deposit E114149V1](https://doi.org/10.3886/E114149V1).

## Requirements

Stata with `hprescott`, and MATLAB with Statistics and Machine Learning
Toolbox. Set `STATA` and `MATLAB` to executable paths when they are not on PATH.

## Prepare the data

From the repository root, with the Python environment activated:

```sh
cd code
python -m applications.atalay.prepare
```

Use `--section benchmark` for only the benchmark calibration. The default
also prepares the elasticity grids. Prepared inputs are written to
`code/.cache/derived/atalay2017/`. The MATLAB exporters are alongside the
preparation script in `code/applications/atalay/`.
