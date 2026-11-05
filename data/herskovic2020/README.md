# Herskovic et al. (2020)

This folder contains code and public inputs from the replication package for
*Firm Volatility in Granular Networks*. Source information is in `source.json`.
Licensed Compustat/CRSP observations and restricted linkage inputs must be
obtained separately.

Our analysis code is in `code/applications/herskovic/`. Its preparation step
requires MATLAB with Statistics and Machine Learning Toolbox and reproduces
the authors' 5,000-firm model-size draw with seed 100. Set `MATLAB` to the
executable path when it is not on PATH.

From the repository root, with the Python environment activated:

```sh
cd code
python -m applications.herskovic.prepare
```

The MATLAB exporter is alongside that preparation script. Generated sizes,
networks, and recovered shocks are stored under `code/.cache/`.
