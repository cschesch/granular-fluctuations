# Herskovic et al. (2020)

`replication_package/` preserves the local author release for
*Firm Volatility in Granular Networks*, DOI 10.1086/710345. It contains the
original MATLAB/Stata code, fitted parameters, linkage file, public deflators,
and licensed Compustat/CRSP observations. Source details are in `source.json`.
The journal currently returns HTML to the programmatic archive request.
Generated execution output and exporter executable-path caches are archived
separately; original code, parameters and source observations are retained.

Our application is in `code/applications/herskovic/` and reads the deposit
without editing it. Its model-size export reproduces the author's 5,000-firm
MATLAB draw using seed 100. Rebuilding that draw needs MATLAB with the Statistics and Machine Learning Toolbox. All generated
sizes, completed networks and recovered shocks live under `code/.cache/`.

The public release retains the original code and source instructions. Its
licensed observations and linkage inputs are supplied separately by users.
