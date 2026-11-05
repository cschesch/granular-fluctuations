# Acemoglu et al. (2012) replication and MGP extension

This directory contains the deposited replication package for Daron Acemoglu,
Vasco Carvalho, Asuman Ozdaglar, and Alireza Tahbaz-Salehi, “The Network Origins
of Aggregate Fluctuations,” *Econometrica* 80(5), 2012. The official source files
are treated as immutable inputs. `code/run_acemoglu2012_mgp.py` checks their
SHA-256 hashes before doing any analysis.

## Exact package run

On 2026-08-31, all deposited MATLAB scripts were run directly with MATLAB R2026a,
without modifying any `.m` or `.mat` file:

```sh
cd data/acemoglu2012
/Applications/MATLAB_R2026a.app/bin/matlab -batch \
  "std_TFP_NBER_Manuf; norm_v2_vector_detailedIO; norm_v2_vector_summaryIO; weighted_indegrees_final; weighted_outdegrees_final; weighted_outdegrees_2ndorder_final;"
```

The run reproduced:

- mean raw and detrended five-factor manufacturing TFP volatility: 0.0595 and 0.0594;
- detailed-IO influence norms for 1972--2002: 0.0980, 0.0912, 0.0883, 0.0881,
  0.0925, 0.0899, 0.0937;
- summary-IO influence norms: 0.1389, 0.1370, 0.1494, 0.1334, 0.1374, 0.1154,
  0.1187;
- all weighted in-degree, first-order out-degree, and second-order out-degree
  calculations. Modern MATLAB emitted only harmless legend warnings.

Run `make acemoglu2012-original` from `code/` to repeat the exact package run.

## Separate MGP extension

The package does not provide a mapping from its detailed BEA commodity matrices
to the four-digit SIC NBER-CES manufacturing panel, nor does it estimate a matched
panel of full-economy primitive shocks. The separate extension therefore uses a
model identity that does not require that crosswalk: equation (5) in the paper
shows that a sector's static network influence equals its equilibrium sales share.

We retain the 451 SIC industries with complete five-factor TFP growth for
1959--2005, reproduce the deposited script's no-intercept linear-time detrending,
and fix each industry's influence at its 1958 shipment share. Eight incomplete
industries are excluded: SIC 2411, 2711, 2721, 2731, 2741, 2771, 3292, and 3732.
The normalization of manufacturing shipment shares is immaterial for a variance
share. This is a manufacturing decomposition, not a full-economy GDP estimate.

The analysis writes derived inputs and diagnostics to `code/.cache/derived/acemoglu2012/`:

- `acemoglu2012_nber_manufacturing_primitive_shocks.csv`: wide shock panel;
- `acemoglu2012_nber_manufacturing_primitive_shocks_long.csv`: auditable long panel;
- `acemoglu2012_nber_manufacturing_influence.csv`: fixed 1958 influence weights;
- `acemoglu2012_replication_manifest.json`: source hashes and package checks.

From `code/`, run `python run_acemoglu2012_mgp.py` for the MGP extension or
`python test_acemoglu2012_mgp.py` for the focused checks.

## 1992 network-weight application

`bea1992/ndn0184.zip` is the BEA's [1992 detailed commodity-by-commodity
total requirements archive](https://apps.bea.gov/industry/zip/ndn0184.zip)
(SHA-256 `54baca22f9eed26abaa9901467c38e7598aeb45ddf41effbe6e51bbcbde7e1a5`).
It supplies the labeled total requirements and the 1987 SIC concordance.
Inverting the first 476 rows and columns of the labeled table and subtracting
the inverse from the identity reproduces the deposited direct requirements
matrix to floating-point precision after zeroing coefficients below
`9.99e-7`, a cutoff between the largest omitted positive coefficient and
the smallest retained coefficient. The extension uses the deposited matrix
itself. Its influence vector reproduces the package's 1992 norm of 0.0924943.

The concordance initially matches 441 of the 451 balanced SIC industries.
The [benchmark documentation](https://apps.bea.gov/scb/pdf/national/inputout/meth/io1992.pdf#page=45)
explains where BEA reallocates the nine zero-output commodity codes linked
to the other ten industries. Following these reassignments retains all 451
industries in 341 common units, covering 344 BEA commodities. In the two
groups with multiple destination commodities, we add the commodity influences
once each. Within each common unit, we combine the original detrended
NBER–CES shocks using fixed 1992 shipment shares. The shipment-weight
comparisons on common units use those same shock series.

We solve for the full-economy influence vector before selecting manufacturing
commodities. The covered units account for 50.4% of the sum of that vector.
The variance denominator is the variance of the aggregate component driven
by those manufacturing shocks. The 1992 network is held fixed over the
1959–2005 panel.

The paper pipeline uses the 341 common units. The SIC-level allocation and
the shipment-only runs are sensitivity checks. From `code/`, run
`python -m run_acemoglu2012_bea1992`. The comparison uses
ten LPT groups and 1,000 moving-block gradient bootstrap draws for the
one-sided 95% upper confidence limit. The results, labeled network nodes,
concordance, and within-unit allocations are saved under
`code/.cache/derived/acemoglu2012/` with the prefix `bea1992_`.
