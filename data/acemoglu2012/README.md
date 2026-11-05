# Acemoglu et al. (2012)

Original replication files for Daron Acemoglu, Vasco Carvalho, Asuman Ozdaglar,
and Alireza Tahbaz-Salehi, “The Network Origins of Aggregate Fluctuations,”
*Econometrica* 80(5), 2012. The deposited MATLAB files and data remain unchanged.

`bea1992/ndn0184.zip` is BEA's original [1992 detailed total-requirements archive](https://apps.bea.gov/industry/zip/ndn0184.zip).
It provides labeled network sectors and the SIC concordance. The
[benchmark documentation](https://apps.bea.gov/scb/pdf/national/inputout/meth/io1992.pdf#page=45)
explains the zero-output commodity reassignments used by the match.

`code/applications/acemoglu/panel.py` reads and validates the NBER–CES panel.
The analysis retains the authors' detrending and matches 451 balanced
manufacturing industries to 341 common units in the deposited 1992 network.
Two sensitivity checks use the original SIC industries, weighted either by
1958 shipments or by network influences allocated to those industries.

From `code/`:

```sh
python -m applications.acemoglu.run_acemoglu2012_bea1992
python -m tests.test_acemoglu2012_mgp
```

Generated shocks, network matches and the three-specification comparison live
in `.cache/derived/acemoglu2012/`; the paper's primary summary lives in
`.cache/run_outputs/`.
