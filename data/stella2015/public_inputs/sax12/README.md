# Published sax12 source and Stella's seasonal-adjustment specification

The [Stata Journal sax12.ado](https://www.stata-journal.com/software/sj12-2/st0255/sax12.ado)
was fetched on 2026-10-20. Its version header is 1.0.1, modified 2012-03-14;
`provenance.json` records its SHA-256.

Stella's deposited Stata script calls:

```stata
sax12 revtq, satype(single) transfunc(log) outauto(all) outmethod(addone) x11seas() sliding history noview
```

The earlier reconstruction incorrectly interpreted `outauto(all)` as automatic
ARIMA model selection. In sax12.ado it controls **outliers** only. The program
emits an `automdl` block only if the separate `ammaxlag()` option is specified;
Stella does not specify that option. The source also emits `types=(all)`,
`method=addone`, forecast horizons 12/0, probability .95, `estimate{}`, and
`x11{seasonalma=msr}`. The earlier reconstruction instead inserted `automdl`
and relied on unspecified outlier/X-11 defaults.

The unmodified published ado was executed in Stata using `justspc` on an entirely
synthetic quarterly series. `synthetic_spec_generated_by_stata.spc` is the output;
`synthetic_revenues.dat` is its synthetic data. The generated SPC confirms these
options and the absence of an explicit ARIMA or automatic-model block. Its
absolute original file path is replaced with the local synthetic fixture path
when running the test.

`code/prepare_stella_sales.py --seasonal-spec sax12` translates these estimation
options to current Census X-13 and saves `sales_growth_sax12.csv`. Sliding-span
and revision-history diagnostics are omitted from production runs. The numerical
test in `code/test_stella_preprocessing.py` confirms that the production
translation and the original Stata-emitted specification give identical D11
adjusted values on the synthetic fixture (including those diagnostics in the
original specification). All synthetic fixtures here are public, contain no
Compustat observations, and serve only to validate software options.

The original executable is still unavailable: this improves the *specification*
match while retaining current X-13 and `saleq` in place of the paper's X-12 and
`revtq`. The archive also treats original `tid=549` specially; its gvkey is
unavailable, so that exception cannot be recreated. The older automatic-model
reconstruction remains available explicitly with `--seasonal-spec automdl` and
keeps its existing `sales_growth_x13.csv` outputs.
