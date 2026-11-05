# Atalay (2017)

`replication_elasticities.zip` is the untouched publication package from the
[author's website](https://enghinatalay.github.io/), corresponding to
[the AEA deposit](https://doi.org/10.3886/E114149V1).
Its SHA-256 is `bea450fd194343a2c13ef48212ff63c9898727dde51ccaac618f95daa77a2ce7`.

`python -m applications.atalay.prepare` from `code/` rebuilds the U.S. data and
filter inputs in `.cache/derived/atalay2017/`. It requires Stata with `hprescott`
and MATLAB with Statistics and Machine Learning Toolbox. Set `STATA` and
`MATLAB` to their executable paths when they are outside PATH.
Our wrappers run against a working copy and preserve the original archive.
