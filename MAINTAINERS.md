# Maintainer tasks

## Release copy

Copy maintained code and explicitly selected public inputs from the research
checkout. Never copy licensed observations, Forbes snapshots, credentials,
solver licenses, caches, checkpoints, generated results or the private archive.
Original third-party packages and their terms remain separate from our wrappers.

After preparing the release, compare it with the research checkout and write
its checksums:

```sh
python code/verify_release.py --source /path/to/research --release /path/to/release --write
python code/verify_release.py --release /path/to/release --exact
```

The exact-file-set option is for release maintenance. Ordinary users should
use the default integrity check, which permits separately obtained inputs.

Reference numerical values are reviewed, fixed public-data benchmarks in
`code/reference_results.json`. Do not replace them automatically with the
output of a new run. Review source hashes, sample definitions and numerical
differences before updating a reference.

## Manuscript output

From a complete research checkout, `make -C code syncdraft` replaces the
manuscript's separate `overleaf/output/` folder with the generated files used
by the paper. It checks the manuscript's references against `paper_outputs.py`.
This is separate from replication and does not edit the manuscript.

## Tests and performance

Keep fast, numerical and source-specific tests discoverable. CI installs the
pinned environment in a clean Python 3.11 checkout, then runs fast/numerical
checks and the synthetic smoke test. Record measured settings and limitations
in `code/REPRODUCIBILITY.md`; do not label smoke timings as full-paper runtimes.
