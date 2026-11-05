# Data

Each folder holds a source's raw snapshots, attribution and, where useful,
acquisition code. Public inputs are bundled. Original publication archives are
kept intact; our code extracts and processes them under `code/.cache/`.

Licensed WRDS inputs and Forbes snapshots stay local. See [WRDS](README_WRDS.md)
and [Forbes](forbes_billionaires/README.md) for acquisition. Missing inputs leave
the corresponding estimates blank. Generated panels, estimates and checkpoints
belong in `code/.cache/`; the GitHub package starts with empty caches.

`SHA256SUMS` in the replication package verifies its exact code and input files.
