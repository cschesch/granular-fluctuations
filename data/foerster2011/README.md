# Foerster, Sarte and Watson (2011)

`source.zip` is the complete original author archive. Its URL and SHA-256 are
in `source.json`. The original files are extracted into the ignored cache.

Run `python -m applications.foerster.prepare` from `code/`. It reads the
original XLS inputs, follows the source GAUSS normalization and quarterly
recurrence, and uses the shared Octave/Sims model solver. Source functions
and Excel ranges are identified beside the translated calculations.
All prepared inputs and exports stay in `code/.cache/derived/foerster2011/`.
Octave and Dynare's `gensys.m` are required. Set `OCTAVE` and `GENSYS_DIR` if
they are outside the usual search paths.
