# Foerster, Sarte and Watson (2011)

`source.zip` contains the original replication package. Its source URL and
checksum are in `source.json`.

## Requirements

Octave and `gensys.m` with its `qzdiv.m` and `qzswitch.m` helpers, available with
Dynare. Set `OCTAVE` to the executable and `GENSYS_DIR` to the directory containing
these three files when they are not found automatically.

## Prepare the data

From the repository root, with the Python environment activated:

```sh
cd code
python -m applications.foerster.prepare
```

This reconstructs model inputs and productivity shocks from the original
Excel files. The Python code follows the original GAUSS calculations; source
files and spreadsheet ranges are identified beside those calculations.
The shared model solver is `code/network_solver.py`.

Results are written to `code/.cache/derived/foerster2011/`.
Preparation validates source, code and output hashes before reusing this cache;
validated results can be reused without Octave. When preparation is unavailable
and the cache cannot be validated, the runner skips dependent analyses.
