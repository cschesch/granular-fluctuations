# WRDS data

Compustat, CRSP, mutual-fund, SDC, and several firm-network applications require
WRDS extracts. These observations are not included. You need a WRDS account
with access to the relevant databases.

## Download data

Activate the replication environment and install the optional WRDS package:

```sh
python -m pip install wrds
```

Download scripts are in the corresponding source folders. For example, from
the repository root:

```sh
python data/compustat/download_quarterly_sales.py --username YOUR_WRDS_USERNAME
```

Use `--help` for options and `--print-sql` to inspect a query. Check the sample
dates before downloading. The scripts save compressed CSV files and JSON
sidecars recording the query, parameters, and checksum. Earlier downloads are
not overwritten.

The download scripts retrieve new extracts, not the paper's historical files.
To run analyses with these downloads, select `--inputs latest` in the replication
runner, as shown below. Reproducing the original inputs requires files matching
the recorded checksums; renaming a new download does not make it the same snapshot.

## Which file is used?

By default, each loader selects the exact filename recorded in its source
folder's `paper_inputs.json`. The metadata also record extraction dates,
sample cutoffs, and SHA-256 checksums. A missing paper input is reported as
unavailable; a newer file is not silently substituted.

To use newly downloaded inputs explicitly, run from the repository root:

```sh
python code/run_replication.py --inputs latest
```

This selects the latest dated export for each WRDS dataset, or its standard
`dataset.csv.gz` filename when no dated export exists. For an individual
analysis module, set `WRDS_INPUTS=latest`. The selected paths are printed and
recorded by the runner. New inputs may produce different results from the paper.

Keep licensed data and login credentials out of the repository.
