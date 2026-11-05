# WRDS data

Source folders contain the acquisition scripts for Compustat, CRSP, mutual funds
and SDC. Each script documents its SQL; `_wrds.py` handles login, streaming
CSV.gz exports and metadata. Install the optional `wrds` Python package, then run,
for example, from the project root:

```sh
python data/compustat/download_quarterly_sales.py --username YOUR_WRDS_USERNAME
```

Scripts accept `--print-sql` and date cutoffs. They use the official WRDS terminal
prompts or an existing `.pgpass`, preserve earlier snapshots, and record SQL,
parameters and hashes in JSON sidecars. Compustat download queries restrict
currency to USD.

The research checkout pins its snapshots in the private `wrds_inventory.json`.
Without that inventory, loaders accept a canonical filename or the latest dated
export from the corresponding downloader. Licensed data and credentials stay
out of GitHub.
