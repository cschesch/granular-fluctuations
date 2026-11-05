# Forbes archive snapshots

The daily research snapshot is reconstructed from Internet Archive captures of
Forbes's real-time billionaire list. `archive_cdx_daily.csv` preserves the capture
URLs and archive digests. `source.json` records the exact local snapshot hash.

The public repository includes the acquisition code and capture manifest.
The snapshot stays local under Forbes's redistribution terms.
Run `python data/forbes_billionaires/build_forbes_billionaires_archive.py --help`
for the replay options. Supply the local daily CSV to populate the application.

Sources include https://www.forbes.com/terms/ and https://www.forbes.com/connect/permissions-data-licensing/.
