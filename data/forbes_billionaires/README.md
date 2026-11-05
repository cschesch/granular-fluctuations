# Forbes billionaire wealth

This application uses Internet Archive captures of Forbes's real-time
billionaire list. The repository includes the reconstruction script and
`archive_cdx_daily.csv`, which lists the selected captures. The reconstructed
wealth observations are not included.

## Required file

The application reads:

```text
data/forbes_billionaires/archive_realtime_daily.csv
```

`source.json` records the checksum of the data file used for the paper.

## Reconstruction

From the repository root:

```sh
python data/forbes_billionaires/build_forbes_billionaires_archive.py --help
```

The analysis code is in `code/applications/forbes/`. The reconstructed snapshots
stay local under [Forbes's data terms](https://www.forbes.com/connect/permissions-data-licensing/).
