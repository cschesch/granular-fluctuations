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

From the repository root, list the options with:

```sh
python data/forbes_billionaires/build_forbes_billionaires_archive.py --help
```

The script defaults to monthly sampling. Build the daily file over the bundled
capture manifest's window with:

```sh
python data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency daily --start-date 2019-08-01 --end-date 2026-06-25 --no-raw
```

The manifest contains captures from August 15, 2019 through June 25, 2026;
the paper's loader uses captures through December 31, 2025. A new archive
download is not guaranteed to match the recorded file checksum.

The analysis code is in `code/applications/forbes/`. The reconstructed snapshots
stay local under [Forbes's data terms](https://www.forbes.com/connect/permissions-data-licensing/).
