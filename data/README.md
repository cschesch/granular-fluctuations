# Data sources

The release contains the frozen public inputs used by the paper. Source folders
retain their attribution and original license notices.

- BLS CES/CPI, BEA production accounts and regional GDP, BEA series distributed
  through FRED, FDIC bank accounts, EPA plant emissions and generation, FDA
  approvals, and Treasury holdings are public government inputs. Maddison GDP,
  Global Carbon Budget emissions and the election deposit retain their source
  attribution and release terms.
- Original publication releases are kept once under their study folder.
  `source.zip`, where available, preserves the received archive, and `source.json`
  records its origin. Our analysis code calls the extracted source from outside
  it. Author licenses and notices remain with the original files.
- The four UN Comtrade snapshots contain 19,098 original records. They are
  supplied under the allowance for free extraction services with up to 100,000
  records, with attribution to UN Comtrade. The Granular Cities workbook is
  Esquierro and Da Silva's Figshare release under CC BY 4.0. [UN Comtrade policy](https://uncomtrade.org/docs/re-dissemination-of-data/), [Granular Cities release](https://doi.org/10.6084/m9.figshare.21702203.v1)
- The Global Macro Database September 2026 file contains only the eleven source
  columns read by our debt application. Its research-use terms allow the data
  used in a paper's replication package, with attribution. The IMF Global Debt
  Database workbook retains IMF attribution. [GMD terms](https://www.globalmacrodata.com/license.html), [IMF terms](https://www.imf.org/en/About/copyright-and-terms)
- `python data/download_public.py` obtains the public French, KPSS, Song Ma and
  OWID files from their upstream releases. `downloads.json` records the expected
  SHA256 hashes. The script verifies existing files and rejects changed
  downloads. Use `--only` to select sources. The same script can fetch the
  Granular Cities workbook if it is missing.
- Licensed Compustat, CRSP and SDC extracts are supplied locally by authorized
  users. Their acquisition scripts remain available. The Forbes folder provides
  the capture manifest, downloader and snapshot hash for a local rebuild.
  Missing inputs leave the corresponding estimates blank.

Generated panels, estimates and checkpoints are created under `code/.cache/`
when the analysis runs. The release starts with empty caches, and Git ignores
their contents.
