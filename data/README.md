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
- French returns, KPSS patents, Song Ma's public firm-year patent panel, and OWID
  energy are bundled as frozen upstream snapshots. Their source links and
  attributions appear below; `SHA256SUMS` verifies the exact bundled files.
- Licensed Compustat, CRSP and SDC extracts are supplied locally by authorized
  users. Their acquisition scripts remain available. The Forbes folder provides
  the capture manifest, downloader and snapshot hash for a local rebuild.
  Missing inputs leave the corresponding estimates blank.

Generated panels, estimates and checkpoints are created under `code/.cache/`
when the analysis runs. The release starts with empty caches, and Git ignores
their contents.

Public snapshot sources:

- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
  `F-F_International_Countries.zip` and `F-F_Research_Data_Factors_CSV.zip`;
  retain Kenneth R. French's copyright notices in the original files.
- [KPSS extended data](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data/tree/fe9d68ecc6f8f4d87b4d54179d14cfcfb2768805):
  the three original 2024 ZIP files, with the authors' README and citation request.
- [Song Ma patent-firm panel](https://github.com/songma/patent-firm-link-share/tree/17f2f473a878036dfa8c87fdf969ac835b6edd24):
  `firm_year_panel_public.csv`, with the author's README and citations.
- [Our World in Data energy](https://github.com/owid/energy-data/tree/e7897aed55b514b8d16e0a3d8f1518789ac3de31):
  the original `owid-energy-data.csv`, stored as `owid_energy_data.csv`, with
  the upstream README and codebook identifying each series' original sources.
  OWID's work is CC BY; third-party series retain their original source terms.
