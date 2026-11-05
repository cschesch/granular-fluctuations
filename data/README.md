# Data

Each folder contains data from one source, source information, and useful
download scripts. Public inputs are bundled. Preparation writes processed data
to `code/.cache/`; original input files are not modified.

## Data obtained separately

WRDS extracts, Forbes wealth snapshots, and restricted inputs from other papers'
replication packages are not included.

- [WRDS](README_WRDS.md): Compustat, CRSP, mutual funds, and SDC.
- [Forbes](forbes_billionaires/README.md): archived billionaire wealth data.
- [Herskovic et al.](herskovic2020/README.md): licensed firm data and linkages.
- [Stella](stella2015/README.md) and [Welburn](welburn2020/README.md): additional WRDS inputs.

Each WRDS source folder has `paper_inputs.json` with the paper's filenames,
extraction dates, sample cutoffs, and checksums. These files contain metadata,
not licensed observations. See [WRDS input selection](README_WRDS.md#which-file-is-used).

The default runner uses the paper inputs and runs everything available.
Unavailable estimates remain empty; `output/run_report.json` lists completed,
skipped, and failed analyses with reasons. Use `--strict` to fail when a
requested analysis is unavailable. See the [main README](../README.md).

## Additional software

| Application | Additional requirements |
| --- | --- |
| [Foerster, Sarte and Watson](foerster2011/README.md) | Octave and `gensys.m` |
| [Atalay](atalay2017/README.md) | Stata with `hprescott`; MATLAB with Statistics and Machine Learning Toolbox |
| [Herskovic et al.](herskovic2020/README.md) | MATLAB with Statistics and Machine Learning Toolbox |
| [Stella](stella2015/README.md) | Octave and `gensys.m`; X-12; GNU Fortran to build the bundled X-12 source |

The source READMEs contain setup instructions. Python-only applications use the
packages in `code/requirements.txt`.

## Sources and reuse

Attribution is recorded in each source's README or metadata files. Included
publication archives preserve the authors' original files. Third-party data
and code remain subject to their original terms.
