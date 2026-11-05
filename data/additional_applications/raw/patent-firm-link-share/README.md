# PatentsView to Compustat / CRSP Firm-Name Match

This database links USPTO PatentsView patents to public-firm identifiers: Compustat `gvkey` and, where available, CRSP `permno`. It is designed for researchers who want to merge patent-level innovation measures and firm-year technology measures with standard financial datasets.

The public release currently includes two analysis-ready CSV files: `patent_level.csv` and `firm_year_panel_public.csv`.

## Citation

If you use this database, please cite:

> Ma, Song. "Technological Obsolescence." *The Review of Financial Studies*, October 2025, hhaf059. [https://doi.org/10.1093/rfs/hhaf059](https://doi.org/10.1093/rfs/hhaf059)

> Ma, Song. "The life cycle of corporate venture capital." *Review of Financial Studies* 33, no. 1 (2020): 358–394. https://doi.org/10.1093/rfs/hhz042

## Data Sources

- **[PatentsView](https://data.uspto.gov/bulkdata/datasets/pvgpatdis?fileDataFromDate=1976-01-01&fileDataToDate=2025-12-31)** : granted patent disambiguated data, source release dated **2026-04-10**, covering grants through **2025-12-30**: patent records, application records, disambiguated assignees, citations, CPC, and IPC files.
- **Compustat North America and Global**: historical firm names and aliases, keyed by `gvkey`.
- **CRSP stocknames**: historical US security names, keyed by `permno`.
- **CRSP/Compustat Merged link table**: bridge between `gvkey` and `permno`.
- **KPSS 2017 patent-firm linkage** ([Kogan, Papanikolaou, Seru, and Stoffman, 2017](https://doi.org/10.1093/qje/qjw040), with extended data through 2024 from the [KPSS GitHub repository](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data)): patent-to-`permno` linkage used as an external benchmark and validated overlay.
- **Manual rescue file**: hand-curated identifiers for high-volume or otherwise important assignees.

## Sample Definition

This release was built from the PatentsView granted-patent source release dated **2026-04-10**:

- **Source release**: **9,454,161** PatentsView source patent records.
- **Patent-level universe**: **8,531,932** cleaned utility patents granted in 1976 or later.
- **Firm-name matching universe**: **7,606,292** patents in the patent-level universe with at least one company/corporation assignee (PatentsView `assignee_type` 2 or 3, US or foreign company/corporation).

## Public Data Files

### `patent_level.csv`

Patent-level panel with selected firm identifiers, primary CPC class, and selected innovation measures.

- Size: **8,531,932 rows, 18 columns**.
- Includes the public patent-level sample described above; it is not restricted to patents successfully matched to Compustat or CRSP.
- Identifiers: `patent_num`, `filing_year`, `grant_year`, `permno`, `gvkey`, `primary_cpc_class`.
- `gvkey` and `permno` are populated when the patent is matched to the public-firm linkage; otherwise they are blank.

Users who need other fields can merge them from the same PatentsView source release using `patent_num`.

Of the patent-level universe, **4,854,640** unique patents are linked to a public-firm identifier (`gvkey` or `permno`) via the underlying patent-firm linkage file `patent_public.csv`. Among these linked patents, **4,826,989** (**99.4%**) carry a Compustat `gvkey` and **3,768,458** (**77.6%**) carry a CRSP `permno`. The remaining **27,651** patents are CRSP-only matches: they have a `permno` but no Compustat `gvkey`. Users can either keep these rows or restrict the sample to `gvkey`-matched patents.

The final linkage covers **42,118** PatentsView organizational assignee IDs, **23,280** Compustat `gvkey`s, and **12,699** CRSP `permno`s. These counts are not expected to be one-to-one: multiple PatentsView assignee IDs, historical names, subsidiaries, and spelling variants can map to the same Compustat firm.

### `firm_year_panel_public.csv`

Balanced firm-year panel for `gvkey`-matched firms, with selected patent-level measures aggregated over the year's newly filed and eventually granted patents, plus the firm-year obsolescence measure.

- Unique key: (`gvkey`, `year`).

- Size: **593,488 rows** across **23,118** unique `gvkey`s, spanning 1976-2025.

- The panel covers the `gvkey`-matched portion of the PatentsView data.

- Count-like measure columns are summed over patents with `filing_year == t`. Index/share measure columns are equal-weight averages over those patents. `n_patents_stock` is the only cumulative stock variable.

- Obsolescence: `obsolescence_hrz_5`, available in the analysis window once the firm has at least 10 cumulative patents; otherwise defined as missing following the paper.

## Selected Measure Definitions

Firm-year aggregation uses the patent `filing_year`, not grant year. Because the source is granted patents, flow counts are counts of patents filed in year `t` that were eventually granted.

A citation link is directed from the later citing patent to the earlier cited patent.

| Column                                      | Patent-level definition                                                                                                                                                                      | Firm-year aggregation                                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `n_patents_flow`                            |                                                                                                                                                                                              | Count of patents filed by the firm in year `t` that were eventually granted.                                        |
| `n_patents_stock`                           |                                                                                                                                                                                              | Cumulative count of eventually granted patents filed by the firm through year `t`; this is the only stock variable. |
| `primary_cpc_class`                         | CPC class on the PatentsView `cpc_sequence == 0` row.                                                                                                                                        |                                                                                                                     |
| `num_cite_forward`                          | Number of forward citation links received by the patent.                                                                                                                                     | Sum over patents filed by the firm in year `t`.                                                                     |
| `frac_cite_forward`                         | Fractional forward citations: each citing patent contributes `1 / total backward citations` to each patent it cites.                                                                         | Sum over patents filed by the firm in year `t`.                                                                     |
| `num_cite_3y`, `num_cite_5y`                | Forward citation counts received within 3-year and 5-year filing-year windows.                                                                                                               | Sum over patents filed by the firm in year `t`; missing patent-level values contribute zero to the sum.             |
| `originality_cpc`, `originality_ipc`        | Originality following Hall, Jaffe, and Trajtenberg (2001): `1 - sum(share^2)` over the CPC/IPC classes of patents cited by this patent; higher values imply broader cited-class dispersion.  | Equal-weight average over patents filed by the firm in year `t`.                                                    |
| `generality_cpc`, `generality_ipc`          | Generality following Hall, Jaffe, and Trajtenberg (2001): `1 - sum(share^2)` over the CPC/IPC classes of patents that cite this patent; higher values imply broader citing-class dispersion. | Equal-weight average over patents filed by the firm in year `t`.                                                    |
| `exploitative_narrow`, `explorative_narrow` | Share of backward citation links that cite patents in the citing firm's own prior patent portfolio by the citing year; `explorative_narrow = 1 - exploitative_narrow`.                       | Equal-weight average over patents filed by the firm in year `t`.                                                    |
| `exploitative_broad`, `explorative_broad`   | Share of backward citation links that cite patents in the broad existing-knowledge base: the firm's own prior patents or patents the firm cited during the previous five years.              | Equal-weight average over patents filed by the firm in year `t`.                                                    |
| `obsolescence_hrz_5`                        |                                                                                                                                                                                              | Ma (2025) five-year firm technology-obsolescence measure; missing below the patent-stock threshold.                 |

Reference for the originality and generality measures:

> Hall, Bronwyn H., Adam B. Jaffe, and Manuel Trajtenberg. "The NBER Patent Citation Data File: Lessons, Insights and Methodological Tools." NBER Working Paper No. 8498, October 2001. [https://doi.org/10.3386/w8498](https://doi.org/10.3386/w8498)

## Construction Overview

The database is built in two broad stages:

1. **Firm-name matching**: link PatentsView organizational assignees to Compustat `gvkey` and CRSP `permno`.
2. **Post-match measure construction**: combine the patent-firm linkage with citation and technology-class data to produce the public panels.

### Firm-Name Matching

- **Normalize company names**: standardize patent-assignee and Compustat/CRSP names before matching by fixing encoding and punctuation, harmonizing abbreviations and corporate suffixes, removing legal boilerplate, and extracting a compact `stem_name`.
- **Build historical reference names** from Compustat North America, CRSP names linked to Compustat through CCM, CRSP names without a Compustat link, and Compustat Global. A patent filed under an older assignee name is compared against historical Compustat and CRSP names, not only the firm's latest name.
- **Run a staged fuzzy-name cascade**: restrict candidate pairs using distinctive character n-grams; score candidates using multiple name-similarity signals; lock high-confidence matches early; let unresolved assignees continue across the remaining reference pools.
- **Apply final linkage priority**: hand-curated manual rescue links; automated fuzzy-cascade matches; validated KPSS-seeded `assignee -> permno` fallback links.

### Technological Obsolescence

- Obsolescence follows Ma (2025).
- A firm's technology base is the set of patents cited by the firm's own patents up to the relevant base year, excluding the firm's own patents from that base.
- The measure compares later citation activity to earlier citation activity for the same fixed technology base.
- The public files include the headline `w = 5` horizon: `obsolescence_hrz_5`.
- The measure is available in the analysis window once the firm has at least 10 cumulative patents; otherwise missing.

## Relationship to KPSS

KPSS (Kogan, Papanikolaou, Seru, and Stoffman, 2017, extended through 2024) is the standard benchmark for patent-to-public-firm linkage. 

As discussed below, our database has ~30% (nearly 900,000) more patents identified with a `permno` and/or `gvkey`. We have high common-sample agreement with KPSS, which is reassuring. At the same time, our database is built from PatentsView assignees, historical Compustat and CRSP names, Compustat Global, manual rescues, and a staged fuzzy-name cascade. This process allows for the additional linkages.

However, Neither linkage is error-free, and some edge-case matches may still be wrong. 

### 1. CRSP `permno` comparison

In the common validation window (1976-2024 grant years), the universe contains **7,298,454** utility patents with at least one organizational assignee. Restricting the comparison to CRSP `permno`, KPSS's primary identifier:

| Measure                            | Patents       | Takeaway                                             |
| ---------------------------------- | ------------- | ---------------------------------------------------- |
| KPSS assigns a `permno`            | **2,791,628** | KPSS CRSP-linked coverage.                           |
| Our linkage assigns a `permno`     | **3,648,288** | Larger CRSP-linked coverage.                         |
| Both assign a `permno`             | **2,751,076** | We cover **98.5%** of KPSS-linked patents.           |
| Same `permno` when both assign one | **2,663,381** | **96.8%** agreement on common coverage.              |
| Our linkage only                   | **897,212**   | Additional CRSP-linked patents not assigned by KPSS. |
| KPSS only                          | **40,552**    | KPSS links not retained by our validation gate.      |

The comparison has two implications: the overlap with KPSS is large, and our `permno` coverage is meaningfully broader. The added coverage comes from matching current PatentsView assignee IDs to historical Compustat and CRSP names with standardized names, staged fuzzy matching, and manual rescues.

### 2. What the `gvkey` linkage adds

The larger extension is outside the `permno` comparison. The public release links **4.85 million** patents to public firms, nearly all with a Compustat `gvkey`; roughly **1.09 million** linked patents have a `gvkey` but no usable `permno`. These are mainly Compustat NA firms without a valid CCM link in the relevant period and non-US listed firms from **Compustat Global**. The file is therefore built for Compustat-based research, not only for applications that require a CRSP security identifier.

## Identifier Notes

- `gvkey` is the Compustat firm identifier.
- `permno` is the CRSP security identifier and is available for the subset of matched patents with CRSP coverage.
- `ass_id` is the PatentsView disambiguated assignee identifier for the source release.

## Updates

The database is updated with PatentsView releases. The current public files were built from the PatentsView granted-patent source release dated **2026-04-10**.

## Contact

Song Ma, Yale School of Management - [song.ma@yale.edu](mailto:song.ma@yale.edu)
