# Figures and tables

Numbers refer to the current manuscript. All files below are written under
`output/`. Run `make all` for the dependencies, or use the section commands in
[Code](README.md). Producer modules are run with `python -m MODULE` from `code/`;
plotting modules read the estimates prepared earlier by the section runner.
Output filenames are retained so existing manuscript references still work.

| Figure/table | Description | Output file | Producer module |
| --- | --- | --- | --- |
| Figure 1 | Granularity across applications | `figure1.pdf` | `plot_expanded_applications` |
| Figure 2 | Compustat LPT buckets | `lpt_compustat_qtr.pdf` | `applications.compustat.run_compustat_quarterly` |
| Figure 3 | Homoskedastic versus heteroskedastic bounds | `figure1_hmgp.pdf` | `build_homoskedastic_applications_table` |
| Figure 4 | Atalay elasticity profiles | `atalay_elasticity_profile.pdf` | `applications.atalay.run_atalay_elasticity_profile` |
| Figure 5 | Factor-rank lower bounds | `factor_count_overview.pdf` | `plot_factor_count_overview` |
| Table A.1 | Application estimates and sample definitions | `table_applications_vertical.tex` | `build_applications_vertical` |
| Figure A.1 | Common factors in Compustat sales | `sales_factor_averaging.pdf` | `applications.compustat.plot_common_shock_diagnostics` |
| Figure A.2 | Foerster influence-weight LPT buckets | `lpt_foerster_net_M10.pdf` | `applications.foerster.run_foerster_epsilon` |
| Figure A.3 | Sensitivity to group count | `msweep_combined.pdf` | `plot_m_sweep` |
| Table A.2 | Matched Foerster comparison | `table_foerster_matched_tabular.tex` | `applications.foerster.run_foerster_matched_comparison` |
| Table A.3 | Stella input moments | `table_stella_input_moments.tex` | `applications.stella.audit_stella_input_moments` |
| Table A.4 | Welburn input moments | `welburn_input_moments.tex` | `applications.welburn.build_welburn_moments` |
| Figure A.4 | Atalay calibration: one-factor estimate and factor-free ceiling | `atalay_surface_comparison.pdf` | `applications.atalay.run_atalay_elasticity_surface` |
| Figure A.5 | Sectoral granularity and GDP volatility | `bea_granular_ceiling_figure1.pdf` | `applications.volatility.run_bea_granular_ceiling` |
| Figure A.6 | Firm granularity and GDP volatility | `firm_granular_volatility.pdf` | `applications.volatility.run_great_diversification` |
| Figure D.1 | Confidence-limit coverage | `mc_new_coverage.pdf` | `plot_inference_mc` |
| Figure D.2 | Sampling distributions | `mc_new_limit_law.pdf` | `plot_inference_mc` |
| Figure D.3 | Finite-sample bias | `mc_new_bias.pdf` | `plot_inference_mc` |
| Figure D.4 | Coverage across simulation designs | `mc_new_designs.pdf` | `plot_inference_mc` |
| Table D.1 | Simulation estimates and coverage | `table_mc_new_tabular.tex` | `plot_inference_mc` |
| Figure D.5 | Near-optimality tolerance | `mc_new_delta.pdf` | `plot_inference_mc` |
| Figure D.6 | Compustat confidence intervals | `mc_new_compustat_intervals.pdf` | `plot_inference_mc` |
| Table D.2 | Alternative grouping schemes | `table_grouping_schemes_mc_new_tabular.tex` | `run_grouping_schemes` |
| Figure D.7 | Tightness under Gaussian factors | `mc_new_tightness.pdf` | `plot_inference_tightness` |
| Table E.1 | Homoskedastic application estimates | `table_applications_homoskedastic.tex` | `build_homoskedastic_applications_table` |
| Figure E.1 | Within-industry dependence | `compustat_industry_dependence_ci.pdf` | `applications.compustat.run_compustat_industry_ci` |
| Table E.2 | Idiosyncratic covariance budgets | `table_sales_covariance_budget_tabular.tex` | `applications.compustat.run_sales_covariance_budget_table` |
| Table F.1 | GIV simulation estimates | `table_grouped_giv_mc_combined_tabular.tex` | `run_grouped_giv_mc` |

Figures B.1 and C.1 are drawn directly in the manuscript and have no external
generated file. `run_inference_mc` produces the simulation estimates used by
`plot_inference_mc` and `plot_inference_tightness`.
