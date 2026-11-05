"""Run the analyses behind the paper's tables and figures."""

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from project_paths import CODE_DIR, DERIVED_CACHE, ROOT, available
from wrds_inputs import wrds_path


DATA = ROOT / "data"
OUT = ROOT / "output"
CACHE = CODE_DIR / ".cache/run_outputs"
STELLA = ".cache/firm_networks/stella_revtq"
STELLA_DATA = DATA / "stella2015"
STELLA_PUBLIC = STELLA_DATA / "public_inputs"
ALIASES = DATA / "compustat/welburn_customer_aliases.csv"
GDPDEF = DATA / "fred/GDPDEF.csv"
GDPC1 = DATA / "fred/GDPC1.csv"
ANNUAL = wrds_path("compustat_annual")
QUARTERLY = wrds_path("compustat_quarterly_sales")
CUSTOMERS = wrds_path("compustat_customers")
CPI = DATA / "bls_cpi"
RI = CPI / "relative_importance_raw"
FRENCH = DATA / "french"
FOERSTER = DERIVED_CACHE / "foerster2011/exports"
FOERSTER_MODEL_INPUTS = DERIVED_CACHE / "foerster2011/model_inputs"
BEA = DATA / "bea_klems"
ATALAY = DERIVED_CACHE / "atalay2017"
HERSKOVIC = DATA / "herskovic2020"
WELBURN = DATA / "welburn2020/wrds_inputs"


@dataclass(frozen=True)
class Analysis:
    name: str
    commands: tuple[tuple[str, ...], ...]
    inputs: tuple[Path, ...] = ()
    exhibits: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()
    missing: tuple[tuple[str, ...], ...] = ()
    prepared: tuple[Path, ...] = ()


def stella_input() -> Path:
    if override := os.environ.get("STELLA_QUARTERLY_FILE"):
        return Path(override).resolve()
    status = DATA / "stella2015/wrds_inputs/latest_acquisition_status.json"
    if not status.is_file():
        return status
    record = json.loads(status.read_text())
    return status.parent / record.get("file", "stella_quarterly.csv.gz")


def run(command: tuple[str, ...], *, plan: bool = False) -> None:
    print(f"  {' '.join(command)}", flush=True)
    if not plan:
        env = dict(os.environ, STELLA_REVENUE_FIELD="revtq")
        subprocess.run([sys.executable, "-u", "-m", *command], cwd=CODE_DIR, env=env, check=True)


def blank_exhibit(name: str) -> None:
    path = OUT / name
    if path.suffix == ".tex":
        path.write_text("% Source data unavailable.\n")
    else:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.savefig(path)
        plt.close(fig)


APPLICATIONS = (
    Analysis("CPI preparation", (
        ("applications.bls_cpi.build_cpi_weights",),
        ("applications.bls_cpi.build_cpi_prices",),
        ("applications.bls_cpi.build_cpi_weight_series",),
        ("applications.bls_cpi.build_cpi_panels",),
    ), (CPI / "cu.data.0.Current.txt", CPI / "cu.item", RI / "historical-relative-importance-1947-1986.xlsx",
        *(RI / f"ri-archive-{years}.zip" for years in ("1990-1999", "2000-2009", "2010-2019")),
        *(RI / f"{year}.xlsx" for year in range(2020, 2026)),
        *(RI / "1987-1989_RI_archive" / f"{year}.txt" for year in range(1987, 1990)),
        *(CPI / "history" / f"cu.data.{file}" for file in (
            "1.AllItems", "2.Summaries", "11.USFoodBeverage", "12.USHousing", "13.USApparel",
            "14.USTransportation", "15.USMedical", "16.USRecreation", "17.USEducationAndCommunication",
            "18.USOtherGoodsAndServices", "20.USCommoditiesServicesSpecial",
        ))),
        prepared=tuple(DERIVED_CACHE / "bls_cpi/panels" / name for name in (
            "cpi_1967_leaves.csv", "cpi_1967_inflation.parquet",
        ))),
    Analysis("Country equity preparation", (("applications.french.prepare",),), (
        FRENCH / "F-F_International_Countries.zip",
        FRENCH / "F-F_Research_Data_Factors_CSV.zip",
        FRENCH / "world_bank_CM.MKT.LCAP.CD_20_markets_1960_2025.json",
    ), prepared=tuple(DERIVED_CACHE / "french" / name for name in (
        "french_country_market_returns_monthly_1975_2025.csv", "world_bank_country_market_cap_weights_1975.csv",
    ))),
    Analysis("Quarterly Compustat", (("applications.compustat.run_compustat_quarterly", "--run-profile", "full"),),
             (QUARTERLY,), ("lpt_compustat_qtr.pdf",), ("compustat_qtr_summary.pkl",)),
    Analysis("Core applications", (("run_multi_source", "--available-only"),)),
    Analysis("Mutual funds", (("applications.mutual_funds.run_mutual_fund_applications",),), tuple(
        wrds_path(f"mutual_fund_{name}") for name in (
            "monthly_assets", "monthly_returns", "daily_returns", "headers", "portfolio_map",
        )), summaries=("mutual_fund_applications_summary.pkl",)),
    Analysis("Other application panels", (("applications.run_panels",),)),
    Analysis("Elections", (("applications.elections.run_election_applications",),),
             (DATA / "elections/algara_amlani_presidential_county_1868_2020.csv",),
             summaries=("election_applications_summary.pkl",)),
    Analysis("Sector output", (("run_sector_output_baseline",),), (
        BEA / "KLEMS-Research-Tables-Underlying-Data_0.xlsx",
        BEA / "KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx",
    ), summaries=("bea_bls_sector_tfp_no_network_M*_summary.pkl",)),
    Analysis("Foerster preprocessing", (("applications.foerster.prepare",),),
             (DATA / "foerster2011/source.zip",)),
    Analysis("Foerster", (
        ("applications.foerster.run_foerster_epsilon",),
        ("applications.foerster.run_foerster_matched_comparison",),
        ("applications.foerster.run_foerster_matched_inference",),
    ), tuple(FOERSTER / f"{stem}.csv" for stem in (
        "eps_q_98_0_1972_2007benchmark_wide", "eps_q_98_0_1972_2007benchmark_columns",
        "l4wq_q_98_0_1972_2007", "pii_a_98_benchmark", "xi_98_benchmark", "cap_rho_98_benchmark",
    )), ("lpt_foerster_net_M10.pdf",),
        ("foerster*_summary*.pkl", "foerster_matched_comparison.pkl"),
        missing=(("applications.foerster.run_foerster_matched_comparison", "--blank"),)),
    Analysis("Acemoglu", (("applications.acemoglu.run_acemoglu2012_bea1992",),), tuple(DATA / "acemoglu2012" / path for path in (
        "CC_allyears_new.mat", "NBER_Manuf_TFP_source_data.mat", "bea1992/ndn0184.zip",
    )), summaries=("acemoglu2012*_summary.pkl",)),
    Analysis("Carvalho–Gabaix", (("applications.carvalho_gabaix.run_carvalho_gabaix_proxy",),), tuple(BEA / path for path in (
        "KLEMS-Research-Tables-Underlying-Data_0.xlsx",
        "KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx",
    )), summaries=("carvalho_gabaix*_summary.pkl",)),
    Analysis("Atalay preprocessing", (("applications.atalay.prepare", "--section", "benchmark", "--available-only"),),
             (DATA / "atalay2017/replication_elasticities.zip",)),
    Analysis("Atalay", (("applications.atalay.run_atalay_epsilon",),), (
        ATALAY / "atalay_benchmark_productivity_shocks.csv", ATALAY / "atalay_benchmark_network_variance_matrix.csv",
    ), summaries=("atalay*_M*_summary.pkl",)),
    Analysis("Welburn", (("applications.welburn.run_welburn_firm_network",), ("applications.welburn.build_welburn_moments",)), (
        ANNUAL, QUARTERLY, CUSTOMERS, GDPDEF, ALIASES, STELLA_DATA / "manual_customer_aliases.csv",
        *(next(iter(sorted(WELBURN.glob(f"{stem}_*.csv.gz"), reverse=True)), WELBURN / f"{stem}.csv.gz")
          for stem in ("wrds_supply_chain_ids", "wrds_company_subsidiaries", "wrds_subsidiary_global_aliases")),
    ), summaries=("welburn_firm_network_M10_summary.pkl", "welburn_input_moments.pkl"),
        missing=(("applications.welburn.build_welburn_moments", "--blank"),)),
    Analysis("Herskovic preprocessing", (("applications.herskovic.prepare", "--available-only"),), (
        HERSKOVIC / "replication_package/data/Compustat Qtrly Sales and GIC July2017.txt",
    )),
    Analysis("Herskovic", (("applications.herskovic.run_herskovic_firm_network",),), (
        HERSKOVIC / "replication_package/data/Compustat Qtrly Sales and GIC July2017.txt",
        HERSKOVIC / "replication_package/data/fred_deflators_annual.xlsx",
        HERSKOVIC / "replication_package/data/linkages2013.mat",
        HERSKOVIC / "replication_package/code/guess/benchmark_parameters.mat",
        DERIVED_CACHE / "herskovic2020/calibrated_sizes_matlab_seed100.mat",
    ), summaries=("herskovic_firm_network_M10_summary.pkl",)),
    Analysis("Stella public preprocessing", (("applications.stella.prepare_public", "--available-only"),),
             (STELLA_DATA / "source.zip",)),
    Analysis("Stella", (
        ("applications.welburn.prepare_welburn_links",),
        ("applications.stella.audit_stella_selection_adf", "--output", f"{STELLA}/selection_adf_pvalues_sax12.csv",
         "--diagnostics-path", f"{STELLA}/selection_adf_diagnostics.json", "--save-adjusted"),
        ("applications.stella.prepare_stella_calibration", "--output-dir", STELLA,
         "--selection-adf", f"{STELLA}/selection_adf_pvalues_sax12.csv",
         "--checks-output", f"{STELLA}/network_checks.csv"),
        ("applications.stella.prepare_stella_unbalanced", "--source-dir", STELLA, "--output-dir", f"{STELLA}_unbalanced"),
        ("applications.stella.prepare_stella_sales", "--calibration-dir", STELLA, "--seasonal-spec", "sax12"),
        ("applications.stella.validate_stella_calibration_source", "--calibration-dir", STELLA,
         "--output", f"{STELLA}/original_constructor_validation.json"),
        ("applications.stella.solve_stella_dynamic", "--input-dir", STELLA),
        ("applications.stella.run_stella_firm_network", "--calibration-dir", STELLA, "--output-stem", "stella_revtq_firm_network"),
        ("applications.stella.audit_stella_input_moments",),
    ), (
        ANNUAL, CUSTOMERS, GDPDEF, ALIASES, stella_input(),
        *(STELLA_DATA / name for name in (
            "bea_summary_to_stella_sid.csv", "manual_firm_sectors.csv", "manual_customer_aliases.csv",
        )),
        *(STELLA_PUBLIC / name for name in (
            "1987_sic_to_1997_naics.xls",
            "bea_1997_summary/IO-CodeSummary.xls", "bea_1997_summary/NAICS-IO.xls",
            "bea_1997_summary/NAICSUseSummary.xls",
        )),
        DERIVED_CACHE / "stella/bls_sectoral_employment.csv",
        DERIVED_CACHE / "stella/deposited_real_gdp_growth.csv",
        STELLA_DATA / "source.zip",
        *(FOERSTER_MODEL_INPUTS / f"{stem}.csv" for stem in ("matrix_io", "alpha", "matrix_capital")),
        *(FOERSTER / f"{stem}_98_benchmark.csv" for stem in ("pii_a", "xi", "cap_rho")),
    ),
        summaries=("stella_revtq_firm_network_M10_summary.pkl", "stella_input_moment_comparison.pkl"),
        missing=(("applications.stella.audit_stella_input_moments", "--blank"),)),
    Analysis("Applications exhibits", (
        ("compute_applications_coefficients",),
        ("plot_expanded_applications",),
        ("build_applications_vertical",),
        ("run_homoskedastic_mgp", "--no-plot"),
        ("build_homoskedastic_applications_table",),
        ("check_factor_rank_figure1", "--refresh-panels", "--max-dimension", "10", "--available-only"),
        ("plot_factor_count_overview",),
        ("plot_m_sweep", "--available-only"),
    )),
)
VARIATIONS = (
    Analysis("Atalay filter grids", (("applications.atalay.prepare", "--available-only"),),
             (DATA / "atalay2017/replication_elasticities.zip",)),
    Analysis("Firm granularity and GDP volatility", (("applications.volatility.run_great_diversification",),),
             (ANNUAL, GDPDEF, GDPC1), ("firm_granular_volatility.pdf",)),
    Analysis("Sector granularity and GDP volatility", (("applications.volatility.run_bea_granular_ceiling",),),
             (BEA / "KLEMS-Research-Tables-Underlying-Data_0.xlsx",
              BEA / "KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx",
              GDPC1),
             ("bea_granular_ceiling_figure1.pdf",)),
    Analysis("Atalay elasticity profile", (("applications.atalay.run_atalay_elasticity_profile",),),
             tuple(ATALAY / "elasticity_profile" / sweep / "grid_index.csv" for sweep in ("eps_M", "eps_Q")), ("atalay_elasticity_profile.pdf",)),
    Analysis("Atalay elasticity surface", (("applications.atalay.run_atalay_elasticity_surface",),),
             tuple(ATALAY / "elasticity_surface" / f"atalay_surface_epsD_{value}.mat"
                   for value in (250, 500, 750, 900, 999, 1250)), ("atalay_elasticity_surface.png",)),
)
SIMULATIONS = (
    Analysis("Simulations", (
        ("run_mc_new", "--jobs", "2"),
        ("plot_mc_new", "--simulation-only"),
        ("plot_mc_new_tightness",),
        ("run_grouped_giv_mc",),
    )),
)
EMPIRICAL = (
    Analysis("Grouping comparison", (("run_grouping_schemes", "--solver", "CLARABEL"),)),
    Analysis("Quarterly Compustat appendix", (
        ("applications.compustat.plot_common_shock_diagnostics",),
        ("plot_mc_new", "--compustat-only"),
        ("applications.compustat.run_compustat_industry_dependence",),
        ("applications.compustat.run_compustat_industry_ci",),
        ("applications.compustat.run_sales_covariance_budget_table",),
    ), (QUARTERLY,), ("sales_factor_averaging.pdf", "mc_new_compustat_intervals.pdf",
        "compustat_industry_dependence_ci.pdf"),
        missing=(("applications.compustat.run_sales_covariance_budget_table", "--blank"),)),
)
SECTIONS = {"applications": APPLICATIONS, "variations": VARIATIONS,
            "simulations": SIMULATIONS, "empirical": EMPIRICAL}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", nargs="?", default="all", choices=("all", *SECTIONS))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    sections = SECTIONS if args.section == "all" else (args.section,)
    if not args.plan:
        OUT.mkdir(exist_ok=True)
    for section in sections:
        for analysis in SECTIONS[section]:
            print(f"[{section}] {analysis.name}", flush=True)
            if args.plan or available(analysis.name, *analysis.inputs):
                for command in analysis.commands:
                    run(command, plan=args.plan)
                continue
            for pattern in analysis.summaries:
                for path in CACHE.glob(pattern):
                    path.unlink()
            for path in analysis.prepared:
                path.unlink(missing_ok=True)
            for name in analysis.exhibits:
                blank_exhibit(name)
            for command in analysis.missing:
                run(command)


if __name__ == "__main__":
    main()
