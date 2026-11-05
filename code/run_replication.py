"""Run the analyses behind the paper's tables and figures."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

from project_paths import CODE_DIR, DERIVED_CACHE, ROOT, find_program
from wrds_inputs import wrds_path, verify_inputs


OUT = ROOT / "output"
CACHE = CODE_DIR / ".cache/run_outputs"


@dataclass(frozen=True)
class Analysis:
    name: str
    commands: tuple[tuple[str, ...], ...]
    inputs: tuple[Path, ...] = ()
    exhibits: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()
    missing: tuple[tuple[str, ...], ...] = ()
    prepared: tuple[Path, ...] = ()
    optional_inputs: tuple[Path, ...] = ()


def run(command: tuple[str, ...], log: Path) -> dict:
    """Stream a command's output and retain its missing-input messages."""
    print(f"  {' '.join(command)}", flush=True)
    started = time.monotonic()
    skipped = []
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", *command], cwd=CODE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                if re.search(r"\[skip\]|\bSKIP\b", line, flags=re.IGNORECASE):
                    skipped.append(line.strip())
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    return dict(command=list(command), returncode=returncode,
                seconds=time.monotonic() - started, skipped=skipped, log=str(log))


def unavailable_exhibit(name: str) -> None:
    path = OUT / name
    message = "Unavailable: required input data or software missing. See run_report.json."
    if path.suffix == ".tex":
        path.write_text(r"\parbox{.95\linewidth}{\textit{Unavailable: required input data or software missing. See \texttt{run\_report.json}.}}" + "\n")
    else:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 3))
        fig.text(.5, .5, message, ha="center", va="center", wrap=True)
        fig.savefig(path)
        plt.close(fig)


def analysis_sections():
    DATA = ROOT / "data"
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
        Analysis("Core applications", (("run_multi_source", "--available-only"),), optional_inputs=(
            ANNUAL, QUARTERLY, wrds_path("crsp_daily_stocks_ciz"), wrds_path("crsp_security_history_ciz"),
        )),
        Analysis("Mutual funds", (("applications.mutual_funds.run_mutual_fund_applications",),), tuple(
            wrds_path(f"mutual_fund_{name}") for name in (
                "monthly_assets", "monthly_returns", "daily_returns", "headers", "portfolio_map",
            )), summaries=("mutual_fund_applications_summary.pkl",)),
        Analysis("Other application panels", (("applications.run_panels",),),
                 optional_inputs=(wrds_path("sdc_new_issues"),)),
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
        ), (DATA / "foerster2011/source.zip", *(FOERSTER / f"{stem}.csv" for stem in (
            "eps_q_98_0_1972_2007benchmark_wide", "eps_q_98_0_1972_2007benchmark_columns",
            "l4wq_q_98_0_1972_2007", "pii_a_98_benchmark", "xi_98_benchmark", "cap_rho_98_benchmark",
        ))), ("lpt_foerster_net_M10.pdf",),
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
            DATA / "atalay2017/replication_elasticities.zip",
            ATALAY / "atalay_benchmark_productivity_shocks.csv", ATALAY / "atalay_benchmark_network_variance_matrix.csv",
        ), summaries=("atalay*_M*_summary.pkl",)),
        Analysis("Welburn", (("applications.welburn.run_welburn_firm_network",), ("applications.welburn.build_welburn_moments",)), (
            ANNUAL, QUARTERLY, CUSTOMERS, GDPDEF, ALIASES, STELLA_DATA / "manual_customer_aliases.csv",
            *(wrds_path(stem)
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
            ("applications.stella.audit_stella_selection_adf", "--save-adjusted"),
            ("applications.stella.prepare_stella_calibration",),
            ("applications.stella.prepare_stella_unbalanced",),
            ("applications.stella.prepare_stella_sales",),
            ("applications.stella.validate_stella_calibration_source",),
            ("applications.stella.solve_stella_dynamic",),
            ("applications.stella.run_stella_firm_network",),
            ("applications.stella.audit_stella_input_moments",),
        ), (
            ANNUAL, CUSTOMERS, GDPDEF, ALIASES, wrds_path("stella_quarterly", path=os.environ.get("STELLA_QUARTERLY_FILE")),
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
            DATA / "foerster2011/source.zip",
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
                 (DATA / "atalay2017/replication_elasticities.zip",
                  *(ATALAY / "elasticity_profile" / sweep / "grid_index.csv" for sweep in ("eps_M", "eps_Q"))), ("atalay_elasticity_profile.pdf",)),
        Analysis("Atalay calibration comparison", (("applications.atalay.run_atalay_elasticity_surface",),),
                 (DATA / "atalay2017/replication_elasticities.zip",
                  *(ATALAY / "elasticity_surface" / f"atalay_surface_epsD_{value}.mat"
                    for value in (667, 999, 1333)),
                  *(ATALAY / "elasticity_surface" / f"one_factor_{value}.csv"
                    for value in (667, 999, 1333))), ("atalay_surface_comparison.pdf",)),
    )
    SIMULATIONS = (
        Analysis("Simulations", (
            ("run_inference_mc", "--jobs", "2"),
            ("plot_inference_mc", "--simulation-only"),
            ("plot_inference_tightness",),
            ("run_grouped_giv_mc",),
        )),
    )
    EMPIRICAL = (
        Analysis("Grouping comparison", (("run_grouping_schemes", "--solver", "CLARABEL"),),
                 (DATA / "simulation/grouping_mc_buckets.json",), ("table_grouping_schemes_mc_new_tabular.tex",)),
        Analysis("Quarterly Compustat appendix", (
            ("applications.compustat.plot_common_shock_diagnostics",),
            ("plot_inference_mc", "--compustat-only"),
            ("applications.compustat.run_compustat_industry_dependence",),
            ("applications.compustat.run_compustat_industry_ci",),
            ("applications.compustat.run_sales_covariance_budget_table",),
        ), (QUARTERLY,), ("sales_factor_averaging.pdf", "mc_new_compustat_intervals.pdf",
            "compustat_industry_dependence_ci.pdf"),
            missing=(("applications.compustat.run_sales_covariance_budget_table", "--blank"),)),
    )
    SECTIONS = {"applications": APPLICATIONS, "variations": VARIATIONS,
                "simulations": SIMULATIONS, "empirical": EMPIRICAL}

    return SECTIONS


def requested_analyses(sections, section):
    """Refresh the quarterly inputs needed by an independently requested appendix."""
    if section == "empirical" and wrds_path("compustat_quarterly_sales").is_file():
        yield "prerequisites", next(a for a in sections["applications"] if a.name == "Quarterly Compustat")
    for name in sections if section == "all" else (section,):
        for analysis in sections[name]:
            yield name, analysis


def missing_requirements(analysis):
    reasons = [f"Missing input: {path}" for path in analysis.inputs if not path.exists()]
    # Other external-tool preparers report their own availability. Stella needs
    # both tools before starting its multi-command reconstruction.
    if analysis.name == "Stella" and not reasons:
        from network_solver import find_gensys
        from applications.stella.prepare_stella_sales import locate_x13

        if find_program("octave") is None:
            reasons.append("Missing software: Octave; set OCTAVE to its executable.")
        for locate in (find_gensys, locate_x13):
            try:
                locate()
            except FileNotFoundError as error:
                reasons.append(f"Missing software: {error}")
    return reasons


def execute(analyses, report_path, *, strict=False, input_mode="paper"):
    OUT.mkdir(parents=True, exist_ok=True)
    logs = CACHE / "logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs.mkdir(parents=True)
    input_log = logs / "inputs.jsonl"
    os.environ["REPLICATION_INPUT_LOG"] = str(input_log)
    report = dict(
        started=datetime.now(timezone.utc).isoformat(), input_mode=input_mode, strict=strict,
        python=sys.version, platform=platform.platform(),
        thread_settings={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        numerical_checks="Not run; use make check-results separately.",
        analyses=[dict(section=section, name=a.name, status="not_run", reasons=[],
                       inputs=[str(p) for p in (*a.inputs, *a.optional_inputs)], commands=[]) for section, a in analyses],
    )
    started = time.monotonic()
    exit_code = 0
    try:
        for index, ((section, analysis), record) in enumerate(zip(analyses, report["analyses"], strict=True)):
            print(f"[{section}] {analysis.name}", flush=True)
            for path in (*analysis.inputs, *analysis.optional_inputs):
                print(f"[input] {path}", flush=True)
            absent = missing_requirements(analysis)
            commands = analysis.commands
            if absent:
                record.update(status="skipped", reasons=absent)
                print("[skip] " + "; ".join(record["reasons"]), flush=True)
                if strict:
                    exit_code = 2
                    break
                for pattern in analysis.summaries:
                    for path in CACHE.glob(pattern):
                        path.unlink()
                for path in analysis.prepared:
                    path.unlink(missing_ok=True)
                for name in analysis.exhibits:
                    unavailable_exhibit(name)
                commands = analysis.missing
            else:
                record["status"] = "running"
                verify_inputs((*analysis.inputs, *analysis.optional_inputs))
            for number, command in enumerate(commands):
                result = run(command, logs / f"{index:02d}-{number:02d}.log")
                record["commands"].append(result)
                record["reasons"].extend(result["skipped"])
                if result["returncode"]:
                    record["status"] = "failed"
                    exit_code = result["returncode"]
                    break
                if result["skipped"] and record["status"] != "skipped":
                    record["status"] = "partial"
                if strict and result["skipped"]:
                    exit_code = 2
                    break
            if record["status"] == "running":
                record["status"] = "completed"
            if exit_code:
                break
    except BaseException as error:
        record.update(status="failed", reasons=[*record["reasons"], f"{type(error).__name__}: {error}"])
        exit_code = 1
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["finished"] = datetime.now(timezone.utc).isoformat()
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report["peak_child_rss_mib"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
        selected = {}
        if input_log.exists():
            for line in input_log.read_text().splitlines():
                item = json.loads(line)
                key = item["dataset"], item["file"], item["mode"]
                selected.setdefault(key, {}).update({k: v for k, v in item.items() if v is not None})
        report["selected_inputs"] = list(selected.values())
        report["status"] = ("failed" if any(r["status"] == "failed" for r in report["analyses"])
                            else "complete" if all(r["status"] == "completed" for r in report["analyses"])
                            else "partial")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n{report['status'].upper()} run of requested analyses: {report_path}", flush=True)
        for row in report["analyses"]:
            print(f"  {row['status']}: {row['name']}", flush=True)
            for reason in row["reasons"]:
                print(f"    {reason}", flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", nargs="?", default="all",
                        choices=("all", "applications", "variations", "simulations", "empirical"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail when any requested analysis is unavailable.")
    parser.add_argument("--inputs", choices=("paper", "latest"), default=os.environ.get("WRDS_INPUTS", "paper"))
    parser.add_argument("--report", type=Path, default=OUT / "run_report.json")
    args = parser.parse_args()
    os.environ["WRDS_INPUTS"] = args.inputs
    os.environ["STELLA_REVENUE_FIELD"] = "revtq"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    analyses = list(requested_analyses(analysis_sections(), args.section))
    if args.plan:
        for section, analysis in analyses:
            print(f"[{section}] {analysis.name}")
            for command in analysis.commands:
                print(f"  {sys.executable} -m {' '.join(command)}")
        return 0
    return execute(analyses, args.report, strict=args.strict, input_mode=args.inputs)


if __name__ == "__main__":
    raise SystemExit(main())
