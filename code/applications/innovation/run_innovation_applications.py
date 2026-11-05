#!/usr/bin/env python3
"""Public innovation-flow applications: patent flows by firm/field and drugs by sponsor."""

from __future__ import annotations

import re

import pandas as pd

from mgp_core import ANALYSIS_BOOTSTRAP_DRAWS, ROOT, choose_solver
from project_paths import load_available
from applications.additional.common import Application, estimate, prepare_flows

RAW = ROOT / "data/additional_applications/raw"
KPSS = RAW / "kpss_extended/KPSS_2024.zip"
CPC = RAW / "kpss_extended/Match_patent_cpc_2024.zip"
PERMCO = RAW / "kpss_extended/Match_patent_permco_permno_2024.zip"
FDA = RAW / "fda_nme_new_biologic_1985_2025.xlsx"


def additive_flow_application(entity: str, outcome: str, source: str, benchmark: str,
                              levels: pd.DataFrame, labels: pd.Series, sample: str,
                              notes: str) -> Application:
    flows, mean = prepare_flows(levels)
    return Application(entity, outcome, source, benchmark, flows, mean,
                       labels.reindex(flows.index), sample, "Annual", "Full-sample mean flow",
                       notes + " Raw flow levels are retained without detrending or winsorization. "
                       "Full-sample mean flows determine sizes for grouping and aggregation; the aggregate is proportional to total flows.",
                       aggregation="sum")


def read_patents(*columns: str) -> pd.DataFrame:
    patents = pd.read_csv(KPSS, usecols=["patent_num", "issue_date", *columns])
    patents["year"] = pd.to_numeric(patents["issue_date"], errors="coerce").floordiv(10_000)
    return patents.loc[patents.year.between(1926, 2024)].dropna(subset=list(columns))


def fractional_cpc_matches(patents: pd.DataFrame) -> pd.DataFrame:
    """Match patents to unique CPC subclasses, retaining each patent's divisor."""
    cpc = pd.read_csv(CPC, usecols=["patent_num", "cpc"])
    cpc["field"] = cpc["cpc"].fillna("").str.split(";")
    cpc = cpc.explode("field")
    cpc["field"] = cpc["field"].str.extract(r"^([A-HY]\d{2}[A-Z])", expand=False)
    cpc = cpc.dropna().drop_duplicates(["patent_num", "field"])
    cpc["n_fields"] = cpc.groupby("patent_num")["field"].transform("size")
    return patents.merge(cpc, on="patent_num", how="inner", validate="one_to_many")


def patent_value_by_field() -> Application:
    d = fractional_cpc_matches(read_patents("xi_real"))
    # Fractional assignment prevents a multi-field patent from being counted repeatedly.
    d["field_value"] = d["xi_real"] / d["n_fields"]
    levels = d.pivot_table(index="field", columns="year", values="field_value", aggfunc="sum")
    levels = levels.reindex(columns=range(1926, 2025), fill_value=0.0)
    labels = pd.Series(levels.index, index=levels.index)
    return additive_flow_application(
        "Patent technology fields", "Real economic value of issued patents",
        "Kogan, Papanikolaou, Seru, and Stoffman (2017), extended KPSS 2024 data",
        "KPSS infer patent-level private value from the stock-market reaction to patent grants.",
        levels, labels, "1926–2024 issue years",
        "Patent-level xi_real (millions of 1982 dollars) is fractionally allocated across unique four-character "
        "CPC subclasses, so fields sum exactly to total value for patents with CPC matches. Public-company patents only."
    )


def patent_count_by_field() -> Application:
    d = fractional_cpc_matches(read_patents())
    d["field_count"] = 1.0 / d["n_fields"]
    levels = d.pivot_table(index="field", columns="year", values="field_count", aggfunc="sum")
    levels = levels.reindex(columns=range(1926, 2025), fill_value=0.0)
    labels = pd.Series(levels.index, index=levels.index)
    return additive_flow_application(
        "Patent-count fields", "Number of issued patents",
        "Kogan, Papanikolaou, Seru, and Stoffman (2017), extended KPSS 2024 data",
        "Issued patents are assigned to four-character CPC subclasses.",
        levels, labels, "1926–2024 issue years",
        "Each issued patent is fractionally allocated across its unique four-character CPC subclasses, "
        "so field counts add exactly to the number of patents with CPC matches."
    )


def patent_value_by_firm() -> Application:
    value = read_patents("xi_real")
    match = pd.read_csv(PERMCO, usecols=["patent_num", "permco"]).drop_duplicates()
    # The authors omit patents assigned to multiple CRSP securities; collapse the
    # remaining security links to CRSP's company identifier before aggregation.
    d = value.merge(match, on="patent_num", how="inner", validate="one_to_one")
    levels = d.pivot_table(index="permco", columns="year", values="xi_real", aggfunc="sum")
    levels = levels.reindex(columns=range(1926, 2025), fill_value=0.0)
    labels = pd.Series("CRSP PERMCO " + levels.index.astype(str), index=levels.index)
    return additive_flow_application(
        "Patent-value firms", "Real economic value of issued patents",
        "Kogan, Papanikolaou, Seru, and Stoffman (2017), extended KPSS 2024 data",
        "KPSS infer patent-level private value from the stock-market reaction to patent grants.",
        levels, labels, "1926–2024 issue years",
        "Patent-level xi_real (millions of 1982 dollars) is summed by CRSP permco and issue year. "
        "The authors exclude patents matched to multiple CRSP securities; zero-flow years are retained."
    )


def clean_sponsor(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    # Remove legal suffixes while preserving FDA's underlying applicant names.
    return re.sub(r"\b(INC|LLC|LTD|CORP|CORPORATION|CO|COMPANY|LP|PLC)\b", "", text).strip()


def novel_drugs_by_sponsor() -> Application:
    d = pd.read_excel(FDA, sheet_name="2025 Compilation", usecols=["Applicant", "Approval Year"])
    d["year"] = pd.to_numeric(d["Approval Year"], errors="coerce")
    d["sponsor"] = d["Applicant"].map(clean_sponsor)
    d = d[d["year"].between(1985, 2025) & d["sponsor"].ne("")]
    levels = d.assign(n=1.0).pivot_table(index="sponsor", columns="year", values="n", aggfunc="sum")
    levels = levels.reindex(columns=range(1985, 2026), fill_value=0.0)
    labels = pd.Series(levels.index.str.title(), index=levels.index)
    return additive_flow_application(
        "Novel-drug sponsors", "Novel drug and new biologic approvals",
        "FDA Compilation of CDER NME Drug and New Biologic Approvals, 1985–2025",
        "FDA defines the file as original CDER Type 1 or Type 1/4 NME and new-biologic approvals.",
        levels, labels, "1985–2025 approval years",
        "One count per FDA compilation row, assigned to the applicant recorded at original approval. Corporate suffixes "
        "and punctuation are normalized, but acquisitions and historical sponsor-name changes are not consolidated."
    )


def main() -> None:
    solver = choose_solver("auto")
    rows = []
    sources = (
        ("Patent-value firms", patent_value_by_firm),
        ("Patent-count fields", patent_count_by_field),
        ("Patent technology fields", patent_value_by_field),
        ("Novel-drug sponsors", novel_drugs_by_sponsor),
    )
    for app in load_available(sources):
        row, _ = estimate(app, solver, bootstrap_b=ANALYSIS_BOOTSTRAP_DRAWS, seed=20260907)
        rows.append(row)
        print(f"{app.entity}: N={row['N']}, T={row['T']}, M={row['M']}, "
              f"share={row['MGP_share']:.4f}, UCL={row['UCL95']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    cache = ROOT / "code/.cache/run_outputs/innovation_applications_summary.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache)


if __name__ == "__main__":
    main()
