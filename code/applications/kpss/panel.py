"""Patent counts and economic values from the KPSS public release."""

import pandas as pd
from project_paths import ROOT
from applications.common import Application, additive_flow_application


RAW = ROOT / "data/kpss"
KPSS = RAW / "KPSS_2024.zip"
CPC = RAW / "Match_patent_cpc_2024.zip"
PERMCO = RAW / "Match_patent_permco_permno_2024.zip"


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
