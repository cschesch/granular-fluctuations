#!/usr/bin/env python3
"""Render the sectioned applications table from current analysis results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from compute_applications_coefficients import CACHE, core_estimates, summary_path, summary_rows
from mgp_core import OUTPUT_DIR as OUT

SECTIONS = [
    ("Firms", [
        "Sales (quarterly)", "Sales (annual)",
        "Employment", "Productivity (quarterly)", "Productivity (annual)",
    ]),
    ("Returns", ["Stock returns", "Mutual-fund returns", "Billionaire wealth"]),
    ("Financial flows", [
        "Bank lending", "Corporate bond issuance", "Mutual-fund flows",
        "Foreign Treasury holdings",
    ]),
    ("Trade", [
        "Exports by product", "Imports by product", "Exports by destination",
        "Imports by origin",
    ]),
    ("Prices and labor", ["Consumer prices", "Sector employment"]),
    ("Innovation", ["Patents by firm", "Patent value by firm", "Patent count by field",
                    "Patent value by field", "Novel-drug approvals"]),
    ("Geography", ["Metro GDP", "County GDP", "State GDP"]),
    ("Elections", ["Democratic votes by state", "Democratic votes by county"]),
    ("Energy and climate", [
        "Plant generation", "Plant CO2 emissions", "Oil production",
        "CO2 emissions by country",
    ]),
    ("World economy", ["World GDP", "Country market cap", "Public debt", "Private debt"]),
    ("Sector networks", ["No I/O network or lags", "Foerster et al. (2011)",
                         "Acemoglu et al. (2012)", "Carvalho--Gabaix (2013)", "Atalay (2017)"]),
    ("Firm networks", ["Stella (2015)", "Herskovic et al. (2020)",
                       "Welburn et al. (2020)"]),
]

CORE_SPECS = {
    "Sales (quarterly)": ("compustat_qtr", "Firms", "Compustat", "Q"),
    "Sales (annual)": ("compustat_ann_sales", "Firms", "Compustat", "A"),
    "Productivity (quarterly)": ("compustat_qtr_tfp", "Firms", "Compustat", "Q"),
    "Productivity (annual)": ("compustat_ann_tfp", "Firms", "Compustat", "A"),
    "Employment": ("compustat_ann_emp", "Firms", "Compustat", "A"),
    "Stock returns": ("crsp", "Stocks", "CRSP", "D"),
    "Billionaire wealth": ("forbes_full", "Billionaires", "Forbes", "D"),
    "Consumer prices": ("cpi1967", "Items", "BLS CPI", "M"),
    "Sector employment": ("bls_ces", "Sectors", "BLS CES", "M"),
}

ADDITIONAL_ENTITIES = {
    "Bank lending": "Banks",
    "Mutual-fund returns": "Mutual fund portfolios: returns",
    "Mutual-fund flows": "Mutual fund portfolios: flows",
    "Corporate bond issuance": "US corporate bond issuers",
    "Foreign Treasury holdings": "Foreign Treasury holders",
    "State GDP": "US states",
    "County GDP": "US counties",
    "Metro GDP": "Metropolitan areas",
    "Exports by product": "Exported products",
    "Imports by product": "Imported products",
    "Exports by destination": "Export destinations",
    "Imports by origin": "Import origins",
    "World GDP": "Countries",
    "Public debt": "Public debt by country",
    "Private debt": "Private nonfinancial debt by country",
    "Country market cap": "Country equity markets",
    "Patents by firm": "Patent assignees",
    "Patent value by firm": "Patent-value firms",
    "Patent count by field": "Patent-count fields",
    "Patent value by field": "Patent technology fields",
    "Novel-drug approvals": "Novel-drug sponsors",
    "Plant generation": "Power plants",
    "Plant CO2 emissions": "Power-plant emissions",
    "Oil production": "Oil-producing countries",
    "CO2 emissions by country": "Country CO2 emissions",
}

ADDITIONAL_DISPLAY = {
    "Bank lending": ("Banks", "FDIC"),
    "Mutual-fund returns": ("Mutual funds", "CRSP"),
    "Mutual-fund flows": ("Mutual funds", "CRSP"),
    "Corporate bond issuance": ("Issuers", "LSEG SDC"),
    "Foreign Treasury holdings": ("Foreign holders", "Treasury TIC"),
    "State GDP": ("States", "BEA"),
    "County GDP": ("Counties", "BEA"),
    "Metro GDP": ("Metros", "BEA"),
    "Exports by product": ("Products", "UN Comtrade"),
    "Imports by product": ("Products", "UN Comtrade"),
    "Exports by destination": ("Countries", "UN Comtrade"),
    "Imports by origin": ("Countries", "UN Comtrade"),
    "World GDP": ("Countries", "Maddison Project"),
    "Public debt": ("Countries", "GMD"),
    "Private debt": ("Countries", "IMF GDD"),
    "Country market cap": ("Countries", "French/WB"),
    "Patents by firm": ("Firms", "Ma"),
    "Patent value by firm": ("Firms", "KPSS"),
    "Patent count by field": ("Fields", "KPSS"),
    "Patent value by field": ("Fields", "KPSS"),
    "Novel-drug approvals": ("Firms", "FDA CDER"),
    "Plant generation": ("Plants", "EPA CAMPD"),
    "Plant CO2 emissions": ("Plants", "EPA CAMPD"),
    "Oil production": ("Countries", "Energy Institute"),
    "CO2 emissions by country": ("Countries", "GCB"),
}

OUTCOME_DISPLAY = {
    "Sales (quarterly)": "Sales", "Sales (annual)": "Sales",
    "Productivity (quarterly)": "Productivity",
    "Productivity (annual)": "Productivity", "Employment": "Employment",
    "Stock returns": "Stock returns", "Mutual-fund returns": "Mutual fund returns",
    "Billionaire wealth": "Wealth", "Bank lending": "Bank lending",
    "Mutual-fund flows": "Mutual fund flows", "Corporate bond issuance": "Bond issuance",
    "Foreign Treasury holdings": "Treasury holdings", "Consumer prices": "Inflation",
    "Sector employment": "Employment", "State GDP": "Real GDP", "County GDP": "Real GDP",
    "Metro GDP": "GDP", "Exports by product": "Exports", "Imports by product": "Imports",
    "Exports by destination": "Exports", "Imports by origin": "Imports",
    "World GDP": "Real GDP", "Public debt": "Public debt", "Patents by firm": "Patent count",
    "Private debt": "Private debt", "Country market cap": "Market returns",
    "Patent value by firm": "Patent value", "Patent count by field": "Patent count",
    "Patent value by field": "Patent value", "Novel-drug approvals": "Drug approvals",
    "Plant generation": "Power generation", "Plant CO2 emissions": "CO2 emissions",
    "Oil production": "Oil production", "CO2 emissions by country": "CO2 emissions",
    "Democratic votes by state": "Democratic votes",
    "Democratic votes by county": "Democratic votes",
    "No I/O network or lags": "TFP shocks",
    "Foerster et al. (2011)": "Primitive shocks",
    "Acemoglu et al. (2012)": "Primitive shocks",
    "Carvalho--Gabaix (2013)": "Primitive shocks",
    "Atalay (2017)": "Primitive shocks",
    "Stella (2015)": "Output shocks",
    "Welburn et al. (2020)": "Output shocks",
    "Herskovic et al. (2020)": "Demand shocks",
}

def short_frequency(value: object) -> str:
    text = str(value)
    if "presidential" in text.lower():
        return "4Y"
    for key, code in (("Quarterly", "Q"), ("Monthly", "M"), ("Weekly", "W"),
                      ("Daily", "D"), ("Annual", "A")):
        if key.lower() in text.lower():
            return code
    return text


def short_sample(value: object) -> str:
    """Reduce a sample description to its leading date range.

    Sources describe their window as e.g. "2000Q4-2025Q4 levels; four-quarter
    growth 2001Q4-2025Q4" or "1926-2024 issue years"; the table wants the range.
    """
    text = str(value).replace("–", "--").split(";")[0].strip()
    return re.sub(r"\s+(?:levels|(?:\w+\s+)?(?:years|dates))$", "", text)


def fmt_int(value: object) -> str:
    return r"\textemdash{}" if pd.isna(value) else f"{int(value):,}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return r"\textemdash{}"
    scaled = 100 * float(value)
    return rf"{scaled:.{digits}f}\%"


def fmt_ucl(high: object) -> str:
    if pd.isna(high):
        return r"\textemdash{}"
    hi = min(100.0, 100 * float(high))
    return rf"{hi:.1f}\%"


def fmt_hhi(value: object) -> str:
    if pd.isna(value):
        return r"\textemdash{}"
    scaled = 100 * float(value)
    digits = 2 if scaled < 0.1 else 1
    return rf"{scaled:.{digits}f}\%"


def split_sample(value: object) -> str:
    """Two physical cells around a fixed intercolumn en dash."""
    if pd.isna(value):
        return r"\multicolumn{2}{c}{\textemdash{}}"
    years = re.findall(r"(?:19|20)\d{2}", str(value))
    if len(years) < 2:
        return rf"\multicolumn{{2}}{{c}}{{{value}}}"
    return f"{years[0]} & {years[-1]}"


def fmt_text(value: object) -> str:
    return r"\textemdash{}" if pd.isna(value) else str(value)


ROW_FIELDS = (
    "outcome", "unit", "sample", "frequency", "T", "N", "M", "hhi",
    "naive", "estimate", "ci_low", "ci_high",
)


def blank_row(section: str, application: str) -> dict[str, object]:
    return {key: pd.NA for key in ROW_FIELDS} | {
        "section": section, "application": application, "source": pd.NA,
        "outcome": OUTCOME_DISPLAY[application],
    }


def core_rows() -> dict[str, dict[str, object]]:
    results = core_estimates()
    out = {}
    for application, (stem, unit, source, frequency) in CORE_SPECS.items():
        fields = out[application] = dict(unit=unit, source=source, frequency=frequency)
        if stem not in results.index:
            continue
        row = results.loc[stem]
        fields.update({
            "sample": f"{row.start}--{row.end}",
            "T": row["T"], "N": row["N"], "M": row["M"],
            "hhi": row.hhi,
            "naive": row.naive_homoscedastic,
            "estimate": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high,
        })
    return out


def additional_rows() -> dict[str, dict[str, object]]:
    by_entity = summary_rows("additional_applications")
    out = {}
    for application, entity in ADDITIONAL_ENTITIES.items():
        unit, source = ADDITIONAL_DISPLAY[application]
        fields = out[application] = dict(unit=unit, source=source)
        if entity not in by_entity.index:
            continue
        row = by_entity.loc[entity]
        m = int(row["M"])
        fields.update({
            "sample": short_sample(row["Sample"]),
            "frequency": short_frequency(row["Frequency"]),
            "T": row["T"], "N": row["N"], "M": m,
            "hhi": float(row["HHI"]),
            "naive": row["Naive_homoscedastic"],
            "estimate": row["MGP_share"],
            "ci_low": row["CI90_low"], "ci_high": row["CI90_high"],
        })
    return out


def election_rows() -> dict[str, dict[str, object]]:
    data = summary_rows("election_applications")
    out = {}
    for application in ("Democratic votes by state", "Democratic votes by county"):
        fields = out[application] = {
            "unit": "States" if application.endswith("state") else "Counties",
            "source": r"Amlani \& Algara",
        }
        if application not in data.index:
            continue
        row = data.loc[application]
        fields.update({
            "sample": row["Sample"], "frequency": short_frequency(row["Frequency"]),
            "T": row["T"], "N": row["N"], "M": row["M"], "hhi": row["HHI"],
            "naive": row["Naive_homoscedastic"],
            "estimate": row["MGP_share"], "ci_low": row["CI90_low"], "ci_high": row["CI90_high"],
        })
    return out


NETWORK_SPECS = {
    "No I/O network or lags": ("bea_bls_sector_tfp_no_network", "Sectors", "BEA--BLS"),
    "Foerster et al. (2011)": ("foerster_net", "Sectors", "Foerster et al."),
    "Acemoglu et al. (2012)": ("acemoglu2012_bea1992", "Sectors", "Acemoglu et al."),
    "Carvalho--Gabaix (2013)": ("carvalho_gabaix_bea_bls_historical", "Sectors", r"Carvalho \& Gabaix"),
    "Atalay (2017)": ("atalay_epsilon_network", "Sectors", "Atalay"),
    "Stella (2015)": ("stella_revtq_firm_network", "Firms", "Stella"),
    "Welburn et al. (2020)": ("welburn_firm_network_M10", "Firms", "Welburn et al."),
    "Herskovic et al. (2020)": ("herskovic_firm_network", "Firms", "Herskovic et al."),
}
APPROXIMATE_NETWORK_APPLICATIONS = {
    "Acemoglu et al. (2012)",
    "Carvalho--Gabaix (2013)",
    "Stella (2015)",
    "Herskovic et al. (2020)",
    "Welburn et al. (2020)",
}


def network_rows() -> dict[str, dict[str, object]]:
    plotted = pd.read_csv(CACHE / "applications_coefficients_expanded_data.csv").set_index("item")
    out = {}
    for application, (stem, unit, source) in NETWORK_SPECS.items():
        display_source = source + ("*" if application in APPROXIMATE_NETWORK_APPLICATIONS else "")
        fields = out[application] = dict(unit=unit, source=display_source)
        path = summary_path(stem)
        if path is None:
            continue
        row = pd.read_pickle(path).iloc[0]
        plot_name = application.replace("--", "–")
        estimate = plotted.loc[plot_name]
        fields.update({
            "sample": f"{row['start_date']}--{row['last_date']}",
            "frequency": short_frequency(row.get("frequency", "Annual")),
            "T": row["T"], "N": row["N"], "M": row["M"], "hhi": row.get("HHI", pd.NA),
            "naive": estimate["homoskedastic"], "estimate": estimate["estimate"],
            "ci_low": estimate["low"], "ci_high": estimate["high"],
        })
    return out


def build_rows() -> list[dict[str, object]]:
    available = core_rows() | additional_rows() | election_rows() | network_rows()
    rows = []
    for section, applications in SECTIONS:
        for application in applications:
            row = blank_row(section, application)
            row.update(available.get(application, {}))
            rows.append(row)
    return rows


def latex_rows(rows: list[dict[str, object]], sections: list[str]) -> str:
    lines = []
    for section in sections:
        section_rows = [item for item in rows if item["section"] == section]
        lines.append(r"\addlinespace[3pt]")
        lines.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\textit{{{section}}}}} \\")
        for row in section_rows:
            outcome = fmt_text(row["outcome"]).replace("CO2", r"CO$_2$")
            lines.append(
                f"{outcome} & {fmt_text(row['unit'])} & {fmt_text(row['source'])} & "
                f"{fmt_text(row['frequency'])} & {split_sample(row['sample'])} & "
                f"{fmt_int(row['T'])} & {fmt_int(row['N'])} & {fmt_int(row['M'])} & "
                f"{fmt_hhi(row['hhi'])} & {fmt_pct(row['naive'])} & "
                f"{fmt_pct(row['estimate'])} & {fmt_ucl(row['ci_high'])} " + r"\\"
            )
    return "\n".join(lines)


def write_table(rows: list[dict[str, object]], sections: list[str], path: Path) -> None:
    header = r"""{\setlength{\tabcolsep}{3pt}%
\renewcommand{\arraystretch}{1.07}%
\begin{tabular}{@{}lll@{\hspace{2pt}}c r@{--}l rrr rrr r@{}}
\toprule
Outcome & Unit & Source & Freq. & \multicolumn{2}{c}{Sample} & $T$ & $N$ & $M$ & HHI & $\sigma_x^2\!\cdot\!\mathrm{HHI}/\sigma_y^2$ & $\hat{\bar\gamma}$ & \multicolumn{1}{c}{95\% UCL} \\
\midrule
"""
    path.write_text(
        "% Generated by code/build_applications_vertical.py. Do not edit by hand.\n"
        + header + latex_rows(rows, sections) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}}" + "\n"
    )


def main() -> None:
    rows = build_rows()
    write_table(rows, [name for name, _ in SECTIONS], OUT / "table_applications_vertical.tex")
    pd.DataFrame(rows).to_csv(CACHE / "table_applications_vertical_data.csv", index=False)
    print(pd.DataFrame(rows)[["section", "application", "N", "T", "M",
                              "estimate", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
