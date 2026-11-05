"""Novel-drug and biologic approvals by sponsor from FDA."""

import re
import pandas as pd
from project_paths import ROOT
from applications.common import Application, additive_flow_application


FDA = ROOT / "data/fda/fda_nme_new_biologic_1985_2025.xlsx"


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
