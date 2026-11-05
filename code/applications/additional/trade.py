"""US imports and exports by product and partner from UN Comtrade."""

from __future__ import annotations

import json

import pandas as pd

from .common import RAW, Application, balanced_log_growth


PARTNERS = RAW / "un_comtrade_partner_areas.json"


def load_trade(flow: str, dimension: str) -> Application:
    d = pd.read_csv(RAW / f"un_comtrade_us_{flow}_{dimension}_1995_2024.csv", dtype={"cmdCode": str, "partnerCode": str})
    is_exports = flow == "exports"
    if dimension == "hs2":
        unit, label = "cmdCode", "cmdDesc"
        entity = "Exported products" if is_exports else "Imported products"
    else:
        valid = {str(r["PartnerCode"]): r["PartnerDesc"] for r in json.loads(PARTNERS.read_text())["results"]
                 if r.get("PartnerCodeIsoAlpha3", "").isalpha() and not r.get("isGroup", False)}
        d = d[d["partnerCode"].isin(valid)]
        unit, label = "partnerCode", "partnerDesc"
        entity = "Export destinations" if is_exports else "Import origins"
    values = d.pivot_table(index=unit, columns="year", values="primaryValue", aggfunc="sum")
    y = balanced_log_growth(values, 1995, 2024)
    labels = d.drop_duplicates(unit).set_index(unit)[label]
    return Application(
        entity, f"US {'export' if is_exports else 'import'}-value growth", "UN Comtrade, annual HS trade",
        "Product and partner-country decompositions complement di Giovanni, Levchenko & Mejean (2014) and Wagner (2014).",
        y, values.loc[y.index, 1995], labels, "1995–2024 levels; 1996–2024 growth", "Annual", "1995",
        f"Balanced {'HS two-digit product' if dimension == 'hs2' else 'partner-country'} panel; fixed 1995 nominal trade weights.",
    )
