#!/usr/bin/env python3
"""Prepare the end-2025 CRSP common-stock panel from the project's CIZ exports.

Use dated security classifications equivalent to SIZ share codes 10/11, as in
WRDS's CIZtoSIZ macro. Keep native dlyret (including delisting records) and dlycap;
do not merge another delisting return. The 2000--2025 history covers the paper's
final 5040 trading days and their pre-sample weight date. No returns are imputed.
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mgp_core import _content_fingerprint
from project_paths import DERIVED_CACHE
from wrds_inputs import wrds_path

TARGET = DERIVED_CACHE / "crsp" / "daily_common_stocks_2000_2025.parquet"


def prepare() -> Path:
    daily = wrds_path("crsp_daily_stocks_ciz")
    history = wrds_path("crsp_security_history_ciz")
    signature = {"version": 1, "daily": _content_fingerprint(daily),
                 "history": _content_fingerprint(history), "end": "2025-12-31"}
    sidecar = TARGET.with_suffix(".json")
    if TARGET.exists() and sidecar.exists() and json.loads(sidecar.read_text())["inputs"] == signature:
        return TARGET
    h = pd.read_csv(history, usecols=["permno", "secinfostartdt", "secinfoenddt",
                    "sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype"])
    h["permno"] = h.permno.astype("int32")
    h["secinfostartdt"] = pd.to_datetime(h.secinfostartdt)
    h["secinfoenddt"] = pd.to_datetime(h.secinfoenddt).fillna(pd.Timestamp("2099-12-31"))
    h["common"] = (h.sharetype.eq("NS") & h.securitytype.eq("EQTY")
                   & h.securitysubtype.eq("COM") & h.usincflg.eq("Y")
                   & h.issuertype.isin(["ACOR", "CORP"]))
    h = h[["permno", "secinfostartdt", "secinfoenddt", "common"]].sort_values("secinfostartdt")
    if h.duplicated(["permno", "secinfostartdt"]).any():
        raise ValueError("Duplicate CRSP security-history start dates")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix(".partial")
    writer = None
    count = 0
    try:
        for j, d in enumerate(pd.read_csv(daily, usecols=["permno", "dlycaldt", "dlyret", "dlycap"],
                              dtype={"permno": "int32", "dlyret": "float64", "dlycap": "float64"},
                              chunksize=2_000_000), 1):
            d = d.loc[d.dlycaldt.between("2000-01-01", "2025-12-31")].copy()
            if d.empty:
                continue
            d["dlycaldt"] = pd.to_datetime(d.dlycaldt)
            d = pd.merge_asof(d.sort_values("dlycaldt"), h, by="permno",
                             left_on="dlycaldt", right_on="secinfostartdt", direction="backward")
            d = d.loc[d.common.eq(True) & d.dlycaldt.le(d.secinfoenddt)]
            d = d.rename(columns={"permno": "PERMNO", "dlycaldt": "date",
                                  "dlyret": "RET", "dlycap": "mkt_cap"})
            d = d.set_index(["date", "PERMNO"])[["RET", "mkt_cap"]]
            if d.empty:
                continue
            if d.index.duplicated().any():
                raise ValueError("Duplicate CRSP security-day records")
            table = pa.Table.from_pandas(d, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="snappy")
            writer.write_table(table)
            count += len(d)
            print(f"[CRSP] scanned {2*j:,} million source rows; retained {count:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if count == 0:
        raise ValueError("No CRSP common-stock observations")
    partial.replace(TARGET)
    sidecar.write_text(json.dumps({"inputs": signature, "rows": count,
        "classification_reference": "https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/",
        "returns": "Native CIZ dlyret; no additional delisting adjustment",
        "market_cap": "Native CIZ dlycap, USD thousands"}, indent=2) + "\n")
    return TARGET


if __name__ == "__main__":
    print(prepare())
