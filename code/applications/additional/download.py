"""Opt-in acquisition of public inputs for additional applications."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .common import RAW
from .finance import FDIC
from .geography import BEA_COUNTIES, BEA_STATES, FIGSHARE_CITIES
from .production import CAMPD, EIA
from .trade import PARTNERS


def fetch(url: str, destination: Path, retries: int = 12, accept: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            headers = {"User-Agent": "granularity-replication/1.0"}
            if accept:
                headers["Accept"] = accept
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def download_campd_data() -> None:
    """Prepend 1999Q4 and 2000 while preserving archived later observations."""
    if CAMPD.exists():
        return
    baseline = RAW / "epa_campd_facility_quarterly_2000_2024.csv"
    legacy = RAW / "epa_campd_facility_quarterly_2001_2024.csv"
    first_year = RAW / "epa_campd_facility_quarterly_2000.csv"
    if not baseline.exists():
        destination = first_year if legacy.exists() else baseline
        if not destination.exists():
            params = urllib.parse.urlencode({
                "api_key": "DEMO_KEY",
                "year": "2000" if legacy.exists() else "|".join(map(str, range(2000, 2025))),
                "quarter": "1|2|3|4",
            })
            fetch("https://api.epa.gov/easey/streaming-services/emissions/apportioned/quarterly/by-facility?" + params,
                  destination, accept="text/csv")
        if legacy.exists():
            frames = [pd.read_csv(path, dtype={"Facility ID": str}) for path in (first_year, legacy)]
            if set(frames[0]["Year"]) != {2000}:
                raise ValueError("The CAMPD 2000 response must contain only 2000 observations")
            pd.concat(frames, ignore_index=True).to_csv(baseline, index=False)
    initial = RAW / "epa_campd_facility_quarterly_1999q4.csv"
    if not initial.exists():
        params = urllib.parse.urlencode({"api_key": "DEMO_KEY", "year": "1999", "quarter": "4"})
        fetch("https://api.epa.gov/easey/streaming-services/emissions/apportioned/quarterly/by-facility?" + params,
              initial, accept="text/csv")
    early = pd.read_csv(initial, dtype={"Facility ID": str})
    if set(early["Year"]) != {1999} or set(early["Quarter"]) != {4}:
        raise ValueError("The initial CAMPD response must contain only 1999Q4 observations")
    combined = pd.concat([early, pd.read_csv(baseline, dtype={"Facility ID": str})], ignore_index=True)
    if combined.duplicated(["Facility ID", "Year", "Quarter"]).any():
        raise ValueError("Duplicate facility-quarter records in the CAMPD extension")
    temporary = CAMPD.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(CAMPD)


def download_fdic_data() -> None:
    """Assemble quarterly inputs, paginating dates with 10,000+ banks."""
    fdic_frames = []
    # The assembled input also serves as a cache, avoiding reads of individual
    # quarter files that Dropbox may have moved out of local storage.
    previous = RAW / "fdic_quarterly_financials_2000q4_2025q4.csv"
    cached_path = FDIC if FDIC.exists() else previous
    cached = (dict(tuple(pd.read_csv(cached_path, dtype={"CERT": str, "REPDTE": str})
                         .groupby("REPDTE"))) if cached_path.exists() else {})
    quarter_ends = [f"{year}{md}" for year in range(1999, 2026)
                    for md in ("0331", "0630", "0930", "1231")]
    for qend in (q for q in quarter_ends if "19991231" <= q <= "20251231"):
        legacy = RAW / "fdic_quarters" / f"fdic_{qend}.csv"
        frame = cached.get(qend)
        if frame is None and legacy.exists():
            frame = pd.read_csv(legacy, dtype={"CERT": str, "REPDTE": str})
        if frame is None or len(frame) >= 10000:
            # Preserve existing raw responses. Fetch sorted pages separately:
            # the legacy 10,000-row response can be truncated at early dates.
            pages = []
            offset = 0
            while True:
                target = RAW / "fdic_quarters" / f"fdic_{qend}_offset{offset}.csv"
                if not target.exists():
                    params = urllib.parse.urlencode({
                        "filters": f"REPDTE:{qend}",
                        "fields": "CERT,REPDTE,NAME,ASSET,LNLSNET",
                        "limit": 10000, "offset": offset, "format": "csv",
                        "sort_by": "CERT", "sort_order": "ASC",
                    })
                    fetch(f"https://api.fdic.gov/banks/financials?{params}", target)
                page = pd.read_csv(target, dtype={"CERT": str, "REPDTE": str})
                pages.append(page)
                if len(page) < 10000:
                    break
                offset += 10000
            frame = pd.concat(pages, ignore_index=True)
        if frame.empty or not frame["REPDTE"].eq(qend).all() or frame["CERT"].duplicated().any():
            raise ValueError(f"Invalid or duplicate FDIC records for {qend}")
        fdic_frames.append(frame)
    pd.concat(fdic_frames, ignore_index=True).to_csv(FDIC, index=False)


def download_public_data(include_electricity: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not FIGSHARE_CITIES.exists():
        fetch("https://ndownloader.figshare.com/files/38492837", FIGSHARE_CITIES)

    download_fdic_data()

    if not PARTNERS.exists():
        fetch("https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json", PARTNERS)
    for flow, flow_name in (("X", "exports"), ("M", "imports")):
        for dimension, cmd, partner in (("hs2", "AG2", 0), ("partners", "TOTAL", None)):
            frames = []
            for year in range(1995, 2025):
                query = {"reporterCode": 842, "period": year, "cmdCode": cmd, "flowCode": flow,
                         "partner2Code": 0, "customsCode": "C00", "motCode": 0, "maxRecords": 500}
                if partner is not None:
                    query["partnerCode"] = partner
                target = RAW / "comtrade_years" / f"us_{flow_name}_{dimension}_{year}.json"
                legacy = RAW / "comtrade_years" / f"us_exports_hs2_{year}.json"
                if not target.exists() and flow_name == "exports" and dimension == "hs2" and legacy.exists():
                    target.write_bytes(legacy.read_bytes())
                if not target.exists():
                    fetch(f"https://comtradeapi.un.org/public/v1/preview/C/A/HS?{urllib.parse.urlencode(query)}", target)
                    time.sleep(1.25)
                records = json.loads(target.read_text()).get("data", [])
                if not records:
                    raise RuntimeError(f"UN Comtrade returned no {flow_name}/{dimension} records for {year}")
                frame = pd.DataFrame(records)
                frame["year"] = year
                frames.append(frame)
            keep = ["year", "cmdCode", "cmdDesc", "partnerCode", "partnerDesc", "primaryValue"]
            pd.concat(frames, ignore_index=True).reindex(columns=keep).to_csv(
                RAW / f"un_comtrade_us_{flow_name}_{dimension}_1995_2024.csv", index=False)

    if not BEA_STATES.exists():
        fetch("https://apps.bea.gov/regional/zip/SAGDP.zip", BEA_STATES)
    if not BEA_COUNTIES.exists():
        fetch("https://apps.bea.gov/regional/zip/CAGDP1.zip", BEA_COUNTIES)
    download_campd_data()

    eia_frames = []
    for year in (range(2001, 2025) if include_electricity else []):
        offset, total = 0, 1
        while offset < total:
            target = RAW / "eia_years" / f"plant_generation_{year}_{offset}.json"
            legacy = RAW / "eia_years" / f"plant_generation_{year}.json"
            if offset == 0 and legacy.exists() and not target.exists():
                target.write_bytes(legacy.read_bytes())
            if not target.exists():
                query = urllib.parse.urlencode({"api_key": "DEMO_KEY", "frequency": "annual", "data[0]": "generation",
                                                "start": year, "end": year, "facets[fuel2002][]": "ALL",
                                                "length": 5000, "offset": offset})
                fetch(f"https://api.eia.gov/v2/electricity/facility-fuel/data/?{query}", target)
                time.sleep(2)
            payload = json.loads(target.read_text())["response"]
            total = int(payload["total"])
            eia_frames.extend(payload["data"])
            offset += 5000
    if eia_frames:
        pd.DataFrame(eia_frames).to_csv(EIA, index=False)
