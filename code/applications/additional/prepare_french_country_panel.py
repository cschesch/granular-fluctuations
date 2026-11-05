#!/usr/bin/env python3
"""Build French/WDI country-equity panels and fixed market-cap weights.

Raw inputs are the Kenneth French international-country and U.S. factor ZIP
archives plus World Bank indicator CM.MKT.LCAP.CD.  The output return panel is
in decimal units and begins with the earliest common French vintage, January
1975; fixed end-1975 market-capitalization weights determine the cross section.
The estimation loader starts in January 1976, after the weight date.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


from project_paths import DERIVED_CACHE, ROOT


RAW = ROOT / "data" / "additional_applications" / "raw"
FRENCH_RAW = RAW / "french_country_returns"
DERIVED = DERIVED_CACHE / "additional_applications"

INTERNATIONAL_ZIP = FRENCH_RAW / "F-F_International_Countries.zip"
US_ZIP = FRENCH_RAW / "F-F_Research_Data_Factors_CSV.zip"
WORLD_BANK_JSON = FRENCH_RAW / "world_bank_CM.MKT.LCAP.CD_20_markets_1960_2025.json"

RETURNS_OUT = DERIVED / "french_country_market_returns_monthly_1975_2025.csv"
WEIGHTS_1975_OUT = DERIVED / "world_bank_country_market_cap_weights_1975.csv"
MANIFEST_OUT = DERIVED / "french_country_panel_manifest.json"

COUNTRIES = {
    "Australia": ("AUS", "Austrlia.Dat"),
    "Austria": ("AUT", "Austria.Dat"),
    "Belgium": ("BEL", "Belgium.Dat"),
    "Canada": ("CAN", "Canada.Dat"),
    "Denmark": ("DNK", "Denmark.Dat"),
    "Finland": ("FIN", "Finland.Dat"),
    "France": ("FRA", "France.Dat"),
    "Germany": ("DEU", "Germany.Dat"),
    "Hong Kong": ("HKG", "HongKong.Dat"),
    "Italy": ("ITA", "Italy.Dat"),
    "Japan": ("JPN", "Japan.Dat"),
    "Netherlands": ("NLD", "Nethrlnd.Dat"),
    "New Zealand": ("NZL", "NewZland.Dat"),
    "Norway": ("NOR", "Norway.Dat"),
    "Singapore": ("SGP", "Singapor.Dat"),
    "Spain": ("ESP", "Spain.Dat"),
    "Sweden": ("SWE", "Sweden.Dat"),
    "Switzerland": ("CHE", "Swtzrlnd.Dat"),
    "United Kingdom": ("GBR", "UK.Dat"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_monthly_market_return(text: str) -> pd.Series:
    """Read Mkt from the first monthly table in a French country .Dat file."""
    observations: dict[pd.Timestamp, float] = {}
    started = False
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{6})\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            started = True
            period, value = match.groups()
            observations[pd.to_datetime(period, format="%Y%m")] = float(value) / 100
        elif started:
            break
    if not observations:
        raise ValueError("No monthly observations found")
    return pd.Series(observations, dtype=float)


def us_market_return(zip_path: Path) -> pd.Series:
    with zipfile.ZipFile(zip_path) as archive:
        member = archive.namelist()[0]
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    rows = []
    for row in csv.reader(io.StringIO("\n".join(lines[header + 1 :]))):
        if not row or not re.fullmatch(r"\d{6}", row[0].strip()):
            break
        rows.append((pd.to_datetime(row[0].strip(), format="%Y%m"),
                     (float(row[1]) + float(row[4])) / 100))
    if not rows:
        raise ValueError("No monthly U.S. factor observations found")
    return pd.Series(dict(rows), dtype=float)


def build_returns(eligible_countries: pd.Index) -> pd.DataFrame:
    series = {}
    with zipfile.ZipFile(INTERNATIONAL_ZIP) as archive:
        for country, (_, member) in COUNTRIES.items():
            text = archive.read(member).decode("latin-1")
            series[country] = first_monthly_market_return(text)
    series["United States"] = us_market_return(US_ZIP)
    expected = pd.date_range("1975-01-01", "2025-12-01", freq="MS")
    panel = pd.DataFrame(series).reindex(expected)
    panel = panel.loc[:, panel.columns.intersection(eligible_countries, sort=False)]
    panel = panel.dropna(axis=1)
    if panel.shape != (612, 12):
        raise ValueError(f"Expected a complete 612 x 12 panel, got {panel.shape}")
    panel.index.name = "date"
    panel.to_csv(RETURNS_OUT, date_format="%Y-%m")
    return panel


def build_weights(year: int, destination: Path) -> pd.DataFrame:
    payload = json.loads(WORLD_BANK_JSON.read_text())
    records = payload[1]
    iso_to_country = {iso: country for country, (iso, _) in COUNTRIES.items()}
    iso_to_country["USA"] = "United States"
    values = {
        record["countryiso3code"]: float(record["value"])
        for record in records
        if record.get("date") == str(year)
        and record.get("countryiso3code") in iso_to_country
        and record.get("value") is not None
    }
    if not values:
        raise ValueError(f"No {year} WDI values")
    weights = pd.DataFrame(
        [
            {"country": iso_to_country[iso], "iso3": iso, "market_cap_current_usd": value}
            for iso, value in values.items()
        ]
    )
    weights["weight"] = weights["market_cap_current_usd"] / weights["market_cap_current_usd"].sum()
    weights = weights.sort_values("weight", ascending=False).reset_index(drop=True)
    weights.to_csv(destination, index=False)
    return weights


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    weights_1975 = build_weights(1975, WEIGHTS_1975_OUT)
    panel = build_returns(pd.Index(weights_1975["country"]))
    wb_metadata = json.loads(WORLD_BANK_JSON.read_text())[0]
    manifest = {
        "created_from": {
            "kenneth_french_international": {
                "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Countries.zip",
                "path": str(INTERNATIONAL_ZIP.relative_to(ROOT)),
                "sha256": sha256(INTERNATIONAL_ZIP),
            },
            "kenneth_french_us_factors": {
                "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
                "path": str(US_ZIP.relative_to(ROOT)),
                "sha256": sha256(US_ZIP),
                "construction": "US market return = (Mkt-RF + RF) / 100",
            },
            "world_bank_market_cap": {
                "indicator": "CM.MKT.LCAP.CD",
                "url": (
                    "https://api.worldbank.org/v2/country/"
                    "AUS;AUT;BEL;CAN;CHE;DEU;DNK;ESP;FIN;FRA;GBR;HKG;ITA;JPN;"
                    "NLD;NOR;NZL;SGP;SWE;USA/indicator/CM.MKT.LCAP.CD?"
                    "format=json&per_page=20000&date=1960:2025"
                ),
                "path": str(WORLD_BANK_JSON.relative_to(ROOT)),
                "sha256": sha256(WORLD_BANK_JSON),
                "api_last_updated": wb_metadata.get("lastupdated"),
            },
        },
        "returns": {
            "path": str(RETURNS_OUT.relative_to(ROOT)),
            "units": "decimal simple monthly return",
            "start": panel.index.min().strftime("%Y-%m"),
            "end": panel.index.max().strftime("%Y-%m"),
            "observations": len(panel),
            "markets": panel.shape[1],
        },
        "weights": {
            "1975": {
                "path": str(WEIGHTS_1975_OUT.relative_to(ROOT)), "date": "1975-12-31",
                "units": "current US dollars", "total_market_cap_usd": weights_1975["market_cap_current_usd"].sum(),
                "weight_sum": weights_1975["weight"].sum(), "hhi": (weights_1975["weight"] ** 2).sum(),
            }
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {RETURNS_OUT.relative_to(ROOT)} ({panel.shape[0]} x {panel.shape[1]})")
    print("Wrote World Bank market-cap weights for 1975")
    print(f"Wrote {MANIFEST_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
