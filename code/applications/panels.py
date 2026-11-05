"""Panel data structures and shared loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mgp_core import ROOT, clean_panel_cache_path, read_clean_panel_cache, write_clean_panel_cache


GDPDEF_PATH = ROOT / "data" / "fred" / "GDPDEF.csv"
WEIGHT_TIMING = "pre_shock_t_minus_1_v1"


@dataclass
class SourcePanel:
    """Already-cleaned units-by-time observations with size-snapshot metadata."""

    name: str
    y: pd.DataFrame  # units x time
    sizes: pd.Series
    labels: pd.Series
    start_date: object
    last_date: object
    frequency: str
    size_name: str
    value_name: str
    coverage: float | None = None
    notes: str = ""
    weight_timing: str = "unspecified"
    weight_date: object | None = None
    clean_panel_path: str | None = None


def source_panel_cache_path(prefix: str, params: dict[str, object], sources: list[Path]) -> Path:
    keyed_params = {"version": 4, **params}
    return clean_panel_cache_path(prefix, keyed_params, sources)


def load_source_panel_cache(path: Path, use_cache: bool, refresh_cache: bool) -> SourcePanel | None:
    if not use_cache or refresh_cache:
        return None
    payload = read_clean_panel_cache(path)
    if payload is None:
        return None
    return SourcePanel(**(payload | {"clean_panel_path": str(path.relative_to(ROOT))}))


def save_source_panel_cache(path: Path, panel: SourcePanel, use_cache: bool) -> None:
    if use_cache:
        panel.clean_panel_path = str(path.relative_to(ROOT))
        write_clean_panel_cache(path, vars(panel))


def load_gdp_deflator(freq: str) -> pd.Series:
    d = pd.read_csv(GDPDEF_PATH, parse_dates=["observation_date"])
    d["GDPDEF"] = pd.to_numeric(d["GDPDEF"], errors="coerce")
    d = d.dropna(subset=["GDPDEF"])
    if freq == "annual":
        d["period"] = pd.PeriodIndex(d["observation_date"], freq="Y")
        return d.groupby("period")["GDPDEF"].mean()
    if freq == "quarterly":
        d["period"] = pd.PeriodIndex(d["observation_date"], freq="Q")
        return d.drop_duplicates("period").set_index("period")["GDPDEF"].sort_index()
    raise ValueError(freq)
