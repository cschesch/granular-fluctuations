#!/usr/bin/env python3
"""Prepare Stella-style seasonally adjusted quarterly firm-sales growth.

Stella's Stata code applies Census X-12 to log nominal ``revtq`` with automatic
outlier selection and takes X-11 table D11, then deflates, log-differences,
and demeans within firm. The emitted sax12 estimation options are translated
literally. The dedicated WRDS extract supplies ``revtq`` directly.
The recovered original X-12 build is preferred when available; X-13 is a
separately labelled engine alternative. Legacy saleq filenames are retained
for downstream compatibility, without filling missing revenue from sales.
"""

from __future__ import annotations

from applications.stella.stella_quarterly_inputs import QFILE, read_quarterly, input_description

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import ROOT
from applications.panels import load_gdp_deflator
from applications.stella.paths import CALIBRATION
from applications.stella.stella_provenance import archive_existing, file_fingerprints


DEFAULT_X13 = ROOT / "code/.cache/tools/x13binary/x13ashtml"
DEFAULT_X12 = ROOT / "code/.cache/tools/x12a_v03/source/x12a"
X12_BUILD_MANIFEST = ROOT / "code/.cache/tools/x12a_v03/build_manifest.json"


def locate_x13(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for variable in ("X12PATH", "X13PATH"):
        if os.environ.get(variable):
            value = Path(os.environ[variable])
            candidates.extend([value, value / "x12a", value / "x13ashtml", value / "x13as"])
    if X12_BUILD_MANIFEST.is_file() and DEFAULT_X12.is_file():
        manifest = json.loads(X12_BUILD_MANIFEST.read_text())
        if manifest.get("verified") and manifest.get("executable_sha256") == hashlib.sha256(DEFAULT_X12.read_bytes()).hexdigest():
            candidates.append(DEFAULT_X12)
    candidates.append(DEFAULT_X13)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Census seasonal-adjustment executable not found. Build X-12, "
        "install CRAN x13binary, or pass --x12/--x13."
    )


def x13_spec(values: np.ndarray, title: str, start: pd.Period,
             seasonal_spec: str = "sax12") -> str:
    data = "\n".join(f"{value:.17g}" for value in values)
    safe_title = title.replace('"', "")
    if seasonal_spec in {"sax12", "sax12_no_outlier"}:
        # The published sax12.ado writes automdl only when ammaxlag() is
        # supplied. Stella never supplies that option: outauto(all) selects
        # outliers, not ARIMA models. Preserve its emitted estimation options.
        model = """outlier{types=(all) method=addone}
forecast{maxlead=12 maxback=0 probability=0.95}
estimate{}
x11{seasonalma=msr save=(d11)}"""
    else:
        raise ValueError(seasonal_spec)
    if seasonal_spec == "sax12_no_outlier":
        # Released data_building.do applies this exception to literal tid=549.
        model = model.replace("outlier{types=(all) method=addone}\n", "")
    return f'''series {{
 data=({data})
 appendbcst=no
 appendfcst=no
 period=4
 start={start.year}.{start.quarter}
 title="{safe_title}"
}}
transform{{function=log}}
{model}
'''


def parse_d11(path: Path, expected: int) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", skiprows=2, names=["date", "value"])
    values = pd.to_numeric(frame["value"], errors="coerce").dropna().to_numpy(float)
    if len(values) != expected or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Invalid X-13 D11 output: expected {expected} positive values, got {len(values)}")
    return values


def adjust_one(
    gvkey: str,
    values: np.ndarray,
    start: pd.Period,
    executable: Path,
    seasonal_spec: str = "sax12",
) -> tuple[str, np.ndarray, str]:
    with tempfile.TemporaryDirectory(prefix=f"stella_x13_{gvkey}_") as tmp:
        directory = Path(tmp)
        base = directory / "series"
        base.with_suffix(".spc").write_text(x13_spec(values, gvkey, start, seasonal_spec))
        completed = subprocess.run(
            [str(executable), base.name], cwd=directory, text=True,
            capture_output=True, timeout=120,
        )
        d11 = base.with_suffix(".d11")
        if completed.returncode != 0 or not d11.exists():
            error_file = next((path for path in (directory / "series.err", directory / "series_err.html") if path.exists()), directory / "series.err")
            details = error_file.read_text(errors="ignore")[-2000:] if error_file.exists() else completed.stderr[-2000:]
            raise RuntimeError(f"X-13 failed for {gvkey}: {details}")
        return gvkey, parse_d11(d11, len(values)), completed.stderr[-500:]


def growth_from_levels(levels: pd.DataFrame, deflator: pd.Series) -> pd.DataFrame:
    real_log = np.log(levels).sub(np.log(deflator), axis=1)
    growth = real_log.diff(axis=1).iloc[:, 1:].T
    return growth.sub(growth.mean(axis=0), axis=1)


def full_consecutive_spells(levels: pd.DataFrame) -> dict[str, tuple[pd.Period, np.ndarray]]:
    """Return each target firm's positive spell ending 2011Q4 and containing 1970Q1.

    Stella seasonally adjusts the full consecutive ``input_nsa`` history before
    restricting the balanced panel to 1970.  The current extract starts earlier,
    so retaining the consecutive pre-1970 tail avoids an artificial X-11 endpoint.
    """
    d = read_quarterly(["gvkey", "fyearq", "fqtr", "fic", "curcdq", "saleq"])
    d["gvkey"] = d["gvkey"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    d = d.loc[
        d["gvkey"].isin(levels.index)
        & d["fic"].eq("USA")
        & d["curcdq"].eq("USD")
        & d["fyearq"].between(1962, 2011)
        & d["fqtr"].between(1, 4)
        & d["saleq"].gt(0)
    ].copy()
    d["quarter"] = pd.PeriodIndex(
        d["fyearq"].astype(int).astype(str) + "Q" + d["fqtr"].astype(int).astype(str), freq="Q"
    )
    collapsed = d.groupby(["gvkey", "quarter"])["saleq"].mean()
    spells: dict[str, tuple[pd.Period, np.ndarray]] = {}
    for gvkey in levels.index:
        series = collapsed.loc[gvkey].sort_index()
        start = pd.Period("1970Q1", freq="Q")
        candidate = start - 1
        while candidate >= pd.Period("1962Q1", freq="Q") and candidate in series.index:
            start = candidate
            candidate -= 1
        periods = pd.period_range(start, "2011Q4", freq="Q")
        spell = series.reindex(periods)
        if spell.isna().any():
            raise ValueError(f"Target firm {gvkey} is not consecutive over {start}--2011Q4")
        sample = spell.reindex(levels.columns)
        if not np.allclose(sample.to_numpy(float), levels.loc[gvkey].to_numpy(float)):
            raise ValueError(f"Full-spell and calibration levels disagree for {gvkey}")
        spells[gvkey] = (start, spell.to_numpy(float))
    return spells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x12", "--x13", dest="x13", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION)
    parser.add_argument("--diagnostics-path", type=Path, default=None)
    args = parser.parse_args()
    executable = locate_x13(args.x13)
    engine = "X-12" if "x12" in executable.name.lower() else "X-13"
    calibration = args.calibration_dir
    variant = "sax12"
    diagnostics_path = args.diagnostics_path or calibration / "sales_preparation_sax12_diagnostics.json"

    levels = pd.read_csv(calibration / "nominal_saleq_levels.csv", index_col=0)
    levels.index = levels.index.astype(str).str.zfill(6)
    levels.columns = pd.PeriodIndex(levels.columns, freq="Q")
    if levels.shape[0] < 1 or levels.shape[1] != 168 or not np.isfinite(levels).all().all() or not levels.gt(0).all().all():
        raise ValueError(f"Expected positive N x 168 Stella levels; got {levels.shape}")
    if not levels.columns.equals(pd.period_range("1970Q1", "2011Q4", freq="Q")):
        raise ValueError("Stella level columns must cover every quarter in 1970--2011")
    selection_provenance = calibration / "selection_adf_pvalues_sax12.provenance.json"
    source_adf = json.loads(selection_provenance.read_text()) if selection_provenance.exists() else {}
    no_outlier_gvkey = source_adf.get("literal_tid549_current_gvkey")
    def firm_spec(gvkey):
        return "sax12_no_outlier" if gvkey == no_outlier_gvkey else "sax12"
    spells = full_consecutive_spells(levels)
    adjusted = pd.DataFrame(index=levels.index, columns=levels.columns, dtype=float)
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(adjust_one, gvkey, values, start, executable, firm_spec(gvkey)): (gvkey, start)
            for gvkey, (start, values) in spells.items()
        }
        for future in as_completed(futures):
            gvkey, start = futures[future]
            try:
                _, values, _ = future.result()
                offset = pd.Period("1970Q1", freq="Q").ordinal - start.ordinal
                adjusted.loc[gvkey] = values[offset:offset + len(levels.columns)]
            except Exception as exc:  # preserve every failure for an auditable stop
                errors[gvkey] = str(exc)
    if errors:
        raise RuntimeError(f"X-13 failed for {len(errors)} firms: {errors}")

    for name in ("sales_growth_sax12.csv", "nominal_saleq_sax12_adjusted.csv"):
        archive_existing(calibration / name)
    archive_existing(diagnostics_path)

    deflator = load_gdp_deflator("quarterly").reindex(levels.columns)
    if deflator.isna().any():
        raise ValueError("GDP deflator does not cover 1970Q1--2011Q4.")
    x13_growth = growth_from_levels(adjusted, deflator)
    x13_growth.to_csv(calibration / f"sales_growth_{variant}.csv")
    adjusted.to_csv(calibration / f"nominal_saleq_{variant}_adjusted.csv")

    prequarters = {
        gvkey: pd.Period("1970Q1", freq="Q").ordinal - start.ordinal
        for gvkey, (start, _) in spells.items()
    }
    payload = {
        "input_variable_in_paper": "Compustat quarterly revtq",
        "calibration_directory": str(calibration),
        "input_variable_in_reconstruction": input_description()["input_revenue_field"],
        **input_description(),
        "paper_seasonal_adjustment": "Stata sax12: log transform, outauto(all), outmethod(addone), no explicit ARIMA model or automdl, X-11 seasonalma=msr and D11; sax12.ado 2012-03-14",
        "seasonal_specification": "sax12",
        "literal_tid549_current_gvkey": no_outlier_gvkey,
        "reconstruction_seasonal_adjustment": f"Published sax12 estimation options translated literally to Census {engine}: log; outlier types=all method=addone; no automdl; forecast maxlead=12 maxback=0 probability=.95; estimate{{}}; X-11 seasonalma=msr and D11. Diagnostic-only sliding-span/history requests omitted.",
        "x13_executable": str(executable),
        "seasonal_adjustment_engine": engine,
        "seasonal_adjustment_executable": str(executable),
        "x13_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "input_file_fingerprints": file_fingerprints([QFILE, executable, calibration / "nominal_saleq_levels.csv"], ROOT),
        "x13_failures": errors,
        "N": int(levels.shape[0]),
        "variant": variant,
        "nominal_level_T": int(levels.shape[1]),
        "demeaned_growth_T": int(x13_growth.shape[0]),
        "sample_levels": "1970Q1--2011Q4",
        "sample_growth": "1970Q2--2011Q4",
        "firms_with_consecutive_pre_1970_history": int(sum(value > 0 for value in prequarters.values())),
        "pre_1970_quarters_median": float(np.median(list(prequarters.values()))),
        "pre_1970_quarters_min": int(min(prequarters.values())),
        "pre_1970_quarters_max": int(max(prequarters.values())),
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
