"""Read Acemoglu et al. source files and the matched NBER–CES TFP panel."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from project_paths import DERIVED_CACHE, ROOT


DATA_ROOT = ROOT / "data" / "acemoglu2012"
DERIVED = DERIVED_CACHE / "acemoglu2012"
NBER_MAT = DATA_ROOT / "NBER_Manuf_TFP_source_data.mat"
DETAILED_IO_MAT = DATA_ROOT / "CC_allyears_new.mat"
SUMMARY_IO_MAT = DATA_ROOT / "CC_2d_allyears_new.mat"

EXPECTED_SOURCE_HASHES = {
    "CC_2d_allyears_new.mat": "d39ddd1cf2a4b3653bc4db73f93c032fc1ae93f0ee9a8970921d4a96001e11d6",
    "CC_allyears_new.mat": "a1be2655b9e97b3347bc63e468e24c1ec1e77419483eebd6b1200907bfef077c",
    "NBER_Manuf_TFP_source_data.mat": "0493f48697e3ff3cd991e73213af740b3f3f08388199301fa249b7379ec25eb2",
    "ksr.m": "bcdf7d11248faf8412810b6f9c3f5f594da18c914a43ae53f49470c14f23267d",
    "norm_v2_vector_detailedIO.m": "8294ac80f1d61afaca4acefcd623110eaea9053b350df48cf7e8461d1fc13ff7",
    "norm_v2_vector_summaryIO.m": "f616ef96284f15bfee898bb67ef1f59aa53c056370078afc8755fb1b77a60d6e",
    "std_TFP_NBER_Manuf.m": "7aea96a0f4ecf38b8e62e7dcbbb0fe917b4ec97007e9848a2012096641cc27a1",
    "w_ind.m": "3dc2d6ead244109a0c73c6b0a4eec25fbf21e7c3827c70fcd993249774de3d10",
    "w_out.m": "fee30dda3488d05f8321333248de1bd07e04c121431071c80a5795fc6426d6b5",
    "w_out_2ord.m": "0ee15cd49988dca09b5e22819d89530c41a4f49722f02575185922a2d3cc131b",
    "weighted_indegrees_final.m": "1de4544f028ca7b50a0c911b313447c23a7d5134e5deb5355abf90a227a3c7ab",
    "weighted_outdegrees_2ndorder_final.m": "78e13046574a0929093973c4a28ae63ba9c384a712068eb65a4c2eb44618d9eb",
    "weighted_outdegrees_final.m": "fce3904c6ee97002b95965657812ca0c9d83bd73fe80c47a6c8aeabe701e6e6a",
}

# These constants make accidental edits to the official deposit fail loudly
# instead of silently changing the extension.
EXPECTED_DETAILED_NORMS = np.array(
    [0.0980050545768956, 0.0911793487330519, 0.08829601675312361,
     0.08814821551077173, 0.09249430126367793, 0.08992407083261413,
     0.09366260931262133]
)
EXPECTED_SUMMARY_NORMS = np.array(
    [0.13893969597639794, 0.13698963998024688, 0.1493887726525457,
     0.13337629862308475, 0.13740032330239207, 0.11538886451816463,
     0.11867294696051352]
)
YEARS_IO = (1972, 1977, 1982, 1987, 1992, 1997, 2002)
EXPECTED_HEADERS = (
    "SIC", "YEAR", "EMP", "PAY", "PRODE", "PRODH", "PRODW", "VSHIP",
    "MATCOST", "VADD", "INVEST", "INVENT", "ENERGY", "CAP", "EQUIP",
    "PLANT", "PISHIP", "PIMAT", "PIINV", "PIEN", "DTFP5", "TFP5",
    "DTFP4", "TFP4", "SHARE",
)


def validate_source_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        path = DATA_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Missing official Acemoglu et al. source: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[name] = digest
        if digest != expected:
            raise ValueError(f"Official source checksum mismatch for {name}: {digest} != {expected}")
    return observed


def _network_norms(path: Path, *, summary: bool) -> tuple[np.ndarray, list[int]]:
    source = loadmat(path)
    suffix = "_2d" if summary else ""
    norms: list[float] = []
    dimensions: list[int] = []
    for year in YEARS_IO:
        key = f"CC{str(year)[-2:]}{suffix}"
        requirements = np.asarray(source[key], dtype=float).copy()
        column_sums = requirements.sum(axis=0)
        positive = column_sums > 0
        if np.any(column_sums < 0):
            raise ValueError(f"{key} contains a negative column total.")
        requirements[:, positive] /= column_sums[positive]
        n = requirements.shape[0]
        alpha = 0.3333
        influence = (alpha / n) * np.linalg.solve(
            np.eye(n) - (1.0 - alpha) * requirements,
            np.ones(n),
        )
        norms.append(float(np.linalg.norm(influence)))
        dimensions.append(int(n))
    return np.asarray(norms), dimensions


def verify_original_replication_calculations(
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Independently reproduce the deposited MATLAB scripts' headline outputs.

    ``diagnostics`` is the second return value of ``load_nber_manufacturing_panel``;
    pass it to reuse an already-parsed panel instead of re-reading the MAT file.
    """

    detailed, detailed_n = _network_norms(DETAILED_IO_MAT, summary=False)
    summary, summary_n = _network_norms(SUMMARY_IO_MAT, summary=True)
    if not np.allclose(detailed, EXPECTED_DETAILED_NORMS, rtol=0, atol=2e-12):
        raise ValueError("Detailed input--output influence norms do not match the package.")
    if not np.allclose(summary, EXPECTED_SUMMARY_NORMS, rtol=0, atol=2e-12):
        raise ValueError("Summary input--output influence norms do not match the package.")

    if diagnostics is None:
        _, diagnostics = load_nber_manufacturing_panel()
    # These are the full-precision values behind MATLAB's displayed 0.0595 and 0.0594.
    if not np.isclose(diagnostics["package_mean_raw_tfp_std"], 0.05946450773767385, atol=2e-14):
        raise ValueError("Raw NBER-CES TFP volatility does not match the package.")
    if not np.isclose(diagnostics["package_mean_detrended_tfp_std"], 0.059449497785336466, atol=2e-14):
        raise ValueError("Detrended NBER-CES TFP volatility does not match the package.")
    return {
        "detailed_io_years": list(YEARS_IO),
        "detailed_io_dimensions": detailed_n,
        "detailed_influence_l2_norms": detailed.tolist(),
        "summary_io_dimensions": summary_n,
        "summary_influence_l2_norms": summary.tolist(),
        "matlab_display_STD_TFPg_RAW": 0.0595,
        "matlab_display_STD_TFPg_RES": 0.0594,
    }


def load_nber_manufacturing_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    source = loadmat(NBER_MAT, squeeze_me=True, struct_as_record=False)
    headers = tuple(str(value) for value in np.asarray(source["colheaders"]).tolist())
    if headers != EXPECTED_HEADERS:
        raise ValueError(f"Unexpected NBER-CES columns: {headers}")
    raw = pd.DataFrame(np.asarray(source["data"], dtype=float), columns=headers)
    raw["SIC"] = raw["SIC"].astype(int)
    raw["YEAR"] = raw["YEAR"].astype(int)
    if raw.duplicated(["SIC", "YEAR"]).any():
        raise ValueError("NBER-CES panel contains duplicate SIC-year observations.")
    counts = raw.groupby("SIC")["YEAR"].nunique()
    if len(counts) != 459 or not counts.eq(48).all():
        raise ValueError("Expected 459 four-digit SIC industries observed for 48 years.")
    years = np.sort(raw["YEAR"].unique())
    if not np.array_equal(years, np.arange(1958, 2006)):
        raise ValueError(f"Unexpected NBER-CES years: {years}")

    tfp = raw.pivot(index="SIC", columns="YEAR", values="DTFP5").sort_index()
    shipments = raw.pivot(index="SIC", columns="YEAR", values="VSHIP").reindex(tfp.index)
    if tfp[1958].notna().any() or shipments[1958].isna().any() or not shipments[1958].gt(0).all():
        raise ValueError("Expected missing 1958 DTFP5 and complete positive 1958 shipments.")

    time = np.arange(1.0, 48.0)
    raw_std: list[float] = []
    residual_std: list[float] = []
    for _, row in tfp.loc[:, 1959:2005].iterrows():
        values = row.to_numpy(float)
        finite = np.isfinite(values)
        raw_std.append(float(np.std(values[finite], ddof=1)))
        slope = float(time[finite] @ values[finite] / (time[finite] @ time[finite]))
        residual_std.append(float(np.std(values[finite] - slope * time[finite], ddof=1)))

    complete = tfp.loc[:, 1959:2005].notna().all(axis=1)
    balanced_tfp = tfp.loc[complete, 1959:2005]
    balanced_shipments = shipments.loc[complete, 1958]
    residual = np.empty_like(balanced_tfp.to_numpy(float))
    slopes = np.empty(len(balanced_tfp))
    for row_number, values in enumerate(balanced_tfp.to_numpy(float)):
        slopes[row_number] = float(time @ values / (time @ time))
        residual[row_number] = values - slopes[row_number] * time

    influence = balanced_shipments / balanced_shipments.sum()
    long = (
        pd.DataFrame(residual, index=balanced_tfp.index, columns=balanced_tfp.columns)
        .rename_axis(index="sic", columns="year")
        .stack()
        .rename("primitive_productivity_shock")
        .reset_index()
    )
    raw_long = (
        balanced_tfp.rename_axis(index="sic", columns="year")
        .stack()
        .rename("raw_dtfp5")
        .reset_index()
    )
    long = long.merge(raw_long, on=["sic", "year"], validate="one_to_one")
    long = long.merge(
        pd.DataFrame(
            {
                "sic": balanced_tfp.index.astype(int),
                "detrending_slope_no_intercept": slopes,
                "fixed_1958_shipments": balanced_shipments.to_numpy(float),
                "fixed_influence_weight": influence.to_numpy(float),
            }
        ),
        on="sic",
        validate="many_to_one",
    )
    long["influence_base_year"] = 1958
    long = long.sort_values(["sic", "year"]).reset_index(drop=True)
    diagnostics = {
        "source_industries": 459,
        "balanced_industries": int(complete.sum()),
        "dropped_incomplete_industries": int((~complete).sum()),
        "first_shock_year": 1959,
        "last_shock_year": 2005,
        "package_mean_raw_tfp_std": float(np.mean(raw_std)),
        "package_mean_detrended_tfp_std": float(np.mean(residual_std)),
        "balanced_mean_raw_tfp_std": float(
            np.mean(np.std(balanced_tfp.to_numpy(float), axis=1, ddof=1))
        ),
        "balanced_mean_detrended_tfp_std": float(np.mean(np.std(residual, axis=1, ddof=1))),
        "dropped_sic_codes": tfp.index[~complete].astype(int).tolist(),
    }
    return long, diagnostics
