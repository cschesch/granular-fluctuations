#!/usr/bin/env python3
"""Acemoglu et al. (2012) replication checks and manufacturing MGP extension.

The deposited MATLAB source and MAT files in ``data/acemoglu2012`` are treated
as immutable inputs.  This separate script independently verifies the paper's
reported network-influence norms and NBER-CES TFP-volatility calculation, then
applies the grouped MGP to a balanced panel of the deposited five-factor TFP
growth series.

In the paper's Cobb--Douglas model, equilibrium sector sales shares equal the
network influence vector exactly (their equation 5).  The deposit does not
provide a crosswalk between the detailed BEA input--output matrices and the
four-digit SIC manufacturing panel, so the extension uses 1958 manufacturing
shipments as fixed, pre-sample sales influence.  It is consequently a
manufacturing application, not a full-economy GDP decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

import gradient_bootstrap as GB
from bias import grouped_bias_terms
from factor_rank import factor_rank_tests
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    OUTPUT_DIR,
    ROOT,
    RUN_OUTPUT_CACHE,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    mean_ar1,
    mean_pairwise_correlation,
    mgp_share,
    plot_lpt_squared_bucket_sizes,
    regularity_report,
    sample_cov,
)
from project_paths import DERIVED_CACHE
from run_multi_source import lpt_diagnostics


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


def export_primitive_shocks(panel: pd.DataFrame) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(DERIVED / "acemoglu2012_nber_manufacturing_primitive_shocks_long.csv", index=False)
    wide = panel.pivot(index="year", columns="sic", values="primitive_productivity_shock")
    wide.columns = [f"shock_sic_{int(sic)}" for sic in wide.columns]
    wide.reset_index().to_csv(
        DERIVED / "acemoglu2012_nber_manufacturing_primitive_shocks.csv", index=False
    )
    (
        panel[["sic", "fixed_1958_shipments", "fixed_influence_weight"]]
        .drop_duplicates()
        .sort_values("sic")
        .to_csv(DERIVED / "acemoglu2012_nber_manufacturing_influence.csv", index=False)
    )


def detrending_sensitivity(
    panel: pd.DataFrame,
    shares: np.ndarray,
    *,
    m: int,
    solver: str,
) -> dict[str, float]:
    """Ceiling under the deposit's trend convention and two conventional ones.

    ``std_TFP_NBER_Manuf.m`` calls MATLAB's ``regress`` with a bare time column,
    so its trend has no intercept; the published paper's 0.058 is closer to a
    conventional intercept-inclusive detrending (0.0588 here) than to the
    script's 0.0594.  The estimate should not, and does not, hinge on which one
    we follow.
    """
    raw = panel.pivot(index="sic", columns="year", values="raw_dtfp5").sort_index().to_numpy(float)
    time = np.arange(1.0, raw.shape[1] + 1.0)
    design = np.column_stack([np.ones_like(time), time])
    with_intercept = raw - (design @ np.linalg.lstsq(design, raw.T, rcond=None)[0]).T
    out: dict[str, float] = {}
    for label, values in (("raw_dtfp5", raw), ("detrended_with_intercept", with_intercept)):
        grouped, *_ = build_lpt_grouped_data(values, shares, m)
        out[f"mgp_share_{label}"] = float(mgp_share(grouped, solver)["mgp_share"])
    return out


def run_mgp(
    panel: pd.DataFrame,
    *,
    m: int,
    bootstrap_b: int,
    solver: str,
    seed: int,
    progress_every: int,
    run_profile: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    primitive = panel.pivot(
        index="sic", columns="year", values="primitive_productivity_shock"
    ).sort_index()
    block_length = analysis_block_length(primitive.shape[1])
    influence = (
        panel.groupby("sic")["fixed_influence_weight"].first().reindex(primitive.index)
    )
    shares = (influence / influence.sum()).to_numpy(float)
    grouped, buckets, loads, _ = build_lpt_grouped_data(
        primitive.to_numpy(float), shares, m
    )
    sigma_grouped = sample_cov(grouped)
    observed = mgp_share(grouped, solver)
    regularity = regularity_report(sigma_grouped, solver)
    bias_summary = grouped_bias_terms(sigma_grouped, grouped.shape[0], solver)
    detrending = detrending_sensitivity(panel, shares, m=m, solver=solver)
    prefix = f"acemoglu2012_nber_manufacturing_M{m}"
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    family: dict[str, object] = {}
    if bootstrap_b > 0:
        family = GB.bootstrap_family(
            grouped,
            b=bootstrap_b,
            block_length=block_length,
            rng=make_panel_rng(seed, f"{prefix}_gradient"),
            solver=solver,
            progress_every=progress_every,
            checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_gradient.pkl",
            checkpoint_every=progress_every,
        )

    try:
        _, rank = factor_rank_tests(grouped, alpha=0.05, progress_label=prefix)
        factor_lb_j = rank.get("factor_lb_J", np.nan)
        regular_certified_j = bool(rank.get("regular_certified_J", False))
    except Exception as exc:
        print(f"[{prefix}] J-test skipped: {type(exc).__name__}: {exc}", flush=True)
        factor_lb_j, regular_certified_j = np.nan, False

    labels = pd.Series([f"SIC {int(sic)}" for sic in primitive.index], index=primitive.index)
    units, bucket_diag = lpt_diagnostics(primitive.index, shares, labels, buckets, loads)
    units["fixed_1958_shipments"] = (
        panel.groupby("sic")["fixed_1958_shipments"].first().reindex(primitive.index).to_numpy()
    )
    units.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_units.pkl")
    bucket_diag.to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_diag_buckets.pkl")
    plot_lpt_squared_bucket_sizes(
        units.rename(columns={"unit_id": "gvkey"}),
        bucket_diag,
        OUTPUT_DIR / f"lpt_{prefix}",
        ylabel=r"Bucket squared-sales-influence load: $\sum_{i\in B_m} v_i^2$",
    )

    values = primitive.to_numpy(float)
    summary: dict[str, object] = {
        "dataset": prefix,
        "table_column": "Acemoglu et al. manufacturing",
        "table_outcome": "static sales-influence primitive-productivity bound",
        "run_profile": run_profile,
        "solver_backend": solver,
        "frequency": "annual",
        "value_name": "detrended NBER-CES five-factor TFP growth",
        "size_name": "fixed 1958 manufacturing shipment share",
        "weight_timing": "pre_sample_1958_sales_influence_v1",
        "weight_date": "1958",
        "start_date": str(int(primitive.columns.min())),
        "last_date": str(int(primitive.columns.max())),
        "N": int(primitive.shape[0]),
        "T": int(primitive.shape[1]),
        "M": int(m),
        "bootstrap_b": int(bootstrap_b),
        "block_bootstrap_b": int(bootstrap_b),
        "requested_subsample_n": 0,
        "block_length": int(block_length),
        "coverage": float(primitive.shape[0] / diagnostics["source_industries"]),
        "HHI": float(np.sum(shares**2)),
        "top1_share": float(np.max(shares)),
        "top10_share": float(np.sort(shares)[-10:].sum()),
        "top50_share": float(np.sort(shares)[-50:].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        "sigma_x": float(np.mean(np.std(values, axis=1, ddof=1))),
        "sigma_y": float(np.sqrt(observed["aggregate_variance"])),
        "mgp_share": float(observed["mgp_share"]),
        "mgp_level": float(observed["mgp_level"]),
        "aggregate_variance": float(observed["aggregate_variance"]),
        "se_share": float(observed["se_share"]),
        "normal_ucl95": float(observed["normal_ucl95"]),
        "ledermann_kbar": regularity["ledermann_kbar"],
        "common_rank": regularity["common_rank"],
        "factor_lb_J": factor_lb_j,
        "regular_certified_J": regular_certified_j,
        "regular": regularity["regular"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
        **bias_summary,
        "gradient_bootstrap_p05": family.get("gradient_low", np.nan),
        "gradient_bootstrap_p95": family.get("gradient_high", np.nan),
        "moving_block_gradient_bootstrap_p05": family.get("block_gradient_low", np.nan),
        "moving_block_gradient_bootstrap_p95": family.get("block_gradient_high", np.nan),
        "bootstrap_mean": family.get("bootstrap_mean", np.nan),
        "bootstrap_p05": family.get("bootstrap_low", np.nan),
        "bootstrap_p95": family.get("bootstrap_high", np.nan),
        "moving_block_bootstrap_mean": family.get("block_mean", np.nan),
        "moving_block_bootstrap_p05": family.get("block_low", np.nan),
        "moving_block_bootstrap_p95": family.get("block_high", np.nan),
        "primitive_mean_pairwise_correlation": mean_pairwise_correlation(values),
        "primitive_mean_ar1": mean_ar1(values),
        **detrending,
        "sales_share_identity": "Acemoglu et al. (2012), equation (5)",
        "notes": (
            "Manufacturing extension using the deposited NBER-CES five-factor TFP growth "
            "and shipments. Exact no-intercept trend residualization follows the deposited "
            "MATLAB script; the balanced 451-industry panel uses fixed 1958 shipment shares. "
            "Because the deposit has no detailed-IO-to-SIC crosswalk, this is not a full-economy "
            "GDP decomposition. In the paper's model, sector sales shares equal network influence."
        ),
        **diagnostics,
    }
    pd.DataFrame([summary]).to_pickle(RUN_OUTPUT_CACHE / f"{prefix}_summary.pkl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=10,
                        help="LPT groups; default 10.")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--run-profile", choices=["medium", "full"], default="medium")
    args = parser.parse_args()

    hashes = validate_source_files()
    panel, diagnostics = load_nber_manufacturing_panel()
    original = verify_original_replication_calculations(diagnostics)
    export_primitive_shocks(panel)
    DERIVED.mkdir(parents=True, exist_ok=True)
    with (DERIVED / "acemoglu2012_replication_manifest.json").open("w") as handle:
        json.dump(
            {
                "source_sha256": hashes,
                "original_package_checks": original,
                "manufacturing_extension": diagnostics,
                "official_matlab_source_modified": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    summary = run_mgp(
        panel,
        m=args.M,
        bootstrap_b=args.bootstrap_b,
        solver=choose_solver(args.solver),
        seed=args.seed,
        progress_every=args.progress_every,
        run_profile=args.run_profile,
        diagnostics=diagnostics,
    )

    display = pd.DataFrame([summary])[
        [
            "dataset", "N", "T", "M", "mgp_share",
            "moving_block_gradient_bootstrap_p05",
            "moving_block_gradient_bootstrap_p95",
            "primitive_mean_pairwise_correlation", "primitive_mean_ar1",
        ]
    ]
    print("\nAcemoglu et al. manufacturing extension:", flush=True)
    print(display.round(6).to_string(index=False), flush=True)
    print(f"Wrote primitive shocks to {DERIVED}", flush=True)


if __name__ == "__main__":
    main()
