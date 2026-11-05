#!/usr/bin/env python3
"""Carvalho--Gabaix primitive-shock decomposition with public BEA--BLS data.

This is a public BEA--BLS proxy, not a replication of the authors' original
77-sector Dale Jorgenson Associates panel.  The main panel uses the experimental
63-industry BEA--BLS integrated production account for 1963--2016.  Government
and real estate are excluded to approximate the paper's private-market sample
without owner-occupied housing, leaving 60 industries and 53 innovations.

The downloaded workbooks in ``data/bea_klems`` are read only and shared with the
separate BEA composition exercise. All derived shocks, diagnostics, and estimates
are written separately.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

import gradient_bootstrap as GB
from factor_rank import factor_rank_tests
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    ROOT,
    RUN_OUTPUT_CACHE,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
    mean_ar1,
    mean_pairwise_correlation,
    mgp_share,
    regularity_report,
    sample_cov,
)
from project_paths import DERIVED_CACHE
from applications.volatility.run_bea_granular_ceiling import NON_MARKET_INDUSTRIES


BEA_KLEMS = ROOT / "data" / "bea_klems"
DERIVED = DERIVED_CACHE / "carvalho_gabaix_proxy"
HISTORICAL_UNDERLYING = BEA_KLEMS / "KLEMS-Research-Tables-Underlying-Data_0.xlsx"
HISTORICAL_TABLES = (
    BEA_KLEMS
    / "KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx"
)
EXPECTED_SOURCE_HASHES = {
    HISTORICAL_UNDERLYING: "2adcff255671f4f2d62a05d272389f1d9186b797a0dea5451a9fca25d6366603",
    HISTORICAL_TABLES: "5c83f99f2894116ac8e11541dfbc1f165d899a6b60588e53c44485886e129a67",
}
# Government plus all real estate. The paper excludes owner-occupied housing
# only; the public 63-industry account does not separate it from market real
# estate. Shared with the frozen-covariance proxy exercise.
EXCLUDED_INDUSTRIES = NON_MARKET_INDUSTRIES

NOMINAL_INPUTS = [
    "ii.",
    "vkit.",
    "vksoft.",
    "vkRD.",
    "vkart.",
    "vkoth.",
    "vlcol.",
    "vln.",
]
QUANTITY_INPUTS = [
    "iiqi.",
    "qkit.",
    "qks.",
    "qkrd.",
    "qka.",
    "qko.",
    "qlindexcol_merge.",
    "qlindexn_merge.",
]


def validate_source_files() -> None:
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(f"Required unmodified source workbook is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Source checksum mismatch for {path.name}: {digest} != {expected}")


@dataclass(frozen=True)
class Calibration:
    """Exact one-shot calibration in Carvalho--Gabaix's online-appendix Table 5.

    Field names follow that table's own labels: ``b`` is *one minus* the share of
    intermediate inputs, i.e. the primary-factor (value-added) share of gross
    output, not the intermediate share itself.  The derived properties below
    reproduce the table's reported "Resulting Values" -- micro-level productivity
    multiplier beta = 2.7, aggregate multiplier mu = 4.5, elasticity Phi = 0.3 --
    and the appendix's c_A = 1.19, which is what pins the implementation.
    """

    alpha: float = 2.0 / 3.0
    value_added_share_b: float = 0.5
    labor_supply_phi: float = 2.0
    product_differentiation_psi: float = 1.2
    variable_labor_share_nu: float = 0.5
    mismeasurement_theta: float = 0.43

    @property
    def beta(self) -> float:
        return 1.0 / (
            self.product_differentiation_psi
            - 1.0
            + self.value_added_share_b * self.alpha * (1.0 - self.variable_labor_share_nu)
        )

    @property
    def phi_capital(self) -> float:
        return self.beta * self.value_added_share_b * self.alpha / (1.0 + self.labor_supply_phi)

    @property
    def gdp_multiplier_mu(self) -> float:
        return (1.0 + self.labor_supply_phi) / self.alpha

    @property
    def measured_tfp_own_loading(self) -> float:
        return (
            1.0
            + self.mismeasurement_theta
            * self.value_added_share_b
            * self.alpha
            * self.variable_labor_share_nu
            * self.beta
        )

    @property
    def measured_tfp_aggregate_loading(self) -> float:
        return (
            self.mismeasurement_theta
            * self.value_added_share_b
            * self.alpha
            * (
                self.labor_supply_phi / (1.0 + self.labor_supply_phi)
                - self.variable_labor_share_nu * self.phi_capital
            )
        )


def _assert_complete_balanced_panel(panel: pd.DataFrame, label: str) -> None:
    counts = panel.groupby("industry_code")["year"].nunique()
    if panel.isna().any().any() or counts.nunique() != 1:
        raise ValueError(f"{label} is not a complete balanced panel.")
    if panel[["nominal_gross_output", "nominal_value_added", "domar_weight"]].le(0).any().any():
        raise ValueError(f"{label} has nonpositive nominal quantities or Domar weights.")


def load_historical_panel() -> tuple[pd.DataFrame, dict[str, float]]:
    """Construct annual gross-output MFP residuals from the underlying KLEMS file."""

    raw = pd.read_excel(HISTORICAL_UNDERLYING, sheet_name="1963-2016", header=1)
    expected = {
        "yr",
        "indnum",
        "Description",
        "go.",
        "ii.",
        "goqi.",
        *NOMINAL_INPUTS,
        *QUANTITY_INPUTS,
    }
    missing = expected.difference(raw.columns)
    if missing:
        raise ValueError(f"Historical workbook is missing columns: {sorted(missing)}")

    cost_identity_error = (
        raw[NOMINAL_INPUTS].sum(axis=1).sub(raw["go."]).abs().div(raw["go."])
    )
    if float(cost_identity_error.max()) > 2e-7:
        raise ValueError("Historical KLEMS nominal input costs do not sum to gross output.")

    records: list[dict[str, object]] = []
    for industry_code, group in raw.groupby("indnum", sort=True):
        group = group.sort_values("yr").reset_index(drop=True)
        cost_shares = group[NOMINAL_INPUTS].div(group["go."], axis=0).to_numpy(float)
        input_growth = np.diff(np.log(group[QUANTITY_INPUTS].to_numpy(float)), axis=0)
        output_growth = np.diff(np.log(group["goqi."].to_numpy(float)))
        avg_cost_shares = 0.5 * (cost_shares[1:] + cost_shares[:-1])
        measured_mfp = output_growth - np.sum(avg_cost_shares * input_growth, axis=1)
        nominal_va = group["go."].to_numpy(float) - group["ii."].to_numpy(float)
        for j, year in enumerate(group["yr"].to_numpy(int)[1:]):
            records.append(
                {
                    "industry_code": int(industry_code),
                    "industry": str(group["Description"].iloc[0]),
                    "year": int(year),
                    "measured_tfp_innovation": float(measured_mfp[j]),
                    "nominal_gross_output": float(group["go."].iloc[j + 1]),
                    "nominal_value_added": float(nominal_va[j + 1]),
                }
            )
    panel = pd.DataFrame(records)

    published = pd.read_excel(HISTORICAL_TABLES, sheet_name="Table11.1", header=None)
    published = published.iloc[5:68, [0, 6]].copy()
    published.columns = ["industry", "published_average_mfp_growth_pct"]
    published["published_average_mfp_growth_pct"] = pd.to_numeric(
        published["published_average_mfp_growth_pct"], errors="raise"
    )
    computed = (
        panel.groupby("industry", as_index=False)["measured_tfp_innovation"]
        .mean()
        .rename(columns={"measured_tfp_innovation": "computed_average_mfp_growth_pct"})
    )
    computed["computed_average_mfp_growth_pct"] *= 100.0
    validation = computed.merge(published, on="industry", validate="one_to_one")
    validation["difference_pct_point"] = (
        validation["computed_average_mfp_growth_pct"]
        - validation["published_average_mfp_growth_pct"]
    )
    if len(validation) != 63:
        raise ValueError(f"Expected validation for 63 industries, found {len(validation)}.")
    DERIVED.mkdir(parents=True, exist_ok=True)
    validation.to_csv(DERIVED / "historical_mfp_validation.csv", index=False)
    diagnostics = {
        "mfp_validation_industries": int(len(validation)),
        "mfp_validation_correlation": float(
            validation[["computed_average_mfp_growth_pct", "published_average_mfp_growth_pct"]]
            .corr()
            .iloc[0, 1]
        ),
        "mfp_validation_max_abs_difference_pct_point": float(
            validation["difference_pct_point"].abs().max()
        ),
        "nominal_cost_identity_max_relative_error": float(cost_identity_error.max()),
    }
    return _restrict_and_add_domar(panel, initial_year=1963, raw_for_initial=raw), diagnostics


def _restrict_and_add_domar(
    panel: pd.DataFrame,
    *,
    initial_year: int,
    raw_for_initial: pd.DataFrame,
) -> pd.DataFrame:
    panel = panel.loc[~panel["industry"].isin(EXCLUDED_INDUSTRIES)].copy()
    panel["aggregate_private_market_value_added"] = panel.groupby("year")[
        "nominal_value_added"
    ].transform("sum")
    panel["domar_weight"] = (
        panel["nominal_gross_output"] / panel["aggregate_private_market_value_added"]
    )

    if "yr" in raw_for_initial.columns:
        raw_for_initial = raw_for_initial.loc[raw_for_initial["yr"].eq(initial_year)].copy()
    initial = raw_for_initial.loc[
        ~raw_for_initial["Description"].astype(str).isin(EXCLUDED_INDUSTRIES)
    ].copy()
    initial["initial_value_added"] = initial["go."] - initial["ii."]
    initial_gdp = float(initial["initial_value_added"].sum())
    fixed = (initial.set_index("indnum")["go."] / initial_gdp).rename("fixed_influence_weight")
    panel = panel.merge(fixed, left_on="industry_code", right_index=True, validate="many_to_one")
    panel["influence_base_year"] = int(initial_year)
    _assert_complete_balanced_panel(panel, f"BEA--BLS {initial_year} influence panel")
    if panel["industry_code"].nunique() != 60:
        raise ValueError(f"Expected 60 retained industries, found {panel['industry_code'].nunique()}.")
    return panel.sort_values(["industry_code", "year"]).reset_index(drop=True)


def invert_measured_tfp(panel: pd.DataFrame, calibration: Calibration) -> tuple[pd.DataFrame, float]:
    """Invert m_t = c_A A_t + c_Y mu 1 (d_t' A_t) year by year."""

    c_a = calibration.measured_tfp_own_loading
    k = calibration.measured_tfp_aggregate_loading * calibration.gdp_multiplier_mu
    result = panel.copy()
    result["primitive_productivity_shock"] = np.nan
    max_error = 0.0
    for _, rows in result.groupby("year", sort=True):
        rows = rows.sort_values("industry_code")
        m = rows["measured_tfp_innovation"].to_numpy(float)
        d = rows["domar_weight"].to_numpy(float)
        primitive = m / c_a - (k / c_a**2) * (d @ m) / (1.0 + k * d.sum() / c_a)
        reconstructed = c_a * primitive + k * np.ones(len(primitive)) * (d @ primitive)
        max_error = max(max_error, float(np.max(np.abs(m - reconstructed))))
        result.loc[rows.index, "primitive_productivity_shock"] = primitive
    if max_error > 1e-12:
        raise ValueError(f"Measured-TFP inversion error is too large: {max_error:g}")
    return result, max_error


def export_panel(panel: pd.DataFrame, vintage: str) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    export = panel.copy()
    export.insert(0, "vintage", vintage)
    export.to_csv(DERIVED / f"bea_bls_{vintage}_primitive_shocks_long.csv", index=False)


def run_mgp(
    panel: pd.DataFrame,
    *,
    vintage: str,
    m: int,
    bootstrap_b: int,
    solver: str,
    seed: int,
    progress_every: int,
    extra_diagnostics: dict[str, object],
) -> dict[str, object]:
    primitive = panel.pivot(
        index="industry_code", columns="year", values="primitive_productivity_shock"
    ).sort_index()
    block_length = analysis_block_length(primitive.shape[1])
    measured = panel.pivot(
        index="industry_code", columns="year", values="measured_tfp_innovation"
    ).reindex(index=primitive.index, columns=primitive.columns)
    influence = panel.groupby("industry_code")["fixed_influence_weight"].first().reindex(primitive.index)
    shares = (influence / influence.sum()).to_numpy(float)
    x_grouped, _, loads, _ = build_lpt_grouped_data(primitive.to_numpy(float), shares, m)
    sigma_grouped = sample_cov(x_grouped)
    observed = mgp_share(x_grouped, solver)
    regularity = regularity_report(sigma_grouped, solver)

    prefix = f"carvalho_gabaix_bea_bls_{vintage}_M{m}"
    RUN_OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    family: dict[str, object] = {}
    if bootstrap_b > 0:
        family = GB.bootstrap_family(
            x_grouped,
            b=bootstrap_b,
            block_length=block_length,
            rng=make_panel_rng(seed, f"{prefix}_gradient"),
            solver=solver,
            progress_every=progress_every,
            checkpoint_path=RUN_OUTPUT_CACHE / f"{prefix}_draws_gradient.pkl",
            checkpoint_every=progress_every,
        )

    try:
        _, rank = factor_rank_tests(x_grouped, alpha=0.05, progress_label=prefix)
        factor_lb_j = rank.get("factor_lb_J", np.nan)
        regular_certified_j = bool(rank.get("regular_certified_J", False))
    except Exception as exc:
        print(f"[{prefix}] J-test skipped: {type(exc).__name__}: {exc}", flush=True)
        factor_lb_j, regular_certified_j = np.nan, False

    weight_sum = float(influence.sum())
    mu = Calibration().gdp_multiplier_mu
    summary: dict[str, object] = {
        "dataset": prefix,
        "table_column": "Carvalho--Gabaix public proxy",
        "table_outcome": "static Domar-propagated primitive-productivity bound",
        "solver_backend": solver,
        "frequency": "annual",
        "value_name": "Carvalho--Gabaix primitive productivity shock",
        "size_name": "initial-year BEA--BLS Domar influence",
        "weight_timing": "pre_sample_initial_domar_v1",
        "weight_date": str(int(panel["influence_base_year"].iloc[0])),
        "start_date": str(int(primitive.columns.min())),
        "last_date": str(int(primitive.columns.max())),
        "N": int(primitive.shape[0]),
        "T": int(primitive.shape[1]),
        "M": int(m),
        "bootstrap_b": int(bootstrap_b),
        "block_bootstrap_b": int(bootstrap_b),
        "requested_subsample_n": 0,
        "block_length": int(block_length),
        "coverage": 1.0,
        "HHI": float(np.sum(shares**2)),
        "top1_share": float(np.max(shares)),
        "top10_share": float(np.sort(shares)[-10:].sum()),
        "top50_share": float(np.sort(shares)[-50:].sum()),
        "bucket_load_cv": float(np.std(loads) / np.mean(loads)),
        "sigma_x": float(np.mean(np.std(primitive.to_numpy(float), axis=1, ddof=1))),
        "sigma_y": float(mu * weight_sum * np.sqrt(observed["aggregate_variance"])),
        "mgp_share": float(observed["mgp_share"]),
        "mgp_level": float(mu**2 * weight_sum**2 * observed["mgp_level"]),
        "aggregate_variance": float(mu**2 * weight_sum**2 * observed["aggregate_variance"]),
        "se_share": float(observed["se_share"]),
        "normal_ucl95": float(observed["normal_ucl95"]),
        "ledermann_kbar": regularity["ledermann_kbar"],
        "common_rank": regularity["common_rank"],
        "factor_lb_J": factor_lb_j,
        "regular_certified_J": regular_certified_j,
        "regular": regularity["regular"],
        "rank_gap_ratio": regularity["rank_gap_ratio"],
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
        "initial_domar_weight_sum": weight_sum,
        "gdp_multiplier_mu": mu,
        "measured_tfp_mean_pairwise_correlation": mean_pairwise_correlation(measured.to_numpy(float)),
        "primitive_mean_pairwise_correlation": mean_pairwise_correlation(primitive.to_numpy(float)),
        "primitive_mean_ar1": mean_ar1(primitive.to_numpy(float)),
        "notes": (
            "BEA--BLS public proxy to Carvalho--Gabaix's 77-sector Jorgenson panel: "
            "BEA--BLS integrated industry production account, excluding government "
            "and all real estate to remove owner-occupied housing. Measured gross-output "
            "TFP is inverted using the exact online-appendix calibration; static Domar "
            "influence is fixed before the first innovation, as in the Foerster design."
        ),
        **extra_diagnostics,
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
    args = parser.parse_args()

    DERIVED.mkdir(parents=True, exist_ok=True)
    validate_source_files()
    calibration = Calibration()
    historical, historical_diag = load_historical_panel()
    historical, historical_error = invert_measured_tfp(historical, calibration)
    historical_diag["tfp_inversion_max_abs_error"] = historical_error
    export_panel(historical, "historical")
    historical_summary = run_mgp(
        historical,
        vintage="historical",
        m=args.M,
        bootstrap_b=args.bootstrap_b,
        solver=choose_solver(args.solver),
        seed=args.seed,
        progress_every=args.progress_every,
        extra_diagnostics=historical_diag,
    )

    display = pd.DataFrame([historical_summary])[
        [
            "dataset",
            "N",
            "T",
            "M",
            "mgp_share",
            "moving_block_gradient_bootstrap_p05",
            "moving_block_gradient_bootstrap_p95",
            "primitive_mean_pairwise_correlation",
            "primitive_mean_ar1",
        ]
    ]
    print("\nCarvalho--Gabaix public-proxy summary:", flush=True)
    print(display.round(6).to_string(index=False), flush=True)
    print(f"Wrote primitive shocks to {DERIVED}", flush=True)


if __name__ == "__main__":
    main()
