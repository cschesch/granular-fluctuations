"""Sector granularity and US GDP volatility using BEA--BLS production accounts."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from project_paths import CODE_DIR

SCRIPT_DIR = CODE_DIR
CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plot_style as ps
from statsmodels.tsa.filters.hp_filter import hpfilter

ps.use()

from mgp_core import (
    OUTPUT_DIR,
    ROOT,
    RUN_OUTPUT_CACHE,
    choose_solver,
)
from .paths import build_paths


BEA_DIR = ROOT / "data" / "bea_klems"
BEA_UNDERLYING = BEA_DIR / "KLEMS-Research-Tables-Underlying-Data_0.xlsx"
BEA_TABLES = (
    BEA_DIR
    / "KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx"
)
GDPC1_PATH = ROOT / "data" / "fred" / "GDPC1.csv"


NOMINAL_QUANTITY_PAIRS = (
    ("ii.", "iiqi.", "intermediate"),
    ("vkit.", "qkit.", "capital_it"),
    ("vksoft.", "qks.", "capital_software"),
    ("vkRD.", "qkrd.", "capital_rd"),
    ("vkart.", "qka.", "capital_art"),
    ("vkoth.", "qko.", "capital_other"),
    ("vlcol.", "qlindexcol_merge.", "labor_college"),
    ("vln.", "qlindexn_merge.", "labor_noncollege"),
)
CAPITAL_COMPONENTS = tuple(x[2] for x in NOMINAL_QUANTITY_PAIRS if x[2].startswith("capital_"))
LABOR_COMPONENTS = tuple(x[2] for x in NOMINAL_QUANTITY_PAIRS if x[2].startswith("labor_"))
# Carvalho and Gabaix's Data Appendix cuts their 88-sector Jorgenson panel to 77 by
# dropping government sectors *and* "the imputed service flow from owner-occupied
# housing".  The BEA--BLS KLEMS account does not separate market real estate from
# owner-occupied housing, so dropping the whole "Real estate" industry is the closest
# available match.  This leaves 60 market industries, and run_carvalho_gabaix_proxy.py
# imports this same set so the two exercises cannot drift apart.
NON_MARKET_INDUSTRIES = frozenset({"Federal", "State and local", "Real estate"})
EXPECTED_UNDERLYING_SHA256 = "2adcff255671f4f2d62a05d272389f1d9186b797a0dea5451a9fca25d6366603"
EXPECTED_TABLES_SHA256 = "5c83f99f2894116ac8e11541dfbc1f165d899a6b60588e53c44485886e129a67"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sources() -> None:
    for path, expected in (
        (BEA_UNDERLYING, EXPECTED_UNDERLYING_SHA256),
        (BEA_TABLES, EXPECTED_TABLES_SHA256),
        (GDPC1_PATH, None),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required source is missing: {path}")
        if expected is not None:
            observed = sha256(path)
            if observed != expected:
                raise ValueError(f"Checksum mismatch for {path.name}: {observed} != {expected}")


def load_bea_panel(include_non_market: bool = False) -> pd.DataFrame:
    """Read and validate the balanced detailed BEA--BLS 1963--2016 panel."""
    d = pd.read_excel(BEA_UNDERLYING, sheet_name="1963-2016", header=1)
    expected = {"yr", "indnum", "Description", "go.", "ii.", "goqi."}
    expected.update(x for pair in NOMINAL_QUANTITY_PAIRS for x in pair[:2])
    missing = sorted(expected.difference(d.columns))
    if missing:
        raise ValueError(f"BEA workbook is missing columns: {missing}")

    numeric = ["yr", "indnum", "go.", "goqi."] + [x for p in NOMINAL_QUANTITY_PAIRS for x in p[:2]]
    for column in dict.fromkeys(numeric):
        d[column] = pd.to_numeric(d[column], errors="raise")
    d["yr"] = d["yr"].astype(int)
    d["indnum"] = d["indnum"].astype(int)
    d = d.sort_values(["indnum", "yr"]).reset_index(drop=True)

    counts = d.groupby("yr", sort=True)["indnum"].nunique()
    if list(counts.index) != list(range(1963, 2017)) or not counts.eq(63).all():
        raise ValueError("Expected a balanced 63-industry panel for every year 1963--2016.")
    if d.duplicated(["yr", "indnum"]).any():
        raise ValueError("Duplicate BEA year-industry rows.")
    quantity_columns = ["goqi."] + [p[1] for p in NOMINAL_QUANTITY_PAIRS]
    if not np.isfinite(d[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite BEA input value.")
    if (d[quantity_columns] <= 0).any(axis=None):
        raise ValueError("KLEMS quantity indexes must be strictly positive.")
    if (d[["go.", "ii."]] <= 0).any(axis=None):
        raise ValueError("Gross output and intermediate inputs must be strictly positive.")

    nominal_inputs = [p[0] for p in NOMINAL_QUANTITY_PAIRS]
    d["nominal_input_sum"] = d[nominal_inputs].sum(axis=1)
    d["nominal_balance_error"] = d["nominal_input_sum"] - d["go."]
    d["nominal_balance_rel_error"] = d["nominal_balance_error"] / d["go."]
    if d["nominal_balance_rel_error"].abs().max() > 2e-6:
        raise ValueError("KLEMS nominal inputs do not reconcile to nominal gross output.")

    d["value_added"] = d["go."] - d["ii."]
    if (d["value_added"] <= 0).any():
        raise ValueError("Industry value added must be positive.")

    if not include_non_market:
        d = d.loc[~d["Description"].astype(str).isin(NON_MARKET_INDUSTRIES)].copy()
        if d["indnum"].nunique() != 60:
            raise ValueError(f"Expected 60 market industries, found {d['indnum'].nunique()}.")
    return d


def reconstruct_tfp(d: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct annual log TFP growth using two-period average cost shares."""
    d = d.sort_values(["indnum", "yr"]).copy()
    group = d.groupby("indnum", sort=False)
    d["output_growth"] = group["goqi."].transform(lambda x: np.log(x).diff())

    contribution_columns: list[str] = []
    for nominal, quantity, label in NOMINAL_QUANTITY_PAIRS:
        share = d[nominal] / d["go."]
        average_share = 0.5 * (share + share.groupby(d["indnum"]).shift(1))
        quantity_growth = group[quantity].transform(lambda x: np.log(x).diff())
        column = f"contribution_{label}"
        d[column] = average_share * quantity_growth
        contribution_columns.append(column)

    d["capital_contribution"] = d[[f"contribution_{x}" for x in CAPITAL_COMPONENTS]].sum(axis=1)
    d["labor_contribution"] = d[[f"contribution_{x}" for x in LABOR_COMPONENTS]].sum(axis=1)
    d["intermediate_contribution"] = d["contribution_intermediate"]
    d["input_contribution"] = d[contribution_columns].sum(axis=1)
    d["tfp_growth"] = d["output_growth"] - d["input_contribution"]

    private_gdp = d.groupby("yr", sort=True)["value_added"].sum()
    d["private_gdp"] = d["yr"].map(private_gdp)
    d["domar_weight"] = d["go."] / d["private_gdp"]
    if (d["domar_weight"] <= 0).any():
        raise ValueError("Domar weights must be positive.")

    growth = d.loc[d["yr"] > d["yr"].min(), "tfp_growth"]
    if growth.isna().any() or not np.isfinite(growth.to_numpy()).all():
        raise ValueError("The balanced panel did not produce complete TFP growth.")
    return d


def validate_against_table_11_1(d_all: pd.DataFrame) -> dict[str, float]:
    """Reconcile reconstructed 1963--2016 average growth with BEA Table 11.1."""
    raw = pd.read_excel(BEA_TABLES, sheet_name="Table11.1", header=None)
    official = raw.iloc[5 : 5 + 63, [0, 1, 3, 4, 5, 6]].copy()
    official.columns = [
        "Description",
        "output_growth",
        "capital_contribution",
        "labor_contribution",
        "intermediate_contribution",
        "tfp_growth",
    ]
    for column in official.columns[1:]:
        official[column] = pd.to_numeric(official[column], errors="raise")

    calculated = (
        d_all.loc[d_all["yr"] > 1963]
        .groupby(["indnum", "Description"], sort=True)[official.columns[1:].tolist()]
        .mean()
        .mul(100.0)
        .reset_index()
    )
    if calculated["Description"].tolist() != official["Description"].tolist():
        raise ValueError("Industry labels do not align with BEA Table 11.1.")
    differences = calculated[official.columns[1:]].to_numpy() - official[official.columns[1:]].to_numpy()
    max_abs = np.max(np.abs(differences), axis=0)
    checks = {f"table11_1_max_abs_{c}": float(v) for c, v in zip(official.columns[1:], max_abs)}
    if max(checks.values()) > 7e-4:
        raise ValueError(f"TFP reconstruction does not reconcile to BEA Table 11.1: {checks}")
    return checks


def build_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return processed long data, unit-by-time TFP, year-by-unit weights, checks."""
    all_sectors = reconstruct_tfp(load_bea_panel(include_non_market=True))
    checks = validate_against_table_11_1(all_sectors)
    private = reconstruct_tfp(load_bea_panel(include_non_market=False))
    checks["nominal_balance_max_abs_relative"] = float(private["nominal_balance_rel_error"].abs().max())

    tfp = (
        private.loc[private["yr"] >= 1964]
        .pivot(index="indnum", columns="yr", values="tfp_growth")
        .sort_index()
    )
    weights = private.pivot(index="yr", columns="indnum", values="domar_weight").sort_index()
    if tfp.shape != (60, 53) or weights.shape != (54, 60):
        raise ValueError(f"Unexpected private-panel shapes: TFP {tfp.shape}, weights {weights.shape}")
    if not tfp.index.equals(weights.columns):
        raise ValueError("TFP industries and Domar-weight industries do not align.")
    return private, tfp, weights, checks


def grouping_matrix(weights: np.ndarray, buckets: list[list[int]]) -> np.ndarray:
    g = np.zeros((len(buckets), weights.size), dtype=float)
    for bucket_id, bucket in enumerate(buckets):
        g[bucket_id, bucket] = weights[bucket]
    return g


def gdp_volatility() -> pd.DataFrame:
    """Recreate the two annual GDP-volatility measures used in CG Figure 1."""
    d = pd.read_csv(GDPC1_PATH, parse_dates=["observation_date"])
    d["GDPC1"] = pd.to_numeric(d["GDPC1"], errors="coerce")
    d = d.dropna(subset=["GDPC1"]).sort_values("observation_date")
    d["period"] = pd.PeriodIndex(d["observation_date"], freq="Q")
    log_gdp = pd.Series(np.log(d["GDPC1"].to_numpy()), index=d["period"], name="log_gdp")
    if not log_gdp.index.is_unique or (log_gdp <= 0).any():
        raise ValueError("Invalid quarterly real-GDP series.")

    cycle, _ = hpfilter(log_gdp, lamb=1600)
    # Carvalho--Gabaix use a centered 41-quarter window in the interior and
    # progressively more one-sided windows at the end of the sample.  Keep the
    # width fixed at 41 quarters while shifting the window inside the observed
    # sample, which implements that endpoint convention directly.
    rolling_values = np.empty(len(cycle), dtype=float)
    for position in range(len(cycle)):
        start = position - 20
        stop = position + 21
        if start < 0:
            stop -= start
            start = 0
        if stop > len(cycle):
            start -= stop - len(cycle)
            stop = len(cycle)
        window = cycle.iloc[max(start, 0):stop]
        rolling_values[position] = 2.0 * window.std(ddof=1) if len(window) == 41 else np.nan
    rolling_quarterly = pd.Series(rolling_values, index=cycle.index)
    rolling_annual = rolling_quarterly.groupby(rolling_quarterly.index.year).mean()

    growth = log_gdp.diff()
    reg = pd.DataFrame({"growth": growth, "lag": growth.shift(1)}).dropna()
    reg = reg.loc[reg.index >= pd.Period("1960Q1", freq="Q")]
    x = np.column_stack([np.ones(len(reg)), reg["lag"].to_numpy()])
    beta = np.linalg.lstsq(x, reg["growth"].to_numpy(), rcond=None)[0]
    residual = pd.Series(reg["growth"].to_numpy() - x @ beta, index=reg.index)
    instantaneous_quarterly = 2.0 * np.sqrt(np.pi / 2.0) * residual.abs()
    instantaneous_annual = instantaneous_quarterly.groupby(instantaneous_quarterly.index.year).mean()
    _, instantaneous_hp = hpfilter(instantaneous_annual, lamb=6.25)

    out = pd.DataFrame(
        {
            "gdp_rolling_volatility": rolling_annual,
            "gdp_instantaneous_hp_volatility": instantaneous_hp,
        }
    )
    out.index.name = "year"
    return out


def plot_figure1(
    results: pd.DataFrame,
    out: Path,
    frozen_paths: pd.DataFrame | None = None,
) -> None:
    """Figure-1-style level comparison, optionally with frozen decompositions."""
    common = results.dropna(
        subset=[
            "gdp_rolling_volatility",
            "gdp_instantaneous_hp_volatility",
            "ceiling_volatility",
            "cg_diagonal_volatility",
        ]
    ).copy()
    if frozen_paths is not None:
        frozen_paths = frozen_paths.reindex(common.index)
        ceiling = frozen_paths["ceiling_volatility"]
    else:
        ceiling = common["ceiling_volatility"]
    cg_fundamental = common["cg_diagonal_volatility"]
    rolling = common["gdp_rolling_volatility"]
    instantaneous = common["gdp_instantaneous_hp_volatility"]

    fig, ax = plt.subplots(figsize=(8.25, 4.5))
    ax.plot(
        common.index,
        100 * rolling,
        color="tab:red",
        lw=1.8,
        label="GDP: rolling",
    )
    ax.plot(
        common.index,
        100 * instantaneous,
        color="tab:orange",
        lw=1.4,
        ls="--",
        label="GDP: instantaneous (HP trend)",
    )
    ax.plot(
        common.index,
        100 * ceiling,
        color="tab:blue",
        lw=1.9,
        label="Granularity ceiling",
    )
    ax.plot(
        common.index,
        100 * cg_fundamental,
        color="tab:green",
        lw=1.6,
        ls="-.",
        label="Naive fundamental volatility",
    )
    plotted = [rolling, instantaneous, ceiling, cg_fundamental]
    if frozen_paths is not None:
        reference_colors = ("tab:blue", "tab:blue", "tab:blue")
        references = sorted(
            int(column.removeprefix("frozen").removesuffix("_volatility"))
            for column in frozen_paths.columns
            if column.startswith("frozen") and column.endswith("_volatility")
            and frozen_paths[column].notna().any()
        )
        years = common.index.to_numpy(dtype=float)
        for reference, color in zip(references, reference_colors):
            column = f"frozen{reference}_volatility"
            series = frozen_paths[column]
            ax.plot(
                years,
                100 * series,
                color=color,
                lw=1.05,
                ls=":",
                label=f"Frozen: {reference}",
                alpha=0.72,
                zorder=2,
            )
            ax.plot(
                [reference],
                [100 * frozen_paths.loc[reference, column]],
                marker="o",
                ms=3.8,
                color=color,
                alpha=0.9,
                zorder=4,
            )
            plotted.append(series)
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Year")
    ax.set_yscale("log")
    all_series = pd.concat(plotted)
    ax.set_ylim(bottom=100 * all_series.min() * 0.85, top=100 * all_series.max() * 1.15)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1, 2, 3, 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", which="both")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        columnspacing=1.1,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ps.save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    require_sources()
    _, tfp, weights, checks = build_inputs()
    print(f"[bea] historical TFP panel: {tfp.shape}; reconciliation: {checks}", flush=True)
    if args.validate_only:
        return

    paths = build_paths(tfp.to_numpy(dtype=float), weights, args.M, choose_solver(args.solver))
    figure_data = paths.rename(columns={"diagonal_volatility": "cg_diagonal_volatility"})
    plot_figure1(figure_data.join(gdp_volatility()), OUTPUT_DIR / "bea_granular_ceiling_figure1",
                 frozen_paths=paths)
    paths.to_csv(RUN_OUTPUT_CACHE / "bea_granular_volatility_paths.csv")


if __name__ == "__main__":
    main()
