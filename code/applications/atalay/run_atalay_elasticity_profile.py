#!/usr/bin/env python3
"""Primitive-granularity profiles in Atalay's production elasticities.

Section~5 reports the grouped-MGP ceiling on primitive granularity at Atalay's
published calibration.  The natural objection is that the ceiling is low only
because the structural filter that recovers the primitives is parameterised
slightly wrongly: observed sectoral comovement identifies the *product* of
propagation and primitive commonality, not either separately, so a lower
elasticity of substitution buys more propagation and leaves less observed
comovement to be explained by common primitive shocks.

This script turns that objection into an estimable object.  For each candidate
elasticity on a grid, the deposited Atalay filter is re-run by
``export_atalay_elasticity_profile.m`` (nothing in the deposited package is
edited) to recover primitive innovations and the aggregate-output variance
matrix ``V``.  We then apply the same $M=10$ grouped bound Section~5 uses and
record the *primitive-granularity profile*

    Gbar(theta) = grouped-MGP ceiling on the primitive granular share,

and invert it into the identified sets

    Theta(g) = {theta : Gbar(theta) >= g},

the elasticities not ruled out by a maintained lower bound ``g`` on primitive
granularity.  Sampling uncertainty is out of scope: every number here is the
point estimate at that parameter value.

The figure follows Atalay's own Figure 3: two panels, one per elasticity, each
holding the others at unity, with both of the inversions ``filter_full.m``
computes -- the direct Equation-(11) recursion, which produces his benchmark
83 percent, and the Kalman filter, which he falls back on wherever the
recursion fails its screens and which produces his 21 percent under unit
elasticities.

Writes ../output/atalay_elasticity_profile.{pdf,png},
.cache/run_outputs/atalay_elasticity_profile.csv and
.cache/run_outputs/atalay_elasticity_profile_identified_sets.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.io import loadmat

from mgp_core import (
    rule_m,
    OUTPUT_DIR,
    ROOT,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    build_lpt_grouped_data,
    choose_solver,
    mean_pairwise_correlation,
)
from applications.foerster.run_foerster_epsilon import network_observed_stats


PROFILE_ROOT = ROOT / "code/.cache/derived/atalay2017/elasticity_profile"
PROFILE_CSV = RUN_OUTPUT_CACHE / "atalay_elasticity_profile.csv"
FIGURE_STEM = OUTPUT_DIR / "atalay_elasticity_profile"

# Unity is the package's stand-in for the Cobb-Douglas case: the model is
# singular at exactly one.
UNITY = 0.999

# Panels, in the order they are drawn.  ``benchmark`` is the value Atalay's
# Section III results actually use; ``xmax`` bounds the drawn grid.
SWEEPS: dict[str, dict[str, object]] = {
    "eps_M": {
        "label": r"Elasticity across material inputs $\varepsilon_M$",
        # Chosen as "the middle of positive portion of the 90 percent confidence
        # interval": the IV point estimates are negative, around -0.1.
        "benchmark": 0.10,
        "benchmark_label": r"benchmark $\varepsilon_M=0.10$",
        "xmax": 1.50,
    },
    "eps_Q": {
        "label": r"Elasticity between value added and intermediate inputs $\varepsilon_Q$",
        # "I will choose the conventional value of 1 for eps_Q for the benchmark
        # parameter configuration"; the IV specifications give 0.8-0.9 and unit
        # elasticities cannot be rejected.  Note this benchmark is *not* his own
        # IV-estimated range: he imposes the conventional value instead.
        "benchmark": 1.0,
        "benchmark_label": r"benchmark $\varepsilon_Q=1$",
        "xmax": 1.55,
    },
}

# Maintained lower bounds on primitive granularity to invert the profiles at.
GRANULARITY_THRESHOLDS = (0.35, 0.40, 0.50, 0.80, 0.90, 1.00)

import plot_style as ps

ps.use()

CEILING_COLOR, FACTOR_COLOR, EXPLOSIVE_COLOR = "tab:blue", "tab:orange", "tab:red"

FILTERS = {"": "Equation-(11) recursion", "_kf": "Kalman filter"}


def grouped_ceiling(shocks: np.ndarray, network_v: np.ndarray, m: int, solver: str) -> dict[str, float]:
    """Section~5's grouped primitive-shock bound for one structural filter.

    ``shocks`` is T x N primitive innovations, ``network_v`` the N x N matrix
    with aggregate-output variance ``trace(V Sigma_eps)``.  Total dynamic
    influence is ``V_i = sqrt(V_ii)``, exactly as in ``run_atalay_epsilon.py``;
    units are sorted by decreasing influence as everywhere else in the paper, so
    the LPT grouping is the same construction.
    """
    n = network_v.shape[0]
    if shocks.shape[1] != n:
        raise ValueError(f"shocks have {shocks.shape[1]} columns, V is {n}x{n}.")
    ids = pd.Index([f"shock_{i + 1:02d}" for i in range(n)], name="shock_id")
    y = pd.DataFrame(shocks.T, index=ids)
    sizes = pd.Series(np.sqrt(np.maximum(np.diag(network_v), 0.0)), index=ids)
    y, sizes = align_sizes_to_panel(y, sizes)
    order = [int(str(idx).split("_")[-1]) - 1 for idx in y.index]
    if sorted(order) != list(range(n)):
        raise ValueError("Influence sort dropped or duplicated an industry.")
    v_sorted = network_v[np.ix_(order, order)]

    shares = (sizes / sizes.sum()).to_numpy(float)
    y_np = y.to_numpy(float)
    x_grouped, *_ = build_lpt_grouped_data(y_np, shares, m)
    observed = network_observed_stats(
        x_grouped, y_np.T, v_sorted, float(sizes.sum() ** 2), solver
    )
    return {
        "mgp_share": float(observed["mgp_share"]),
        "mgp_level": float(observed["mgp_level"]),
        "aggregate_variance": float(observed["aggregate_variance"]),
        "se_share": float(observed["se_share"]),
        "influence_hhi": float(np.sum(shares**2)),
        "top1_influence_share": float(shares[0]),
        "mean_pairwise_corr": mean_pairwise_correlation(y_np),
    }


def load_grid(profile_dir: Path, sweep: str) -> pd.DataFrame:
    index_path = profile_dir / "grid_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}.  Run\n"
            f"  matlab -batch \"ATALAY_SWEEP='{sweep}'; "
            f"run('{ROOT / 'code' / 'export_atalay_elasticity_profile.m'}')\"\n"
            "to re-run the deposited Atalay filter over that grid first."
        )
    index = pd.read_csv(index_path)
    if not index["sweep"].eq(sweep).all():
        raise ValueError(f"{index_path} is not a {sweep} grid.")
    return index


def rule_m_from_grid(profile_root: Path) -> int:
    """Paper-wide M=10, capped by N, read from the first exported shock panel."""
    for sweep in SWEEPS:
        index = load_grid(profile_root / sweep, sweep)
        ok = index[index["status"].astype(str).eq("ok") & index["file"].astype(str).str.strip().ne("")]
        if not ok.empty:
            shocks = np.asarray(loadmat(profile_root / sweep / str(ok.iloc[0]["file"]), squeeze_me=True)["shocks"], dtype=float)
            return rule_m(shocks.shape[1])   # shocks are T x N
    raise FileNotFoundError(f"No successful filter export under {profile_root}")


def build_profile(profile_root: Path, m: int, solver: str) -> pd.DataFrame:
    frames = []
    for sweep in SWEEPS:
        profile_dir = profile_root / sweep
        index = load_grid(profile_dir, sweep)
        rows: list[dict[str, object]] = []
        for _, entry in index.iterrows():
            value = float(entry["value"])
            status = str(entry["status"])
            if status != "ok" or not str(entry["file"]).strip():
                rows.append({"sweep": sweep, "value": value, "filter_status": status})
                print(f"[{sweep}={value:.3f}] filter failed: {status}", flush=True)
                continue
            payload = loadmat(profile_dir / str(entry["file"]), squeeze_me=True)
            network_v = np.asarray(payload["V"], dtype=float)
            network_v = 0.5 * (network_v + network_v.T)

            # Atalay's own screens on the recursive Equation-(11) filter: a
            # spectral radius below one (otherwise the recursion is explosive
            # and, in his words, "data on output changes alone cannot fully
            # identify the productivity shocks") and a state loading that is not
            # numerically singular.  filter_full.m applies both before it will
            # use the Equation-(11) innovations at all.
            rho = float(payload["max_abs_eig_final_filter_3_A"])
            min_eig = float(payload["min_abs_eig_filter_matrix_1_A"])
            eq11_ok = rho < 1.0 and min_eig > 0.01
            row: dict[str, object] = {
                "sweep": sweep,
                "value": value,
                "filter_status": "ok",
                "filter_rho": rho,
                "filter_min_eig": min_eig,
                "eq11_screen_passes": eq11_ok,
                # Atalay's own object at the same parameter: the share of
                # aggregate-output variance his one-factor split assigns to
                # sector-specific shocks, against his fitted covariance.
                "atalay_one_factor_share": float(payload["one_factor_share"]),
                "atalay_one_factor_share_kf": float(payload["one_factor_share_kf"]),
            }
            for suffix, key in (("", "shocks"), ("_kf", "shocks_kf")):
                if suffix == "" and not eq11_ok:
                    continue
                shocks = np.asarray(payload[key], dtype=float)
                row.update(
                    {f"{name}{suffix}": v
                     for name, v in grouped_ceiling(shocks, network_v, m, solver).items()}
                )
            rows.append(row)
            eq11 = row.get("mgp_share", np.nan)
            print(
                f"[{sweep}={value:.3f}] eq11_screen={eq11_ok} "
                f"Gbar(eq11)={'n/a  ' if not eq11_ok else f'{eq11:.4f}'} "
                f"Gbar(kf)={row['mgp_share_kf']:.4f}",
                flush=True,
            )
        frames.append(pd.DataFrame(rows).sort_values("value"))
    return pd.concat(frames, ignore_index=True)


def crossings(x: np.ndarray, share: np.ndarray, level: float) -> list[float]:
    """Parameter values at which a profile crosses ``level``, interpolated."""
    out: list[float] = []
    for a, b, ga, gb in zip(x[:-1], x[1:], share[:-1], share[1:]):
        if ga == level:
            out.append(float(a))
        if (ga - level) * (gb - level) < 0.0:
            out.append(float(a + (b - a) * (level - ga) / (gb - ga)))
    if share.size and share[-1] == level:
        out.append(float(x[-1]))
    return sorted(set(out))


def identified_sets(profile: pd.DataFrame, sweep: str, column: str) -> pd.DataFrame:
    """Theta(g) = {value on the grid : Gbar(value) >= g}, as interval endpoints."""
    d = profile[profile["sweep"].eq(sweep)].dropna(subset=[column]).sort_values("value")
    x = d["value"].to_numpy(float)
    share = d[column].to_numpy(float)
    rows = []
    for level in GRANULARITY_THRESHOLDS:
        inside = share >= level
        segments: list[tuple[float, float]] = []
        start: float | None = None
        for i, flag in enumerate(inside):
            if flag and start is None:
                start = float(x[i])
            if start is not None and (not flag or i == len(inside) - 1):
                segments.append((start, float(x[i] if flag else x[i - 1])))
                start = None
        rows.append(
            {
                "sweep": sweep,
                "column": column,
                "threshold": level,
                "attained": bool(inside.any()),
                "n_grid_points": int(inside.sum()),
                "grid_min_value": segments[0][0] if segments else np.nan,
                "grid_max_value": segments[-1][1] if segments else np.nan,
                "crossings": ";".join(f"{c:.4f}" for c in crossings(x, share, level)),
                "profile_max": float(share.max()) if share.size else np.nan,
                "argmax_value": float(x[int(np.argmax(share))]) if share.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def render_plot(profile: pd.DataFrame, m: int, stem: Path) -> None:
    """Atalay's Figure 3 layout and marker convention: panel A is eps_M, panel B
    is eps_Q, both as markers rather than lines.  Per his note, "Hollow circles
    denote figures that result from the model filter, with omega_0 fixed at 0,
    iteratively applying equation (11).  '+' signs denote the figures that
    result from the Kalman filter."
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.25))
    for ax, (sweep, spec) in zip(axes, SWEEPS.items()):
        d = profile[profile["sweep"].eq(sweep)].sort_values("value")
        explosive = d.loc[d["filter_rho"] >= 1.0, "value"]
        if not explosive.empty:
            ax.axvspan(0.0, float(explosive.max()), color=EXPLOSIVE_COLOR, alpha=0.22,
                       linewidth=0.0, zorder=1)
        ax.axvline(float(spec["benchmark"]), color="0.35", linewidth=1.1,
                   linestyle="--", zorder=2)

        # Atalay's own Figure 3 convention: hollow circles for the direct
        # Equation-(11) recursion (his "model filter, with omega_0 fixed at 0"),
        # "+" signs for the Kalman filter.
        for column, color in (("mgp_share", CEILING_COLOR),
                              ("atalay_one_factor_share", FACTOR_COLOR)):
            ax.plot(
                d["value"], d[column], color=color, linestyle="none", marker="o",
                markersize=4.2, markerfacecolor="none", markeredgewidth=1.1, zorder=4,
            )
        for column, color in (("mgp_share_kf", CEILING_COLOR),
                              ("atalay_one_factor_share_kf", FACTOR_COLOR)):
            ax.plot(
                d["value"], d[column], color=color, linestyle="none", marker="+",
                markersize=5.6, markeredgewidth=1.1, zorder=5,
            )

        ax.set_title(spec["label"])
        ax.set_xlim(0.0, float(spec["xmax"]))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


    axes[0].set_ylabel("Granular share of aggregate variance")
    handles = [
        Line2D([], [], color=FACTOR_COLOR, linestyle="none", marker="s", markersize=6,
               label="One-factor purging"),
        Line2D([], [], color=CEILING_COLOR, linestyle="none", marker="s", markersize=6,
               label="Factor-free ceiling"),
        Line2D([], [], color="0.35", linestyle="none", marker="o", markersize=4.2,
               markerfacecolor="none", markeredgewidth=1.1, label="Recursion"),
        Line2D([], [], color="0.35", linestyle="none", marker="+", markersize=5.6,
               markeredgewidth=1.1, label="Kalman filter"),
        Patch(facecolor=EXPLOSIVE_COLOR, alpha=0.22, edgecolor="none",
              label="Explosive dynamics"),
        Line2D([], [], color="0.35", linestyle="--", linewidth=1.1,
               label="Benchmark"),
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=7.5, frameon=True,
                   facecolor="white", edgecolor="0.45", framealpha=0.93)
    fig.tight_layout()
    ps.save(fig, stem)


def report(profile: pd.DataFrame, sets: pd.DataFrame, m: int) -> None:
    print("\n" + "=" * 78)
    print(f"Primitive-granularity profiles, Atalay (2017) filter, M={m}")
    print("=" * 78)
    for sweep, spec in SWEEPS.items():
        d = profile[profile["sweep"].eq(sweep)]
        print(f"\n### {sweep}: {len(d)} grid points in "
              f"[{d['value'].min():.3f}, {d['value'].max():.3f}]")
        explosive = d[d["filter_rho"] >= 1.0]["value"]
        if not explosive.empty:
            print(f"  Equation-(11) recursion explosive for {sweep} <= {explosive.max():.3f}")
        for suffix, label in FILTERS.items():
            column = f"mgp_share{suffix}"
            usable = d.dropna(subset=[column])
            print(f"  -- {label} ({len(usable)}/{len(d)} usable)")
            hit = usable[np.isclose(usable["value"], float(spec["benchmark"]))
                         | np.isclose(usable["value"], UNITY)
                         & np.isclose(float(spec["benchmark"]), 1.0)]
            for _, row in hit.iterrows():
                print(
                    f"     at benchmark {row['value']:.3f}: Gbar={row[column]:.4f}, "
                    f"Atalay's own share={row['atalay_one_factor_share' + suffix]:.4f}"
                )
            if usable.empty:
                continue
            best = usable.loc[usable[column].idxmax()]
            print(f"     max over the grid: {best[column]:.4f} at {sweep}={best['value']:.3f}")
            for _, row in sets[sets["sweep"].eq(sweep) & sets["column"].eq(column)].iterrows():
                if not row["attained"]:
                    print(f"     Theta({row['threshold']:.2f}): empty")
                else:
                    print(
                        f"     Theta({row['threshold']:.2f}): {row['n_grid_points']} points, "
                        f"{row['grid_min_value']:.3f} to {row['grid_max_value']:.3f}; "
                        f"crossing(s) at {row['crossings'] or 'none'}"
                    )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=None,
                        help="LPT groups; default 10, capped by N.")
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    parser.add_argument("--out-csv", type=Path, default=PROFILE_CSV)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    parser.add_argument("--from-cache", action="store_true", help="Redraw from the cached CSV.")
    args = parser.parse_args()

    if args.from_cache:
        profile = pd.read_csv(args.out_csv)
    else:
        if args.M is None:
            args.M = rule_m_from_grid(args.profile_root)
        profile = build_profile(args.profile_root, args.M, choose_solver(args.solver))
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.out_csv, index=False)

    sets = pd.concat(
        [identified_sets(profile, sweep, f"mgp_share{suffix}")
         for sweep in SWEEPS for suffix in FILTERS],
        ignore_index=True,
    )
    sets.to_csv(args.out_csv.with_name(args.out_csv.stem + "_identified_sets.csv"), index=False)
    args.figure_stem.parent.mkdir(parents=True, exist_ok=True)
    render_plot(profile, args.M, args.figure_stem)
    report(profile, sets, args.M)
    print(f"Wrote {args.figure_stem.with_suffix('.pdf')} and {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
