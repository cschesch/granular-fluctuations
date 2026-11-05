#!/usr/bin/env python3
"""Compare one-factor shares and granularity ceilings over Atalay's elasticities.

Each profile holds the other elasticities at unity and shows both the direct
recursion (where its stability and conditioning checks pass) and Kalman filter.
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

from mgp_core import OUTPUT_DIR, RUN_OUTPUT_CACHE, choose_solver
from applications.atalay.common import PROFILE_ROOT, grouped_ceiling


PROFILE_CSV = RUN_OUTPUT_CACHE / "atalay_elasticity_profile.csv"
FIGURE_STEM = OUTPUT_DIR / "atalay_elasticity_profile"

# Panels, in the order they are drawn.  ``benchmark`` is the value Atalay's
# Section III results actually use; ``xmax`` bounds the drawn grid.
SWEEPS: dict[str, dict[str, object]] = {
    "eps_M": {
        "label": r"Elasticity between material inputs $\varepsilon_M$",
        # Chosen as "the middle of positive portion of the 90 percent confidence
        # interval": the IV point estimates are negative, around -0.1.
        "benchmark": 0.10,
        "xmax": 1.50,
    },
    "eps_Q": {
        "label": r"Elasticity between the capital-labor bundle" "\n" r"and intermediate inputs $\varepsilon_Q$",
        # "I will choose the conventional value of 1 for eps_Q for the benchmark
        # parameter configuration"; the IV specifications give 0.8-0.9 and unit
        # elasticities cannot be rejected.  Note this benchmark is *not* his own
        # IV-estimated range: he imposes the conventional value instead.
        "benchmark": 1.0,
        "xmax": 1.55,
    },
}

import plot_style as ps

ps.use()

CEILING_COLOR, FACTOR_COLOR, EXPLOSIVE_COLOR = "tab:blue", "tab:orange", "tab:red"


def load_grid(profile_dir: Path, sweep: str) -> pd.DataFrame:
    index_path = profile_dir / "grid_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}; run python -m applications.atalay.prepare --section profile."
        )
    index = pd.read_csv(index_path)
    if not index["sweep"].eq(sweep).all():
        raise ValueError(f"{index_path} is not a {sweep} grid.")
    return index


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
                raise RuntimeError(f"Atalay {sweep}={value:g}: filter failed ({status}).")
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
                "M": m,
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

        ax.set_xlabel(spec["label"])
        ax.set_xlim(0.0, float(spec["xmax"]))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


    axes[0].set_ylabel("Granular share of aggregate fluctuations")
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
              label="Unstable shock recovery"),
        Line2D([], [], color="0.35", linestyle="--", linewidth=1.1,
               label="Benchmark"),
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=7.5, frameon=True,
                   facecolor="white", edgecolor="0.45", framealpha=0.93)
    fig.tight_layout()
    ps.save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=10, help="LPT groups; default 10.")
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    parser.add_argument("--out-csv", type=Path, default=PROFILE_CSV)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    parser.add_argument("--from-cache", action="store_true", help="Redraw from the cached CSV.")
    args = parser.parse_args()

    if args.from_cache:
        profile = pd.read_csv(args.out_csv)
        if "M" in profile and not profile["M"].eq(args.M).all():
            raise ValueError(f"Cached profile does not use M={args.M}; rebuild it without --from-cache.")
    else:
        profile = build_profile(args.profile_root, args.M, choose_solver(args.solver))
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.out_csv, index=False)

    args.figure_stem.parent.mkdir(parents=True, exist_ok=True)
    render_plot(profile, args.M, args.figure_stem)
    print(f"Atalay profile: {len(profile)} grid points, M={args.M}; wrote {args.figure_stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
