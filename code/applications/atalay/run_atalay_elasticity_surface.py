#!/usr/bin/env python3
"""Primitive granularity over Atalay's whole elasticity rectangle, not one line of it.

``run_atalay_elasticity_profile.py`` walks one elasticity at a time -- eps_M with
(eps_D, eps_Q) at unity, then eps_Q with (eps_D, eps_M) at unity -- which is
Atalay's own Figure 3 and answers "how does the ceiling move along this line?".
It cannot answer "how high does the ceiling go anywhere in the parameter space
his estimation leaves open?", because the two elasticities his data actually
speak to, eps_M and eps_Q, are never varied together.

This script does.  ``export_atalay_elasticity_surface.m`` re-runs the deposited,
unmodified filter over a three-dimensional grid: one slab per value of eps_D --
the elasticity Atalay does not attempt to identify, and for which he "will
choose a wide range" -- and, within each slab, a full (eps_M, eps_Q) rectangle
from 0 to 1.5.  Here we price every grid point with the same
$M=10$ grouped bound
Section~5 uses and draw the result as one surface per eps_D.

WHICH FILTER.  Two coexist in the deposited code: the direct Equation-(11)
recursion, which is only defined where it passes Atalay's own screens (spectral
radius below one, state loading not numerically singular), and the Kalman filter
he falls back on otherwise.  On a one-dimensional profile it makes sense to plot
both and mark where each applies.  On a surface it does not: the recursion's
screens switch on and off along a band of diagonal stripes, so a surface that
splices the two is discontinuous wherever the stripes are, for a reason that has
nothing to do with the economics.  The Kalman filter is defined at EVERY
parameter vector, so we use it everywhere and the surface is one smooth object.
Little is given up by it: at Atalay's own benchmark the two are close, and
``report`` prints how far apart they
are across every cell where both are defined.  The recursion's ceilings stay in
the CSV as ``mgp_share``.

Atalay's own one-factor share is not on this grid: the export runs a truncated
copy of his filter that stops before the factor analyses, which is what makes a
15,900-point grid affordable.  The one-dimensional profile still carries it.

The script reports two numbers the draft quotes:

  * the (eps_D, eps_M, eps_Q) that maximizes the ceiling over the whole grid;
  * the maximum over Atalay's own credible set, ``CREDIBLE`` below -- the range
    his estimation and the papers he calibrates to actually support.

Writes ../output/atalay_elasticity_surface.{pdf,png} (the draft includes the 300-dpi PNG) and
.cache/run_outputs/atalay_elasticity_surface.csv.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per worker: every solve here is on a 10x10 matrix, and the
# parallelism that pays is across grid points, not inside them.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mgp_core import OUTPUT_DIR, ROOT, RUN_OUTPUT_CACHE, choose_solver
from applications.atalay.run_atalay_elasticity_profile import PROFILE_ROOT, UNITY, grouped_ceiling, rule_m_from_grid

# Imported after mgp_core, which points MPLCONFIGDIR at a writable cache.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import plot_style as ps

ps.use()

SURFACE_ROOT = ROOT / "code/.cache/derived/atalay2017/elasticity_surface"
SURFACE_CSV = RUN_OUTPUT_CACHE / "atalay_elasticity_surface.csv"
FIGURE_STEM = OUTPUT_DIR / "atalay_elasticity_surface"

# Atalay's benchmark calibration.
BENCHMARK = {"eps_D": 1.0, "eps_M": 0.10, "eps_Q": 1.00}

# Atalay's own credible set, from his Table 1 and Section II.C calibration
# discussion.  eps_M: "one cannot reject 0 (or slightly positive values) ... the
# right endpoint of a 90 percent confidence interval is approximately 0.2".
# eps_Q: Rotemberg-Woodford's 0.7, Oberfield-Raval's 0.6-0.9, his own online
# Appendix B's 0.4-0.8, and the conventional 1 he imposes for the benchmark.
# eps_D: not identified in the paper; Herrendorf-Rogerson-Valentinyi's 0.9 and
# Oberfield-Raval's 0.8-1.1, plus the benchmark 1.
CREDIBLE = {"eps_D": (0.80, 1.10), "eps_M": (0.00, 0.20), "eps_Q": (0.40, 1.00)}

FILTERS = {"": "Equation-(11) recursion", "_kf": "Kalman filter"}


def load_slabs(root: Path) -> list[Path]:
    """Paths to every slab file on disk, main pass and any tagged extra passes.

    Discovered by glob rather than from ``slab_index.csv``: the export checkpoints
    each slab as it goes and writes the index only when a process finishes, so
    globbing is what lets this run against a partial or still-running export.
    Rows the export has not reached are NaN, and ``price_slab`` skips them.
    """
    slabs = sorted(root.glob("atalay_surface_epsD*_*.mat"))
    if not slabs:
        raise FileNotFoundError(
            f"No slabs in {root}.  Run\n"
            f"  matlab -batch \"run('{ROOT / 'code' / 'export_atalay_elasticity_surface.m'}')\"\n"
            "to re-run the deposited Atalay filter over the 3-D grid first "
            "(or `make atalay-surface-export`, which runs the six slabs in parallel)."
        )
    return slabs


def _price_slab_path(payload: tuple[Path, int, str]) -> pd.DataFrame:
    """Worker entry point: load one slab from disk and price it."""
    path, m, solver = payload
    return price_slab(loadmat(path, squeeze_me=True), m, solver)


def price_slab(slab: dict[str, object], m: int, solver: str) -> pd.DataFrame:
    """The grouped ceiling at every (eps_M, eps_Q) cell of one eps_D slab."""
    eps_d = float(np.asarray(slab["eps_D"]))
    eps_m = np.atleast_1d(np.asarray(slab["eps_M_grid"], dtype=float))
    eps_q = np.atleast_1d(np.asarray(slab["eps_Q_grid"], dtype=float))
    ok = np.atleast_2d(np.asarray(slab["ok"], dtype=bool))
    rho = np.atleast_2d(np.asarray(slab["rho"], dtype=float))
    min_eig = np.atleast_2d(np.asarray(slab["min_eig"], dtype=float))
    shocks = np.asarray(slab["shocks"], dtype=float)
    shocks_kf = np.asarray(slab["shocks_kf"], dtype=float)
    network = np.asarray(slab["V"], dtype=float)

    rows: list[dict[str, object]] = []
    for i, value_m in enumerate(eps_m):
        for j, value_q in enumerate(eps_q):
            row: dict[str, object] = {
                "eps_D": eps_d, "eps_M": value_m, "eps_Q": value_q,
                "M": m,
                "solved": bool(ok[i, j]), "filter_rho": rho[i, j],
                "filter_min_eig": min_eig[i, j],
            }
            if ok[i, j]:
                # Atalay's own screens on the Equation-(11) recursion.
                row["eq11_screen_passes"] = bool(rho[i, j] < 1.0 and min_eig[i, j] > 0.01)
                v = 0.5 * (network[i, j] + network[i, j].T)
                for suffix, series in (("", shocks[i, j]), ("_kf", shocks_kf[i, j])):
                    if suffix == "" and not row["eq11_screen_passes"]:
                        continue
                    stats = grouped_ceiling(series, v, m, solver)
                    row.update({f"{k}{suffix}": v_ for k, v_ in stats.items()})
            else:
                row["eq11_screen_passes"] = False
            rows.append(row)
        print(f"[surface] eps_D={eps_d:.3f} eps_M={value_m:.3f}: "
              f"{int(ok[i].sum())}/{len(eps_q)} solved", flush=True)
    return pd.DataFrame(rows)


def preferred(table: pd.DataFrame) -> pd.DataFrame:
    """The Kalman ceiling at every cell, which is the surface we draw.

    Either ceiling column is optional, because a partial export can contain no cell
    at which that filter was usable at all.
    """
    for column in ("mgp_share", "mgp_share_kf"):
        if column not in table:
            table[column] = np.nan
    return table.assign(mgp_share_preferred=table["mgp_share_kf"])


def common_rectangle(table: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the (eps_M, eps_Q) box every slab actually priced.

    The export walks eps_M upwards and stops on a wall-clock budget, and the
    deposited filter's own factor analyses get dramatically more expensive at the
    top of that range, so different slabs can end at different eps_M.  Comparing
    one slab's maximum with another's only means something over a box they share.
    """
    priced = table[table["mgp_share_preferred"].notna()]
    if priced.empty:
        return table
    limit_m = priced.groupby("eps_D")["eps_M"].max().min()
    limit_q = priced.groupby("eps_D")["eps_Q"].max().min()
    dropped = int((table["eps_M"] > limit_m).sum() + (table["eps_Q"] > limit_q).sum())
    if dropped:
        print(f"[surface] restricting to the common rectangle eps_M <= {limit_m:g}, "
              f"eps_Q <= {limit_q:g}: {dropped} cells outside it are kept in the CSV only",
              flush=True)
    return table[table["eps_M"].le(limit_m) & table["eps_Q"].le(limit_q)]


def _within(table: pd.DataFrame, box: dict[str, tuple[float, float]]) -> pd.Series:
    inside = pd.Series(True, index=table.index)
    for name, (low, high) in box.items():
        value = table[name].replace(UNITY, 1.0)
        inside &= value.between(low - 1e-9, high + 1e-9)
    return inside


def render(table: pd.DataFrame, m: int, stem: Path) -> Path:
    """One panel per eps_D, each a surface over (eps_M, eps_Q)."""
    slabs = sorted(table["eps_D"].unique())
    rows = int(np.ceil(len(slabs) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(9.4, 3.1 * rows + 0.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    levels = np.linspace(0.0, float(np.nanmax(table["mgp_share_preferred"])), 21)
    mesh = None
    for ax, eps_d in zip(axes, slabs):
        block = table[table["eps_D"].eq(eps_d)]
        grid = block.pivot(index="eps_Q", columns="eps_M", values="mgp_share_preferred")
        x, y = grid.columns.to_numpy(float), grid.index.to_numpy(float)
        mesh = ax.contourf(x, y, grid.to_numpy(float), levels=levels, cmap=ps.HEAT)
        # His credible set for the two elasticities he does estimate.
        (m_lo, m_hi), (q_lo, q_hi) = CREDIBLE["eps_M"], CREDIBLE["eps_Q"]
        ax.add_patch(Rectangle((m_lo, q_lo), m_hi - m_lo, q_hi - q_lo, fill=False,
                               edgecolor="white", lw=1.8, zorder=4))
        ax.add_patch(Rectangle((m_lo, q_lo), m_hi - m_lo, q_hi - q_lo, fill=False,
                               edgecolor=ps.INK, lw=0.9, zorder=5))
        if np.isclose(1.0 if eps_d == UNITY else eps_d, BENCHMARK["eps_D"]):
            ax.plot([BENCHMARK["eps_M"]], [BENCHMARK["eps_Q"]], marker="*", ms=7,
                    color="white", markeredgecolor=ps.INK, markeredgewidth=0.7, zorder=6)
        inside = _within(block, {k: v for k, v in CREDIBLE.items() if k != "eps_D"})
        credible_max = block.loc[inside, "mgp_share_preferred"].max()
        eps_d_in = CREDIBLE["eps_D"][0] <= (1.0 if eps_d == UNITY else eps_d) <= CREDIBLE["eps_D"][1]
        # A dagger marks the eps_D panels Atalay's own calibration range covers;
        # the two summary numbers go inside the panel, where there is room.
        ax.set_title(rf"$\varepsilon_D={1.0 if eps_d == UNITY else eps_d:g}$"
                     + (r"$^{\dagger}$" if eps_d_in else ""))
        ax.text(0.97, 0.95,
                f"panel max {block['mgp_share_preferred'].max():.0%}\n"
                f"in box {credible_max:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.2),
                zorder=7)
        ax.grid(False)
    for ax in axes[len(slabs):]:
        ax.set_visible(False)
    # Bound the axes by what was actually priced: the deposited filter's factor
    # analyses get expensive at the top of the eps_M range and the export stops on
    # a wall-clock budget, so the grid can end short of 1.5 there.
    priced = table[table["mgp_share_preferred"].notna()]
    for ax in axes[: len(slabs)]:
        ax.set_xlim(0.0, float(priced["eps_M"].max()))
        ax.set_ylim(0.0, float(priced["eps_Q"].max()))
    for ax in axes[len(slabs) - 2 : len(slabs)]:
        ax.set_xlabel(r"$\varepsilon_M$: across material inputs")
    for ax in axes[: len(slabs)][::2]:
        ax.set_ylabel(r"$\varepsilon_Q$: value added vs. intermediates")

    # Wrapped: a single long line would widen the whole figure, since the style
    # saves with a tight bounding box.
    fig.text(0.5, 0.005,
             "$\\dagger$: $\\varepsilon_D$ inside the range Atalay calibrates to, so only those "
             "panels' 'in box' figures sit inside his credible set.\n"
             "White box: his credible set for $(\\varepsilon_M,\\varepsilon_Q)$; star: his "
             "benchmark calibration.",
             ha="center", va="bottom", fontsize=7.6, color="0.25", linespacing=1.5)
    bar = fig.colorbar(mesh, ax=axes[: len(slabs)].tolist(), shrink=0.75, pad=0.02)
    bar.set_label(rf"Granularity ceiling $\hat{{\bar\gamma}}$, $M={m}$, Kalman filter")
    bar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    bar.update_ticks()
    return ps.save(fig, stem, suffix=".png", dpi=300)


def report(table: pd.DataFrame, m: int) -> None:
    print("\n" + "=" * 78)
    print(f"Primitive-granularity surface, Atalay (2017) filter, M={m}")
    print("=" * 78)
    solved = table[table["mgp_share_preferred"].notna()]
    print(f"{len(solved)}/{len(table)} grid points priced over "
          f"{table['eps_D'].nunique()} eps_D slabs")

    best = solved.loc[solved["mgp_share_preferred"].idxmax()]
    both = solved.dropna(subset=["mgp_share"])
    if len(both):
        gap = (both["mgp_share_kf"] - both["mgp_share"]).abs()
        print(f"Atalay's Equation-(11) recursion is also defined at "
              f"{len(both)}/{len(solved)} of them; where both exist the two ceilings "
              f"differ by {gap.median():.2%} at the median and {gap.max():.2%} at worst")

    print(f"\nGRID MAXIMUM: {best['mgp_share_preferred']:.2%} at "
          f"eps_D={best['eps_D']:g}, eps_M={best['eps_M']:g}, eps_Q={best['eps_Q']:g}")

    inside = solved[_within(solved, CREDIBLE)]
    if len(inside):
        top = inside.loc[inside["mgp_share_preferred"].idxmax()]
        print(f"\nOVER ATALAY'S CREDIBLE SET "
              f"(eps_D in {CREDIBLE['eps_D']}, eps_M in {CREDIBLE['eps_M']}, "
              f"eps_Q in {CREDIBLE['eps_Q']}), {len(inside)} grid points:")
        print(f"  max {top['mgp_share_preferred']:.2%} at eps_D={top['eps_D']:g}, "
              f"eps_M={top['eps_M']:g}, eps_Q={top['eps_Q']:g}")
        print(f"  min {inside['mgp_share_preferred'].min():.2%}, "
              f"median {inside['mgp_share_preferred'].median():.2%}")

    bench = solved[
        np.isclose(solved["eps_M"], BENCHMARK["eps_M"])
        & np.isclose(solved["eps_Q"].replace(UNITY, 1.0), BENCHMARK["eps_Q"])
        & np.isclose(solved["eps_D"].replace(UNITY, 1.0), BENCHMARK["eps_D"])
    ]
    for _, row in bench.iterrows():
        print(f"\nAt Atalay's benchmark (1, 0.1, 1): {row['mgp_share_preferred']:.2%} "
              f"(Kalman), against {row['mgp_share']:.2%} for his Equation-(11) recursion")

    print("\nPer-slab maxima:")
    for eps_d, block in solved.groupby("eps_D"):
        row = block.loc[block["mgp_share_preferred"].idxmax()]
        print(f"  eps_D={eps_d:g}: max {row['mgp_share_preferred']:.2%} at "
              f"eps_M={row['eps_M']:g}, eps_Q={row['eps_Q']:g}")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=None,
                        help="LPT groups; default 10, capped by N.")
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--surface-root", type=Path, default=SURFACE_ROOT)
    parser.add_argument("--out-csv", type=Path, default=SURFACE_CSV)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="Slabs to price in parallel (one process each).")
    parser.add_argument("--from-cache", action="store_true",
                        help="Redraw from the cached CSV.")
    args = parser.parse_args()

    if args.M is None:
        args.M = rule_m_from_grid(PROFILE_ROOT)
    if args.from_cache:
        table = pd.read_csv(args.out_csv)
        if "M" not in table or not table["M"].eq(args.M).all():
            raise ValueError("Cached surface has a different or unrecorded M; "
                             "reprice the saved filter exports without --from-cache.")
    else:
        solver = choose_solver(args.solver)
        payloads = [(path, args.M, solver) for path in load_slabs(args.surface_root)]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                frames = list(pool.map(_price_slab_path, payloads))
        else:
            frames = [_price_slab_path(payload) for payload in payloads]
        table = pd.concat(frames, ignore_index=True).sort_values(
            ["eps_D", "eps_M", "eps_Q"], ignore_index=True
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
    table = common_rectangle(preferred(table))
    figure = render(table, args.M, args.figure_stem)
    report(table, args.M)
    print(f"Wrote {figure} and {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
