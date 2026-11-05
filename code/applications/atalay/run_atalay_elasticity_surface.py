#!/usr/bin/env python3
"""Compare Atalay's one-factor estimate with the grouped ceiling across calibrations.

Both columns use Kalman-filter shocks on the same three elasticity grids.
The one-factor fits come from the deposited MATLAB routine; Python computes
MGP ceilings and draws the calibration comparison used in the appendix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import os
from pathlib import Path

# Parallelize across slabs, with one BLAS thread per worker.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mgp_core import OUTPUT_DIR, RUN_OUTPUT_CACHE, choose_solver
from applications.atalay.common import SURFACE_ROOT, SURFACE_SLABS, UNITY, grouped_ceiling

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import plot_style as ps


SURFACE_CSV = RUN_OUTPUT_CACHE / "atalay_surface_comparison.csv"
FIGURE_STEM = OUTPUT_DIR / "atalay_surface_comparison"
PARAMETERS = ["eps_D", "eps_M", "eps_Q"]


def load_slabs(root: Path) -> list[Path]:
    paths = [root / f"atalay_surface_epsD_{tag}.mat" for tag in SURFACE_SLABS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepare Atalay's surface first; missing " + ", ".join(missing))
    return paths


def price_slab(payload: tuple[Path, int, str]) -> pd.DataFrame:
    """Compute each Kalman ceiling and join its corresponding one-factor fit."""
    path, m, solver = payload
    slab = loadmat(path, squeeze_me=True)
    eps_d = float(slab["eps_D"])
    eps_m = np.atleast_1d(slab["eps_M_grid"])
    eps_q = np.atleast_1d(slab["eps_Q_grid"])
    shape = (len(eps_m), len(eps_q))
    ok = np.asarray(slab["ok"]).reshape(shape)
    if int(slab["rows_done"]) != len(eps_m) or not ok.all():
        raise ValueError(f"Incomplete Atalay surface: {path}")
    shocks = np.asarray(slab["shocks_kf"], dtype=float).reshape(*shape, -1, 30)
    network = np.asarray(slab["V"], dtype=float).reshape(*shape, 30, 30)
    rows = []
    for i, value_m in enumerate(eps_m):
        for j, value_q in enumerate(eps_q):
            v = 0.5 * (network[i, j] + network[i, j].T)
            share = grouped_ceiling(shocks[i, j], v, m, solver)["mgp_share"]
            rows.append((eps_d, float(value_m), float(value_q), share))
    ceilings = pd.DataFrame(rows, columns=[*PARAMETERS, "mgp_share_kf"])
    tag = path.stem.rsplit("_", 1)[-1]
    fits = pd.read_csv(path.with_name(f"one_factor_{tag}.csv"))
    # MATLAB and CSV round-trips can differ at the final decimal place.
    keys = [f"join_{name}" for name in PARAMETERS]
    fits[keys] = fits[PARAMETERS].round(12)
    ceilings[keys] = ceilings[PARAMETERS].round(12)
    table = fits.merge(ceilings[[*keys, "mgp_share_kf"]], on=keys,
                       how="outer", validate="one_to_one", indicator=True)
    if len(table) != len(rows) or not table["_merge"].eq("both").all():
        raise ValueError(f"One-factor fits do not match the complete grid: {path}")
    table = table[[*PARAMETERS, "one_factor", "mgp_share_kf", "fit_limit"]].assign(M=m)
    if not np.isfinite(table.to_numpy(float)).all():
        raise ValueError(f"Non-finite Atalay surface estimates: {path}")
    print(f"[atalay] eps_D={eps_d:g}: {len(table)} calibration points", flush=True)
    return table


def build_surface(root: Path, m: int, solver: str, jobs: int = 3) -> pd.DataFrame:
    payloads = [(path, m, solver) for path in load_slabs(root)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(payloads))) as pool:
            frames = list(pool.map(price_slab, payloads))
    else:
        frames = [price_slab(payload) for payload in payloads]
    return pd.concat(frames, ignore_index=True).sort_values(PARAMETERS, ignore_index=True)


def render(table: pd.DataFrame, stem: Path) -> Path:
    """Draw the one-factor and factor-free surfaces with a common color scale."""
    slabs = sorted(table.eps_D.unique())
    if len(slabs) != 3 or not np.allclose(slabs, [2 / 3, UNITY, 4 / 3], rtol=0, atol=1e-12):
        raise ValueError("The paper surface requires eps_D = 2/3, 0.999, 4/3.")
    ps.use()
    levels = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(len(slabs), 2, figsize=(9.8, 12), sharex=True,
                             sharey=True, layout="constrained")
    fig.get_layout_engine().set(hspace=.09)
    for row, d in enumerate(slabs):
        block = table[table.eps_D == d]
        label = "0.67" if np.isclose(d, 2 / 3) else ("1.00" if d == UNITY else "1.33")
        for col, field in enumerate(("one_factor", "mgp_share_kf")):
            ax = axes[row, col]
            grid = block.pivot(index="eps_Q", columns="eps_M", values=field)
            if grid.isna().any().any():
                raise ValueError(f"Incomplete {field} surface at eps_D={d:g}.")
            mesh = ax.contourf(grid.columns, grid.index, grid, levels=levels,
                               cmap=ps.HEAT, norm=None, vmin=0, vmax=1)
            for color, width in (("white", 1.8), (ps.INK, .8)):
                ax.add_patch(Rectangle((0, .4), .2, .6, fill=False,
                                       edgecolor=color, lw=width))
            if np.isclose(d, UNITY):
                ax.plot(.1, 1, marker="*", linestyle="none", markersize=8,
                        color="white", markeredgecolor=ps.INK, markeredgewidth=.8, zorder=6)
            ax.set_title("Elasticity between final consumption goods " + rf"$\varepsilon_D={label}$",
                         fontsize=11, pad=9)
            ax.grid(False)
            ax.set_xlim(0, block.eps_M.max())
            ax.set_ylim(0, block.eps_Q.max())
            ax.tick_params(axis="y", labelleft=(col == 0), labelright=False)
    fig.supylabel("Elasticity between the capital-labor bundle and intermediate inputs "
                  + r"$\varepsilon_Q$", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel(r"Elasticity between material inputs $\varepsilon_M$")
    for ax, heading in zip(axes[0], ("After removing one factor", "Factor-free upper bound")):
        ax.text(.5, 1.18, heading, transform=ax.transAxes, ha="center", va="bottom", fontsize=13)
    bar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal",
                       fraction=.045, pad=.025, shrink=.8, spacing="uniform", ticks=np.arange(0, 1.01, .2))
    bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.set_label("Granular share of aggregate fluctuations")
    output = stem.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Resolve constrained layout at the preview's raster resolution before PDF export.
    with BytesIO() as raster:
        fig.savefig(raster, format="png", dpi=140, facecolor="white")
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--surface-root", type=Path, default=SURFACE_ROOT)
    parser.add_argument("--out-csv", type=Path, default=SURFACE_CSV)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--from-cache", action="store_true", help="Redraw the saved comparison CSV.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.from_cache:
        table = pd.read_csv(args.out_csv)
        if "M" not in table or not table.M.eq(args.M).all():
            raise ValueError("Cached surface has a different or unrecorded M; rebuild it first.")
    else:
        table = build_surface(args.surface_root, args.M, choose_solver(args.solver), args.jobs)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
    figure = render(table, args.figure_stem)
    print(f"[atalay] {len(table)} cells; {int(table.fit_limit.sum())} one-factor fits reached an optimizer limit.")
    print(f"Wrote {figure} and {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
