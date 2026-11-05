#!/usr/bin/env python3
"""90% two-sided directional-bootstrap CIs and horizontal NAICS ceiling plot.

Uses cached grouped nominal-sales panels. No resampling of firms, changes to
classification assignments, or changes to main.tex. The same time-block draws
are used for each classification. N in plot labels counts pre-LPT units.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import gradient_bootstrap as gb
from mgp_core import (OUTPUT_DIR, RUN_OUTPUT_CACHE, analysis_block_length,
                      atomic_write_pickle, make_panel_rng)
from applications.compustat.run_compustat_industry_dependence import PREFIX


def row_label(row):
    name = "Individual firms" if row.key == "firms" else f"NAICS {row.key[-1]}-digit"
    return f"{name} (N={int(row.clusters)})"


def render(frame):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(8.7, 4.7))
    position = np.arange(len(frame))
    for j, row in enumerate(frame.itertuples()):
        color = "C0"
        lo, hi, value = 100*row.ci_low, 100*row.ci_high, 100*row.mgp_share
        ax.hlines(j, 0, max(0, lo), color="#B9D9EE", linewidth=1.8, zorder=2)
        ax.hlines(j, lo, hi, color=color, linewidth=1.8, zorder=3)
        ax.scatter(value, j, color=color, s=35, zorder=3)
    ax.set_yticks(position, [row_label(row) for row in frame.itertuples()])
    ax.set_ylim(len(frame)-0.5, -0.5)
    ax.set_xlim(min(0, float(100*frame.ci_low.min())), float(100*frame.ci_high.max())*1.06)
    ax.set_xlabel("Granular share of aggregate fluctuations")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    stem = OUTPUT_DIR / f"{PREFIX}_ci"
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-b", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()
    cache = RUN_OUTPUT_CACHE / f"{PREFIX}_ci_summary.pkl"
    if args.from_cache:
        frame = pd.read_pickle(cache)
        draws = pd.read_pickle(RUN_OUTPUT_CACHE / f"{PREFIX}_ci_draws.pkl")
        current = pd.read_pickle(RUN_OUTPUT_CACHE / f"{PREFIX}_summary.pkl").set_index("key")
        for row in frame.itertuples(index=False):
            source = current.loc[row.key]
            for field in ("N", "T", "M", "clusters", "start_date", "last_date", "weight_date"):
                assert getattr(row, field) == source[field], f"Stale {row.key} CI: {field}"
            assert abs(row.mgp_share - source.mgp_share) < 1e-10, f"Stale {row.key} point"
            assert len(draws[row.key]) == row.bootstrap_b
        for index, row in frame.iterrows():
            low, high = gb._ci(row.mgp_share, draws[row.key], row["T"], lo=.05, hi=.95)
            frame.loc[index, ["ci_low", "ci_high", "ci_level"]] = [low, high, .90]
        atomic_write_pickle(frame, cache)
        atomic_write_pickle(frame, RUN_OUTPUT_CACHE / f"{PREFIX}_ci_progress.pkl")
    else:
        summary = pd.read_pickle(RUN_OUTPUT_CACHE / f"{PREFIX}_summary.pkl")
        panels = pd.read_pickle(RUN_OUTPUT_CACHE / f"{PREFIX}_panels.pkl")
        rows = []; draws = {}
        for row in summary.itertuples(index=False):
            x = panels[row.key]
            ell = analysis_block_length(x.shape[0])
            result = gb.moving_block_gradient_ci(
                x, args.bootstrap_b, ell,
                make_panel_rng(args.seed, f"{PREFIX}_ci"), args.solver,
                lo=0.05, hi=0.95)
            assert abs(result["g_hat"]-row.mgp_share) < 1e-7
            assert np.isfinite(result["directional_stats"]).all()
            assert result["fallbacks"] == 0
            draws[row.key] = result["directional_stats"]
            rows.append({**row._asdict(), "ci_low": result["ci_low"],
                         "ci_high": result["ci_high"], "bootstrap_b": args.bootstrap_b,
                         "block_length": ell, "ci_level": .90,
                         "delta_T": result["delta_T"], "fallbacks": result["fallbacks"]})
            atomic_write_pickle(pd.DataFrame(rows), RUN_OUTPUT_CACHE / f"{PREFIX}_ci_progress.pkl")
            print(f"{row_label(row)}: {100*row.mgp_share:.2f}% "
                  f"[{100*result['ci_low']:.2f}, {100*result['ci_high']:.2f}]", flush=True)
        frame = pd.DataFrame(rows)
        atomic_write_pickle(frame, cache)
        atomic_write_pickle(draws, RUN_OUTPUT_CACHE / f"{PREFIX}_ci_draws.pkl")
    render(frame)


if __name__ == "__main__":
    main()
