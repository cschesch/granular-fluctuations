#!/usr/bin/env python3
"""Plot the sensitivity of granularity bounds to the number of groups.

For monthly and daily panels, repeat LPT grouping and estimation for
M = 5, ..., floor(sqrt(T)). Report point estimates and 90% moving-block
gradient-bootstrap intervals.

Write output/msweep_combined.pdf and cache the estimates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import gradient_bootstrap as GB
from mosek_face import licensed_mosek
from mgp_core import (
    ANALYSIS_BOOTSTRAP_DRAWS,
    analysis_block_length,
    OUTPUT_DIR,
    RUN_OUTPUT_CACHE,
    align_sizes_to_panel,
    atomic_write_pickle,
    build_lpt_grouped_data,
    choose_solver,
    make_panel_rng,
)
from applications.bls_ces.panel import load_bls_ces_panel
from applications.bls_cpi.panel import load_bls_cpi_panel
from applications.crsp.panel import load_crsp_panel
from applications.figure_layout import hierarchy_positions
from applications.forbes.panel import FORBES_ARCHIVE_DAILY, load_forbes_billionaire_panels
from run_multi_source import DAILY_HIGH_M_THRESHOLD
from compute_applications_coefficients import available_items

# Import plotting only after mgp_core has configured its writable cache directory.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


TEAL = "#2f6f73"
PANELS = {
    "BLS CES Employment": ("Employment by sector", load_bls_ces_panel),
    "BLS CPI Inflation (1967 Weights)": ("Price by item", load_bls_cpi_panel),
    "CRSP Daily Returns": ("Stocks", lambda: load_crsp_panel(20 * 252)),
    "Forbes Billionaire Wealth": (
        "Forbes Billionaires",
        lambda: load_forbes_billionaire_panels(None, FORBES_ARCHIVE_DAILY)[0],
    ),
}
SWEEP_COLUMNS = ("panel", "M", "share", "ci_low", "ci_high", "bootstrap_B", "block_length")
INTERVAL_COLUMNS = ("share", "ci_low", "ci_high")
# Bump for changes to grouping or inference in this script; plotting does not
# change cached estimates. Shared numerical implementations are hashed below.
SWEEP_VERSION = "validated-inference-v1"


def inference_fingerprint() -> str:
    """Invalidate saved intervals when their numerical implementation changes."""
    digest = hashlib.sha256(SWEEP_VERSION.encode())
    for name in ("gradient_bootstrap.py", "near_optimal.py", "fast_mgp.py",
                 "mgp_kernels.py", "mgp_core.py", "mosek_face.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def checked_results(out: pd.DataFrame) -> pd.DataFrame:
    """Do not redraw or retain intervals from an unchecked numerical method."""
    if out.empty:
        return out
    required = {*SWEEP_COLUMNS, "inference_fingerprint"}
    if (not required.issubset(out.columns)
            or not out["inference_fingerprint"].eq(inference_fingerprint()).all()
            or not np.isfinite(out[list(INTERVAL_COLUMNS)]).all().all()):
        raise ValueError(
            "Cached M-sweep intervals are unverified, stale or nonfinite. "
            "Rerun without --from-cache or --only to rebuild them."
        )
    return out


def m_grid(t: int) -> list[int]:
    m_max = int(np.floor(np.sqrt(t)))
    return list(range(5, m_max + 1))


def sweep(
    y: np.ndarray,
    shares: np.ndarray,
    grid: list[int],
    solver: str,
    label: str,
    *,
    is_daily: bool,
    bootstrap_b: int,
    seed: int,
    start_date: str,
    last_date: str,
) -> pd.DataFrame:
    rows = []
    method_hash = inference_fingerprint()
    checkpoint_dir = RUN_OUTPUT_CACHE / "msweep"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for m in grid:
        x_grouped, *_ = build_lpt_grouped_data(y, shares, m)
        job_solver = (
            "MOSEK"
            if is_daily and m >= DAILY_HIGH_M_THRESHOLD and licensed_mosek()
            else solver
        )
        block_length = analysis_block_length(x_grouped.shape[0])
        input_hash = hashlib.sha256(np.ascontiguousarray(x_grouped).tobytes()).hexdigest()
        stamp = {
            "input_fingerprint": input_hash,
            "inference_fingerprint": method_hash,
            "bootstrap_B": int(bootstrap_b),
            "block_length": int(block_length),
            "seed": int(seed),
            "solver": job_solver,
            "N": y.shape[0], "T": y.shape[1],
            "start_date": start_date, "last_date": last_date,
        }
        key = hashlib.sha256(f"{label}_M{m}".encode()).hexdigest()[:20]
        checkpoint = checkpoint_dir / f"{key}.pkl"
        if checkpoint.exists():
            saved = pd.read_pickle(checkpoint)
            if (all(saved.get(k) == v for k, v in stamp.items())
                    and all(np.isfinite(saved.get(k, np.nan)) for k in INTERVAL_COLUMNS)):
                rows.append(saved)
                print(f"[{label}] M={m:>3}: verified cached interval", flush=True)
                continue
        boot = GB.moving_block_gradient_ci(
            x_grouped,
            b=bootstrap_b,
            block_length=block_length,
            rng=make_panel_rng(seed, f"msweep_{label}_M{m}_block_gradient"),
            solver=job_solver,
        )
        row = {
            "panel": label,
            "M": m,
            "share": boot["g_hat"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "bootstrap_B": boot["B"],
            "block_length": boot["block_length"],
            "common_rank": boot["common_rank"],
            "regular": boot["regular"],
            "fallbacks": boot["fallbacks"],
            **stamp,
        }
        rows.append(row)
        atomic_write_pickle(row, checkpoint)
        print(
            f"[{label}] M={m:>3}: share={boot['g_hat']:.4f}, "
            f"90% block-gradient CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]",
            flush=True,
        )
    return pd.DataFrame(rows)


def compute_sweep_results(solver: str, bootstrap_b: int, seed: int,
                          only: list[str] | None = None,
                          available: set[str] | None = None) -> pd.DataFrame:
    results = []
    cache = RUN_OUTPUT_CACHE / "msweep_shares.csv"
    if only and cache.exists():
        prior = pd.read_csv(cache)
        results.append(checked_results(prior.loc[~prior.panel.isin(only)]))
    for label, (item, loader) in PANELS.items():
        if only and label not in only:
            continue
        if available is not None and item not in available:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            panel = loader()
        y, sizes = align_sizes_to_panel(panel.y, panel.sizes)
        shares = (sizes / sizes.sum()).to_numpy(float)
        results.append(sweep(
            y.to_numpy(float), shares, m_grid(y.shape[1]), solver, label,
            is_daily=panel.frequency.lower() == "daily",
            bootstrap_b=bootstrap_b, seed=seed,
            start_date=str(y.columns[0]), last_date=str(y.columns[-1]),
        ))
        # Preserve completed panels if a later, larger panel is interrupted.
        pd.concat(results, ignore_index=True).to_csv(cache, index=False)
    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=SWEEP_COLUMNS)
    out.to_csv(cache, index=False)
    return out


def render_plot(out: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    positions, _, _ = hierarchy_positions()
    labels = sorted(PANELS, key=lambda label: positions[PANELS[label][0]])
    for ax, label in zip(axes.ravel(), labels):
        d = out[out["panel"].eq(label)].sort_values("M")
        if not d.empty:
            ax.plot(d["M"], d["share"].to_numpy(), marker="o", markersize=3.0, linewidth=1.7, color=TEAL)
            ax.fill_between(d["M"], d["ci_low"], d["ci_high"], alpha=0.16, color=TEAL)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"Number of groups $M$")
        ax.set_ylabel(r"Granularity upper bound $\hat{\tilde{\gamma}}$")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "msweep_combined.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--bootstrap-b", type=int, default=ANALYSIS_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--from-cache", action="store_true", help="Redraw the figure from cached msweep_shares.csv.")
    parser.add_argument("--available-only", action="store_true", help="Use applications in the current Figure 1 results.")
    parser.add_argument("--only", nargs="+", choices=PANELS,
                        help="Recompute selected panels and retain the other cached panels.")
    args = parser.parse_args()
    if args.from_cache:
        out = checked_results(pd.read_csv(RUN_OUTPUT_CACHE / "msweep_shares.csv"))
    else:
        solver = choose_solver(args.solver)
        available = available_items() if args.available_only else None
        out = compute_sweep_results(solver, args.bootstrap_b, args.seed, args.only, available)
    render_plot(out)
    print(f"Wrote {OUTPUT_DIR / 'msweep_combined.pdf'} and {RUN_OUTPUT_CACHE / 'msweep_shares.csv'}", flush=True)


if __name__ == "__main__":
    main()
