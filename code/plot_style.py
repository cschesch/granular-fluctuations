"""Shared Matplotlib style, saving helpers and LPT bucket figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "use",
    "save",
    "plot_lpt_squared_bucket_sizes",
    "INK", "TEAL", "RUST", "SLATE", "OLIVE", "PLUM", "GREY", "LIGHT",
    "SERIES", "HEAT", "HEAT_LINE",
]

# Series colours.  INK first so a single-series figure is black; TEAL and RUST
# are the two the empirical figures already used for "ours" and "theirs".
INK = "#1a1a1a"
TEAL = "#2f6f73"
RUST = "#a8552f"
SLATE = "#40506b"
OLIVE = "#6f7d3f"
PLUM = "#6c5b7b"
GREY = "#8a8a8a"
LIGHT = "#d9d9d9"

SERIES = (INK, TEAL, RUST, SLATE, OLIVE, PLUM)

# Sequential colour map for every heat map / surface in the paper, and a colour
# that reads on top of it at both ends.
HEAT = "magma_r"
HEAT_LINE = "#1b7f4b"

_RC = {
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.0,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.0,
    "axes.titlepad": 7.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#3a3a3a",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
    "grid.color": "#c8c8c8",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.borderaxespad": 0.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "image.cmap": HEAT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use() -> None:
    """Install the house style.  Idempotent; safe to call from several modules."""
    mpl.rcParams.update(_RC)


def save(fig, stem: Path | str, *, suffix: str = ".pdf", dpi: int = 220) -> Path:
    path = Path(stem).with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_lpt_squared_bucket_sizes(
    firm_diag: pd.DataFrame,
    bucket_diag: pd.DataFrame,
    path_base: Path,
    ylabel: str | None = None,
    xlabel: str = "LPT bucket",
    figsize_scale: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Stack each unit's S_i^2 contribution within its LPT bucket."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered = firm_diag.sort_values("size_share_sq", ascending=False)
    color_by_index = {idx: colors[j % len(colors)] for j, idx in enumerate(ordered.index)}

    def name_column() -> str | None:
        for candidate in ("name", "label", "description", "unit_id", "gvkey"):
            if candidate in firm_diag.columns:
                return candidate
        return None

    display_aliases = {
        "WALMART": "Walmart",
        "GENERAL MOTORS": "General Motors",
        "FORD MOTOR": "Ford Motor",
        "GE AEROSPACE": "GE Aerospace",
        "INTL BUSINESS MACHINES": "IBM",
        "DISNEY (WALT)": "Disney",
        "LOCKHEED MARTIN": "Lockheed Martin",
        "XEROX HOLDINGS": "Xerox",
        "COCA-COLA": "Coca-Cola",
        "ALTRIA GROUP": "Altria",
        "HOME DEPOT": "Home Depot",
        "PFIZER": "Pfizer",
        "MACY'S": "Macy's",
        "AUTONATION": "AutoNation",
        "VERIZON COMMUNICATIONS": "Verizon",
        "CARDINAL HEALTH": "Cardinal Health",
        "COSTCO WHOLESALE": "Costco",
        "MONDELEZ INTERNATIONAL": "Mondelez",
        "CONAGRA BRANDS": "Conagra",
        "AT&T": "AT&T",
        "PROCTER & GAMBLE": "Procter & Gamble",
        "INTEL": "Intel",
        "JOHNSON & JOHNSON": "Johnson & Johnson",
        "RTX": "RTX",
        "HP": "HP",
        "BOEING": "Boeing",
        "MCKESSON": "McKesson",
        "DUPONT DE NEMOURS": "DuPont",
        "INTL PAPER": "International Paper",
        "KROGER": "Kroger",
        "TARGET": "Target",
        "MOTOROLA SOLUTIONS": "Motorola",
        "UNITED PARCEL SERVICE": "UPS",
        "HONEYWELL INTERNATIONAL": "Honeywell",
    }
    acronym_aliases = {
        "AT&T": "AT&T",
        "GE": "GE",
        "GM": "GM",
        "HP": "HP",
        "IBM": "IBM",
        "P&G": "P&G",
        "RTX": "RTX",
        "UPS": "UPS",
        "US": "US",
        "USA": "USA",
    }

    def capital_case(text: str) -> str:
        def convert_token(token: str) -> str:
            upper = token.upper()
            if token.lower() == "and":
                return "&"
            if upper in acronym_aliases:
                return acronym_aliases[upper]
            if "-" in token:
                return "-".join(convert_token(part) for part in token.split("-"))
            if "'" in token:
                parts = token.lower().split("'")
                return "'".join(part[:1].upper() + part[1:] if part else part for part in parts)
            lower = token.lower()
            return lower[:1].upper() + lower[1:] if lower else token

        return " ".join(convert_token(token) for token in text.split())

    def short_name(value: object, max_chars: int) -> str:
        text = str(value).strip()
        for old, new in (
            ("IP: ", ""),
            ("(SA, 2002=100)", ""),
            ("_other", " other"),
            ("  ", " "),
        ):
            text = text.replace(old, new)
        if "/" in text:
            parts = []
            seen = set()
            for part in text.split("/"):
                part = part.strip()
                key = part.lower()
                if part and key not in seen:
                    parts.append(part)
                    seen.add(key)
            if len(parts) > 2:
                text = f"{parts[0]} / {parts[-1]}"
            elif parts:
                text = " / ".join(parts)
        text = " ".join(text.split())
        text = display_aliases.get(text.upper(), capital_case(text))
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    n_buckets = int(bucket_diag["bucket"].nunique())
    fig_width = max(12.0, 0.7 * n_buckets)
    fig, ax = plt.subplots(figsize=(fig_width * figsize_scale[0], 5.2 * figsize_scale[1]))
    bottoms = {int(b): 0.0 for b in bucket_diag["bucket"]}
    for bucket in sorted(bottoms):
        rows = firm_diag.loc[firm_diag["bucket"] == bucket].sort_values("size_share_sq", ascending=False)
        for idx, row in rows.iterrows():
            height = float(row["size_share_sq"])
            ax.bar(
                bucket + 1,
                height,
                width=0.74,
                bottom=bottoms[bucket],
                color=color_by_index[idx],
                edgecolor=(1, 1, 1, 0.45),
                linewidth=0.12,
            )
            bottoms[bucket] += height

    y_max = max(bottoms.values())
    max_chars = 22 if n_buckets <= 12 else 16 if n_buckets <= 30 else 12
    label_fontsize = 5.0 if n_buckets <= 12 else 4.2 if n_buckets <= 30 else 3.6
    name_col = name_column()
    max_label_lines = 1
    for _, row in bucket_diag.iterrows():
        bucket = int(row["bucket"])
        total = float(row["total_sq_share"])
        n_firms = int(row["n_firms"])
        bucket_rows = firm_diag.loc[firm_diag["bucket"] == bucket].sort_values("size_share_sq", ascending=False)
        n_names = 1 if n_firms == 1 else 3
        names = []
        if name_col is not None:
            names = [
                short_name(value, max_chars)
                for value in bucket_rows[name_col].head(n_names)
            ]
        lines = [f"n={n_firms}", *names]
        if names and n_firms > len(names):
            lines.append("...")
        max_label_lines = max(max_label_lines, len(lines))
        ax.text(
            bucket + 1,
            total + 0.025 * y_max,
            "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            linespacing=1.05,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or r"Bucket squared size load: $\sum_{i\in B_m} S_i^2$")
    ax.set_xticks(np.arange(1, len(bottoms) + 1))
    ax.set_xlim(0.35, len(bottoms) + 0.65)
    ax.set_ylim(0, y_max * (1.18 + 0.11 * max_label_lines))
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
