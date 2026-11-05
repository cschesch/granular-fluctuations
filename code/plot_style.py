"""One plain-Matplotlib house style for every figure in the paper.

The exhibits accumulated three or four private looks as they were written --
left-aligned bold titles with a grey subtitle line, half-open axes with the top
and right spines dropped, ad-hoc hex colours, legends parked under the axes.
This module fixes one look and one palette so a reader flipping through the
paper sees a single set of figures:

  * a closed box (all four spines), because that is what the LaTeX float and
    its caption sit inside;
  * no in-figure title -- the caption is the title, and duplicating it wastes a
    line of every figure;
  * a faint horizontal grid behind the data;
  * the same six ink/teal/rust/slate/olive/plum series colours everywhere, and
    ``magma_r`` for every heat map;
  * one save helper for the file format used by each exhibit.

Import ``plot_style`` AFTER ``mgp_core`` (which points ``MPLCONFIGDIR`` at a
writable cache) and call :func:`use` once, at module import or at the top of
``main()``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "use",
    "save",
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
