"""Shared matplotlib style: one palette, recessive axes, consistent marks.

Colour rules used across all figures
  * categorical series use CATEGORICAL in fixed order, at most 8 per axes
    (more cells -> small multiples, one panel per experiment group);
  * magnitude (e.g. cycle age) uses the single-hue SEQUENTIAL blue ramp;
  * correlations use the DIVERGING blue-grey-red map;
  * risk levels use STATUS colours and are always labelled with text.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
RISK_COLORS = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}

SEQ_CMAP = LinearSegmentedColormap.from_list("voltara_seq", SEQUENTIAL[1:])
DIV_CMAP = LinearSegmentedColormap.from_list("voltara_div", ["#184f95", "#6da7ec", "#f0efec",
                                                             "#ec8a89", "#b42a2a"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(path.parents[2])}")
