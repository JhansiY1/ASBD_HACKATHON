"""
Shared plotting style for every figure in the project.

* Categorical colours follow a validated, colour-blind-safe fixed order
  (blue, orange, aqua, yellow, magenta, green, violet, red); hues are never
  cycled - extra categories are folded into "Other".
* Magnitude uses a single-hue (blue) sequential ramp; signed values
  (correlations) use a two-hue diverging map with a neutral grey midpoint.
* Thin marks, recessive grid, readable 10-12 pt fonts, 150 dpi PNG output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#9a9893"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_orange", ["#184f95", "#6da7ec", "#e7e6e2", "#f2a37f", "#b2401a"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c7c1", "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.color": INK2, "ytick.color": INK2, "legend.fontsize": 9.5, "legend.frameon": False,
        "axes.grid": True, "grid.color": "#e7e6e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "font.family": "DejaVu Sans", "axes.prop_cycle": matplotlib.cycler(color=CAT),
    })


def save(fig, path: Path, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
