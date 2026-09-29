"""Matplotlib style matching the paper figures (AAAS / Science palette)."""

from __future__ import annotations

PALETTE = {
    "ink": "#1B1919",
    "navy": "#3B4992",
    "red": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "teal": "#008280",
    "crimson": "#BB0021",
    "violet": "#5F559B",
    "magenta": "#A20056",
    "gray": "#808180",
}
VOICE = {"say": PALETTE["purple"], "echo": PALETTE["crimson"], "do": PALETTE["navy"], "follow": PALETTE["green"]}


def use_science_style() -> None:
    """Apply the paper's figure style to matplotlib (thin open axes, sans-serif, outward ticks)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": PALETTE["ink"],
        "axes.prop_cycle": cycler(color=[PALETTE[c] for c in ("navy", "crimson", "green", "purple", "teal", "magenta")]),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def panel_label(ax, letter: str) -> None:
    """Bold lower-case panel letter in the top-left corner, Nature/Science style."""
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
