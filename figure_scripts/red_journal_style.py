"""Reusable IJROBP/Red Journal matplotlib styling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt

FigureSizeKind = Literal["single", "double", "km", "multi"]

FIGURE_SIZES = {
    "single": (3.5, 3.0),
    "double": (7.0, 4.5),
    "km": (7.0, 5.5),
    "multi": (7.0, 6.0),
}

RED_JOURNAL_RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COHORT_COLORS = {
    "RTOG-conv": "#0072B2",
    "RTOG-IMRT": "#D55E00",
    "UKHD-SBRT": "#009E73",
    "RTOG-other": "#999999",
    "UKHD-train": "#CC79A7",
}

MARKERS = ["o", "s", "D", "^", "v", "P"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
HATCHES = ["", "///", "\\\\\\", "xx", "..", "--"]


def apply_red_journal_style() -> None:
    """Apply repository default publication plotting parameters."""
    plt.rcParams.update(RED_JOURNAL_RCPARAMS)


def get_figure_size(kind: FigureSizeKind = "double") -> tuple[float, float]:
    """Return a Red Journal-compatible final-print figure size."""
    return FIGURE_SIZES[kind]


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.04,
    fontsize: float = 13,
) -> None:
    """Add a bold panel label in the upper-left corner of an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def clean_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    """Apply clean white-background axes styling."""
    ax.set_facecolor("white")
    ax.tick_params(width=1.0, length=3.0, colors="black")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.grid(axis="x", visible=False)


def save_publication_figure(
    fig: plt.Figure,
    output_base: Path,
    *,
    dpi: int = 600,
    formats: Sequence[str] = ("png", "pdf"),
    bbox_inches: str = "tight",
) -> list[Path]:
    """Save a figure with explicit publication-ready formats."""
    saved: list[Path] = []
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path, dpi=dpi, facecolor="white", transparent=False, bbox_inches=bbox_inches)
        saved.append(path)
    return saved


def apply_distinguishable_styles(
    artists: Iterable[object],
    *,
    colors: Sequence[str] | None = None,
) -> None:
    """Assign color plus non-color encodings to line-like artists when possible."""
    color_cycle = list(colors) if colors is not None else list(COHORT_COLORS.values())
    for idx, artist in enumerate(artists):
        if hasattr(artist, "set_color"):
            artist.set_color(color_cycle[idx % len(color_cycle)])
        if hasattr(artist, "set_marker"):
            artist.set_marker(MARKERS[idx % len(MARKERS)])
        if hasattr(artist, "set_linestyle"):
            artist.set_linestyle(LINE_STYLES[idx % len(LINE_STYLES)])
