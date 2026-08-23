#!/usr/bin/env python3
"""Create the four-panel study workflow figure with representative inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPT_DIR = REPO_ROOT / "figure_scripts"
if str(FIGURE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPT_DIR))

import red_journal_style  # noqa: E402


DEFAULT_PATIENT_DIR = Path(
    "/data/pgsal/doseae/test_cohorts_cache/ukhd-imrt/cache/processed_images/test/ANON_0024"
)
DEFAULT_CACHE = (
    REPO_ROOT / "oliver_paper" / "figure_data" / "Figure1_representative_planning_input.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "oliver_paper" / "figures" / "Figure1_study_workflow"

COHORT_COLORS = {
    "UKHD-train": "#CC79A7",
    "UKHD-SBRT": "#009E73",
    "RTOG-conv": "#0072B2",
    "RTOG-IMRT": "#D55E00",
}
INK = "#202020"
MUTED = "#666666"
PALE_BLUE = "#EAF3F8"
PALE_GOLD = "#FFF3D6"
PALE_GREEN = "#E7F4EE"
PALE_GRAY = "#F3F3F3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", type=Path, default=DEFAULT_PATIENT_DIR)
    parser.add_argument("--slice-index", type=int, default=429)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_nrrd_slice(path: Path, slice_index: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size = list(reader.GetSize())
    if not 0 <= slice_index < size[2]:
        raise IndexError(f"Slice {slice_index} outside z range [0, {size[2] - 1}] for {path}")
    reader.SetExtractIndex([0, 0, slice_index])
    reader.SetExtractSize([size[0], size[1], 0])
    return sitk.GetArrayFromImage(reader.Execute())


def load_or_create_representative_input(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    if args.image_cache.exists():
        cached = np.load(args.image_cache)
        return cached["ct"], cached["dose"], cached["lung_mask"]

    patient_id = args.patient_dir.name
    ct = read_nrrd_slice(
        args.patient_dir / f"{patient_id}_ct_processed.nrrd", args.slice_index
    ).astype(np.float32)
    dose = read_nrrd_slice(
        args.patient_dir / f"{patient_id}_dose_processed.nrrd", args.slice_index
    ).astype(np.float32)
    lung_mask = np.zeros(ct.shape, dtype=bool)
    lobe_paths = sorted(args.patient_dir.glob(f"{patient_id}_*_lobe_mask.nrrd"))
    if not lobe_paths:
        raise FileNotFoundError(f"No lobe masks found in {args.patient_dir}")
    for path in lobe_paths:
        lung_mask |= read_nrrd_slice(path, args.slice_index) > 0
    if not lung_mask.any():
        raise ValueError("Representative slice has an empty lung mask.")

    y, x = np.where(lung_mask)
    padding = 35
    y0 = max(0, int(y.min()) - padding)
    y1 = min(ct.shape[0], int(y.max()) + padding + 1)
    x0 = max(0, int(x.min()) - padding)
    x1 = min(ct.shape[1], int(x.max()) + padding + 1)
    ct = ct[y0:y1, x0:x1]
    dose = dose[y0:y1, x0:x1]
    lung_mask = lung_mask[y0:y1, x0:x1]

    args.image_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.image_cache, ct=ct, dose=dose, lung_mask=lung_mask)
    return ct, dose, lung_mask


def setup_diagram_axis(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        label,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.10,
        0.965,
        title,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = "#777777",
    linewidth: float = 0.9,
    radius: float = 0.02,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#555555",
    linewidth: float = 1.1,
    mutation_scale: float = 9,
    connectionstyle: str = "arc3",
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )


def cohort_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    name: str,
    role: str,
    lines: Sequence[str],
) -> None:
    color = COHORT_COLORS[name]
    rounded_box(ax, xy, width, height, facecolor="white", edgecolor="#9A9A9A")
    ax.add_patch(
        Rectangle(
            xy,
            0.018,
            height,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
    )
    x, y = xy
    ax.text(
        x + 0.035,
        y + height - 0.035,
        name,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="top",
    )
    ax.text(
        x + 0.035,
        y + height - 0.083,
        role,
        transform=ax.transAxes,
        fontsize=6.4,
        fontweight="bold",
        color=MUTED,
        ha="left",
        va="top",
    )
    ax.text(
        x + 0.035,
        y + height - 0.132,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=6.8,
        color=INK,
        linespacing=1.18,
        ha="left",
        va="top",
    )


def draw_cohort_panel(ax: plt.Axes) -> None:
    setup_diagram_axis(ax, "A", "Study cohorts and RT domains")
    cohort_card(
        ax,
        (0.04, 0.52),
        0.44,
        0.34,
        name="UKHD-train",
        role="Training",
        lines=("n = 145", "Mixed thoracic RT", "Mixed fractionation"),
    )
    cohort_card(
        ax,
        (0.52, 0.52),
        0.44,
        0.34,
        name="UKHD-SBRT",
        role="External test",
        lines=("n = 56", "Early-stage T1–T2 N0 M0", "SBRT / hypofractionated"),
    )
    cohort_card(
        ax,
        (0.04, 0.11),
        0.44,
        0.34,
        name="RTOG-conv",
        role="Validation / selection",
        lines=("n = 234", "Locally advanced NSCLC", "3D-CRT", "Conventional", "chemoradiation"),
    )
    cohort_card(
        ax,
        (0.52, 0.11),
        0.44,
        0.34,
        name="RTOG-IMRT",
        role="External test",
        lines=("n = 212", "Locally advanced NSCLC", "IMRT", "Conventional", "chemoradiation"),
    )


def draw_imaging_panel(
    ax: plt.Axes,
    ct: np.ndarray,
    dose: np.ndarray,
    lung_mask: np.ndarray,
) -> None:
    setup_diagram_axis(ax, "B", "Representative imaging and dose inputs")
    ct_ax = ax.inset_axes([0.04, 0.22, 0.44, 0.61])
    dose_ax = ax.inset_axes([0.52, 0.22, 0.44, 0.61])

    for image_ax in (ct_ax, dose_ax):
        image_ax.imshow(ct, cmap="gray", vmin=0.27, vmax=0.77, interpolation="bilinear")
        image_ax.contour(lung_mask.astype(float), levels=[0.5], colors=["#00A6D6"], linewidths=1.0)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.7)

    alpha = np.where(dose > 0.025, np.clip(0.15 + 0.72 * dose, 0, 0.78), 0)
    dose_artist = dose_ax.imshow(
        dose,
        cmap="inferno",
        vmin=0,
        vmax=1,
        alpha=alpha,
        interpolation="bilinear",
    )
    ct_ax.set_title("Planning CT + lung mask", fontsize=8, pad=3)
    dose_ax.set_title("Planned-dose overlay", fontsize=8, pad=3)

    colorbar_ax = ax.inset_axes([0.66, 0.10, 0.25, 0.023])
    colorbar = plt.colorbar(dose_artist, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
    colorbar.ax.text(
        0.5,
        1.65,
        "Normalized dose",
        transform=colorbar.ax.transAxes,
        fontsize=6.8,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.05,
        0.11,
        "Representative UKHD-SBRT planning study",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        ha="left",
        va="center",
    )


def draw_lung_patch_icon(ax: plt.Axes, center: tuple[float, float]) -> None:
    cx, cy = center
    left = Circle((cx - 0.026, cy), 0.040, facecolor="#DDE8EC", edgecolor="#557784", linewidth=0.8)
    right = Circle((cx + 0.026, cy), 0.040, facecolor="#DDE8EC", edgecolor="#557784", linewidth=0.8)
    ax.add_patch(left)
    ax.add_patch(right)
    for offset_x, offset_y in ((-0.040, 0.012), (0.002, 0.020), (0.030, -0.016)):
        ax.add_patch(
            Rectangle(
                (cx + offset_x, cy + offset_y),
                0.028,
                0.028,
                facecolor="none",
                edgecolor="#D55E00",
                linewidth=0.8,
                transform=ax.transAxes,
            )
        )


def draw_network_stack(
    ax: plt.Axes,
    center: tuple[float, float],
    widths: Iterable[float],
    *,
    color: str,
) -> None:
    cx, cy = center
    widths = list(widths)
    spacing = 0.036
    for index, width in enumerate(widths):
        y = cy + (index - (len(widths) - 1) / 2) * spacing
        ax.add_patch(
            Rectangle(
                (cx - width / 2, y - 0.013),
                width,
                0.026,
                facecolor=color,
                edgecolor="#667077",
                linewidth=0.6,
                transform=ax.transAxes,
            )
        )


def draw_doseae_panel(ax: plt.Axes) -> None:
    setup_diagram_axis(ax, "C", "Dose representation learning")
    draw_lung_patch_icon(ax, (0.10, 0.58))
    ax.text(
        0.10,
        0.44,
        "Lung patches\n(50³ voxels)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
    )

    rounded_box(ax, (0.20, 0.49), 0.13, 0.08, facecolor=PALE_GOLD)
    rounded_box(ax, (0.20, 0.61), 0.13, 0.08, facecolor=PALE_BLUE)
    ax.text(0.265, 0.53, "Dose", transform=ax.transAxes, ha="center", va="center", fontsize=7.5)
    ax.text(
        0.265,
        0.65,
        "Dose + CT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
    )
    arrow(ax, (0.15, 0.58), (0.195, 0.58))
    arrow(ax, (0.335, 0.58), (0.395, 0.58))

    draw_network_stack(ax, (0.44, 0.58), (0.11, 0.09, 0.07, 0.05), color="#BFCBD1")
    ax.text(0.44, 0.76, "Encoder", transform=ax.transAxes, ha="center", fontsize=7.5)
    arrow(ax, (0.495, 0.58), (0.545, 0.58))

    rounded_box(ax, (0.55, 0.50), 0.12, 0.16, facecolor="#F1E6F3", edgecolor="#9A6DA1")
    for row in range(4):
        for col in range(4):
            ax.add_patch(
                Circle(
                    (0.575 + col * 0.022, 0.535 + row * 0.027),
                    0.005,
                    facecolor="#9A6DA1",
                    edgecolor="none",
                    transform=ax.transAxes,
                )
            )
    ax.text(
        0.61,
        0.44,
        "128-D / patch",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="#74487A",
    )
    arrow(ax, (0.675, 0.58), (0.72, 0.58))

    draw_network_stack(ax, (0.77, 0.58), (0.05, 0.07, 0.09, 0.11), color="#D7DEE1")
    ax.text(0.77, 0.76, "Decoder", transform=ax.transAxes, ha="center", fontsize=7.5)
    arrow(ax, (0.83, 0.58), (0.87, 0.58))

    rounded_box(ax, (0.875, 0.50), 0.09, 0.16, facecolor="#2D1545", edgecolor="#69358C")
    ax.add_patch(
        Circle(
            (0.92, 0.58),
            0.028,
            facecolor="#FFB000",
            edgecolor="#FFF2AA",
            linewidth=0.7,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.92,
        0.44,
        "Reconstructed\ndose",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
    )

    rounded_box(ax, (0.35, 0.13), 0.30, 0.13, facecolor=PALE_GREEN, edgecolor="#7AAE98")
    ax.text(
        0.50,
        0.195,
        "Patient-level reconstruction\nMSE  •  gamma pass rate",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.3,
    )
    arrow(ax, (0.92, 0.48), (0.65, 0.26), connectionstyle="arc3,rad=-0.10")


def draw_latent_tiles(ax: plt.Axes, origin: tuple[float, float], count: int = 5) -> None:
    x, y = origin
    for index in range(count):
        offset_y = (index - (count - 1) / 2) * 0.055
        rounded_box(
            ax,
            (x, y + offset_y),
            0.095,
            0.032,
            facecolor="#F1E6F3",
            edgecolor="#9A6DA1",
            radius=0.006,
        )
        for dot in range(5):
            ax.add_patch(
                Circle(
                    (x + 0.018 + dot * 0.015, y + offset_y + 0.016),
                    0.0035,
                    facecolor="#8C5A94",
                    edgecolor="none",
                    transform=ax.transAxes,
                    zorder=3,
                )
            )


def draw_prediction_panel(ax: plt.Axes) -> None:
    setup_diagram_axis(ax, "D", "Patient-level prediction and evaluation")
    draw_latent_tiles(ax, (0.035, 0.55))
    ax.text(
        0.082,
        0.36,
        "Patch\nlatents",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
    )
    arrow(ax, (0.135, 0.63), (0.18, 0.63))

    rounded_box(ax, (0.185, 0.53), 0.15, 0.19, facecolor=PALE_BLUE, edgecolor="#7A9BAB")
    ax.text(
        0.26,
        0.625,
        "Patient\naggregation",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        linespacing=1.25,
    )
    rounded_box(ax, (0.39, 0.31), 0.15, 0.09, facecolor=PALE_GREEN, edgecolor="#7AAE98")
    ax.text(
        0.465,
        0.355,
        "Age + smoking",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.7,
    )
    arrow(ax, (0.34, 0.63), (0.39, 0.63))

    rounded_box(ax, (0.395, 0.53), 0.12, 0.19, facecolor=PALE_GOLD, edgecolor="#C7A35A")
    ax.text(
        0.455,
        0.625,
        "PCA\n(4 PCs)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        linespacing=1.3,
    )
    arrow(ax, (0.52, 0.63), (0.57, 0.63))

    rounded_box(ax, (0.575, 0.53), 0.14, 0.19, facecolor=PALE_GRAY, edgecolor="#8A8A8A")
    ax.text(
        0.645,
        0.625,
        "Classifier",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        linespacing=1.25,
    )
    arrow(ax, (0.465, 0.41), (0.60, 0.52), connectionstyle="arc3,rad=-0.12")
    arrow(ax, (0.72, 0.63), (0.77, 0.63))

    rounded_box(ax, (0.775, 0.53), 0.19, 0.19, facecolor="#FCE8E2", edgecolor="#C96A4C")
    ax.text(
        0.87,
        0.625,
        "Any-RILI\nrisk",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#9E3F25",
        linespacing=1.25,
    )

    ax.text(
        0.50,
        0.23,
        "Configuration fixed using the validation domain",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        color=MUTED,
    )
    rounded_box(ax, (0.10, 0.07), 0.27, 0.11, facecolor="white", edgecolor=COHORT_COLORS["RTOG-conv"])
    ax.text(
        0.235,
        0.125,
        "RTOG-conv\nmodel selection",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=COHORT_COLORS["RTOG-conv"],
        fontweight="bold",
    )
    rounded_box(ax, (0.43, 0.07), 0.47, 0.11, facecolor="white", edgecolor="#777777")
    ax.text(
        0.665,
        0.125,
        "RTOG-IMRT  +  UKHD-SBRT\nexternal testing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
        fontweight="bold",
    )


def add_panel_border(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0.005, 0.005),
            0.99,
            0.99,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#B5B5B5",
            linewidth=0.8,
            zorder=20,
        )
    )


def main() -> None:
    args = parse_args()
    ct, dose, lung_mask = load_or_create_representative_input(args)
    red_journal_style.apply_red_journal_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    draw_cohort_panel(axes[0, 0])
    draw_imaging_panel(axes[0, 1], ct, dose, lung_mask)
    draw_doseae_panel(axes[1, 0])
    draw_prediction_panel(axes[1, 1])
    for ax in axes.ravel():
        add_panel_border(ax)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.995, bottom=0.015, wspace=0.035, hspace=0.045)
    red_journal_style.save_publication_figure(fig, args.output_base, dpi=args.dpi)
    plt.close(fig)
    print(f"[saved] {args.output_base.with_suffix('.png')}")
    print(f"[saved] {args.output_base.with_suffix('.pdf')}")
    print(f"[saved] {args.image_cache}")
    print("[note] Follow-up CT omitted: no image was verifiably linked to the representative study patient.")


if __name__ == "__main__":
    main()
