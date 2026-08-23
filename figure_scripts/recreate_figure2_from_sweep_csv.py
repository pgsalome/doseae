#!/usr/bin/env python3
"""Recreate the manuscript validation-sweep heatmaps from exported cell data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import red_journal_style  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "oliver_paper"
    / "figure_data"
    / "figureS_dose_patch_aggregation_sweep_heatmaps_data.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "oliver_paper" / "figures" / "Figure2_validation_sweep_heatmaps"

AGGREGATION_ORDER = [
    "ipsi_mean",
    "ipsi_std",
    "ipsi_mean_std",
    "ipsi_dose_weighted_mean",
    "ipsi_highdose_mean",
    "ipsi_q95",
    "ipsi_highdose_q95",
    "ipsi_max",
]
AGGREGATION_LABELS = {
    "ipsi_mean": "Mean",
    "ipsi_std": "SD",
    "ipsi_mean_std": "Mean+SD",
    "ipsi_dose_weighted_mean": "Dose-wt mean",
    "ipsi_highdose_mean": "HD mean",
    "ipsi_q95": "Q95",
    "ipsi_highdose_q95": "HD Q95",
    "ipsi_max": "Max",
}
PCA_ORDER = ["none", "4", "8", "16", "24", "33", "48", "64"]
SAMPLER_ORDER = [
    "None",
    "ROS",
    "RUS",
    "SMOTE",
    "BLSMOTE",
    "ADASYN",
    "TL",
    "OSS",
    "ENN",
    "NCL",
    "SMOTE+TL",
    "SMOTE+ENN",
]
SAMPLER_LABELS = {
    "SMOTE": "SM",
    "BLSMOTE": "BLSM",
    "ADASYN": "ADA",
    "SMOTE+TL": "SM+TL",
    "SMOTE+ENN": "SM+ENN",
}
CLASSIFIER_ORDER = ["LR-L1", "LR-L2", "LR-EN", "kNN", "SVM", "ET", "GTB", "RF", "BRF", "HGB"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def matrix_from_long(
    data: pd.DataFrame,
    *,
    panel_title: str,
    row_axis: str,
    col_axis: str,
    row_values: Sequence[str],
    col_values: Sequence[str],
) -> np.ndarray:
    panel = data.loc[data["panel_title"].eq(panel_title)].copy()
    panel["row_value"] = panel["row_value"].astype(str)
    panel["col_value"] = panel["col_value"].astype(str)
    matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=float)
    for row_index, row_value in enumerate(row_values):
        for col_index, col_value in enumerate(col_values):
            cell = panel.loc[
                panel["row_value"].eq(str(row_value))
                & panel["col_value"].eq(str(col_value))
                & panel["status"].astype(str).eq("ok"),
                "validation_auc",
            ]
            if not cell.empty:
                matrix[row_index, col_index] = float(pd.to_numeric(cell, errors="coerce").iloc[0])
    return matrix


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    rotate_x: int,
):
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=rotate_x, ha="right" if rotate_x else "center")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing sweep cell data: {args.input_csv}")
    data = pd.read_csv(args.input_csv, low_memory=False, keep_default_na=False)
    required = {"panel_title", "row_value", "col_value", "status", "validation_auc"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Sweep cell data is missing columns: {missing}")

    aggregation_labels = [AGGREGATION_LABELS[value] for value in AGGREGATION_ORDER]
    specs = [
        (
            "Patch aggregation",
            "aggregation",
            "pca_components",
            AGGREGATION_ORDER,
            PCA_ORDER,
            aggregation_labels,
            PCA_ORDER,
            "PCA components",
            "Aggregation",
            0,
        ),
        (
            "PCA dimensionality",
            "pca_components",
            "classifier",
            PCA_ORDER,
            CLASSIFIER_ORDER,
            PCA_ORDER,
            CLASSIFIER_ORDER,
            "Classifier",
            "PCA components",
            45,
        ),
        (
            "Class-imbalance sampler",
            "sampler",
            "classifier",
            SAMPLER_ORDER,
            CLASSIFIER_ORDER,
            [SAMPLER_LABELS.get(value, value) for value in SAMPLER_ORDER],
            CLASSIFIER_ORDER,
            "Classifier",
            "Sampler",
            45,
        ),
        (
            "Classifier",
            "classifier",
            "aggregation",
            CLASSIFIER_ORDER,
            AGGREGATION_ORDER,
            CLASSIFIER_ORDER,
            aggregation_labels,
            "Aggregation",
            "Classifier",
            45,
        ),
    ]

    matrices = [
        matrix_from_long(
            data,
            panel_title=title,
            row_axis=row_axis,
            col_axis=col_axis,
            row_values=row_values,
            col_values=col_values,
        )
        for title, row_axis, col_axis, row_values, col_values, *_rest in specs
    ]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()])
    vmin = float(max(0.35, np.nanpercentile(finite, 2)))
    vmax = float(min(0.85, np.nanpercentile(finite, 98)))
    if vmax <= vmin:
        vmin, vmax = 0.35, 0.85

    red_journal_style.apply_red_journal_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=red_journal_style.get_figure_size("multi"),
        constrained_layout=True,
        facecolor="white",
    )
    image = None
    for index, (ax, matrix, spec) in enumerate(zip(axes.ravel(), matrices, specs)):
        title, _row_axis, _col_axis, _rows, _cols, row_labels, col_labels, xlabel, ylabel, rotation = spec
        image = draw_heatmap(
            ax,
            matrix,
            row_labels=row_labels,
            col_labels=col_labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            vmin=vmin,
            vmax=vmax,
            rotate_x=rotation,
        )
        red_journal_style.add_panel_label(ax, chr(ord("A") + index), x=-0.20, y=1.06)

    colorbar = fig.colorbar(image, ax=axes.ravel(), fraction=0.023, pad=0.012)
    colorbar.set_label("RTOG-conv validation AUROC")
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    saved = red_journal_style.save_publication_figure(
        fig,
        args.output_base,
        dpi=args.dpi,
        formats=("png", "pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
