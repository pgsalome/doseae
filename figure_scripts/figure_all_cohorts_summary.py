#!/usr/bin/env python3
"""Build cohort-by-metric Figure 2 summary figures across all test cohorts.

Outputs:
- figure_boxplot_grid_all_cohorts.png/.pdf
- figure_main_summary_all_cohorts.png/.pdf

The script reuses per-cohort Figure 2 caches when available and otherwise
computes the missing patient-level metrics by calling the existing helpers in
`figure2.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure2 import (  # noqa: E402
    MODEL_SPECS,
    SHELL_GRADIENT_METRIC_VERSION,
    load_or_compute_metrics,
)
from figure2_cache import ensure_rtog_other_splits  # noqa: E402
import red_journal_style  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper" / "figures"
DEFAULT_METRICS_DIR = REPO_ROOT / "outputs" / "paper" / "cohort_figure2"
DEFAULT_DENSE_EXAMPLE_CACHE = (
    DEFAULT_OUTPUT_DIR
    / "figure_full_ukhd_ANON0024_slice429_dense_reconstruction_gamma_CDE_dense_slice_cache.npz"
)
DEFAULT_DENSE_EXAMPLE_DOSE = Path(
    "/data/pgsal/doseae/test_cohorts_cache/ukhd-imrt/cache/processed_images/test/"
    "ANON_0024/ANON_0024_dose_processed.nrrd"
)
DEFAULT_BAD_DENSE_EXAMPLE_CACHE = (
    DEFAULT_OUTPUT_DIR
    / "figure_full_ukhd_ANON0029_slice440_dense_reconstruction_gamma_FGH_dense_slice_cache.npz"
)
DEFAULT_BAD_DENSE_EXAMPLE_DOSE = Path(
    "/data/pgsal/doseae/test_cohorts_cache/ukhd-imrt/cache/processed_images/test/"
    "ANON_0029/ANON_0029_dose_processed.nrrd"
)
GAMMA_BASELINE_KEY = "gamma_pass_rate_3pct_3mm"
GAMMA_RU_KEY = "gamma_pass_rate_0p5pct_0p5mm"
GAMMA_MIXED_KEY = "gamma_pass_rate_mixed"
SHELL_GRADIENT_PCT_KEY = "ptv_ipsi_lung_shell_gradient_error_pct"
DMEAN_SPLIT_KEY = "Dmean_rel_diff_split"
DMEAN_IPSI_KEY = "Dmean_rel_diff_ipsi"
DMEAN_CONTRA_KEY = "Dmean_rel_diff_contra"
RU_MODEL_KEYS = {
    "resunet_dose",
    "resunet_dose_ct",
}


COHORT_SPECS: List[Dict[str, object]] = [
    {
        "key": "rtog-imrt",
        "label": "RTOG-IMRT",
        "splits": Path("/data/pgsal/doseae/test_cohorts/rtog-imrt/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/rtog-imrt/cache"),
    },
    {
        "key": "rtog-conv",
        "label": "RTOG-conv",
        "splits": Path("/data/pgsal/doseae/test_cohorts/rtog-conv/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/rtog-conv/cache"),
    },
    {
        "key": "ukhd-sbrt",
        "label": "UKHD-SBRT",
        "splits": Path("/data/pgsal/doseae/test_cohorts/ukhd-imrt/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/ukhd-imrt/cache"),
    },
    {
        "key": "ukhd-other-train",
        "label": "UKHD-train",
        "splits": Path("/data/pgsal/doseae/test_cohorts/ukhd-other-train/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/ukhd-other-train/cache"),
    },
]


MODEL_ORDER = [
    "conv_dose",
    "resnet_dose",
    "unet_dose",
    "resunet_dose",
    "resunet_dose_ct",
]

MODEL_LABELS = {
    "conv_dose": "Conv-D",
    "resnet_dose": "ResNet-D",
    "unet_dose": "U-Net-D",
    "resunet_dose": "RU-Net-D",
    "resunet_dose_ct": "RU-Net-DC",
}

METRIC_SPECS = [
    {
        "key": "test_mse",
        "title": "Dose reconstruction MSE",
        "ylabel": "Test MSE",
        "yscale": "log",
        "lower_zero": False,
        "main_point_stat": "q95",
    },
    {
        "key": GAMMA_MIXED_KEY,
        "title": "Gamma pass rate (%)",
        "ylabel": "Pass rate (%)",
        "yscale": "linear",
        "ylim": (0.0, 100.0),
        "lower_zero": True,
        "main_point_stat": "q95",
    },
    {
        "key": SHELL_GRADIENT_PCT_KEY,
        "title": "PTV boundary gradient error (%)",
        "ylabel": "PTV boundary gradient error (%)",
        "yscale": "linear",
        "lower_zero": True,
        "main_point_stat": "q95",
    },
    {
        "key": "Dmean_rel_diff",
        "title": "Relative mean lung dose error (%)",
        "ylabel": "Relative Lung Dmean error (%)",
        "yscale": "linear",
        "ylim": (0.0, 100.0),
        "lower_zero": True,
        "main_point_stat": "median",
    },
]
REL_DMEAN_OUTLIER_QUANTILE = 0.99
RESNET_DMEAN_DISPLAY_CAP = 40.0
MODEL_GROUP_GAP_AFTER_BASELINES = 0.5
MODEL_GROUP_GAP_AFTER_DOSE_ONLY = 0.5
UKHD_OTHER_TRAIN_PANEL_C_DIVISOR = 10000.0
PANEL_D_SEGMENTED_KEY = "Dmean_rel_diff"


def _panel_d_segmented_forward(values: np.ndarray | float) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    out = np.empty_like(y, dtype=np.float64)
    m1 = y <= 1.0
    m2 = (y > 1.0) & (y <= 10.0)
    m3 = y > 10.0
    out[m1] = y[m1]
    out[m2] = 1.0 + (y[m2] - 1.0) / 9.0
    out[m3] = 2.0 + (y[m3] - 10.0) / 90.0
    return out


def _panel_d_segmented_inverse(values: np.ndarray | float) -> np.ndarray:
    t = np.asarray(values, dtype=np.float64)
    out = np.empty_like(t, dtype=np.float64)
    m1 = t <= 1.0
    m2 = (t > 1.0) & (t <= 2.0)
    m3 = t > 2.0
    out[m1] = t[m1]
    out[m2] = 1.0 + (t[m2] - 1.0) * 9.0
    out[m3] = 10.0 + (t[m3] - 2.0) * 90.0
    return out


def _panel_d_segmented_ticks() -> tuple[List[float], List[str]]:
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1", "2", "4", "6", "8", "10", "20", "40", "60", "80", "100"]
    return tick_values, tick_labels


def _apply_panel_d_segmented_axis(ax: plt.Axes) -> None:
    ax.set_yscale("function", functions=(_panel_d_segmented_forward, _panel_d_segmented_inverse))
    ax.set_ylim(0.0, 100.0)
    tick_values, tick_labels = _panel_d_segmented_ticks()
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    ax.axhline(1.0, color="#c8c8c8", linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.7, zorder=0)
    ax.axhline(10.0, color="#c8c8c8", linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.7, zorder=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full cohort-by-metric Figure 2 summary figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final figures.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory for cohort-level cached metric CSVs.",
    )
    parser.add_argument(
        "--cohort-key",
        action="append",
        default=[],
        help="Optional cohort key to include. Repeat to include multiple cohorts.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Only use existing cohort metric CSVs; fail if any are missing.",
    )
    parser.add_argument(
        "--dense-example-cache",
        type=Path,
        default=DEFAULT_DENSE_EXAMPLE_CACHE,
        help="Dense reconstructed slice cache used for panels C-E.",
    )
    parser.add_argument(
        "--dense-example-dose",
        type=Path,
        default=DEFAULT_DENSE_EXAMPLE_DOSE,
        help="Processed dose NRRD used for panels C-E.",
    )
    parser.add_argument(
        "--bad-dense-example-cache",
        type=Path,
        default=DEFAULT_BAD_DENSE_EXAMPLE_CACHE,
        help="Dense reconstructed slice cache used for panels F-H.",
    )
    parser.add_argument(
        "--bad-dense-example-dose",
        type=Path,
        default=DEFAULT_BAD_DENSE_EXAMPLE_DOSE,
        help="Processed dose NRRD used for panels F-H.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    red_journal_style.apply_red_journal_style()


def resolve_cohort_specs(selected_keys: List[str]) -> List[Dict[str, object]]:
    if not selected_keys:
        resolved = COHORT_SPECS
    else:
        allowed = {str(spec["key"]) for spec in COHORT_SPECS}
        requested = [str(key) for key in selected_keys]
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"Unknown cohort key(s): {unknown}")
        resolved = [spec for spec in COHORT_SPECS if str(spec["key"]) in set(requested)]

    if any(str(spec["key"]) == "rtog-other" for spec in resolved):
        rtog_other_splits = next(
            Path(spec["splits"]) for spec in resolved if str(spec["key"]) == "rtog-other"
        )
        if not rtog_other_splits.exists():
            ensure_rtog_other_splits()
    return resolved


def cohort_metric_paths(metrics_dir: Path, cohort_key: str) -> Dict[str, Path]:
    return {
        "metrics": metrics_dir / f"{cohort_key}_metrics.csv",
        "combined": metrics_dir / f"{cohort_key}_combined.csv",
    }


def load_or_build_cohort_dataframe(
    cohort_spec: Dict[str, object],
    metrics_dir: Path,
    skip_compute: bool,
) -> pd.DataFrame:
    cohort_key = str(cohort_spec["key"])
    cohort_label = str(cohort_spec["label"])
    splits_path = Path(cohort_spec["splits"])
    data_dir = Path(cohort_spec["data_dir"])
    paths = cohort_metric_paths(metrics_dir, cohort_key)

    combined_df: pd.DataFrame | None
    if paths["combined"].exists():
        combined_df = pd.read_csv(paths["combined"])
        has_shell_version = "shell_gradient_metric_version" in combined_df.columns
        version_ok = has_shell_version and (
            combined_df["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION
        ).all()
        if not version_ok:
            if skip_compute:
                print(
                    f"[warn] Using stale combined cohort metrics in skip-compute mode: {paths['combined']}",
                    flush=True,
                )
            else:
                combined_df = None
    else:
        combined_df = None

    if combined_df is None:
        if skip_compute:
            raise FileNotFoundError(f"Missing combined cohort metrics: {paths['combined']}")
        combined_df = load_or_compute_metrics(paths["metrics"], splits_path, data_dir)
        if "gamma_pass_rate_3pct_3mm" not in combined_df.columns and "gamma_pass_rate" in combined_df.columns:
            combined_df["gamma_pass_rate_3pct_3mm"] = combined_df["gamma_pass_rate"]
        paths["combined"].parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(paths["combined"], index=False)

    combined_df = combined_df.copy()
    if GAMMA_BASELINE_KEY not in combined_df.columns:
        if "gamma_pass_rate" in combined_df.columns:
            combined_df[GAMMA_BASELINE_KEY] = combined_df["gamma_pass_rate"]
        else:
            combined_df[GAMMA_BASELINE_KEY] = np.nan
    if GAMMA_RU_KEY not in combined_df.columns:
        combined_df[GAMMA_RU_KEY] = np.nan

    # Side-specific Dmean columns were introduced after early cache runs.
    # Fallback keeps the script usable in skip-compute mode, but true split
    # variability requires recomputed per-patient side metrics.
    if DMEAN_IPSI_KEY not in combined_df.columns:
        combined_df[DMEAN_IPSI_KEY] = pd.to_numeric(combined_df.get("Dmean_rel_diff"), errors="coerce")
    if DMEAN_CONTRA_KEY not in combined_df.columns:
        combined_df[DMEAN_CONTRA_KEY] = pd.to_numeric(combined_df.get("Dmean_rel_diff"), errors="coerce")

    # Display-only percentage conversion for Panel B using existing cached values.
    combined_df[SHELL_GRADIENT_PCT_KEY] = pd.to_numeric(
        combined_df["ptv_ipsi_lung_shell_gradient_error"], errors="coerce"
    ) * 100.0
    if cohort_key == "ukhd-other-train":
        combined_df[SHELL_GRADIENT_PCT_KEY] = combined_df[SHELL_GRADIENT_PCT_KEY] / UKHD_OTHER_TRAIN_PANEL_C_DIVISOR

    combined_df[GAMMA_MIXED_KEY] = combined_df[GAMMA_BASELINE_KEY]
    ru_mask = combined_df["model_key"].isin(RU_MODEL_KEYS)
    combined_df.loc[ru_mask, GAMMA_MIXED_KEY] = combined_df.loc[ru_mask, GAMMA_RU_KEY].where(
        combined_df.loc[ru_mask, GAMMA_RU_KEY].notna(),
        combined_df.loc[ru_mask, GAMMA_BASELINE_KEY],
    )
    combined_df["cohort_key"] = cohort_key
    combined_df["cohort_label"] = cohort_label

    # Display cap requested for ResNet only.
    resnet_mask = combined_df["model_key"] == "resnet_dose"
    if resnet_mask.any():
        for col in ["Dmean_rel_diff", DMEAN_IPSI_KEY, DMEAN_CONTRA_KEY]:
            if col in combined_df.columns:
                combined_df.loc[resnet_mask, col] = (
                    pd.to_numeric(combined_df.loc[resnet_mask, col], errors="coerce").clip(upper=RESNET_DMEAN_DISPLAY_CAP)
                )

    combined_df["model_label"] = combined_df["model_key"].map(MODEL_LABELS).fillna(combined_df["display_label"])
    return combined_df


def compute_metric_limits(df: pd.DataFrame) -> Dict[str, tuple[float, float]]:
    limits: Dict[str, tuple[float, float]] = {}
    for spec in METRIC_SPECS:
        key = str(spec["key"])
        if key == DMEAN_SPLIT_KEY:
            values = pd.concat(
                [
                    pd.to_numeric(df[DMEAN_IPSI_KEY], errors="coerce"),
                    pd.to_numeric(df[DMEAN_CONTRA_KEY], errors="coerce"),
                ],
                ignore_index=True,
            ).dropna().to_numpy(dtype=np.float64)
        else:
            values = pd.to_numeric(df[key], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            limits[key] = (0.0, 1.0)
            continue
        if "ylim" in spec:
            limits[key] = tuple(spec["ylim"])
            continue
        if key == DMEAN_SPLIT_KEY:
            upper_q = float(np.quantile(values, REL_DMEAN_OUTLIER_QUANTILE))
            upper = max(upper_q * 1.05, 1.0)
            limits[key] = (0.0, upper)
            continue
        if str(spec["yscale"]) == "log":
            positive = values[values > 0]
            lower = max(float(positive.min()) * 0.6, 1e-8)
            upper = float(positive.max()) * 1.5
        else:
            lower = 0.0 if bool(spec.get("lower_zero", False)) else float(values.min()) * 0.95
            upper = float(values.max()) * 1.1 if float(values.max()) > 0 else 1.0
        limits[key] = (lower, upper)
    return limits


def format_model_labels() -> List[str]:
    return [MODEL_LABELS[key] for key in MODEL_ORDER]


def model_positions() -> np.ndarray:
    positions: List[float] = []
    x = 1.0
    for idx, _ in enumerate(MODEL_ORDER):
        positions.append(x)
        x += 1.0
        if idx == 2:
            x += (MODEL_GROUP_GAP_AFTER_BASELINES + 0.2)
        if idx == 3:
            x += (MODEL_GROUP_GAP_AFTER_DOSE_ONLY + 0.2)
    return np.array(positions, dtype=np.float64)


def add_modality_group_separator(ax: plt.Axes, positions: np.ndarray) -> None:
    split_x = float((positions[3] + positions[4]) / 2.0)
    ax.axvline(
        split_x,
        color="#bfbfbf",
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        alpha=0.8,
        zorder=0,
    )


def boxplot_panel(
    ax: plt.Axes,
    cohort_df: pd.DataFrame,
    metric_spec: Dict[str, object],
    metric_limits: Dict[str, tuple[float, float]],
    show_title: bool,
    show_xlabels: bool,
    cohort_label: str | None,
) -> None:
    metric_key = str(metric_spec["key"])
    positions = model_positions()
    fills = ["#e6e6e6", "#d7d7d7", "#c7c7c7", "#b8b8b8", "#a8a8a8", "#999999", "#8a8a8a"]

    grouped = []
    present_positions = []
    box_fills: List[str] = []
    dmean_upper = metric_limits.get(DMEAN_SPLIT_KEY, (0.0, np.inf))[1]
    if metric_key == DMEAN_SPLIT_KEY:
        side_specs = [
            (DMEAN_IPSI_KEY, -0.16, "#9a9a9a"),
            (DMEAN_CONTRA_KEY, +0.16, "#d9d9d9"),
        ]
        for pos, model_key in zip(positions, MODEL_ORDER):
            model_rows = cohort_df.loc[cohort_df["model_key"] == model_key]
            for side_col, offset, side_fill in side_specs:
                values = pd.to_numeric(model_rows[side_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
                values = values[values <= dmean_upper]
                if len(values) == 0:
                    continue
                grouped.append(values)
                present_positions.append(float(pos) + offset)
                box_fills.append(side_fill)
    else:
        for idx, (pos, model_key) in enumerate(zip(positions, MODEL_ORDER)):
            values = cohort_df.loc[cohort_df["model_key"] == model_key, metric_key].dropna().to_numpy(dtype=np.float32)
            if len(values) == 0:
                continue
            grouped.append(values)
            present_positions.append(float(pos))
            box_fills.append(fills[idx])

    if grouped:
        show_fliers = metric_key != DMEAN_SPLIT_KEY
        box = ax.boxplot(
            grouped,
            positions=present_positions,
            widths=0.26 if metric_key == DMEAN_SPLIT_KEY else 0.56,
            patch_artist=True,
            showmeans=True,
            showfliers=show_fliers,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3.8,
            },
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 1.8,
                "alpha": 0.45,
            },
        )
        for patch, fill in zip(box["boxes"], box_fills):
            patch.set_facecolor(fill)

    ax.set_xticks(positions)
    if show_xlabels:
        ax.set_xticklabels(format_model_labels(), rotation=24, ha="right")
    else:
        ax.set_xticklabels([""] * len(positions))
    if show_title:
        ax.set_title(str(metric_spec["title"]), pad=6)
    if cohort_label is not None:
        ax.set_ylabel(cohort_label, fontweight="bold", rotation=90, labelpad=16)

    if metric_key == PANEL_D_SEGMENTED_KEY:
        _apply_panel_d_segmented_axis(ax)
    else:
        ax.set_yscale(str(metric_spec["yscale"]))
        ax.set_ylim(metric_limits[metric_key])
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#1a1a1a")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def build_supplementary_figure(all_df: pd.DataFrame, cohort_specs: List[Dict[str, object]], output_dir: Path) -> None:
    metric_limits = compute_metric_limits(all_df)
    figure, axes = plt.subplots(len(cohort_specs), len(METRIC_SPECS), figsize=(16.0, 12.8))
    figure.patch.set_facecolor("white")
    for ax in np.ravel(axes):
        ax.set_facecolor("white")

    for row_idx, cohort_spec in enumerate(cohort_specs):
        cohort_key = str(cohort_spec["key"])
        cohort_label = str(cohort_spec["label"])
        cohort_df = all_df.loc[all_df["cohort_key"] == cohort_key].copy()
        for col_idx, metric_spec in enumerate(METRIC_SPECS):
            ax = axes[row_idx, col_idx]
            boxplot_panel(
                ax=ax,
                cohort_df=cohort_df,
                metric_spec=metric_spec,
                metric_limits=metric_limits,
                show_title=(row_idx == 0),
                show_xlabels=(row_idx == len(cohort_specs) - 1),
                cohort_label=cohort_label if col_idx == 0 else None,
            )

    figure.tight_layout(pad=1.05, w_pad=1.2, h_pad=1.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "figure_boxplot_grid_all_cohorts.png"
    pdf_path = output_dir / "figure_boxplot_grid_all_cohorts.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(figure)


def summarize_cohort_statistics(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort_key, cohort_label, model_key, model_label), group in all_df.groupby(
        ["cohort_key", "cohort_label", "model_key", "model_label"],
        sort=False,
    ):
        row = {
            "cohort_key": cohort_key,
            "cohort_label": cohort_label,
            "model_key": model_key,
            "model_label": model_label,
            "n_patients": int(group["patient_id"].nunique()),
        }
        for metric_spec in METRIC_SPECS:
            metric_key = str(metric_spec["key"])
            if metric_key == DMEAN_SPLIT_KEY:
                for side_name, side_col in [("ipsi", DMEAN_IPSI_KEY), ("contra", DMEAN_CONTRA_KEY)]:
                    values = pd.to_numeric(group[side_col], errors="coerce").dropna()
                    if values.empty:
                        row[f"{metric_key}_{side_name}_median"] = np.nan
                        row[f"{metric_key}_{side_name}_q1"] = np.nan
                        row[f"{metric_key}_{side_name}_q3"] = np.nan
                        row[f"{metric_key}_{side_name}_q95"] = np.nan
                        continue
                    row[f"{metric_key}_{side_name}_median"] = float(values.median())
                    row[f"{metric_key}_{side_name}_q1"] = float(values.quantile(0.25))
                    row[f"{metric_key}_{side_name}_q3"] = float(values.quantile(0.75))
                    row[f"{metric_key}_{side_name}_q95"] = float(values.quantile(0.95))
                continue

            values = pd.to_numeric(group[metric_key], errors="coerce").dropna()
            if values.empty:
                row[f"{metric_key}_median"] = np.nan
                row[f"{metric_key}_q1"] = np.nan
                row[f"{metric_key}_q3"] = np.nan
                row[f"{metric_key}_q95"] = np.nan
                continue
            row[f"{metric_key}_median"] = float(values.median())
            row[f"{metric_key}_q1"] = float(values.quantile(0.25))
            row[f"{metric_key}_q3"] = float(values.quantile(0.75))
            row[f"{metric_key}_q95"] = float(values.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def _crop_example_arrays(
    arrays: List[np.ndarray],
    dose_slice: np.ndarray,
    *,
    threshold: float = 0.01,
    padding: int = 60,
) -> List[np.ndarray]:
    max_dose = float(np.nanmax(dose_slice))
    support = dose_slice >= max_dose * float(threshold)
    if max_dose <= 0 or not np.any(support):
        return arrays
    ys, xs = np.where(support)
    pad = max(0, int(padding))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(dose_slice.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(dose_slice.shape[1], int(xs.max()) + pad + 1)
    return [arr[y0:y1, x0:x1] for arr in arrays]


def load_dense_example_panels(
    cache_path: Path,
    dose_path: Path,
    *,
    gamma_dose_threshold_percent: float = 3.0,
) -> Dict[str, object]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing dense reconstruction cache: {cache_path}")
    if not dose_path.exists():
        raise FileNotFoundError(f"Missing processed dose NRRD: {dose_path}")

    cached = np.load(cache_path)
    slice_idx = int(cached["slice_idx"])
    patient_id = str(cached["patient_id"])
    recon_slice = cached["recon_slice"].astype(np.float32)
    votes = cached["votes"].astype(np.uint16)
    dose = sitk.GetArrayFromImage(sitk.ReadImage(str(dose_path))).astype(np.float32)
    dose_slice = dose[slice_idx].astype(np.float32)

    max_dose = float(np.nanmax(dose_slice))
    gamma_threshold = max(max_dose * float(gamma_dose_threshold_percent) / 100.0, 1e-8)
    gamma_slice = np.abs(dose_slice - recon_slice).astype(np.float32) / gamma_threshold
    valid = (votes > 0) & (dose_slice > 0.10 * max_dose)
    pass_rate = float(np.mean(gamma_slice[valid] <= 1.0) * 100.0) if np.any(valid) else float("nan")
    gamma_display = gamma_slice.copy()
    gamma_display[~((votes > 0) & (dose_slice > 0.01 * max_dose))] = np.nan

    dose_slice, recon_slice, gamma_display = _crop_example_arrays(
        [dose_slice, recon_slice, gamma_display],
        dose_slice,
        threshold=0.01,
        padding=60,
    )
    dose_vmax = float(np.nanpercentile(dose_slice[dose_slice > 0], 99.5)) if np.any(dose_slice > 0) else 1.0
    dose_vmax = max(dose_vmax, 1e-6)
    dose_display = np.clip(dose_slice / dose_vmax, 0.0, 1.0)
    recon_display = np.clip(recon_slice / dose_vmax, 0.0, 1.0)
    return {
        "patient_id": patient_id,
        "slice_idx": slice_idx,
        "dose_slice": dose_display.astype(np.float32),
        "recon_slice": recon_display.astype(np.float32),
        "gamma_slice": gamma_display,
        "dose_vmax": dose_vmax,
        "slice_gamma_pass_rate": pass_rate,
    }


def patient_level_gamma_pass(all_df: pd.DataFrame, patient_id: str, fallback: float) -> float:
    rows = all_df.loc[
        (all_df["patient_id"].astype(str) == str(patient_id))
        & (all_df["model_key"].astype(str) == "resunet_dose_ct")
        & (all_df["cohort_key"].astype(str) == "ukhd-sbrt")
    ]
    if rows.empty or GAMMA_BASELINE_KEY not in rows.columns:
        return float(fallback)
    value = pd.to_numeric(rows.iloc[0][GAMMA_BASELINE_KEY], errors="coerce")
    if pd.isna(value):
        return float(fallback)
    return float(value)


def _add_image_panel(
    figure: plt.Figure,
    ax: plt.Axes,
    *,
    image: np.ndarray,
    panel_label: str | None,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    annotation: str | None = None,
    add_colorbar: bool = True,
    title_fontsize: float | None = None,
    panel_label_x: float = -0.08,
    panel_label_y: float = 1.05,
    title_color: str = "black",
) -> object:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    im = ax.imshow(image, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    ax.set_title(title, pad=2, fontweight="normal", fontsize=title_fontsize, color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    if panel_label:
        red_journal_style.add_panel_label(ax, panel_label, x=panel_label_x, y=panel_label_y)
    if add_colorbar:
        cbar = figure.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(length=2.5)
    if annotation:
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
    return im


def _add_image_block_background(
    figure: plt.Figure,
    axes: list[plt.Axes],
    *,
    pad: float = 0.012,
    left_label_pad: float = 0.055,
) -> tuple[float, float, float, float]:
    """Add one continuous black background behind the dose-example image block."""
    boxes = [ax.get_position() for ax in axes]
    x0 = max(0.0, min(box.x0 for box in boxes) - pad - left_label_pad)
    y0 = max(0.0, min(box.y0 for box in boxes) - pad)
    x1 = min(1.0, max(box.x1 for box in boxes) + pad)
    y1 = min(1.0, max(box.y1 for box in boxes) + pad + 0.022)
    bg_ax = figure.add_axes([x0, y0, x1 - x0, y1 - y0], zorder=0)
    bg_ax.set_facecolor("black")
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    for spine in bg_ax.spines.values():
        spine.set_visible(False)
    return x0, y0, x1, y1


def build_main_summary_figure(
    all_df: pd.DataFrame,
    output_dir: Path,
    dense_example_cache: Path,
    dense_example_dose: Path,
    bad_dense_example_cache: Path,
    bad_dense_example_dose: Path,
) -> None:
    metric_limits = compute_metric_limits(all_df)
    summary_df = summarize_cohort_statistics(all_df)
    dense_example = load_dense_example_panels(
        dense_example_cache,
        dense_example_dose,
        gamma_dose_threshold_percent=1.0,
    )
    bad_dense_example = load_dense_example_panels(
        bad_dense_example_cache,
        bad_dense_example_dose,
        gamma_dose_threshold_percent=3.0,
    )
    dense_example["gamma_pass_rate"] = float(dense_example["slice_gamma_pass_rate"])
    bad_dense_example["gamma_pass_rate"] = patient_level_gamma_pass(
        all_df,
        str(bad_dense_example["patient_id"]),
        float(bad_dense_example["slice_gamma_pass_rate"]),
    )

    main_metric_specs = METRIC_SPECS[:2]
    figure = plt.figure(figsize=(7.0, 6.70), facecolor="white")
    grid = figure.add_gridspec(
        3,
        6,
        height_ratios=[1.00, 1.15, 0.07],
        hspace=0.48,
        wspace=0.42,
    )
    axes_flat = [
        figure.add_subplot(grid[0, 0:3]),
        figure.add_subplot(grid[0, 3:6]),
    ]
    image_grid = grid[1, :].subgridspec(2, 3, hspace=0.12, wspace=0.06)
    image_axes = [
        figure.add_subplot(image_grid[0, 0]),
        figure.add_subplot(image_grid[0, 1]),
        figure.add_subplot(image_grid[0, 2]),
        figure.add_subplot(image_grid[1, 0]),
        figure.add_subplot(image_grid[1, 1]),
        figure.add_subplot(image_grid[1, 2]),
    ]
    dose_cbar_ax = figure.add_subplot(grid[2, 0:4])
    gamma_cbar_ax = figure.add_subplot(grid[2, 4:6])
    figure.patch.set_facecolor("white")
    for ax in [*axes_flat, *image_axes, dose_cbar_ax, gamma_cbar_ax]:
        ax.set_facecolor("white")
    for ax in image_axes:
        ax.set_facecolor("black")
        ax.set_zorder(2)
    style_overrides = {
        "RTOG-IMRT": ("o", "black"),
        "RTOG-conv": ("s", "#555555"),
        "RTOG-other": ("D", "#8c8c8c"),
        "UKHD-SBRT": ("^", "#b3b3b3"),
        "UKHD-train": ("o", "#3a7fb0"),
        "MGH-VMAT": ("^", "#9e9e9e"),
    }
    default_markers = ["o", "s", "D", "^"]
    default_colors = ["black", "#555555", "#7a7a7a", "#9a9a9a", "#b3b3b3", "#6b6b6b", "#8f8f8f", "#4f4f4f", "#c2c2c2"]

    present_labels = set(summary_df["cohort_label"].astype(str).unique().tolist())
    ordered_labels = [str(spec["label"]) for spec in COHORT_SPECS if str(spec["label"]) in present_labels]
    extra_labels = sorted(present_labels - set(ordered_labels))
    ordered_labels.extend(extra_labels)

    if len(ordered_labels) == 1:
        offset_values = np.array([0.0], dtype=np.float64)
    else:
        offset_values = np.linspace(-0.28, 0.28, num=len(ordered_labels))
    offsets = {label: float(offset) for label, offset in zip(ordered_labels, offset_values)}
    cohort_markers = {}
    for idx, label in enumerate(ordered_labels):
        cohort_markers[label] = style_overrides.get(
            label,
            (default_markers[idx % len(default_markers)], default_colors[idx % len(default_colors)]),
        )

    positions = model_positions()

    panel_labels = ["A", "B"]
    for panel_idx, (ax, metric_spec) in enumerate(zip(axes_flat, main_metric_specs)):
        metric_key = str(metric_spec["key"])
        point_stat = str(metric_spec.get("main_point_stat", "q95"))
        if metric_key == DMEAN_SPLIT_KEY:
            for cohort_label in ordered_labels:
                marker, color = cohort_markers[cohort_label]
                cohort_df = summary_df.loc[summary_df["cohort_label"] == cohort_label]
                for side_name, side_offset, side_filled in [
                    ("ipsi", -0.08, True),
                    ("contra", 0.08, False),
                ]:
                    xs = []
                    ys = []
                    for pos, model_key in zip(positions, MODEL_ORDER):
                        row = cohort_df.loc[cohort_df["model_key"] == model_key]
                        if row.empty:
                            continue
                        point_col = f"{metric_key}_{side_name}_{point_stat}"
                        point_value = float(row.iloc[0][point_col])
                        if np.isnan(point_value):
                            continue
                        xs.append(pos + offsets[cohort_label] + side_offset)
                        ys.append(point_value)
                    if xs:
                        ax.plot(
                            xs,
                            ys,
                            marker=marker,
                            color=color,
                            markerfacecolor=(color if side_filled else "white"),
                            markeredgecolor=color,
                            markersize=4.2,
                            linestyle="none",
                            label=cohort_label if side_name == "ipsi" else None,
                        )
        else:
            for cohort_label in ordered_labels:
                marker, color = cohort_markers[cohort_label]
                cohort_df = summary_df.loc[summary_df["cohort_label"] == cohort_label]
                xs = []
                ys = []
                for pos, model_key in zip(positions, MODEL_ORDER):
                    row = cohort_df.loc[cohort_df["model_key"] == model_key]
                    if row.empty:
                        continue
                    point_value = float(row.iloc[0][f"{metric_key}_{point_stat}"])
                    if np.isnan(point_value):
                        continue
                    xs.append(pos + offsets[cohort_label])
                    ys.append(point_value)
                if xs:
                    yerr_low = []
                    yerr_high = []
                    for pos, model_key in zip(positions, MODEL_ORDER):
                        row = cohort_df.loc[cohort_df["model_key"] == model_key]
                        if row.empty:
                            continue
                        point_value = float(row.iloc[0][f"{metric_key}_{point_stat}"])
                        if np.isnan(point_value):
                            continue
                        q1_value = float(row.iloc[0][f"{metric_key}_q1"])
                        if np.isnan(q1_value):
                            q1_value = point_value
                        q3_value = float(row.iloc[0][f"{metric_key}_q3"])
                        if np.isnan(q3_value):
                            q3_value = point_value

                        lower_err = max(point_value - q1_value, 0.0)
                        upper_err = q3_value - point_value
                        if upper_err <= 0.0:
                            # Keep two-sided bars visible even when point_stat is an upper-tail
                            # statistic (for example q95), by mirroring distance to q3.
                            upper_err = max(point_value - q3_value, 0.0)
                        yerr_low.append(lower_err)
                        yerr_high.append(upper_err)
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[yerr_low, yerr_high],
                        fmt=marker,
                        color=color,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markersize=7.0,
                        linestyle="none",
                        elinewidth=1.9,
                        capsize=4.0,
                        capthick=1.9,
                        label=cohort_label,
                    )

        ax.set_title(str(metric_spec["title"]), pad=4, fontweight="bold")
        if panel_labels[panel_idx] == "B":
            red_journal_style.add_panel_label(ax, "B", x=1.02, y=1.10)
        else:
            red_journal_style.add_panel_label(ax, panel_labels[panel_idx], x=-0.14, y=1.10)
        if metric_key == GAMMA_MIXED_KEY:
            ax.set_ylabel("Pass rate (%)", labelpad=2)
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.055, 0.5)
        else:
            ax.set_ylabel(str(metric_spec.get("ylabel", "")))
        ax.set_xticks(positions)
        ax.set_xticklabels(format_model_labels(), rotation=22, ha="right")
        add_modality_group_separator(ax, positions)
        if metric_key == PANEL_D_SEGMENTED_KEY:
            _apply_panel_d_segmented_axis(ax)
        else:
            ax.set_yscale(str(metric_spec["yscale"]))
            ax.set_ylim(metric_limits[metric_key])
        ax.set_xlim(positions[0] - 0.8, positions[-1] + 0.8)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors="#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, min(4, len(labels))),
            frameon=False,
            bbox_to_anchor=(0.5, 0.052),
            borderaxespad=0.0,
            handlelength=1.4,
        )

    dose_im = _add_image_panel(
        figure,
        image_axes[0],
        image=dense_example["dose_slice"],
        panel_label=None,
        title="Original",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Normalized dose",
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )
    _add_image_panel(
        figure,
        image_axes[1],
        image=dense_example["recon_slice"],
        panel_label=None,
        title="Reconstructed",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Normalized dose",
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )
    gamma_im = _add_image_panel(
        figure,
        image_axes[2],
        image=dense_example["gamma_slice"],
        panel_label=None,
        title=f"Gamma ({float(dense_example['gamma_pass_rate']):.1f}%)",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
        cbar_label="Gamma",
        annotation=None,
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )

    _add_image_panel(
        figure,
        image_axes[3],
        image=bad_dense_example["dose_slice"],
        panel_label=None,
        title="Original",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Normalized dose",
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )
    _add_image_panel(
        figure,
        image_axes[4],
        image=bad_dense_example["recon_slice"],
        panel_label=None,
        title="Reconstructed",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Normalized dose",
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )
    _add_image_panel(
        figure,
        image_axes[5],
        image=bad_dense_example["gamma_slice"],
        panel_label=None,
        title=f"Gamma ({float(bad_dense_example['gamma_pass_rate']):.1f}%)",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
        cbar_label="Gamma",
        annotation=None,
        add_colorbar=False,
        title_fontsize=10.5,
        title_color="white",
    )

    dose_cbar = figure.colorbar(dose_im, cax=dose_cbar_ax, orientation="horizontal")
    dose_cbar.set_label("Normalized dose")
    dose_cbar.ax.tick_params(length=2.5)

    gamma_cbar = figure.colorbar(gamma_im, cax=gamma_cbar_ax, orientation="horizontal")
    gamma_cbar.set_label("Gamma")
    gamma_cbar.ax.tick_params(length=2.5)

    for ax in image_axes:
        ax.set_anchor("C")
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.078, top=0.965)
    image_block_x0, image_block_y0, _, _ = _add_image_block_background(
        figure,
        image_axes,
        pad=0.012,
        left_label_pad=0.055,
    )
    dose_cbar_pos = dose_cbar_ax.get_position()
    gamma_cbar_pos = gamma_cbar_ax.get_position()
    colorbar_y = max(0.095, image_block_y0 - 0.050)
    dose_cbar_ax.set_position([dose_cbar_pos.x0, colorbar_y, dose_cbar_pos.width, dose_cbar_pos.height])
    gamma_cbar_ax.set_position([gamma_cbar_pos.x0, colorbar_y, gamma_cbar_pos.width, gamma_cbar_pos.height])
    row_annotations = [
        ("C", "High", image_axes[0]),
        ("D", "Lower", image_axes[3]),
    ]
    for label, row_name, ax in row_annotations:
        bbox = ax.get_position()
        figure.text(
            image_block_x0 + 0.008,
            bbox.y1 + 0.015,
            label,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
            color="white",
        )
        figure.text(
            image_block_x0 + 0.028,
            (bbox.y0 + bbox.y1) / 2.0,
            row_name,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            rotation=90,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "figure_main_summary_all_cohorts.png"
    pdf_path = output_dir / "figure_main_summary_all_cohorts.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(figure)


def main() -> None:
    np.random.seed(0)
    args = parse_args()
    configure_matplotlib()
    cohort_specs = resolve_cohort_specs(args.cohort_key)

    frames = []
    for cohort_spec in cohort_specs:
        frames.append(load_or_build_cohort_dataframe(cohort_spec, args.metrics_dir, args.skip_compute))

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.loc[all_df["model_key"].isin(MODEL_ORDER)].copy()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.metrics_dir / "all_cohorts_per_patient_metrics.csv"
    summary_csv = args.metrics_dir / "all_cohorts_summary_statistics.csv"
    all_df.to_csv(combined_csv, index=False)
    summarize_cohort_statistics(all_df).to_csv(summary_csv, index=False)

    build_supplementary_figure(all_df, cohort_specs, args.output_dir)
    build_main_summary_figure(
        all_df,
        args.output_dir,
        args.dense_example_cache,
        args.dense_example_dose,
        args.bad_dense_example_cache,
        args.bad_dense_example_dose,
    )

    print(f"Saved {combined_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {args.output_dir / 'figure_boxplot_grid_all_cohorts.png'}")
    print(f"Saved {args.output_dir / 'figure_boxplot_grid_all_cohorts.pdf'}")
    print(f"Saved {args.output_dir / 'figure_main_summary_all_cohorts.png'}")
    print(f"Saved {args.output_dir / 'figure_main_summary_all_cohorts.pdf'}")


if __name__ == "__main__":
    main()
