#!/usr/bin/env python3
"""Recreate Figure 3 from exported actual/predicted probabilities.

This script intentionally does not refit models or read patch caches. It only
uses new_fig/figure3_recreate_actual_pred_by_panel.csv, which contains the
patient-level actual labels and predicted probabilities needed to recreate the
boxplot and ROC panels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import red_journal_style  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "new_fig" / "figure3_recreate_actual_pred_by_panel.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "new_fig"

PANEL_DOMAINS = {"A": "rtog-conv", "B": "rtog-imrt", "C": "ukhd-sbrt"}
PANEL_LABELS = {"A": "RTOG-conv", "B": "RTOG-IMRT", "C": "UKHD-SBRT"}
PANEL_ROLES = {"A": "Selection cohort", "B": "External test", "C": "External test"}
COHORT_LABELS = {"rtog-conv": "RTOG-conv", "rtog-imrt": "RTOG-IMRT", "ukhd-sbrt": "UKHD-SBRT"}
COHORT_LINESTYLES = {"rtog-conv": "--", "rtog-imrt": "--", "ukhd-sbrt": "--"}
MODEL_ORDER = ["Clinical", "D", "DC", "DC+Clinical"]
MODEL_DISPLAY = {
    "Clinical": "Clinical",
    "D": "Dose",
    "DC": "Dose+CT",
    "DC+Clinical": "Dose+CT+Clinical",
}
ROC_MODEL = "DC+Clinical"


def cohort_color_for_panel(panel: str) -> str:
    domain = PANEL_DOMAINS.get(panel, "")
    cohort = COHORT_LABELS.get(domain, domain)
    return red_journal_style.COHORT_COLORS.get(cohort, "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def stable_seed(*parts: object, base_seed: int) -> int:
    acc = int(base_seed)
    for char in "|".join(str(p) for p in parts):
        acc = (acc * 131 + ord(char)) % 2_147_483_647
    return int(acc)


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def bootstrap_auc_values(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, y.size, size=y.size)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], p[idx])))
    return np.asarray(vals, dtype=float)


def bootstrap_roc_band(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    grid_size: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patient-level stratified bootstrap ROC band on a common FPR grid."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    fpr_grid = np.linspace(0.0, 1.0, int(grid_size), dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        nan = np.full_like(fpr_grid, np.nan, dtype=float)
        return fpr_grid, nan, nan

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        nan = np.full_like(fpr_grid, np.nan, dtype=float)
        return fpr_grid, nan, nan

    rng = np.random.default_rng(seed)
    tpr_samples: List[np.ndarray] = []
    for _ in range(int(n_boot)):
        boot_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        fpr_b, tpr_b, _thresholds = roc_curve(y[boot_idx], p[boot_idx])

        # roc_curve can repeat FPR values at vertical steps; keep the maximum
        # TPR at each FPR before interpolation to preserve the upper envelope.
        step_map: Dict[float, float] = {}
        for ff, tt in zip(fpr_b, tpr_b):
            f = float(ff)
            step_map[f] = max(step_map.get(f, 0.0), float(tt))
        fpr_u = np.asarray(sorted(step_map), dtype=float)
        tpr_u = np.asarray([step_map[float(ff)] for ff in fpr_u], dtype=float)
        interp = np.interp(fpr_grid, fpr_u, tpr_u)
        interp[0] = 0.0
        interp[-1] = 1.0
        tpr_samples.append(interp)

    if not tpr_samples:
        nan = np.full_like(fpr_grid, np.nan, dtype=float)
        return fpr_grid, nan, nan
    arr = np.vstack(tpr_samples)
    return (
        fpr_grid,
        np.nanpercentile(arr, 2.5, axis=0),
        np.nanpercentile(arr, 97.5, axis=0),
    )


def build_summary_and_bootstrap(df: pd.DataFrame, *, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    box = df.loc[df["panel_type"].eq("bootstrap_boxplot_input")].copy()
    summary_rows: List[Dict[str, object]] = []
    boot_rows: List[Dict[str, object]] = []
    for (panel, domain, model), d in box.groupby(["panel", "domain", "model"], sort=False):
        y = d["actual"].astype(int).to_numpy()
        p = d["predicted_probability"].astype(float).to_numpy()
        boot = bootstrap_auc_values(
            y,
            p,
            n_boot=n_boot,
            seed=stable_seed(panel, domain, model, base_seed=seed),
        )
        summary_rows.append(
            {
                "panel": panel,
                "domain": domain,
                "cohort": COHORT_LABELS.get(domain, domain),
                "model": model,
                "auroc": safe_auc(y, p),
                "auroc_ci_lower": float(np.nanpercentile(boot, 2.5)) if boot.size else float("nan"),
                "auroc_ci_upper": float(np.nanpercentile(boot, 97.5)) if boot.size else float("nan"),
                "pr_auc": safe_pr_auc(y, p),
                "n": int(len(d)),
                "events": int(np.sum(y)),
            }
        )
        for idx, value in enumerate(boot.tolist()):
            boot_rows.append(
                {
                    "panel": panel,
                    "domain": domain,
                    "cohort": COHORT_LABELS.get(domain, domain),
                    "model": model,
                    "bootstrap_index": int(idx),
                    "auroc": float(value),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(boot_rows)


def build_roc_df(df: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    roc_input = df.loc[df["panel"].eq("D") & df["panel_type"].eq("roc_input") & df["model"].eq(ROC_MODEL)].copy()
    rows: List[Dict[str, object]] = []
    for domain, d in roc_input.groupby("domain", sort=False):
        y = d["actual"].astype(int).to_numpy()
        p = d["predicted_probability"].astype(float).to_numpy()
        if y.size == 0 or np.unique(y).size < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        boot = bootstrap_auc_values(
            y,
            p,
            n_boot=n_boot,
            seed=stable_seed("roc", domain, ROC_MODEL, base_seed=seed),
        )
        auc_ci_lower = float(np.nanpercentile(boot, 2.5)) if boot.size else float("nan")
        auc_ci_upper = float(np.nanpercentile(boot, 97.5)) if boot.size else float("nan")
        fpr_grid, tpr_lower, tpr_upper = bootstrap_roc_band(
            y,
            p,
            n_boot=n_boot,
            seed=stable_seed("roc-band", domain, ROC_MODEL, base_seed=seed),
        )
        for idx, (ff, tt, th) in enumerate(zip(fpr, tpr, thresholds)):
            rows.append(
                {
                    "panel": "D",
                    "row_type": "curve",
                    "domain": domain,
                    "cohort": COHORT_LABELS.get(domain, domain),
                    "model": ROC_MODEL,
                    "point_index": int(idx),
                    "fpr": float(ff),
                    "tpr": float(tt),
                    "threshold": float(th),
                    "auroc": auc,
                    "auroc_ci_lower": auc_ci_lower,
                    "auroc_ci_upper": auc_ci_upper,
                    "tpr_ci_lower": float("nan"),
                    "tpr_ci_upper": float("nan"),
                }
            )
        for idx, (ff, lo, hi) in enumerate(zip(fpr_grid, tpr_lower, tpr_upper)):
            rows.append(
                {
                    "panel": "D",
                    "row_type": "band",
                    "domain": domain,
                    "cohort": COHORT_LABELS.get(domain, domain),
                    "model": ROC_MODEL,
                    "point_index": int(idx),
                    "fpr": float(ff),
                    "tpr": float("nan"),
                    "threshold": float("nan"),
                    "auroc": auc,
                    "auroc_ci_lower": auc_ci_lower,
                    "auroc_ci_upper": auc_ci_upper,
                    "tpr_ci_lower": float(lo),
                    "tpr_ci_upper": float(hi),
                }
            )
    return pd.DataFrame(rows)


def plot_box_panel(ax: plt.Axes, *, panel: str, summary_df: pd.DataFrame, boot_df: pd.DataFrame) -> None:
    data: List[np.ndarray] = []
    for model in MODEL_ORDER:
        vals = boot_df.loc[boot_df["panel"].eq(panel) & boot_df["model"].eq(model), "auroc"].to_numpy(dtype=float)
        data.append(vals[np.isfinite(vals)])
    positions = np.arange(1, len(MODEL_ORDER) + 1, dtype=float)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        whis=(5, 95),
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": cohort_color_for_panel(panel), "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 1.8,
            "markerfacecolor": "0.35",
            "markeredgecolor": "0.35",
            "alpha": 0.45,
        },
    )
    cohort_color = cohort_color_for_panel(panel)
    alphas = [0.35, 0.48, 0.61, 0.74]
    for box, alpha in zip(bp["boxes"], alphas):
        box.set_facecolor(cohort_color)
        box.set_alpha(alpha)
    actual = (
        summary_df.loc[summary_df["panel"].eq(panel)]
        .set_index("model")
        .reindex(MODEL_ORDER)["auroc"]
        .to_numpy(dtype=float)
    )
    ax.scatter(positions, actual, s=16, color="black", zorder=4)
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.0)
    title = f"{PANEL_LABELS.get(panel, panel)} - {PANEL_ROLES.get(panel, '')}"
    ax.set_title(title, loc="left", fontweight="normal", pad=5)
    red_journal_style.clean_axes(ax)


def plot_roc_panel(ax: plt.Axes, roc_df: pd.DataFrame) -> None:
    for domain in ["rtog-conv", "rtog-imrt", "ukhd-sbrt"]:
        d_all = roc_df.loc[roc_df["domain"].eq(domain)].copy()
        if d_all.empty:
            continue
        if "row_type" in d_all.columns:
            band = d_all.loc[d_all["row_type"].eq("band")].copy()
            d = d_all.loc[d_all["row_type"].eq("curve")].copy()
        else:
            band = pd.DataFrame()
            d = d_all
        if d.empty:
            continue
        cohort = COHORT_LABELS.get(domain, domain)
        auc = float(d["auroc"].iloc[0])
        ci_lower = float(d["auroc_ci_lower"].iloc[0])
        ci_upper = float(d["auroc_ci_upper"].iloc[0])
        ci_label = f"{auc:.2f} ({ci_lower:.2f}-{ci_upper:.2f})" if np.isfinite(ci_lower) and np.isfinite(ci_upper) else f"{auc:.2f}"
        color = red_journal_style.COHORT_COLORS.get(cohort, "black")
        if not band.empty:
            ax.fill_between(
                band["fpr"].to_numpy(dtype=float),
                band["tpr_ci_lower"].to_numpy(dtype=float),
                band["tpr_ci_upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            d["fpr"],
            d["tpr"],
            color=color,
            linestyle=COHORT_LINESTYLES.get(domain, "--"),
            linewidth=2.0,
            label=f"{cohort} {ci_label}",
            zorder=2,
        )
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1.1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(MODEL_DISPLAY.get(ROC_MODEL, ROC_MODEL), loc="left", fontweight="normal", pad=5)
    red_journal_style.clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")


def plot_figure(summary_df: pd.DataFrame, boot_df: pd.DataFrame, roc_df: pd.DataFrame, *, out_dir: Path, dpi: int) -> None:
    red_journal_style.apply_red_journal_style()
    fig, axes = plt.subplots(2, 2, figsize=red_journal_style.get_figure_size("multi"), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, panel in zip(axes_flat[:3], ["A", "B", "C"]):
        plot_box_panel(ax, panel=panel, summary_df=summary_df, boot_df=boot_df)
    plot_roc_panel(axes_flat[3], roc_df)
    for label, ax in zip(["A", "B", "C", "D"], axes_flat):
        red_journal_style.add_panel_label(ax, label, x=-0.18, y=1.13)
    for base in ("figure3_from_recreate_csv", "figure3_classification"):
        red_journal_style.save_publication_figure(fig, out_dir / base, dpi=dpi, formats=("png", "pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, dtype={"patient_key": str, "patient_id": str})
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")
    df = df.dropna(subset=["actual", "predicted_probability"]).copy()

    summary_df, boot_df = build_summary_and_bootstrap(df, n_boot=int(args.bootstrap_samples), seed=int(args.seed))
    roc_df = build_roc_df(df, n_boot=int(args.bootstrap_samples), seed=int(args.seed))
    summary_path = args.out_dir / "figure3_from_recreate_csv_summary.csv"
    boot_path = args.out_dir / "figure3_from_recreate_csv_bootstrap_boxplots.csv"
    roc_path = args.out_dir / "figure3_from_recreate_csv_roc_curves.csv"
    summary_df.to_csv(summary_path, index=False)
    boot_df.to_csv(boot_path, index=False)
    roc_df.to_csv(roc_path, index=False)
    plot_figure(summary_df, boot_df, roc_df, out_dir=args.out_dir, dpi=int(args.dpi))

    print(f"[saved] {args.out_dir / 'figure3_from_recreate_csv.png'}", flush=True)
    print(f"[saved] {args.out_dir / 'figure3_from_recreate_csv.pdf'}", flush=True)
    print(f"[saved] {args.out_dir / 'figure3_classification.png'}", flush=True)
    print(f"[saved] {args.out_dir / 'figure3_classification.pdf'}", flush=True)
    print(summary_df[["panel", "model", "domain", "auroc", "pr_auc", "n", "events"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
