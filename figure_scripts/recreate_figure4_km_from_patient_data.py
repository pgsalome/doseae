#!/usr/bin/env python3
"""Recreate Figure 4 KM artwork from frozen patient-level risk-group data.

This script does not refit models or recompute risk groups. It only redraws the
Kaplan-Meier figure from the exported patient table and summary statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import red_journal_style  # noqa: E402
from create_figure5_ukhd_sbrt_km_risk_groups import (  # noqa: E402
    km_curve,
    survival_at,
)


DEFAULT_PATIENT_DATA = REPO_ROOT / "oliver_paper" / "figures" / "Figure4_ukhd_sbrt_km_risk_groups_patient_data.csv"
DEFAULT_SUMMARY = REPO_ROOT / "oliver_paper" / "figures" / "Figure4_ukhd_sbrt_km_risk_groups_summary.csv"
DEFAULT_OUT_BASE = REPO_ROOT / "oliver_paper" / "figures" / "Figure4_ukhd_sbrt_km_risk_groups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-data", type=Path, default=DEFAULT_PATIENT_DATA)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--display-max-months",
        type=float,
        default=36.0,
        help=(
            "Maximum displayed follow-up time. This crops the artwork and risk table only; "
            "HR and log-rank estimates continue to use all available follow-up."
        ),
    )
    return parser.parse_args()


def plot_km_from_data(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    out_base: Path,
    dpi: int,
    display_max_months: float,
) -> None:
    red_journal_style.apply_red_journal_style()
    fig = plt.figure(figsize=red_journal_style.get_figure_size("km"))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.2, 1.15], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    risk_ax = fig.add_subplot(gs[1], sharex=ax)
    ax.set_zorder(2)
    risk_ax.set_zorder(1)
    risk_ax.patch.set_alpha(0.0)

    colors = {"High risk": "#D55E00", "Low risk": "#0072B2"}
    styles = {"High risk": "-", "Low risk": "--"}
    curve_rows = []

    data = data.copy()
    data["time_months"] = pd.to_numeric(data["time_months"], errors="coerce")
    data["event"] = pd.to_numeric(data["event"], errors="coerce").fillna(0).astype(int)
    data = data.dropna(subset=["time_months", "risk_group"]).copy()

    max_time = float(display_max_months)
    if not np.isfinite(max_time) or max_time <= 0:
        raise ValueError("--display-max-months must be positive and finite")
    tick_step = 6.0 if max_time <= 36.0 else 12.0
    x_ticks = np.arange(0.0, max_time + 0.01, tick_step)

    for group in ["Low risk", "High risk"]:
        group_df = data.loc[data["risk_group"].eq(group)].copy()
        curve = km_curve(group_df["time_months"].to_numpy(), group_df["event"].to_numpy())
        max_group_time = float(group_df["time_months"].max())
        if not curve.empty and max_group_time > float(curve["time"].max()):
            last = curve.iloc[-1].copy()
            last["time"] = max_group_time
            last["n_at_risk"] = float(np.sum(group_df["time_months"].to_numpy() >= max_group_time))
            last["events"] = 0.0
            curve = pd.concat([curve, last.to_frame().T], ignore_index=True)
        curve["risk_group"] = group
        curve_rows.append(curve)

        label = f"{group} (n={len(group_df)}, events={int(group_df['event'].sum())})"
        ax.step(
            curve["time"],
            curve["survival"],
            where="post",
            color=colors[group],
            linestyle=styles[group],
            linewidth=2.2,
            label=f"{group} (n={len(group_df)}, events={int(group_df['event'].sum())})",
        )
        ax.fill_between(
            curve["time"],
            curve["lower"],
            curve["upper"],
            step="post",
            color=colors[group],
            alpha=0.14,
            linewidth=0,
        )
        censored = group_df.loc[group_df["event"].eq(0)]
        if not censored.empty:
            ax.scatter(
                censored["time_months"],
                survival_at(censored["time_months"].to_numpy(), curve),
                marker="+",
                s=28,
                color=colors[group],
                linewidths=1.0,
                alpha=0.9,
                zorder=4,
            )

    red_journal_style.clean_axes(ax, grid=True)
    ax.set_ylabel("RILI-free survival probability")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(0.0, max_time)
    ax.set_title("UKHD-SBRT risk stratification", fontweight="normal", pad=5)
    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Time from treatment, months", labelpad=4)
    ax.tick_params(axis="x", labelbottom=True, bottom=True, length=3.0, pad=2)

    first = summary.iloc[0]
    annotation = (
        f"HR={float(first['hazard_ratio']):.2f}\n"
        f"log-rank p={float(first['p_value']):.3f}"
    )
    ax.text(
        0.95,
        0.04,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#BFBFBF", "linewidth": 0.6, "boxstyle": "round,pad=0.25"},
    )

    risk_ax.set_ylim(0, 1)
    risk_ax.set_yticks([])
    risk_ax.set_xticks(x_ticks)
    risk_ax.set_xlim(0.0, max_time)
    for spine in risk_ax.spines.values():
        spine.set_visible(False)
    risk_ax.tick_params(axis="x", bottom=False, labelbottom=False, top=False, labeltop=False, length=0)

    label_x = -0.11
    risk_ax.text(label_x, 0.78, "No. at risk", transform=risk_ax.transAxes, ha="right", va="center", fontsize=8.5)
    y_positions = {"Low risk": 0.52, "High risk": 0.22}
    for group in ["Low risk", "High risk"]:
        group_df = data.loc[data["risk_group"].eq(group)].copy()
        risk_ax.text(
            label_x,
            y_positions[group],
            group,
            transform=risk_ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color=colors[group],
        )
        for tick in x_ticks:
            n_risk = int(np.sum(group_df["time_months"].to_numpy() >= tick))
            risk_ax.text(tick, y_positions[group], str(n_risk), ha="center", va="center", fontsize=8.5)

    red_journal_style.save_publication_figure(fig, out_base, dpi=dpi, formats=("png", "pdf"), bbox_inches="tight")
    plt.close(fig)

    curve_data = pd.concat(curve_rows, ignore_index=True)
    curve_data.to_csv(out_base.with_name(out_base.name + "_curve_data.csv"), index=False)

    risk_rows = []
    for group in ["High risk", "Low risk"]:
        group_df = data.loc[data["risk_group"].eq(group)].copy()
        for tick in x_ticks:
            risk_rows.append(
                {
                    "risk_group": group,
                    "time_months": float(tick),
                    "n_at_risk": int(np.sum(group_df["time_months"].to_numpy() >= tick)),
                }
            )
    pd.DataFrame(risk_rows).to_csv(out_base.with_name(out_base.name + "_risk_table.csv"), index=False)


def main() -> None:
    args = parse_args()
    patient_data = pd.read_csv(args.patient_data)
    summary = pd.read_csv(args.summary_csv)
    plot_km_from_data(
        patient_data,
        summary,
        args.out_base,
        int(args.dpi),
        float(args.display_max_months),
    )
    print(f"[saved] {args.out_base.with_suffix('.png')}")
    print(f"[saved] {args.out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
