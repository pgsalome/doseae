#!/usr/bin/env python3
"""Create UKHD-SBRT Kaplan-Meier risk-group split from locked Figure 3 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from red_journal_style import apply_red_journal_style, clean_axes, get_figure_size, save_publication_figure


DEFAULT_PREDICTIONS = Path("new_fig/figure3_recreate_actual_pred_by_panel.csv")
DEFAULT_OUTCOMES = Path("clinical_features/outcomes_controlrates.csv")
DEFAULT_ID_MAP = Path("clinical_features/test_subset_early_sbrt_tp1_ct_dose_anonymized_id_map_private.csv")
DEFAULT_OUT_DIR = Path("new_fig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the locked Figure 3 risk model to split UKHD-SBRT into high- and "
            "low-risk groups, then generate a Kaplan-Meier analysis."
        )
    )
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--outcomes-csv", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--id-map-csv", type=Path, default=DEFAULT_ID_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="DC+Clinical", help="Figure 3 model label to use.")
    parser.add_argument("--validation-cohort", default="RTOG-conv")
    parser.add_argument("--target-cohort", default="UKHD-SBRT")
    parser.add_argument(
        "--risk-threshold-method",
        choices=["validation_youden", "target_median", "target_upper_tertile", "target_upper_quartile"],
        default="validation_youden",
        help=(
            "How to split high/low risk. validation_youden is fully locked from the validation cohort; "
            "target_* methods use only the target cohort score distribution, not survival labels."
        ),
    )
    parser.add_argument("--probability-col", default=None, help="Prediction probability column; inferred if omitted.")
    parser.add_argument("--actual-col", default=None, help="Binary label column for validation threshold; inferred if omitted.")
    parser.add_argument("--panel-type", default="bootstrap_boxplot_input", help="Optional panel_type filter if present.")
    parser.add_argument("--output-stem", default="figure5_ukhd_sbrt_km_risk_groups")
    parser.add_argument("--time-col", default="time_tox")
    parser.add_argument("--event-col", default="status_tox")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score.astype(float))
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    return {
        "threshold": float(thresholds[idx]),
        "youden_j": float(youden[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1.0 - fpr[idx]),
    }


def km_curve(times: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    event_times = np.sort(np.unique(times[events == 1]))
    rows: list[dict[str, float]] = [
        {"time": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0, "n_at_risk": float(len(times)), "events": 0.0}
    ]
    survival = 1.0
    greenwood_sum = 0.0
    for t in event_times:
        n_at_risk = float(np.sum(times >= t))
        d = float(np.sum((times == t) & (events == 1)))
        if n_at_risk <= 0 or d <= 0:
            continue
        if n_at_risk - d > 0:
            greenwood_sum += d / (n_at_risk * (n_at_risk - d))
        survival *= max(0.0, 1.0 - d / n_at_risk)
        se = survival * np.sqrt(greenwood_sum) if greenwood_sum > 0 else 0.0
        rows.append(
            {
                "time": float(t),
                "survival": float(survival),
                "lower": float(np.clip(survival - 1.96 * se, 0.0, 1.0)),
                "upper": float(np.clip(survival + 1.96 * se, 0.0, 1.0)),
                "n_at_risk": n_at_risk,
                "events": d,
            }
        )
    return pd.DataFrame(rows)


def survival_at(query_times: np.ndarray, curve: pd.DataFrame) -> np.ndarray:
    values = []
    event_times = curve["time"].to_numpy(dtype=float)
    survival = curve["survival"].to_numpy(dtype=float)
    for t in np.asarray(query_times, dtype=float):
        idx = np.searchsorted(event_times, t, side="right") - 1
        values.append(float(survival[max(idx, 0)]))
    return np.asarray(values, dtype=float)


def logrank_test(data: pd.DataFrame, group_col: str, time_col: str, event_col: str) -> dict[str, float]:
    groups = sorted(data[group_col].dropna().unique().tolist())
    if len(groups) != 2:
        raise ValueError(f"Expected two risk groups, found {groups}")
    high_group = "High risk" if "High risk" in groups else groups[-1]
    event_times = np.sort(data.loc[data[event_col] == 1, time_col].dropna().unique())
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for t in event_times:
        at_risk = data[data[time_col] >= t]
        events = data[(data[time_col] == t) & (data[event_col] == 1)]
        n = float(len(at_risk))
        d = float(len(events))
        n_high = float(np.sum(at_risk[group_col] == high_group))
        d_high = float(np.sum(events[group_col] == high_group))
        if n <= 1 or d <= 0:
            continue
        observed += d_high
        expected += d * n_high / n
        variance += (n_high * (n - n_high) * d * (n - d)) / (n * n * (n - 1.0))
    chi_square = (observed - expected) ** 2 / variance if variance > 0 else np.nan
    p_value = float(chi2.sf(chi_square, 1)) if np.isfinite(chi_square) else np.nan
    return {
        "observed_high": float(observed),
        "expected_high": float(expected),
        "variance": float(variance),
        "chi_square": float(chi_square),
        "p_value": p_value,
    }


def cox_hr(data: pd.DataFrame, time_col: str, event_col: str) -> dict[str, float]:
    fit_df = data[[time_col, event_col, "high_risk_binary"]].dropna().copy()
    times = fit_df[time_col].to_numpy(dtype=float)
    events = fit_df[event_col].to_numpy(dtype=int)
    x = fit_df["high_risk_binary"].to_numpy(dtype=float)
    event_times = np.sort(np.unique(times[events == 1]))
    beta = 0.0
    information = np.nan

    # Univariate Cox proportional hazards model with Breslow handling of tied event times.
    for _ in range(100):
        score = 0.0
        information = 0.0
        for t in event_times:
            event_mask = (times == t) & (events == 1)
            risk_mask = times >= t
            d = float(np.sum(event_mask))
            if d <= 0:
                continue
            xb = beta * x[risk_mask]
            xb = np.clip(xb, -50.0, 50.0)
            weights = np.exp(xb)
            s0 = float(np.sum(weights))
            s1 = float(np.sum(weights * x[risk_mask]))
            s2 = float(np.sum(weights * x[risk_mask] * x[risk_mask]))
            if s0 <= 0:
                continue
            mean_x = s1 / s0
            var_x = max(0.0, s2 / s0 - mean_x * mean_x)
            score += float(np.sum(x[event_mask])) - d * mean_x
            information += d * var_x
        if information <= 0 or not np.isfinite(information):
            break
        step = score / information
        beta += step
        if abs(step) < 1e-8:
            break

    if not np.isfinite(information) or information <= 0:
        se = np.nan
        lower = np.nan
        upper = np.nan
        p_value = np.nan
    else:
        se = float(np.sqrt(1.0 / information))
        lower = beta - 1.96 * se
        upper = beta + 1.96 * se
        z = beta / se if se > 0 else np.nan
        p_value = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "hazard_ratio": float(np.exp(beta)),
        "hr_ci_lower": float(np.exp(lower)) if np.isfinite(lower) else np.nan,
        "hr_ci_upper": float(np.exp(upper)) if np.isfinite(upper) else np.nan,
        "cox_beta": float(beta),
        "cox_beta_se": float(se),
        "cox_p_value": p_value,
    }


def concordance_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)
    comparable = 0
    concordant = 0.0
    for i in range(len(times)):
        for j in range(len(times)):
            if times[i] < times[j] and events[i] == 1:
                comparable += 1
                if scores[i] > scores[j]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable else np.nan


def load_analysis_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = pd.read_csv(args.predictions_csv)
    if "panel_type" in predictions.columns and args.panel_type:
        predictions = predictions[predictions["panel_type"] == args.panel_type].copy()
    predictions = predictions[predictions["model"] == args.model].copy()
    if predictions.empty:
        raise ValueError(f"No rows found for model={args.model!r} in {args.predictions_csv}")

    probability_col = args.probability_col
    if probability_col is None:
        for candidate in ["predicted_probability", "pred_probability", "predicted_risk", "probability"]:
            if candidate in predictions.columns:
                probability_col = candidate
                break
    if probability_col is None or probability_col not in predictions.columns:
        raise ValueError("Could not infer prediction probability column. Pass --probability-col.")

    actual_col = args.actual_col
    if actual_col is None:
        for candidate in ["actual", "y_true", "toxicity_label_0p70"]:
            if candidate in predictions.columns:
                actual_col = candidate
                break
    if actual_col is None or actual_col not in predictions.columns:
        raise ValueError("Could not infer binary label column. Pass --actual-col.")

    validation = predictions[predictions["cohort"] == args.validation_cohort].copy()
    if validation.empty:
        raise ValueError(f"No validation rows found for cohort={args.validation_cohort!r}")
    threshold_info = youden_threshold(
        validation[actual_col].to_numpy(dtype=int),
        validation[probability_col].to_numpy(dtype=float),
    )

    target = predictions[predictions["cohort"] == args.target_cohort].copy()
    if target.empty:
        raise ValueError(f"No target rows found for cohort={args.target_cohort!r}")
    target = target.drop_duplicates(subset=["patient_key"]).copy()
    target = target.rename(columns={probability_col: "predicted_probability", actual_col: "actual"})

    outcomes = pd.read_csv(args.outcomes_csv)
    id_map = pd.read_csv(args.id_map_csv)
    outcomes["sub_key"] = outcomes["sub"].astype(str).str.strip()
    id_map["orig_key"] = id_map["original_patient_id"].astype(str).str.strip()

    data = target.merge(
        id_map[["anon_id", "original_patient_id", "orig_key"]],
        left_on="patient_key",
        right_on="anon_id",
        how="left",
    ).merge(outcomes, left_on="orig_key", right_on="sub_key", how="left")

    required = ["predicted_probability", args.time_col, args.event_col]
    missing = data[required].isna().any(axis=1)
    if missing.any():
        missing_ids = data.loc[missing, "patient_key"].tolist()
        print(f"[warn] Dropping {len(missing_ids)} patients with missing prediction/time/event: {missing_ids}")
        data = data.loc[~missing].copy()

    if args.risk_threshold_method == "validation_youden":
        threshold = threshold_info["threshold"]
    elif args.risk_threshold_method == "target_median":
        threshold = float(data["predicted_probability"].median())
    elif args.risk_threshold_method == "target_upper_tertile":
        threshold = float(data["predicted_probability"].quantile(2.0 / 3.0))
    elif args.risk_threshold_method == "target_upper_quartile":
        threshold = float(data["predicted_probability"].quantile(0.75))
    else:
        raise ValueError(args.risk_threshold_method)
    threshold_info["risk_threshold_method"] = args.risk_threshold_method
    data["risk_threshold_source"] = args.validation_cohort
    data["risk_threshold"] = threshold
    data["risk_group"] = np.where(data["predicted_probability"].astype(float) >= threshold, "High risk", "Low risk")
    data["high_risk_binary"] = (data["risk_group"] == "High risk").astype(int)
    data["event"] = data[args.event_col].astype(int)
    data["time_months"] = data[args.time_col].astype(float)
    return data, threshold_info


def plot_km(
    data: pd.DataFrame,
    threshold_info: dict[str, float],
    logrank: dict[str, float],
    hr: dict[str, float],
    output_base: Path,
    dpi: int,
) -> None:
    apply_red_journal_style()
    fig = plt.figure(figsize=get_figure_size("km"))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.2, 1.15], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    risk_ax = fig.add_subplot(gs[1], sharex=ax)

    colors = {"High risk": "#D55E00", "Low risk": "#0072B2"}
    styles = {"High risk": "-", "Low risk": "--"}
    marker_styles = {"High risk": "+", "Low risk": "+"}
    curve_rows = []

    max_time = float(np.ceil(data["time_months"].max() / 12.0) * 12.0)
    x_ticks = np.arange(0.0, max_time + 0.01, 12.0)

    for group in ["Low risk", "High risk"]:
        group_df = data[data["risk_group"] == group].copy()
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
            label=label,
            linewidth=2.2,
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
        censored = group_df[group_df["event"] == 0]
        if not censored.empty:
            censor_y = survival_at(censored["time_months"].to_numpy(), curve)
            ax.scatter(
                censored["time_months"],
                censor_y,
                marker=marker_styles[group],
                s=28,
                color=colors[group],
                linewidths=1.0,
                alpha=0.9,
                zorder=4,
            )

    clean_axes(ax, grid=True)
    ax.set_ylabel("RILI-free survival probability")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(0.0, max_time)
    ax.set_title("UKHD-SBRT risk stratification", fontweight="normal", pad=5)
    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="x", labelbottom=False, bottom=False, length=0)
    annotation = (
        f"HR={hr['hazard_ratio']:.2f}\n"
        f"log-rank p={logrank['p_value']:.3f}"
    )
    ax.text(
        0.98,
        0.92,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#BFBFBF", "linewidth": 0.6, "boxstyle": "round,pad=0.25"},
    )

    risk_ax.set_ylim(0, 1)
    risk_ax.set_yticks([])
    risk_ax.set_xlabel("Time from treatment, months")
    risk_ax.set_xticks(x_ticks)
    risk_ax.set_xlim(0.0, max_time)
    for side, spine in risk_ax.spines.items():
        spine.set_visible(side == "bottom")
    risk_ax.spines["bottom"].set_linewidth(0.8)
    risk_ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, length=3.0, pad=2)
    label_x = -0.11
    risk_ax.text(label_x, 0.78, "No. at risk", transform=risk_ax.transAxes, ha="right", va="center", fontsize=8.5)
    y_positions = {"Low risk": 0.55, "High risk": 0.26}
    for group in ["Low risk", "High risk"]:
        group_df = data[data["risk_group"] == group].copy()
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

    save_publication_figure(fig, output_base, dpi=dpi, formats=("png", "pdf"), bbox_inches="tight")
    plt.close(fig)

    curve_data = pd.concat(curve_rows, ignore_index=True)
    curve_data.to_csv(output_base.with_name(output_base.name + "_curve_data.csv"), index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.out_dir / args.output_stem

    data, threshold_info = load_analysis_data(args)
    logrank = logrank_test(data, "risk_group", "time_months", "event")
    hr = cox_hr(data, "time_months", "event")

    group_summary = (
        data.groupby("risk_group", observed=True)
        .agg(
            n=("patient_key", "size"),
            events=("event", "sum"),
            median_predicted_risk=("predicted_probability", "median"),
            min_predicted_risk=("predicted_probability", "min"),
            max_predicted_risk=("predicted_probability", "max"),
            median_time_months=("time_months", "median"),
        )
        .reset_index()
    )
    group_summary["validation_threshold"] = threshold_info["threshold"]
    group_summary["applied_risk_threshold"] = float(data["risk_threshold"].iloc[0])
    group_summary["risk_threshold_method"] = threshold_info["risk_threshold_method"]
    group_summary["survival_c_index"] = concordance_index(
        data["time_months"].to_numpy(dtype=float),
        data["event"].to_numpy(dtype=int),
        data["predicted_probability"].to_numpy(dtype=float),
    )
    for key, value in {**threshold_info, **logrank, **hr}.items():
        group_summary[key] = value

    patient_out = output_base.with_name(output_base.name + "_patient_data.csv")
    summary_out = output_base.with_name(output_base.name + "_summary.csv")
    risk_table_out = output_base.with_name(output_base.name + "_risk_table.csv")

    data[
        [
            "patient_key",
            "original_patient_id",
            "actual",
            "predicted_probability",
            "risk_threshold",
            "risk_group",
            "event",
            "time_months",
            "status_pnn",
            "time_pnn",
            "status_fib",
            "time_fib",
            "status_fibpnn",
            "time_fibpnn",
        ]
    ].to_csv(patient_out, index=False)
    group_summary.to_csv(summary_out, index=False)

    max_time = float(np.ceil(data["time_months"].max() / 12.0) * 12.0)
    risk_rows = []
    for group, group_df in data.groupby("risk_group", observed=True):
        for tick in np.arange(0.0, max_time + 0.01, 12.0):
            risk_rows.append(
                {"risk_group": group, "time_months": tick, "n_at_risk": int(np.sum(group_df["time_months"] >= tick))}
            )
    pd.DataFrame(risk_rows).to_csv(risk_table_out, index=False)

    plot_km(data, threshold_info, logrank, hr, output_base, dpi=args.dpi)

    print(f"[saved] {output_base.with_suffix('.png')}")
    print(f"[saved] {output_base.with_suffix('.pdf')}")
    print(f"[saved] {patient_out}")
    print(f"[saved] {summary_out}")
    print(f"[saved] {risk_table_out}")
    print(
        "[summary] "
        f"threshold={float(data['risk_threshold'].iloc[0]):.3f}; "
        f"method={threshold_info['risk_threshold_method']}; "
        f"HR={hr['hazard_ratio']:.2f} ({hr['hr_ci_lower']:.2f}-{hr['hr_ci_upper']:.2f}); "
        f"log-rank p={logrank['p_value']:.3f}"
    )


if __name__ == "__main__":
    main()
