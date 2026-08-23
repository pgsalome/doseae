#!/usr/bin/env python3
"""Create DoseAE reconstruction performance tables for MSE and gamma pass rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

COHORT_ORDER = [
    "rtog-imrt",
    "rtog-conv",
    "ukhd-sbrt",
    "ukhd-other-train",
]

COHORT_LABELS = {
    "rtog-imrt": "RTOG-IMRT",
    "rtog-conv": "RTOG-conv",
    "ukhd-sbrt": "UKHD-SBRT",
    "ukhd-other-train": "UKHD-train",
}

METRIC_SPECS = [
    {
        "metric": "test_mse",
        "metric_label": "Test MSE",
        "column": "test_mse",
        "criterion": "",
    },
    {
        "metric": "gamma_pass_rate_mixed",
        "metric_label": "Gamma pass rate (%)",
        "column": "gamma_pass_rate_mixed",
        "criterion": "3%/3mm for Conv/ResNet/U-Net; 0.5%/0.5mm for RU-Net variants",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/paper/cohort_figure2/all_cohorts_per_patient_metrics.csv"),
        help="Per-patient Figure 2 metric cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper/tables"),
        help="Directory for generated table outputs.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def bootstrap_mean_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_values(values: pd.Series, *, n_bootstrap: int, rng: np.random.Generator) -> dict[str, float | int | str]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "mean_ci_lower": np.nan,
            "mean_ci_upper": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p2p5": np.nan,
            "p97p5": np.nan,
            "range": "",
            "empirical_95_range": "",
            "mean_95ci": "",
        }

    ci_lower, ci_upper = bootstrap_mean_ci(arr, n_bootstrap=n_bootstrap, rng=rng)
    row = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "mean_ci_lower": ci_lower,
        "mean_ci_upper": ci_upper,
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p2p5": float(np.percentile(arr, 2.5)),
        "p97p5": float(np.percentile(arr, 97.5)),
    }
    row["range"] = f"{row['min']:.6g}-{row['max']:.6g}"
    row["empirical_95_range"] = f"{row['p2p5']:.6g}-{row['p97p5']:.6g}"
    row["mean_95ci"] = f"{row['mean']:.6g} ({row['mean_ci_lower']:.6g}-{row['mean_ci_upper']:.6g})"
    return row


def make_table(df: pd.DataFrame, *, group_cols: list[str], n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for keys, group_df in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for spec in METRIC_SPECS:
            summary = summarize_values(group_df[spec["column"]], n_bootstrap=n_bootstrap, rng=rng)
            rows.append(
                {
                    **base,
                    "metric": spec["metric"],
                    "metric_label": spec["metric_label"],
                    "criterion": spec["criterion"],
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    preferred_cols = [
        *group_cols,
        "metric",
        "metric_label",
        "criterion",
        "n",
        "mean",
        "mean_ci_lower",
        "mean_ci_upper",
        "mean_95ci",
        "median",
        "q1",
        "q3",
        "min",
        "max",
        "range",
        "p2p5",
        "p97p5",
        "empirical_95_range",
    ]
    return out.loc[:, preferred_cols]


def write_markdown(table: pd.DataFrame, path: Path, *, group_cols: list[str]) -> None:
    display = table.copy()
    keep = [*group_cols, "metric_label", "criterion", "n", "mean_95ci", "median", "range", "empirical_95_range"]
    display = display.loc[:, keep]
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df[df["model_key"].isin(MODEL_ORDER)].copy()
    df = df[df["cohort_key"].isin(COHORT_ORDER)].copy()
    df["model_label"] = df["model_key"].map(MODEL_LABELS)
    df["cohort_label"] = df["cohort_key"].map(COHORT_LABELS)
    df["model_order"] = pd.Categorical(df["model_key"], categories=MODEL_ORDER, ordered=True)
    df["cohort_order"] = pd.Categorical(df["cohort_key"], categories=COHORT_ORDER, ordered=True)
    df = df.sort_values(["model_order", "cohort_order", "patient_id"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_model = make_table(
        df,
        group_cols=["model_key", "model_label"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    by_model_cohort = make_table(
        df,
        group_cols=["cohort_key", "cohort_label", "model_key", "model_label"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )

    by_model_csv = args.output_dir / "table_doseae_mse_gamma_by_model.csv"
    by_model_cohort_csv = args.output_dir / "table_doseae_mse_gamma_by_model_cohort.csv"
    by_model_md = args.output_dir / "table_doseae_mse_gamma_by_model.md"
    by_model_cohort_md = args.output_dir / "table_doseae_mse_gamma_by_model_cohort.md"

    by_model.to_csv(by_model_csv, index=False)
    by_model_cohort.to_csv(by_model_cohort_csv, index=False)
    write_markdown(by_model, by_model_md, group_cols=["model_label"])
    write_markdown(by_model_cohort, by_model_cohort_md, group_cols=["cohort_label", "model_label"])

    print(f"Saved {by_model_csv}")
    print(f"Saved {by_model_cohort_csv}")
    print(f"Saved {by_model_md}")
    print(f"Saved {by_model_cohort_md}")


if __name__ == "__main__":
    main()
