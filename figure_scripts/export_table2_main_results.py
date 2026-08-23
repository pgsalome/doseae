#!/usr/bin/env python3
"""Export Table 2 discrimination, operating metrics, and DeLong comparisons.

This script is cache-only. It uses the current Figure 3 patient-level
prediction CSV for the learned models and the Figure 5 prediction CSV for the
DVH-NTCP baseline. This keeps Table 2 numerically tied to the manuscript
figures rather than silently retraining a different model recipe.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "new_fig"
DEFAULT_FIG3_PREDICTIONS = DEFAULT_OUT_DIR / "figure3_recreate_actual_pred_by_panel.csv"
DEFAULT_FIG5_PREDICTIONS = DEFAULT_OUT_DIR / "figure5_panelA_predictions.csv"
DEFAULT_COMBINED_PREDICTIONS = (
    REPO_ROOT / "oliver_paper" / "tables" / "table2_predictions_used.csv"
)

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026

DOMAIN_ORDER = ["rtog-conv", "rtog-imrt", "ukhd-sbrt"]
COHORT_LABELS = {
    "rtog-conv": "RTOG-conv",
    "rtog-imrt": "RTOG-IMRT",
    "ukhd-sbrt": "UKHD-SBRT",
}

MODEL_ORDER = [
    "DVH-NTCP",
    "Clinical-only",
    "Dose",
    "Dose+CT",
    "Dose+CT+Clinical",
]
FIG3_MODEL_MAP = {
    "Clinical": "Clinical-only",
    "D": "Dose",
    "DC": "Dose+CT",
    "DC+Clinical": "Dose+CT+Clinical",
}

DELONG_COMPARISONS = [
    ("Dose+CT+Clinical", "DVH-NTCP", "Dose+CT+Clinical vs. DVH-NTCP"),
    ("Dose+CT+Clinical", "Clinical-only", "Dose+CT+Clinical vs. Clinical-only"),
    ("Dose+CT+Clinical", "Dose", "Dose+CT+Clinical vs. Dose"),
    ("Dose+CT", "Dose", "Dose+CT vs. Dose"),
    ("Dose", "DVH-NTCP", "Dose vs. DVH-NTCP"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig3-predictions", type=Path, default=DEFAULT_FIG3_PREDICTIONS)
    parser.add_argument("--fig5-predictions", type=Path, default=DEFAULT_FIG5_PREDICTIONS)
    parser.add_argument(
        "--combined-predictions",
        type=Path,
        default=None,
        help=(
            "Optional self-contained prediction table containing all five models. "
            "When supplied, the historical Figure 3/Figure 5 inputs are not read."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def normalize_patient_key(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.upper().startswith("ANON_"):
        return text.upper()
    if any(ch.isalpha() for ch in text):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return str(int(digits))
    return text


def stable_seed(base_seed: int, *parts: object) -> int:
    acc = int(base_seed)
    for char in "|".join(str(part) for part in parts):
        acc = (acc * 131 + ord(char)) % 2_147_483_647
    return int(acc)


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def youden_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return float("nan")
    j = tpr[valid] - fpr[valid]
    thresholds_valid = thresholds[valid]
    return float(thresholds_valid[int(np.nanargmax(j))])


def classification_metrics_at_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "Sens": float(sens),
        "Spec": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        idx = rng.integers(0, y_true.size, size=y_true.size)
    else:
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        rng.shuffle(idx)
    return idx


def percentile_ci(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def bootstrap_metric_cis(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {
        "AUROC": [],
        "Sens": [],
        "Spec": [],
        "PPV": [],
        "NPV": [],
        "Brier": [],
    }
    for _ in range(int(n_boot)):
        idx = stratified_bootstrap_indices(y, rng)
        yb = y[idx]
        pb = p[idx]
        if np.unique(yb).size == 2:
            values["AUROC"].append(float(roc_auc_score(yb, pb)))
        cm = classification_metrics_at_threshold(yb, pb, threshold)
        for key in ("Sens", "Spec", "PPV", "NPV"):
            if np.isfinite(cm[key]):
                values[key].append(float(np.clip(cm[key], 0.0, 1.0)))
        values["Brier"].append(float(brier_score_loss(yb, np.clip(pb, 0.0, 1.0))))
    return {metric: percentile_ci(vals) for metric, vals in values.items()}


def compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    if np.ndim(delong_cov) == 0:
        delong_cov = np.asarray([[float(delong_cov)]])
    return aucs, delong_cov


def delong_auc_covariance(y_true: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    preds = np.asarray(predictions, dtype=float)
    if np.unique(y).size < 2:
        raise ValueError("DeLong requires both classes.")
    order = np.argsort(-y)
    label_1_count = int(np.sum(y))
    return fast_delong(preds[:, order], label_1_count)


def normal_two_sided_pvalue(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def load_current_figure3_model_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_key": str, "patient_id": str})
    if "panel_type" in df.columns:
        df = df.loc[df["panel_type"].eq("bootstrap_boxplot_input")].copy()
        df = df.rename(
            columns={
                "actual": "y_true",
                "predicted_probability": "pred_probability",
            }
        )
    keep = df.loc[df["domain"].isin(DOMAIN_ORDER) & df["model"].isin(FIG3_MODEL_MAP)].copy()
    keep["model"] = keep["model"].map(FIG3_MODEL_MAP)
    keep["patient_key_norm"] = keep["patient_key"].map(normalize_patient_key)
    keep["y_true"] = pd.to_numeric(keep["y_true"], errors="coerce")
    keep["pred_probability"] = pd.to_numeric(keep["pred_probability"], errors="coerce")
    keep = keep.dropna(subset=["y_true", "pred_probability"]).copy()
    keep["y_true"] = keep["y_true"].astype(int)
    keep["pred_probability"] = keep["pred_probability"].astype(float).clip(0.0, 1.0)
    keep["source"] = str(path)
    return keep[
        [
            "model",
            "domain",
            "cohort",
            "patient_key",
            "patient_key_norm",
            "y_true",
            "pred_probability",
            "feature_set",
            "patch_model_key",
            "patch_model_label",
            "patch_aggregation",
            "clinical_block",
            "locked_classifier",
            "locked_sampler",
            "locked_pca_components",
            "locked_fit_seed",
            "source",
        ]
    ]


def load_dvh_ntcp_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_key": str, "patient_key_norm": str})
    keep = df.loc[df["domain"].isin(DOMAIN_ORDER) & df["model_variant"].eq("DVH-NTCP")].copy()
    keep["model"] = "DVH-NTCP"
    keep["pred_probability"] = pd.to_numeric(keep["predicted_probability"], errors="coerce")
    keep["y_true"] = pd.to_numeric(keep["y_true"], errors="coerce")
    keep = keep.dropna(subset=["y_true", "pred_probability"]).copy()
    keep["y_true"] = keep["y_true"].astype(int)
    keep["pred_probability"] = keep["pred_probability"].astype(float).clip(0.0, 1.0)
    keep["patient_key_norm"] = keep["patient_key_norm"].map(normalize_patient_key)
    keep["feature_set"] = "MLD+V5+V13+V20+V25"
    keep["patch_model_key"] = np.nan
    keep["patch_model_label"] = np.nan
    keep["patch_aggregation"] = np.nan
    keep["clinical_block"] = "None"
    keep["locked_classifier"] = "LogisticRegression"
    keep["locked_sampler"] = "None"
    keep["locked_pca_components"] = 0
    keep["locked_fit_seed"] = np.nan
    keep["source"] = str(path)
    return keep[
        [
            "model",
            "domain",
            "cohort",
            "patient_key",
            "patient_key_norm",
            "y_true",
            "pred_probability",
            "feature_set",
            "patch_model_key",
            "patch_model_label",
            "patch_aggregation",
            "clinical_block",
            "locked_classifier",
            "locked_sampler",
            "locked_pca_components",
            "locked_fit_seed",
            "source",
        ]
    ]


def load_combined_prediction_cache(path: Path) -> pd.DataFrame:
    """Load the manuscript prediction cache without refitting any model."""
    df = pd.read_csv(path, dtype={"patient_key": str, "patient_key_norm": str})
    required = {"model", "domain", "patient_key", "y_true", "pred_probability"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Combined prediction cache is missing columns: {missing}")

    keep = df.loc[df["domain"].isin(DOMAIN_ORDER) & df["model"].isin(MODEL_ORDER)].copy()
    keep["patient_key_norm"] = keep.get("patient_key_norm", keep["patient_key"]).map(
        normalize_patient_key
    )
    keep["y_true"] = pd.to_numeric(keep["y_true"], errors="coerce")
    keep["pred_probability"] = pd.to_numeric(keep["pred_probability"], errors="coerce")
    keep = keep.dropna(subset=["y_true", "pred_probability"]).copy()
    keep["y_true"] = keep["y_true"].astype(int)
    keep["pred_probability"] = keep["pred_probability"].astype(float).clip(0.0, 1.0)
    keep["source"] = str(path)

    metadata_defaults = {
        "cohort": keep["domain"].map(COHORT_LABELS),
        "feature_set": np.nan,
        "patch_model_key": np.nan,
        "patch_model_label": np.nan,
        "patch_aggregation": np.nan,
        "clinical_block": np.nan,
        "locked_classifier": np.nan,
        "locked_sampler": np.nan,
        "locked_pca_components": np.nan,
        "locked_fit_seed": np.nan,
    }
    for column, default in metadata_defaults.items():
        if column not in keep.columns:
            keep[column] = default

    return keep[
        [
            "model",
            "domain",
            "cohort",
            "patient_key",
            "patient_key_norm",
            "y_true",
            "pred_probability",
            "feature_set",
            "patch_model_key",
            "patch_model_label",
            "patch_aggregation",
            "clinical_block",
            "locked_classifier",
            "locked_sampler",
            "locked_pca_components",
            "locked_fit_seed",
            "source",
        ]
    ]


def validate_prediction_table(pred: pd.DataFrame) -> None:
    missing = []
    for domain in DOMAIN_ORDER:
        for model in MODEL_ORDER:
            d = pred.loc[pred["domain"].eq(domain) & pred["model"].eq(model)]
            if d.empty:
                missing.append((domain, model))
            dupes = d.duplicated(["patient_key_norm"]).sum()
            if dupes:
                raise ValueError(f"Duplicate predictions for {domain} {model}: {dupes}")
    if missing:
        raise ValueError(f"Missing model/cohort predictions: {missing}")

    for domain in DOMAIN_ORDER:
        wide_y = pred.loc[pred["domain"].eq(domain)].pivot_table(
            index="patient_key_norm",
            columns="model",
            values="y_true",
            aggfunc="first",
        )
        if wide_y.nunique(axis=1).gt(1).any():
            bad = wide_y.loc[wide_y.nunique(axis=1).gt(1)].head()
            raise ValueError(f"Label mismatch across models in {domain}:\n{bad}")


def summarize_model_cohort(
    pred: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    concerns: List[str] = []
    for domain in DOMAIN_ORDER:
        for model in MODEL_ORDER:
            d = pred.loc[pred["domain"].eq(domain) & pred["model"].eq(model)].copy()
            y = d["y_true"].astype(int).to_numpy()
            p = d["pred_probability"].astype(float).to_numpy()
            threshold = youden_threshold(y, p)
            metrics = classification_metrics_at_threshold(y, p, threshold)
            auc = safe_auc(y, p)
            brier = float(brier_score_loss(y, p))
            ci = bootstrap_metric_cis(
                y,
                p,
                threshold,
                n_boot=n_boot,
                seed=stable_seed(seed, domain, model, "table2"),
            )
            pred_min = float(np.nanmin(p))
            pred_max = float(np.nanmax(p))
            threshold_in_range = bool(pred_min <= threshold <= pred_max)
            if not threshold_in_range:
                concerns.append(
                    f"{COHORT_LABELS[domain]} {model}: Youden threshold {threshold:.6f} outside "
                    f"observed prediction range [{pred_min:.6f}, {pred_max:.6f}]."
                )
            auc_low, auc_high = ci["AUROC"]
            if np.isfinite(auc_low) and np.isfinite(auc_high) and auc_low <= 0.5 <= auc_high:
                concerns.append(
                    f"{COHORT_LABELS[domain]} {model}: AUROC CI straddles 0.5 "
                    f"({auc_low:.3f}-{auc_high:.3f})."
                )
            meta = d.iloc[0].to_dict()
            row: Dict[str, object] = {
                "model": model,
                "cohort": COHORT_LABELS[domain],
                "domain": domain,
                "n": int(len(d)),
                "n_events": int(y.sum()),
                "AUROC_point": auc,
                "AUROC_lower": auc_low,
                "AUROC_upper": auc_high,
                "Sens_point": metrics["Sens"],
                "Sens_lower": ci["Sens"][0],
                "Sens_upper": ci["Sens"][1],
                "Spec_point": metrics["Spec"],
                "Spec_lower": ci["Spec"][0],
                "Spec_upper": ci["Spec"][1],
                "PPV_point": metrics["PPV"],
                "PPV_lower": ci["PPV"][0],
                "PPV_upper": ci["PPV"][1],
                "NPV_point": metrics["NPV"],
                "NPV_lower": ci["NPV"][0],
                "NPV_upper": ci["NPV"][1],
                "Brier_point": brier,
                "Brier_lower": ci["Brier"][0],
                "Brier_upper": ci["Brier"][1],
                "Youden_threshold": threshold,
                "pred_probability_min": pred_min,
                "pred_probability_max": pred_max,
                "threshold_in_observed_range": threshold_in_range,
                "feature_set": meta.get("feature_set"),
                "patch_model_key": meta.get("patch_model_key"),
                "patch_model_label": meta.get("patch_model_label"),
                "patch_aggregation": meta.get("patch_aggregation"),
                "clinical_block": meta.get("clinical_block"),
                "locked_classifier": meta.get("locked_classifier"),
                "locked_sampler": meta.get("locked_sampler"),
                "locked_pca_components": meta.get("locked_pca_components"),
                "locked_fit_seed": meta.get("locked_fit_seed"),
                "prediction_source": meta.get("source"),
            }
            rows.append(row)
    return pd.DataFrame(rows), concerns


def build_delong_table(pred: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    concerns: List[str] = []
    for domain in DOMAIN_ORDER:
        wide = pred.loc[pred["domain"].eq(domain)].pivot_table(
            index=["patient_key_norm", "y_true"],
            columns="model",
            values="pred_probability",
            aggfunc="first",
        )
        wide = wide.reset_index()
        for model_a, model_b, comparison in DELONG_COMPARISONS:
            if model_a not in wide.columns or model_b not in wide.columns:
                continue
            d = wide.dropna(subset=[model_a, model_b]).copy()
            y = d["y_true"].astype(int).to_numpy()
            preds = d[[model_a, model_b]].to_numpy(dtype=float).T
            aucs, cov = delong_auc_covariance(y, preds)
            diff = float(aucs[0] - aucs[1])
            contrast = np.array([1.0, -1.0])
            variance = float(contrast @ cov @ contrast.T)
            if variance > 0:
                se = math.sqrt(variance)
                z = diff / se
                p_value = normal_two_sided_pvalue(z)
                ci_low = diff - 1.96 * se
                ci_high = diff + 1.96 * se
            else:
                se = float("nan")
                z = float("nan")
                p_value = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")
            if np.isfinite(p_value) and (p_value == 0.0 or p_value == 1.0):
                concerns.append(
                    f"{COHORT_LABELS[domain]} {comparison}: DeLong p-value is exactly {p_value}."
                )
            rows.append(
                {
                    "cohort": COHORT_LABELS[domain],
                    "domain": domain,
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "auroc_model_a": float(aucs[0]),
                    "auroc_model_b": float(aucs[1]),
                    "auroc_diff": diff,
                    "ci_lower_diff": ci_low,
                    "ci_upper_diff": ci_high,
                    "delong_z": z,
                    "delong_p": p_value,
                    "n": int(len(d)),
                    "n_events": int(y.sum()),
                }
            )
    return pd.DataFrame(rows), concerns


def check_expected_values(summary: pd.DataFrame, concerns: List[str]) -> None:
    expected_locked = {
        "rtog-conv": 0.709135,
        "rtog-imrt": 0.728632,
        "ukhd-sbrt": 0.812016,
    }
    expected_dvh = {
        "rtog-conv": 0.435096,
        "rtog-imrt": 0.432692,
        "ukhd-sbrt": 0.517442,
    }
    for domain, expected in expected_locked.items():
        got = float(
            summary.loc[
                summary["domain"].eq(domain) & summary["model"].eq("Dose+CT+Clinical"),
                "AUROC_point",
            ].iloc[0]
        )
        if not np.isclose(got, expected, atol=5e-6):
            concerns.append(
                f"Dose+CT+Clinical AUROC for {COHORT_LABELS[domain]} is {got:.6f}, "
                f"expected Figure 3 value ~{expected:.6f}."
            )
    for domain, expected in expected_dvh.items():
        got = float(
            summary.loc[
                summary["domain"].eq(domain) & summary["model"].eq("DVH-NTCP"),
                "AUROC_point",
            ].iloc[0]
        )
        if not np.isclose(got, expected, atol=5e-6):
            concerns.append(
                f"DVH-NTCP AUROC for {COHORT_LABELS[domain]} is {got:.6f}, "
                f"expected Figure 5 value ~{expected:.6f}."
            )


def write_summary(
    path: Path,
    table2: pd.DataFrame,
    delong: pd.DataFrame,
    concerns: Sequence[str],
    *,
    n_boot: int,
    seed: int,
    prediction_source: str,
) -> None:
    lines: List[str] = []
    lines.append("Table 2 quantitative result summary")
    lines.append("=" * 42)
    lines.append("")
    lines.append(f"Bootstrap: patient-level stratified resampling, n={n_boot}, seed={seed}.")
    lines.append(f"Prediction input: {prediction_source}.")
    lines.append("")
    lines.append("Important implementation note:")
    lines.append(
        "The current Figure 3 locked learned-model cache uses LR-L2 + SMOTE + PCA4 and "
        "clinical block Age+Smoking. This differs from the requested LR-EN + SMOTE+TL + "
        "PCA4 and age+smoking+COPD. The cache was used to preserve exact Figure 3 AUROC matching."
    )
    lines.append(
        "COPD is not available for the RTOG cohorts in the local clinical source used for Table 1, "
        "so the current manuscript clinical block is Age+Smoking."
    )
    lines.append("")
    lines.append("AUROC by model and cohort:")
    for model in MODEL_ORDER:
        lines.append(f"- {model}:")
        for domain in DOMAIN_ORDER:
            row = table2.loc[table2["model"].eq(model) & table2["domain"].eq(domain)].iloc[0]
            lines.append(
                f"  {row['cohort']}: {row['AUROC_point']:.3f} "
                f"({row['AUROC_lower']:.3f}-{row['AUROC_upper']:.3f}), "
                f"n={int(row['n'])}, events={int(row['n_events'])}"
            )
    lines.append("")
    lines.append("Primary DeLong comparisons:")
    for _, row in delong.iterrows():
        lines.append(
            f"- {row['cohort']} {row['comparison']}: diff={row['auroc_diff']:.3f}, "
            f"95% CI {row['ci_lower_diff']:.3f}-{row['ci_upper_diff']:.3f}, p={row['delong_p']:.4g}"
        )
    lines.append("")
    lines.append("Flags / concerns:")
    if concerns:
        for item in concerns:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.combined_predictions is not None:
        pred = load_combined_prediction_cache(args.combined_predictions)
        prediction_source = str(args.combined_predictions)
    else:
        fig3_pred = load_current_figure3_model_predictions(args.fig3_predictions)
        dvh_pred = load_dvh_ntcp_predictions(args.fig5_predictions)
        pred = pd.concat([dvh_pred, fig3_pred], ignore_index=True)
        prediction_source = f"{args.fig3_predictions} and {args.fig5_predictions}"
    validate_prediction_table(pred)

    table2, concerns = summarize_model_cohort(
        pred,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    delong, delong_concerns = build_delong_table(pred)
    concerns.extend(delong_concerns)
    check_expected_values(table2, concerns)

    table2["model"] = pd.Categorical(table2["model"], categories=MODEL_ORDER, ordered=True)
    table2["domain"] = pd.Categorical(table2["domain"], categories=DOMAIN_ORDER, ordered=True)
    table2 = table2.sort_values(["domain", "model"]).reset_index(drop=True)
    table2["model"] = table2["model"].astype(str)
    table2["domain"] = table2["domain"].astype(str)

    pred_out = pred.copy()
    pred_out["model"] = pd.Categorical(pred_out["model"], categories=MODEL_ORDER, ordered=True)
    pred_out["domain"] = pd.Categorical(pred_out["domain"], categories=DOMAIN_ORDER, ordered=True)
    pred_out = pred_out.sort_values(["domain", "model", "patient_key_norm"]).reset_index(drop=True)
    pred_out["model"] = pred_out["model"].astype(str)
    pred_out["domain"] = pred_out["domain"].astype(str)

    table2_path = args.out_dir / "table2_main_data.csv"
    delong_path = args.out_dir / "table2_delong_pvalues.csv"
    summary_path = args.out_dir / "table2_summary.txt"
    pred_path = args.out_dir / "table2_predictions_used.csv"

    table2.to_csv(table2_path, index=False)
    delong.to_csv(delong_path, index=False)
    pred_out.to_csv(pred_path, index=False)
    write_summary(
        summary_path,
        table2,
        delong,
        concerns,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        prediction_source=prediction_source,
    )

    print(f"[saved] {table2_path}")
    print(f"[saved] {delong_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {pred_path}")
    print("")
    print("Headline AUROC values:")
    pivot = table2.pivot(index="model", columns="cohort", values="AUROC_point").reindex(MODEL_ORDER)
    print(pivot.round(3).to_string())
    if concerns:
        print("")
        print("Flags:")
        for item in concerns:
            print(f"- {item}")


if __name__ == "__main__":
    main()
