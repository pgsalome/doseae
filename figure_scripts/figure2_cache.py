#!/usr/bin/env python3
"""Build and cache all per-patient Figure 2 metrics across cohorts.

This script computes or reuses the patient-level patch-evaluation metrics needed
for Figure 2-style analyses:
- test_mse
- ptv_ipsi_lung_shell_gradient_error
- gamma_pass_rate_3pct_3mm
- Dmean_rel_diff
- Dmean_rel_diff_ipsi
- Dmean_rel_diff_contra

Outputs are written per cohort plus one combined CSV for downstream plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure2 import (  # noqa: E402
    MODEL_SPECS,
    SHELL_GRADIENT_METRIC_VERSION,
    load_or_compute_metrics,
    load_test_patient_ids,
    normalize_patient_id,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper" / "cohort_figure2"
RTOG_IMRT_SPLITS = Path("/data/pgsal/doseae/test_cohorts/rtog-imrt/splits.json")
RTOG_CONV_SPLITS = Path("/data/pgsal/doseae/test_cohorts/rtog-conv/splits.json")
RTOG_OTHER_ROOT = Path("/data/pgsal/doseae/test_cohorts/rtog-other")
RTOG_OTHER_SPLITS = RTOG_OTHER_ROOT / "splits.json"
RTOG_OTHER_PATIENTS_CSV = RTOG_OTHER_ROOT / "patients.csv"
RTOG_OTHER_MAPPING = REPO_ROOT / "data" / "ct_dose_file_mapping.json"
RTOG_OTHER_CACHE_ROOT = Path("/data/pgsal/NSCLC-Cetuximab_AE_cache")
RTOG_OTHER_SOURCE_CACHE_ROOT = Path("/data/pgsal/NSCLC-Cetuximab_AE_cache")
LEGACY_PATCH_METRICS_CACHE_DIR = REPO_ROOT / "outputs" / "paper" / "figure2_patch_per_patient_metrics_cache"

COHORT_SPECS: List[Dict[str, object]] = [
    {
        "key": "rtog-imrt",
        "label": "RTOG-IMRT",
        "splits": RTOG_IMRT_SPLITS,
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/rtog-imrt/cache"),
    },
    {
        "key": "rtog-conv",
        "label": "RTOG-conv",
        "splits": RTOG_CONV_SPLITS,
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/rtog-conv/cache"),
    },
    {
        "key": "rtog-other",
        "label": "RTOG-other",
        "splits": RTOG_OTHER_SPLITS,
        "data_dir": RTOG_OTHER_CACHE_ROOT,
    },
    {
        "key": "ukhd-sbrt",
        "label": "UKHD-SBRT",
        "splits": Path("/data/pgsal/doseae/test_cohorts/ukhd-imrt/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/ukhd-imrt/cache"),
    },
    {
        "key": "ukhd-other-train",
        "label": "UKHD-other (train)",
        "splits": Path("/data/pgsal/doseae/test_cohorts/ukhd-other-train/splits.json"),
        "data_dir": Path("/data/pgsal/doseae/test_cohorts_cache/ukhd-other-train/cache"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-patient Figure 2 caches for all cohorts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cached per-cohort and combined metric CSVs.",
    )
    parser.add_argument(
        "--cohort-key",
        action="append",
        default=[],
        help="Optional cohort key to process. Repeat to include multiple cohorts.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Require existing cohort cache CSVs; do not compute missing values.",
    )
    parser.add_argument(
        "--cohort-major",
        action="store_true",
        help="Process cohorts one-by-one (legacy order). Default is model-major.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip writing all_cohorts aggregate CSVs (useful for parallel per-cohort runs).",
    )
    return parser.parse_args()


def _load_test_entries(splits_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(splits_path.read_text())
    test_entries = payload.get("test", [])
    if not isinstance(test_entries, list):
        raise ValueError(f"Invalid test split format in {splits_path}")
    return [entry for entry in test_entries if isinstance(entry, dict) and entry.get("patient_id")]


def _load_exclusion_keys(paths: List[Path]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        for entry in _load_test_entries(path):
            keys.add(normalize_patient_id(entry["patient_id"]))
    return keys


def _load_mapping_entries(mapping_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(mapping_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid mapping format in {mapping_path}")
    entries: Dict[str, Dict[str, object]] = {}
    for raw_patient_id, raw_entry in payload.items():
        if not isinstance(raw_entry, dict):
            continue
        ct_path = raw_entry.get("ct_path")
        dose_path = raw_entry.get("dose_path")
        if not ct_path or not dose_path:
            continue
        patient_id = str(raw_patient_id).strip()
        if not patient_id:
            continue
        patient_key = normalize_patient_id(patient_id)
        entries[patient_key] = {
            "patient_id": patient_id,
            "ct_path": str(ct_path),
            "dose_path": str(dose_path),
            "prescribed_dose": raw_entry.get("prescribed_dose"),
            "rtplan_path": raw_entry.get("rtplan_path"),
        }
    return entries


def _infer_source_split(patient_id: str) -> str:
    for split_name in ("train", "val", "test"):
        patch_cache = (
            RTOG_OTHER_SOURCE_CACHE_ROOT
            / "processed_patches"
            / split_name
            / "patient_cache"
            / f"{patient_id}.h5"
        )
        if patch_cache.exists():
            return split_name
    return "test"


def ensure_rtog_other_splits() -> int:
    if not RTOG_OTHER_MAPPING.exists():
        raise FileNotFoundError(f"Missing mapping file for rtog-other: {RTOG_OTHER_MAPPING}")

    mapping_entries = _load_mapping_entries(RTOG_OTHER_MAPPING)
    exclude_keys = _load_exclusion_keys([RTOG_IMRT_SPLITS, RTOG_CONV_SPLITS])

    filtered_test: List[Dict[str, object]] = []
    for patient_key in sorted(mapping_entries.keys()):
        if patient_key in exclude_keys:
            continue
        entry = mapping_entries[patient_key]
        split_entry: Dict[str, object] = {
            "patient_id": str(entry["patient_id"]),
            "ct_path": str(entry["ct_path"]),
            "dose_path": str(entry["dose_path"]),
            "prescribed_dose": entry.get("prescribed_dose"),
            "source_mode": "ct_dose_file_mapping",
            "source_split": _infer_source_split(str(entry["patient_id"])),
        }
        rtplan_path = entry.get("rtplan_path")
        if rtplan_path:
            split_entry["rtplan_path"] = str(rtplan_path)
        filtered_test.append(split_entry)

    payload = {"train": [], "val": [], "test": filtered_test}
    RTOG_OTHER_ROOT.mkdir(parents=True, exist_ok=True)
    RTOG_OTHER_SPLITS.write_text(json.dumps(payload, indent=2))

    patients_df = pd.DataFrame(
        [
            {
                "patid": entry.get("patient_id"),
                "patient_id": entry.get("patient_id"),
                "group": "rtog_other",
                "label": "other",
                "arm": "",
                "ct_path": entry.get("ct_path"),
                "dose_path": entry.get("dose_path"),
                "prescribed_dose": entry.get("prescribed_dose"),
            }
            for entry in filtered_test
        ]
    )
    patients_df.to_csv(RTOG_OTHER_PATIENTS_CSV, index=False)

    print(
        "[cohort-setup] rtog-other prepared: "
        f"{len(filtered_test)} patients (mapping_total={len(mapping_entries)}, excluded={len(exclude_keys)})",
        flush=True,
    )
    return len(filtered_test)


def resolve_cohorts(selected_keys: List[str]) -> List[Dict[str, object]]:
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
        ensure_rtog_other_splits()
    return resolved


def cohort_paths(output_dir: Path, cohort_key: str) -> Dict[str, Path]:
    return {
        "metrics": output_dir / f"{cohort_key}_metrics.csv",
        "combined": output_dir / f"{cohort_key}_combined.csv",
        "summary": output_dir / f"{cohort_key}_summary.csv",
    }


def build_legacy_seed_rows(
    *,
    model_key: str,
    display_label: str,
    splits_path: Path,
    cohort_key: str,
    cohort_label: str,
) -> pd.DataFrame | None:
    legacy_path = LEGACY_PATCH_METRICS_CACHE_DIR / f"{model_key}.csv"
    if not legacy_path.exists():
        return None

    legacy_df = pd.read_csv(legacy_path, dtype={"patient_id": str})
    required_cols = {
        "model_key",
        "display_label",
        "patient_id",
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
        "shell_gradient_metric_version",
    }
    if not required_cols.issubset(legacy_df.columns):
        return None
    if not (legacy_df["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION).all():
        return None

    target_ids = load_test_patient_ids(splits_path)
    key_to_target_id = {normalize_patient_id(pid): str(pid) for pid in target_ids}
    target_keys = set(key_to_target_id.keys())
    if not target_keys:
        return None

    legacy_df = legacy_df.copy()
    legacy_df["__patient_key"] = legacy_df["patient_id"].map(normalize_patient_id)
    legacy_df = legacy_df.loc[legacy_df["__patient_key"].isin(target_keys)]
    if legacy_df.empty:
        return None
    legacy_df = legacy_df.drop_duplicates(subset="__patient_key", keep="last")

    if len(legacy_df["__patient_key"].unique()) != len(target_keys):
        return None

    legacy_df["patient_id"] = legacy_df["__patient_key"].map(key_to_target_id)
    legacy_df["model_key"] = model_key
    legacy_df["display_label"] = display_label
    legacy_df["gamma_pass_rate_3pct_3mm"] = legacy_df["gamma_pass_rate"]
    legacy_df["shell_gradient_metric_version"] = SHELL_GRADIENT_METRIC_VERSION
    legacy_df["skip_reason"] = ""
    legacy_df["cohort_key"] = cohort_key
    legacy_df["cohort_label"] = cohort_label
    legacy_df = legacy_df.drop(columns=["__patient_key"])
    return legacy_df


def load_existing_combined(path: Path) -> pd.DataFrame | None:
    required = {
        "cohort_key",
        "cohort_label",
        "model_key",
        "display_label",
        "patient_id",
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate_3pct_3mm",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
        "shell_gradient_metric_version",
    }
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"patient_id": str})
    if "gamma_pass_rate_3pct_3mm" not in df.columns and "gamma_pass_rate" in df.columns:
        df["gamma_pass_rate_3pct_3mm"] = df["gamma_pass_rate"]
    if "skip_reason" not in df.columns:
        df["skip_reason"] = ""
    if not required.issubset(df.columns):
        return None
    numeric_cols = [
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate_3pct_3mm",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
    ]
    valid_rows = df["skip_reason"].fillna("").astype(str).str.strip() == ""
    if df.loc[valid_rows, numeric_cols].isna().any().any():
        return None
    if not (df["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION).all():
        return None
    return df


def summarize_per_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort_key, cohort_label, model_key, display_label), group in df.groupby(
        ["cohort_key", "cohort_label", "model_key", "display_label"],
        sort=False,
    ):
        row = {
            "cohort_key": cohort_key,
            "cohort_label": cohort_label,
            "model_key": model_key,
            "display_label": display_label,
            "n_patients": int(group["patient_id"].nunique()),
            "n_skipped": int((group.get("skip_reason", "").fillna("").astype(str).str.strip() != "").sum()) if "skip_reason" in group.columns else 0,
        }
        for metric in [
            "test_mse",
            "ptv_ipsi_lung_shell_gradient_error",
            "gamma_pass_rate_3pct_3mm",
            "Dmean_rel_diff",
            "Dmean_rel_diff_ipsi",
            "Dmean_rel_diff_contra",
        ]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_or_load_cohort_cache(cohort_spec: Dict[str, object], output_dir: Path, skip_compute: bool) -> pd.DataFrame:
    cohort_key = str(cohort_spec["key"])
    cohort_label = str(cohort_spec["label"])
    splits_path = Path(cohort_spec["splits"])
    data_dir = Path(cohort_spec["data_dir"])
    paths = cohort_paths(output_dir, cohort_key)

    combined_df = load_existing_combined(paths["combined"])
    if combined_df is None:
        if skip_compute:
            raise FileNotFoundError(f"Missing or invalid combined cache: {paths['combined']}")
        combined_df = load_or_compute_metrics(paths["metrics"], splits_path, data_dir)
        if "gamma_pass_rate_3pct_3mm" not in combined_df.columns:
            if "gamma_pass_rate" in combined_df.columns:
                combined_df["gamma_pass_rate_3pct_3mm"] = combined_df["gamma_pass_rate"]
            else:
                raise KeyError("Base metrics are missing gamma_pass_rate needed to populate gamma_pass_rate_3pct_3mm")
        combined_df["cohort_key"] = cohort_key
        combined_df["cohort_label"] = cohort_label
        paths["combined"].parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(paths["combined"], index=False)
    else:
        combined_df = combined_df.copy()
        if "gamma_pass_rate_3pct_3mm" not in combined_df.columns and "gamma_pass_rate" in combined_df.columns:
            combined_df["gamma_pass_rate_3pct_3mm"] = combined_df["gamma_pass_rate"]
        combined_df["cohort_key"] = cohort_key
        combined_df["cohort_label"] = cohort_label

    summary_df = summarize_per_model(combined_df)
    summary_df.to_csv(paths["summary"], index=False)
    return combined_df


def normalize_ids_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" in df.columns:
        df = df.copy()
        df["patient_id"] = df["patient_id"].astype(str)
    return df


def append_model_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([existing_df, new_df], ignore_index=True)
    out = out.drop_duplicates(subset=["model_key", "patient_id"], keep="last")
    return out


def save_cohort_outputs(cohort_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    paths["combined"].parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(paths["combined"], index=False)
    summarize_per_model(cohort_df).to_csv(paths["summary"], index=False)


def run_model_major(cohort_specs: List[Dict[str, object]], output_dir: Path, skip_compute: bool) -> pd.DataFrame:
    cohort_state: Dict[str, pd.DataFrame] = {}
    cohort_expected_counts: Dict[str, int] = {}

    for cohort_spec in cohort_specs:
        cohort_key = str(cohort_spec["key"])
        splits_path = Path(cohort_spec["splits"])
        cohort_expected_counts[cohort_key] = len(load_test_patient_ids(splits_path))
        paths = cohort_paths(output_dir, cohort_key)
        existing = load_existing_combined(paths["combined"])
        if existing is not None:
            cohort_state[cohort_key] = normalize_ids_for_merge(existing)
        else:
            cohort_state[cohort_key] = pd.DataFrame()

    total_models = len(MODEL_SPECS)
    total_cohorts = len(cohort_specs)
    for model_idx, model_spec in enumerate(MODEL_SPECS, start=1):
        model_key = str(model_spec["key"])
        model_label = str(model_spec["display_label"])
        print(
            f"[model {model_idx}/{total_models}] {model_label} ({model_key})",
            flush=True,
        )
        for cohort_idx, cohort_spec in enumerate(cohort_specs, start=1):
            cohort_key = str(cohort_spec["key"])
            cohort_label = str(cohort_spec["label"])
            splits_path = Path(cohort_spec["splits"])
            data_dir = Path(cohort_spec["data_dir"])
            paths = cohort_paths(output_dir, cohort_key)
            existing = cohort_state[cohort_key]
            expected_patients = cohort_expected_counts[cohort_key]

            if not existing.empty and "model_key" in existing.columns:
                existing_model_rows = existing.loc[existing["model_key"] == model_key]
            else:
                existing_model_rows = pd.DataFrame()
            existing_model_count = int(existing_model_rows["patient_id"].nunique()) if not existing_model_rows.empty else 0

            if existing_model_count >= expected_patients:
                print(
                    f"[model {model_idx}/{total_models}][cohort {cohort_idx}/{total_cohorts}] {cohort_label}: cached {existing_model_count}/{expected_patients}",
                    flush=True,
                )
                continue

            seeded_df = build_legacy_seed_rows(
                model_key=model_key,
                display_label=model_label,
                splits_path=splits_path,
                cohort_key=cohort_key,
                cohort_label=cohort_label,
            )
            if seeded_df is not None:
                cohort_state[cohort_key] = append_model_rows(existing, normalize_ids_for_merge(seeded_df))
                existing = cohort_state[cohort_key]
                save_cohort_outputs(existing, paths)
                seeded_count = int(
                    cohort_state[cohort_key]
                    .loc[cohort_state[cohort_key]["model_key"] == model_key, "patient_id"]
                    .nunique()
                )
                print(
                    f"[model {model_idx}/{total_models}][cohort {cohort_idx}/{total_cohorts}] {cohort_label}: seeded {seeded_count}/{expected_patients} from legacy cache",
                    flush=True,
                )
                continue

            if skip_compute:
                raise FileNotFoundError(
                    f"Missing computed rows for {cohort_key}:{model_key} and --skip-compute is set."
                )

            print(
                f"[model {model_idx}/{total_models}][cohort {cohort_idx}/{total_cohorts}] {cohort_label}: computing {model_key}",
                flush=True,
            )
            model_df = load_or_compute_metrics(
                paths["metrics"],
                splits_path,
                data_dir,
                selected_model_keys=[model_key],
            )
            if "gamma_pass_rate_3pct_3mm" not in model_df.columns and "gamma_pass_rate" in model_df.columns:
                model_df["gamma_pass_rate_3pct_3mm"] = model_df["gamma_pass_rate"]
            model_df["cohort_key"] = cohort_key
            model_df["cohort_label"] = cohort_label
            model_df = normalize_ids_for_merge(model_df)

            cohort_state[cohort_key] = append_model_rows(existing, model_df)
            existing = cohort_state[cohort_key]
            save_cohort_outputs(existing, paths)

    frames = []
    for cohort_spec in cohort_specs:
        cohort_key = str(cohort_spec["key"])
        cohort_label = str(cohort_spec["label"])
        df = cohort_state[cohort_key].copy()
        if not df.empty:
            if "cohort_key" not in df.columns:
                df["cohort_key"] = cohort_key
            if "cohort_label" not in df.columns:
                df["cohort_label"] = cohort_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_specs = resolve_cohorts(args.cohort_key)
    if args.cohort_major:
        frames = []
        total_cohorts = len(cohort_specs)
        for cohort_idx, cohort_spec in enumerate(cohort_specs, start=1):
            cohort_key = str(cohort_spec["key"])
            cohort_label = str(cohort_spec["label"])
            print(
                f"[cohort {cohort_idx}/{total_cohorts}] {cohort_label} ({cohort_key})",
                flush=True,
            )
            frames.append(build_or_load_cohort_cache(cohort_spec, output_dir, args.skip_compute))
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = run_model_major(cohort_specs, output_dir, args.skip_compute)
    combined_path = output_dir / "all_cohorts_per_patient_metrics.csv"
    summary_path = output_dir / "all_cohorts_summary.csv"
    if not args.no_aggregate:
        all_df.to_csv(combined_path, index=False)
        summarize_per_model(all_df).to_csv(summary_path, index=False)
        print(f"Saved {combined_path}")
        print(f"Saved {summary_path}")
    for cohort_spec in cohort_specs:
        cohort_key = str(cohort_spec['key'])
        paths = cohort_paths(output_dir, cohort_key)
        print(f"Saved {paths['combined']}")
        print(f"Saved {paths['summary']}")


if __name__ == "__main__":
    main()
