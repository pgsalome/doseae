#!/usr/bin/env python3
# Figure 2. Patient-level boxplot comparison of patch-based autoencoder models for 3D dose representation.
# (A) Test MSE, (B) high-dose shell gradient error, (C) gamma pass rate, and
# (D) relative Dmean error across test patients for patch-level architectures and multimodal
# ResNet U-Net variants. These panels summarize voxel error, preservation of boundary dose
# fall-off, gamma-based spatial agreement, and summary-dose discrepancy on reconstructed
# full patient volumes.

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from scipy import ndimage as ndi


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from datasets.loaders import create_dataset  # noqa: E402
from models import get_model  # noqa: E402
from core.training.trainer import Trainer  # noqa: E402
from utils.clinical_metrics import ClinicalMetricsCalculator  # noqa: E402
import evaluate_testset as patch_eval  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper" / "figures"
DEFAULT_METRICS_OUTPUT = REPO_ROOT / "outputs" / "paper" / "figure2_patch_per_patient_metrics.csv"
DEFAULT_GAMMA_OUTPUT = REPO_ROOT / "outputs" / "paper" / "figure2_patch_gamma_1pct_1mm.csv"
DEFAULT_SPLITS = REPO_ROOT / "data" / "full_splits_auto.json"
DEFAULT_DATA_DIR = Path("/data/pgsal/NSCLC-Cetuximab_AE_cache")
HIGH_DOSE_CORE_REL_THRESHOLD = 0.90
HIGH_GRADIENT_SHELL_PERCENTILE = 50.0
SHELL_GRADIENT_ERROR_PERCENTILE = 95.0
SHELL_GRADIENT_RELATIVE_FLOOR_PERCENTILE = 10.0
SHELL_GRADIENT_MODE = os.environ.get("DOSEAE_SHELL_GRADIENT_MODE", "v2").strip().lower()
if SHELL_GRADIENT_MODE not in {"v1", "v2"}:
    raise ValueError(
        f"Unsupported DOSEAE_SHELL_GRADIENT_MODE={SHELL_GRADIENT_MODE!r}. Expected 'v1' or 'v2'."
    )
if SHELL_GRADIENT_MODE == "v1":
    HIGH_DOSE_SHELL_DILATION_MM_CANDIDATES = (3.0,)
    SHELL_GRADIENT_METRIC_VERSION = "high_dose_core90_dilate3_top50_p95_v1"
else:
    HIGH_DOSE_SHELL_DILATION_MM_CANDIDATES = (3.0, 5.0, 7.0)
    SHELL_GRADIENT_METRIC_VERSION = "high_dose_core90_dilate3to7_top50_p95_relativepct_v2"
SIMPLIFIED_PANEL_C_GAMMA_DOSE_THRESHOLD = 1.0
SIMPLIFIED_PANEL_C_GAMMA_DISTANCE_THRESHOLD = 1.0
STANDARD_GAMMA_DOSE_THRESHOLD = 3.0
STANDARD_GAMMA_DISTANCE_THRESHOLD = 3.0

MODEL_SPECS: List[Dict[str, object]] = [
    {
        "key": "conv_dose",
        "display_label": "Conv AE",
        "config_path": REPO_ROOT / "config" / "baselines" / "conv_dose_gpu2.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/baselines/conv_dose_gpu2"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/baselines/conv_dose_gpu2/models/lung_best_model.pth"),
    },
    {
        "key": "resnet_dose",
        "display_label": "ResNet AE",
        "config_path": None,
        "results_dir": Path("/data/pgsal/nsclc_doseae/baselines/resnet_dose_gpu0"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/baselines/resnet_dose_gpu0/models/lung_best_model.pth"),
    },
    {
        "key": "unet_dose",
        "display_label": "U-Net AE",
        "config_path": REPO_ROOT / "config" / "baselines" / "unet_dose_gpu1.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/baselines/unet_dose_gpu1"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/baselines/unet_dose_gpu1/models/lung_best_model.pth"),
    },
    {
        "key": "resunet_dose",
        "display_label": "ResNet U-Net AE",
        "config_path": REPO_ROOT / "config" / "baselines" / "resnet_unet_gpu0.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/baselines/resnet_unet_gpu0"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/baselines/resnet_unet_gpu0/models/lung_best_model.pth"),
    },
    {
        "key": "resunet_dose_ct",
        "display_label": "ResNet U-Net AE\n+ CT",
        "config_path": REPO_ROOT / "config" / "baselines" / "resnet_unet_patch_dose_ct.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_ct"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_ct/models/lung_best_model.pth"),
    },
    {
        "key": "resunet_dose_fused",
        "display_label": "ResNet U-Net AE\n+ Fused",
        "config_path": REPO_ROOT / "config" / "baselines" / "resnet_unet_patch_dose_fused.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_fused"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_fused/models/lung_best_model.pth"),
    },
    {
        "key": "resunet_dose_ct_fused",
        "display_label": "ResNet U-Net AE\n+ CT + Fused",
        "config_path": REPO_ROOT / "config" / "baselines" / "resnet_unet_patch_dose_ct_fused.yaml",
        "results_dir": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_ct_fused"),
        "checkpoint": Path("/data/pgsal/nsclc_doseae/experiments/resnet_unet_patch_dose_ct_fused/models/lung_best_model.pth"),
    },
]

PANEL_SPECS = [
    {
        "panel": "A",
        "metric": "test_mse",
        "ylabel": "Test MSE",
        "yscale": "log",
    },
    {
        "panel": "B",
        "metric": "ptv_ipsi_lung_shell_gradient_error",
        "ylabel": "High-dose shell gradient error (%)\n(P95, top50% GT-gradient shell)",
        "yscale": "linear",
    },
    {
        "panel": "C",
        "metric": "gamma_pass_rate_3pct_3mm",
        "ylabel": "Gamma pass rate (%)\n(simplified 3%/3 mm)",
        "yscale": "linear",
        "ylim": (0.0, 100.0),
    },
    {
        "panel": "D",
        "metric": "Dmean_rel_diff",
        "ylabel": "Relative Dmean error (%)\n(global)",
        "yscale": "linear",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patch-based Figure 2 boxplots.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for figure outputs.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to save or reload cached per-patient patch metrics.",
    )
    parser.add_argument(
        "--gamma-output",
        type=Path,
        default=DEFAULT_GAMMA_OUTPUT,
        help="Path to save or reload cached per-patient simplified 1%%/1 mm gamma values for panel C.",
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="Path to the train/val/test split JSON.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Patch/image cache root.")
    parser.add_argument(
        "--model-key",
        action="append",
        default=[],
        help="Optional patch model key to compute. Repeat the flag to provide multiple keys.",
    )
    return parser.parse_args()


def load_test_patient_ids(splits_path: Path) -> List[str]:
    splits = patch_eval.load_splits(splits_path)
    patients = splits.get("test", [])
    patient_ids = []
    for patient in patients:
        patient_id = patient.get("patient_id")
        if patient_id:
            patient_ids.append(str(patient_id))
    if not patient_ids:
        raise ValueError(f"No test patients found in {splits_path}")
    return patient_ids


def normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    compact = text.replace("-", "")
    if compact.isdigit():
        return str(int(compact))
    return text


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "hatch.linewidth": 0.6,
        }
    )


def load_checkpoint_config(checkpoint_path: Path, results_dir: Path, config_path: Path | None = None) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path is not None and config_path.exists():
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)
    else:
        config = copy.deepcopy(checkpoint["config"])
    config.setdefault("output", {})
    config["output"]["results_dir"] = str(results_dir)
    config["output"]["model_dir"] = str(checkpoint_path.parent)
    config["output"].setdefault("log_dir", str(results_dir / "logs"))

    config.setdefault("wandb", {})
    config["wandb"]["use_wandb"] = False

    pre_cfg = config.get("pretrained_encoder")
    if isinstance(pre_cfg, dict):
        pre_cfg["use_pretrained_ct_encoder"] = False

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["test_mode"] = False
    dataset_cfg["num_workers"] = 0
    dataset_cfg["pin_memory"] = False
    dataset_cfg.pop("n_test_samples", None)
    dataset_cfg.pop("n_test_batches", None)

    model_cfg = config.setdefault("model", {})
    if (
        str(model_cfg.get("type", "")).strip() == "resnet_ae"
        and "encoder.4.shortcut.0.weight" in checkpoint.get("model_state_dict", {})
    ):
        model_cfg["legacy_resnet_first_block_downsample"] = True

    return config


def override_dataset_cache_paths(config: Dict[str, object], data_dir: Path) -> None:
    """Force dataset/cache resolution to use the user-provided cache root.

    Several baseline configs hardcode `dataset.data_h5` to the original NSCLC
    cache. For external cohorts we want `--data-dir` to be authoritative for
    both `create_dataset(...)` and `evaluate_testset.resolve_cache_root(...)`.
    """
    dataset_cfg = config.setdefault("dataset", {})
    base_dir = Path(data_dir)
    dataset_type = str(dataset_cfg.get("dataset_type", "patches")).lower()

    if dataset_type.startswith("patch"):
        h5_map: Dict[str, str] = {}
        for split_name in ("test", "train", "val"):
            candidate = (base_dir / "processed_patches" / f"{split_name}.h5").resolve()
            if candidate.exists():
                h5_map[split_name] = str(candidate)
        if not h5_map:
            h5_map["test"] = str((base_dir / "processed_patches" / "test.h5").resolve())
        dataset_cfg["data_h5"] = h5_map
    else:
        h5_map = {}
        for split_name in ("test", "train", "val"):
            candidate = (base_dir / "processed_images" / f"{split_name}.h5").resolve()
            if candidate.exists():
                h5_map[split_name] = str(candidate)
        if not h5_map:
            h5_map["test"] = str((base_dir / "processed_images" / "test.h5").resolve())
        dataset_cfg["data_h5"] = h5_map
    dataset_cfg.pop("h5_path", None)
    dataset_cfg["use_single_file"] = False
    dataset_cfg["data_file"] = None


def infer_model_channels(dataset) -> int:
    input_channels = getattr(dataset, "input_channels", None)
    if isinstance(input_channels, (list, tuple)):
        return len(input_channels)
    if isinstance(input_channels, int):
        return input_channels
    sample = dataset[0]
    input_tensor = sample.get("input")
    if input_tensor is None or not torch.is_tensor(input_tensor):
        raise ValueError("Unable to infer model input channels from dataset sample.")
    return int(input_tensor.shape[0])


def select_device(config: Dict[str, object]) -> torch.device:
    training_cfg = config.get("training", {})
    gpu_ids = training_cfg.get("gpu_ids")
    gpu_id = training_cfg.get("gpu_id")
    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        if isinstance(gpu_ids, list) and gpu_ids:
            requested_gpu = int(gpu_ids[0])
            if requested_gpu >= visible_gpu_count:
                requested_gpu = 0
            return torch.device(f"cuda:{requested_gpu}")
        if isinstance(gpu_id, int):
            requested_gpu = int(gpu_id)
            if requested_gpu >= visible_gpu_count:
                requested_gpu = 0
            return torch.device(f"cuda:{requested_gpu}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_patch_datasets_by_split(config: Dict[str, object], data_dir: Path) -> Dict[str, object]:
    datasets_by_split: Dict[str, object] = {}
    for split_name in ("test", "train", "val"):
        try:
            dataset = create_dataset(config, "lung", split_name, str(data_dir), transform=None, apply_transforms=False)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            if "No patches available after applying dataset filters" in str(exc):
                continue
            raise
        datasets_by_split[split_name] = dataset
    if not datasets_by_split:
        raise FileNotFoundError(f"No patch datasets found under {data_dir}")
    return datasets_by_split


def choose_patient_patch_dataset(patient: Dict[str, object], datasets_by_split: Dict[str, object]) -> Tuple[str, object]:
    patient_id = str(patient.get("patient_id", ""))
    if not patient_id:
        raise ValueError("Patient entry is missing patient_id.")

    preferred_splits: List[str] = []
    source_split = str(patient.get("source_split", "")).strip().lower()
    if source_split in datasets_by_split:
        preferred_splits.append(source_split)
    if "test" in datasets_by_split and "test" not in preferred_splits:
        preferred_splits.append("test")
    for split_name in ("train", "val"):
        if split_name in datasets_by_split and split_name not in preferred_splits:
            preferred_splits.append(split_name)
    for split_name in datasets_by_split:
        if split_name not in preferred_splits:
            preferred_splits.append(split_name)

    for split_name in preferred_splits:
        dataset = datasets_by_split[split_name]
        if dataset.get_patient_indices(patient_id, restrict_to_active=False):
            return split_name, dataset

    # Fallback to the first available split; caller will handle missing indices.
    fallback_split = preferred_splits[0]
    return fallback_split, datasets_by_split[fallback_split]


def close_dataset_views(datasets_by_split: Dict[str, object]) -> None:
    for dataset in datasets_by_split.values():
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            close_fn()


def build_high_dose_shell_mask(
    gt_volume: np.ndarray,
    voxel_spacing_xyz: Tuple[float, float, float],
    valid_mask: np.ndarray,
) -> np.ndarray:
    max_dose = float(np.max(gt_volume))
    if max_dose <= 0:
        raise ValueError("Ground-truth volume has non-positive maximum dose.")

    high_dose_core = gt_volume >= (HIGH_DOSE_CORE_REL_THRESHOLD * max_dose)
    high_dose_core = np.logical_and(high_dose_core, valid_mask)
    if not np.any(high_dose_core):
        raise ValueError("High-dose core is empty after restricting to reconstructed voxels.")

    spacing_zyx = (float(voxel_spacing_xyz[2]), float(voxel_spacing_xyz[1]), float(voxel_spacing_xyz[0]))
    distance_mm = ndi.distance_transform_edt(~high_dose_core, sampling=spacing_zyx)
    for dilation_mm in HIGH_DOSE_SHELL_DILATION_MM_CANDIDATES:
        dilated_core = distance_mm <= float(dilation_mm)
        shell = np.logical_and(dilated_core, ~high_dose_core)
        shell = np.logical_and(shell, valid_mask)
        if np.any(shell):
            return shell

    raise ValueError("High-dose shell is empty after fallback dilations.")


def gradient_magnitude(volume: np.ndarray, voxel_spacing_xyz: Tuple[float, float, float]) -> np.ndarray:
    spacing_zyx = (float(voxel_spacing_xyz[2]), float(voxel_spacing_xyz[1]), float(voxel_spacing_xyz[0]))
    gz, gy, gx = np.gradient(volume.astype(np.float32), *spacing_zyx)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def high_dose_shell_gradient_error(
    gt_grad_mag: np.ndarray,
    pred_grad_mag: np.ndarray,
    shell_mask: np.ndarray,
) -> float:
    grad_err = np.abs(pred_grad_mag - gt_grad_mag)
    shell_gt_grad = gt_grad_mag[shell_mask]
    if shell_gt_grad.size == 0:
        raise ValueError("High-dose shell contains no voxels for gradient evaluation.")

    grad_threshold = float(np.percentile(shell_gt_grad, HIGH_GRADIENT_SHELL_PERCENTILE))
    high_grad_shell_mask = np.logical_and(shell_mask, gt_grad_mag >= grad_threshold)
    if not np.any(high_grad_shell_mask):
        raise ValueError("Top-50% GT-gradient shell region is empty.")

    if SHELL_GRADIENT_MODE == "v1":
        return float(np.percentile(grad_err[high_grad_shell_mask], SHELL_GRADIENT_ERROR_PERCENTILE))

    gt_selected = gt_grad_mag[high_grad_shell_mask]
    floor_value = float(np.percentile(shell_gt_grad, SHELL_GRADIENT_RELATIVE_FLOOR_PERCENTILE))
    floor_value = max(floor_value, 1e-6)
    denom = np.maximum(gt_selected, floor_value)
    rel_err_pct = (grad_err[high_grad_shell_mask] / denom) * 100.0
    return float(np.percentile(rel_err_pct, SHELL_GRADIENT_ERROR_PERCENTILE))


def relative_dmean_error(mask: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    gt_mean = float(np.mean(gt[mask]))
    pred_mean = float(np.mean(pred[mask]))
    if gt_mean == 0.0:
        return np.nan
    return float(abs(pred_mean - gt_mean) / abs(gt_mean) * 100.0)


def get_ipsi_contra_masks(
    mask_dict: Dict[str, np.ndarray],
    gt_volume: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    ipsi = mask_dict.get("ipsi_lung")
    contra = mask_dict.get("contra_lung")
    if ipsi is not None and contra is not None:
        return np.logical_and(ipsi.astype(bool), valid_mask), np.logical_and(contra.astype(bool), valid_mask)

    left = mask_dict.get("left_lung")
    right = mask_dict.get("right_lung")
    if left is None or right is None:
        return None, None

    left = left.astype(bool)
    right = right.astype(bool)
    left_valid = np.logical_and(left, valid_mask)
    right_valid = np.logical_and(right, valid_mask)
    left_mean = float(np.mean(gt_volume[left_valid])) if np.any(left_valid) else 0.0
    right_mean = float(np.mean(gt_volume[right_valid])) if np.any(right_valid) else 0.0
    if left_mean >= right_mean:
        ipsi_mask, contra_mask = left_valid, right_valid
    else:
        ipsi_mask, contra_mask = right_valid, left_valid
    return ipsi_mask, contra_mask


def get_patient_spacing_xyz(processed_images_dir: Path, patient_id: str) -> Tuple[float, float, float]:
    dose_path = processed_images_dir / patient_id / f"{patient_id}_dose_processed.nrrd"
    if not dose_path.exists():
        raise FileNotFoundError(f"Missing processed dose image for {patient_id}: {dose_path}")
    image = sitk.ReadImage(str(dose_path))
    spacing = image.GetSpacing()
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def build_clinical_calculator(
    voxel_spacing_xyz: Tuple[float, float, float],
    gamma_dose_threshold: float = 3.0,
    gamma_distance_threshold: float = 3.0,
) -> ClinicalMetricsCalculator:
    clinical_cfg = {
        "clinical_metrics": {
            "spacing": [voxel_spacing_xyz[2], voxel_spacing_xyz[1], voxel_spacing_xyz[0]],
            "use_pymedphys_gamma": False,
            "gamma_dose_threshold": float(gamma_dose_threshold),
            "gamma_distance_threshold": float(gamma_distance_threshold),
            "dose_threshold_cutoff": 10,
            "calculate_dvh": True,
            "dvh_num_bins": 1000,
        }
    }
    return ClinicalMetricsCalculator(clinical_cfg)


def load_existing_full_volume_patient_metrics(results_dir: Path) -> Dict[str, Dict[str, float]]:
    candidate_files = [
        results_dir / "full_volume_summary.json",
        results_dir / "test_metrics.json",
    ]

    for path in candidate_files:
        if not path.exists():
            continue

        with path.open("r") as handle:
            payload = json.load(handle)

        full_volume_summary = payload
        if path.name == "test_metrics.json":
            full_volume_summary = payload.get("full_volume_summary", {})

        patients = full_volume_summary.get("patients", []) if isinstance(full_volume_summary, dict) else []
        patient_rows: Dict[str, Dict[str, float]] = {}
        for patient_entry in patients:
            patient_id = patient_entry.get("patient_id")
            overall = patient_entry.get("overall", {})
            if not patient_id or not isinstance(overall, dict):
                continue
            test_mse = float(overall.get("mse", np.nan))
            gamma_pass_rate = float(overall.get("gamma_pass_rate", np.nan))
            dmean_rel_diff = float(overall.get("Dmean_rel_diff", np.nan))
            if any(np.isnan(value) for value in (test_mse, gamma_pass_rate, dmean_rel_diff)):
                continue
            patient_rows[str(patient_id)] = {
                "test_mse": test_mse,
                "gamma_pass_rate": gamma_pass_rate,
                "Dmean_rel_diff": dmean_rel_diff,
            }

        if len(patient_rows) == 47:
            return patient_rows

    return {}


def compute_metrics_for_spec(
    spec: Dict[str, object],
    splits_path: Path,
    data_dir: Path,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    display_label = str(spec["display_label"])
    model_key = str(spec["key"])
    checkpoint_path = Path(spec["checkpoint"])
    results_dir = Path(spec["results_dir"])
    config_path = spec.get("config_path")
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_checkpoint_config(
        checkpoint_path,
        results_dir,
        Path(config_path) if config_path is not None else None,
    )
    override_dataset_cache_paths(config, data_dir)
    datasets_by_split = build_patch_datasets_by_split(config, data_dir)
    reference_dataset = datasets_by_split.get("test") or next(iter(datasets_by_split.values()))
    config.setdefault("model", {})["in_channels"] = infer_model_channels(reference_dataset)
    config.setdefault("model", {}).setdefault("output_channels", 1)

    device = select_device(config)
    model = get_model(config).to(device)
    trainer = Trainer(model, config, "lung")
    trainer.load_checkpoint(str(checkpoint_path))
    existing_full_volume_metrics = load_existing_full_volume_patient_metrics(results_dir)

    test_patients = patch_eval.load_splits(splits_path).get("test", [])
    cache_root = patch_eval.resolve_cache_root(config)
    mask_cache_dir = results_dir / "mask_cache"
    preprocessor_holder: Dict[str, object] = {"instance": None}
    patch_shape = tuple(int(v) for v in reference_dataset.patch_shape)
    configured_batch_size = int(config.get("training", {}).get("batch_size", 8) or 8)
    inference_batch_size = max(1, min(configured_batch_size, 32))

    rows: List[Dict[str, object]] = []
    completed_patient_ids = set()
    if cache_path is not None and cache_path.exists():
        cached_partial = pd.read_csv(cache_path, dtype={"patient_id": str})
        if {"patient_id", "model_key", "display_label"}.issubset(cached_partial.columns):
            cached_partial = cached_partial.drop_duplicates(subset="patient_id", keep="last")
            for record in cached_partial.to_dict(orient="records"):
                rows.append(record)
            completed_patient_ids = {
                normalize_patient_id(pid) for pid in cached_partial["patient_id"].astype(str).tolist()
            }

    total_patients = len(test_patients)
    print(
        f"[{display_label}] metrics cache: {len(completed_patient_ids)}/{total_patients} patients already cached",
        flush=True,
    )

    for patient_idx, patient in enumerate(test_patients, start=1):
        patient_id = patient.get("patient_id")
        if not patient_id:
            continue
        patient_key = normalize_patient_id(patient_id)
        if patient_key in completed_patient_ids:
            continue

        print(
            f"[{display_label}] patient {patient_idx}/{total_patients}: {patient_id}",
            flush=True,
        )

        def append_patient_row(
            *,
            test_mse: float = np.nan,
            shell_gradient_error: float = np.nan,
            gamma_pass_rate: float = np.nan,
            dmean_rel_diff: float = np.nan,
            dmean_rel_diff_ipsi: float = np.nan,
            dmean_rel_diff_contra: float = np.nan,
            skip_reason: str = "",
        ) -> None:
            rows.append(
                {
                    "model_key": model_key,
                    "display_label": display_label,
                    "patient_id": patient_id,
                    "test_mse": test_mse,
                    "ptv_ipsi_lung_shell_gradient_error": shell_gradient_error,
                    "gamma_pass_rate": gamma_pass_rate,
                    "gamma_pass_rate_3pct_3mm": gamma_pass_rate,
                    "Dmean_rel_diff": dmean_rel_diff,
                    "Dmean_rel_diff_ipsi": dmean_rel_diff_ipsi,
                    "Dmean_rel_diff_contra": dmean_rel_diff_contra,
                    "shell_gradient_metric_version": SHELL_GRADIENT_METRIC_VERSION,
                    "skip_reason": skip_reason,
                }
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                partial_df = pd.DataFrame(rows)
                if not partial_df.empty:
                    partial_df["__patient_key"] = partial_df["patient_id"].map(normalize_patient_id)
                    partial_df = partial_df.drop_duplicates(subset="__patient_key", keep="last").drop(columns=["__patient_key"])
                rows[:] = partial_df.to_dict(orient="records")
                partial_df.to_csv(cache_path, index=False)

        patient_split, patient_dataset = choose_patient_patch_dataset(patient, datasets_by_split)

        indices = patient_dataset.get_patient_indices(patient_id, restrict_to_active=True)
        if not indices:
            indices = patient_dataset.get_patient_indices(patient_id, restrict_to_active=False)
            if indices:
                print(
                    f"[{display_label}] patient {patient_idx}/{total_patients}: {patient_id} has 0 active patches in split={patient_split}, using all patches ({len(indices)})",
                    flush=True,
                )
        if not indices:
            append_patient_row(skip_reason="no_patches_available")
            continue

        try:
            min_coords, volume_shape = patch_eval.build_patient_bbox(patient_dataset, indices)
            recon, gt, votes = patch_eval.reconstruct_from_cached_dataset(
                trainer,
                patient_dataset,
                indices,
                patch_shape,
                inference_batch_size,
                min_coords,
                volume_shape,
            )

            valid_mask = votes > 0
            if not np.any(valid_mask):
                append_patient_row(skip_reason="no_reconstructed_voxels")
                continue

            patient_processed_images_dir = cache_root / "processed_images" / patient_split
            mask_dict = patch_eval.get_or_compute_masks(
                patient,
                config,
                results_dir,
                mask_cache_dir,
                patient_processed_images_dir,
                min_coords,
                volume_shape,
                preprocessor_holder,
            )
            voxel_spacing_xyz = get_patient_spacing_xyz(patient_processed_images_dir, patient_id)

            shell_mask = build_high_dose_shell_mask(gt, voxel_spacing_xyz, valid_mask)
            gt_grad_mag = gradient_magnitude(gt, voxel_spacing_xyz)
            recon_grad_mag = gradient_magnitude(recon, voxel_spacing_xyz)
            shell_gradient_error = high_dose_shell_gradient_error(gt_grad_mag, recon_grad_mag, shell_mask)
            ipsi_mask, contra_mask = get_ipsi_contra_masks(mask_dict, gt, valid_mask)

            existing_metrics = existing_full_volume_metrics.get(patient_id)
            if existing_metrics is None:
                calculator = build_clinical_calculator(
                    voxel_spacing_xyz,
                    gamma_dose_threshold=STANDARD_GAMMA_DOSE_THRESHOLD,
                    gamma_distance_threshold=STANDARD_GAMMA_DISTANCE_THRESHOLD,
                )
                test_mse = float(np.mean((recon[valid_mask] - gt[valid_mask]) ** 2))
                gamma_results = calculator.calculate_gamma(gt, recon, valid_mask)
                gamma_pass_rate = float(gamma_results.get("gamma_pass_rate", np.nan))
                gt_mean = float(np.mean(gt[valid_mask]))
                recon_mean = float(np.mean(recon[valid_mask]))
                if gt_mean != 0.0:
                    dmean_rel_diff = float(abs(recon_mean - gt_mean) / abs(gt_mean) * 100.0)
                else:
                    dmean_rel_diff = np.nan
            else:
                test_mse = float(existing_metrics["test_mse"])
                gamma_pass_rate = float(existing_metrics["gamma_pass_rate"])
                dmean_rel_diff = float(existing_metrics["Dmean_rel_diff"])

            dmean_rel_diff_ipsi = (
                relative_dmean_error(ipsi_mask, gt, recon) if ipsi_mask is not None else np.nan
            )
            dmean_rel_diff_contra = (
                relative_dmean_error(contra_mask, gt, recon) if contra_mask is not None else np.nan
            )

            append_patient_row(
                test_mse=test_mse,
                shell_gradient_error=shell_gradient_error,
                gamma_pass_rate=gamma_pass_rate,
                dmean_rel_diff=dmean_rel_diff,
                dmean_rel_diff_ipsi=dmean_rel_diff_ipsi,
                dmean_rel_diff_contra=dmean_rel_diff_contra,
            )
        except Exception as exc:
            print(
                f"[{display_label}] patient {patient_idx}/{total_patients}: {patient_id} skipped ({exc})",
                flush=True,
            )
            append_patient_row(skip_reason=str(exc))
            continue

    if device.type == "cuda":
        torch.cuda.empty_cache()
    close_dataset_views(datasets_by_split)

    per_patient_df = pd.DataFrame(rows)
    if not per_patient_df.empty:
        per_patient_df["__patient_key"] = per_patient_df["patient_id"].map(normalize_patient_id)
        per_patient_df = per_patient_df.drop_duplicates(subset="__patient_key", keep="last").drop(columns=["__patient_key"])
    expected_patients = len(test_patients)
    if len(per_patient_df) != expected_patients:
        split_keys = {normalize_patient_id(p.get("patient_id")) for p in test_patients if p.get("patient_id")}
        row_keys = {normalize_patient_id(pid) for pid in per_patient_df["patient_id"].astype(str).tolist()}
        missing = sorted(split_keys - row_keys)
        for missing_pid in missing:
            per_patient_df = pd.concat(
                [
                    per_patient_df,
                    pd.DataFrame(
                        [
                            {
                                "model_key": model_key,
                                "display_label": display_label,
                                "patient_id": missing_pid,
                                "test_mse": np.nan,
                                "ptv_ipsi_lung_shell_gradient_error": np.nan,
                                "gamma_pass_rate": np.nan,
                                "gamma_pass_rate_3pct_3mm": np.nan,
                                "Dmean_rel_diff": np.nan,
                                "Dmean_rel_diff_ipsi": np.nan,
                                "Dmean_rel_diff_contra": np.nan,
                                "shell_gradient_metric_version": SHELL_GRADIENT_METRIC_VERSION,
                                "skip_reason": "missing_patient_row_after_processing",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        print(
            f"[{display_label}] warning: added {len(missing)} missing patient placeholders for {model_key}",
            flush=True,
        )
    return per_patient_df


def compute_gamma_only_for_spec(
    spec: Dict[str, object],
    splits_path: Path,
    data_dir: Path,
    cache_path: Path | None = None,
    gamma_dose_threshold: float = SIMPLIFIED_PANEL_C_GAMMA_DOSE_THRESHOLD,
    gamma_distance_threshold: float = SIMPLIFIED_PANEL_C_GAMMA_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    display_label = str(spec["display_label"])
    model_key = str(spec["key"])
    checkpoint_path = Path(spec["checkpoint"])
    results_dir = Path(spec["results_dir"])
    config_path = spec.get("config_path")
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_checkpoint_config(
        checkpoint_path,
        results_dir,
        Path(config_path) if config_path is not None else None,
    )
    override_dataset_cache_paths(config, data_dir)
    datasets_by_split = build_patch_datasets_by_split(config, data_dir)
    reference_dataset = datasets_by_split.get("test") or next(iter(datasets_by_split.values()))
    config.setdefault("model", {})["in_channels"] = infer_model_channels(reference_dataset)
    config.setdefault("model", {}).setdefault("output_channels", 1)

    device = select_device(config)
    model = get_model(config).to(device)
    trainer = Trainer(model, config, "lung")
    trainer.load_checkpoint(str(checkpoint_path))

    test_patients = patch_eval.load_splits(splits_path).get("test", [])
    patch_shape = tuple(int(v) for v in reference_dataset.patch_shape)
    configured_batch_size = int(config.get("training", {}).get("batch_size", 8) or 8)
    inference_batch_size = max(1, min(configured_batch_size, 32))
    cache_root = patch_eval.resolve_cache_root(config)

    rows: List[Dict[str, object]] = []
    completed_patient_ids = set()
    if cache_path is not None and cache_path.exists():
        cached_partial = pd.read_csv(cache_path, dtype={"patient_id": str})
        required_cached = {"patient_id", "model_key", "display_label", "gamma_pass_rate_1pct_1mm"}
        if required_cached.issubset(cached_partial.columns):
            cached_partial = cached_partial.drop_duplicates(subset="patient_id", keep="last")
            for record in cached_partial.to_dict(orient="records"):
                rows.append(record)
            completed_patient_ids = {
                normalize_patient_id(pid) for pid in cached_partial["patient_id"].astype(str).tolist()
            }

    for patient in test_patients:
        patient_id = patient.get("patient_id")
        if not patient_id:
            continue
        patient_key = normalize_patient_id(patient_id)
        if patient_key in completed_patient_ids:
            continue

        patient_split, patient_dataset = choose_patient_patch_dataset(patient, datasets_by_split)
        indices = patient_dataset.get_patient_indices(patient_id, restrict_to_active=True)
        if not indices:
            indices = patient_dataset.get_patient_indices(patient_id, restrict_to_active=False)
        if not indices:
            continue

        min_coords, volume_shape = patch_eval.build_patient_bbox(patient_dataset, indices)
        recon, gt, votes = patch_eval.reconstruct_from_cached_dataset(
            trainer,
            patient_dataset,
            indices,
            patch_shape,
            inference_batch_size,
            min_coords,
            volume_shape,
        )

        valid_mask = votes > 0
        if not np.any(valid_mask):
            continue

        voxel_spacing_xyz = get_patient_spacing_xyz(cache_root / "processed_images" / patient_split, patient_id)
        calculator = build_clinical_calculator(
            voxel_spacing_xyz,
            gamma_dose_threshold=gamma_dose_threshold,
            gamma_distance_threshold=gamma_distance_threshold,
        )
        gamma_results = calculator.calculate_gamma(gt, recon, valid_mask)
        gamma_pass_rate = float(gamma_results.get("gamma_pass_rate", np.nan))

        rows.append(
            {
                "model_key": model_key,
                "display_label": display_label,
                "patient_id": patient_id,
                "gamma_pass_rate_1pct_1mm": gamma_pass_rate,
            }
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            partial_df = pd.DataFrame(rows).drop_duplicates(subset="patient_id", keep="last")
            rows = partial_df.to_dict(orient="records")
            pd.DataFrame(rows).to_csv(cache_path, index=False)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    close_dataset_views(datasets_by_split)

    per_patient_df = pd.DataFrame(rows)
    if not per_patient_df.empty:
        per_patient_df["__patient_key"] = per_patient_df["patient_id"].map(normalize_patient_id)
        per_patient_df = per_patient_df.drop_duplicates(subset="__patient_key", keep="last").drop(columns=["__patient_key"])
    expected_patients = len(test_patients)
    if len(per_patient_df) != expected_patients:
        split_keys = {normalize_patient_id(p.get("patient_id")) for p in test_patients if p.get("patient_id")}
        row_keys = {normalize_patient_id(pid) for pid in per_patient_df["patient_id"].astype(str).tolist()}
        missing = sorted(split_keys - row_keys)
        raise ValueError(
            f"Expected {expected_patients} gamma rows for {model_key}, found {len(per_patient_df)}. Missing patients: {missing}"
        )
    return per_patient_df


def get_model_cache_path(metrics_path: Path, model_key: str) -> Path:
    cache_dir = metrics_path.parent / f"{metrics_path.stem}_cache"
    return cache_dir / f"{model_key}.csv"


def load_valid_cached_metrics(path: Path, expected_patients: int) -> pd.DataFrame | None:
    required_columns = {
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
    if not path.exists():
        return None

    cached = pd.read_csv(path, dtype={"patient_id": str})
    if not required_columns.issubset(cached.columns):
        return None
    if "skip_reason" not in cached.columns:
        cached["skip_reason"] = ""
    if "gamma_pass_rate_3pct_3mm" not in cached.columns:
        cached["gamma_pass_rate_3pct_3mm"] = cached["gamma_pass_rate"]
    metric_columns = [
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate",
        "gamma_pass_rate_3pct_3mm",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
    ]
    valid_rows = cached["skip_reason"].fillna("").astype(str).str.strip() == ""
    if cached.loc[valid_rows, metric_columns].isna().any().any():
        return None
    if not (cached["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION).all():
        return None

    counts = cached.groupby("model_key")["patient_id"].apply(lambda s: s.map(normalize_patient_id).nunique()).to_dict()
    if len(counts) != 1 or next(iter(counts.values())) != expected_patients:
        return None
    return cached


def load_valid_combined_metrics(path: Path, expected_patients: int) -> pd.DataFrame | None:
    required_columns = {
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
    if not path.exists():
        return None

    cached = pd.read_csv(path, dtype={"patient_id": str})
    if not required_columns.issubset(cached.columns):
        return None
    if "skip_reason" not in cached.columns:
        cached["skip_reason"] = ""
    if "gamma_pass_rate_3pct_3mm" not in cached.columns:
        cached["gamma_pass_rate_3pct_3mm"] = cached["gamma_pass_rate"]
    metric_columns = [
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate",
        "gamma_pass_rate_3pct_3mm",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
    ]
    valid_rows = cached["skip_reason"].fillna("").astype(str).str.strip() == ""
    if cached.loc[valid_rows, metric_columns].isna().any().any():
        return None
    if not (cached["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION).all():
        return None

    counts = cached.groupby("model_key")["patient_id"].apply(lambda s: s.map(normalize_patient_id).nunique()).to_dict()
    if not counts or any(count != expected_patients for count in counts.values()):
        return None
    return cached


def load_existing_boxplot_metrics(path: Path, expected_patients: int) -> pd.DataFrame | None:
    required_columns = {
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
    if not path.exists():
        return None

    cached = pd.read_csv(path, dtype={"patient_id": str})
    if not required_columns.issubset(cached.columns):
        return None
    if "skip_reason" not in cached.columns:
        cached["skip_reason"] = ""
    if "gamma_pass_rate_3pct_3mm" not in cached.columns:
        cached["gamma_pass_rate_3pct_3mm"] = cached["gamma_pass_rate"]
    metric_columns = [
        "test_mse",
        "ptv_ipsi_lung_shell_gradient_error",
        "gamma_pass_rate",
        "gamma_pass_rate_3pct_3mm",
        "Dmean_rel_diff",
        "Dmean_rel_diff_ipsi",
        "Dmean_rel_diff_contra",
    ]
    valid_rows = cached["skip_reason"].fillna("").astype(str).str.strip() == ""
    if cached.loc[valid_rows, metric_columns].isna().any().any():
        return None
    if not (cached["shell_gradient_metric_version"].astype(str) == SHELL_GRADIENT_METRIC_VERSION).all():
        return None

    counts = cached.groupby("model_key")["patient_id"].apply(lambda s: s.map(normalize_patient_id).nunique()).to_dict()
    if set(counts.keys()) != {str(spec["key"]) for spec in MODEL_SPECS}:
        return None
    if any(count != expected_patients for count in counts.values()):
        return None
    return cached


def load_valid_gamma_cached_metrics(path: Path, expected_patients: int) -> pd.DataFrame | None:
    required_columns = {
        "model_key",
        "display_label",
        "patient_id",
        "gamma_pass_rate_1pct_1mm",
    }
    if not path.exists():
        return None

    cached = pd.read_csv(path)
    if not required_columns.issubset(cached.columns):
        return None
    if cached[["gamma_pass_rate_1pct_1mm"]].isna().any().any():
        return None

    counts = cached.groupby("model_key")["patient_id"].nunique().to_dict()
    if len(counts) != 1 or next(iter(counts.values())) != expected_patients:
        return None
    return cached


def load_valid_combined_gamma_metrics(path: Path, expected_patients: int) -> pd.DataFrame | None:
    required_columns = {
        "model_key",
        "display_label",
        "patient_id",
        "gamma_pass_rate_1pct_1mm",
    }
    if not path.exists():
        return None

    cached = pd.read_csv(path)
    if not required_columns.issubset(cached.columns):
        return None
    if cached[["gamma_pass_rate_1pct_1mm"]].isna().any().any():
        return None

    counts = cached.groupby("model_key")["patient_id"].nunique().to_dict()
    if set(counts.keys()) != {str(spec["key"]) for spec in MODEL_SPECS}:
        return None
    if any(count != expected_patients for count in counts.values()):
        return None
    return cached


def load_or_compute_gamma_metrics(
    gamma_path: Path,
    splits_path: Path,
    data_dir: Path,
    selected_model_keys: List[str] | None = None,
) -> pd.DataFrame:
    expected_patients = len(load_test_patient_ids(splits_path))
    all_expected_keys = {str(spec["key"]) for spec in MODEL_SPECS}
    if selected_model_keys:
        ordered_keys = [str(key) for key in selected_model_keys]
        invalid_keys = sorted(set(ordered_keys) - all_expected_keys)
        if invalid_keys:
            raise ValueError(f"Unknown model keys requested: {invalid_keys}")
        target_specs = [spec for spec in MODEL_SPECS if str(spec["key"]) in set(ordered_keys)]
    else:
        target_specs = MODEL_SPECS

    requested_keys = {str(spec["key"]) for spec in target_specs}
    if requested_keys == all_expected_keys:
        cached = load_valid_combined_gamma_metrics(gamma_path, expected_patients)
        if cached is not None and set(cached["model_key"].unique()) == all_expected_keys:
            return cached

    frames = []
    for spec in target_specs:
        model_key = str(spec["key"])
        model_cache_path = get_model_cache_path(gamma_path, model_key)
        cached_model_df = load_valid_gamma_cached_metrics(model_cache_path, expected_patients)
        if cached_model_df is None:
            computed_df = compute_gamma_only_for_spec(spec, splits_path, data_dir, model_cache_path)
            model_cache_path.parent.mkdir(parents=True, exist_ok=True)
            computed_df.to_csv(model_cache_path, index=False)
            frames.append(computed_df)
        else:
            frames.append(cached_model_df)

    full_df = pd.concat(frames, ignore_index=True)
    gamma_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_keys == all_expected_keys:
        full_df.to_csv(gamma_path, index=False)
    return full_df


def load_or_compute_metrics(
    metrics_path: Path,
    splits_path: Path,
    data_dir: Path,
    selected_model_keys: List[str] | None = None,
) -> pd.DataFrame:
    expected_patients = len(load_test_patient_ids(splits_path))
    all_expected_keys = {str(spec["key"]) for spec in MODEL_SPECS}
    if selected_model_keys:
        ordered_keys = [str(key) for key in selected_model_keys]
        invalid_keys = sorted(set(ordered_keys) - all_expected_keys)
        if invalid_keys:
            raise ValueError(f"Unknown model keys requested: {invalid_keys}")
        target_specs = [spec for spec in MODEL_SPECS if str(spec["key"]) in set(ordered_keys)]
    else:
        target_specs = MODEL_SPECS

    requested_keys = {str(spec["key"]) for spec in target_specs}
    if requested_keys == all_expected_keys:
        cached = load_valid_combined_metrics(metrics_path, expected_patients)
        if cached is not None and set(cached["model_key"].unique()) == all_expected_keys:
            return cached

    frames = []
    for spec in target_specs:
        model_key = str(spec["key"])
        display_label = str(spec["display_label"])
        model_cache_path = get_model_cache_path(metrics_path, model_key)
        cached_model_df = load_valid_cached_metrics(model_cache_path, expected_patients)
        if cached_model_df is None:
            print(
                f"[metrics] computing {display_label} ({model_key}) for {expected_patients} patients",
                flush=True,
            )
            computed_df = compute_metrics_for_spec(spec, splits_path, data_dir, model_cache_path)
            model_cache_path.parent.mkdir(parents=True, exist_ok=True)
            computed_df.to_csv(model_cache_path, index=False)
            frames.append(computed_df)
        else:
            print(
                f"[metrics] using cached {display_label} ({model_key}) with {cached_model_df['patient_id'].nunique()}/{expected_patients} patients",
                flush=True,
            )
            frames.append(cached_model_df)

    full_df = pd.concat(frames, ignore_index=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_keys == all_expected_keys:
        full_df.to_csv(metrics_path, index=False)
    return full_df


def print_console_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["model_key", "display_label"])
        .agg(
            n_patients=("patient_id", "count"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_min=("test_mse", "min"),
            test_mse_max=("test_mse", "max"),
            shell_grad_mean=("ptv_ipsi_lung_shell_gradient_error", "mean"),
            shell_grad_min=("ptv_ipsi_lung_shell_gradient_error", "min"),
            shell_grad_max=("ptv_ipsi_lung_shell_gradient_error", "max"),
            gamma_mean=("gamma_pass_rate_3pct_3mm", "mean"),
            gamma_min=("gamma_pass_rate_3pct_3mm", "min"),
            gamma_max=("gamma_pass_rate_3pct_3mm", "max"),
            dmean_rel_mean=("Dmean_rel_diff", "mean"),
            dmean_rel_min=("Dmean_rel_diff", "min"),
            dmean_rel_max=("Dmean_rel_diff", "max"),
        )
        .reset_index()
    )
    print("Figure 2 patch boxplot input summary")
    print(summary.to_string(index=False))


def plot_panel(ax: plt.Axes, df: pd.DataFrame, spec: Dict[str, object], expected_patients: int) -> None:
    metric = str(spec["metric"])
    labels = []
    positions = np.arange(1, len(MODEL_SPECS) + 1)
    grayscale = ["#e0e0e0", "#c9c9c9", "#b1b1b1", "#969696", "#7f7f7f", "#686868", "#555555", "#3f3f3f"]
    hatches = ["", "//", "..", "xx", "\\\\", "++", "--", "oo"]

    grouped = []
    for model_spec in MODEL_SPECS:
        key = str(model_spec["key"])
        values = df.loc[df["model_key"] == key, metric].dropna().to_numpy(dtype=np.float32)
        if len(values) != expected_patients:
            raise ValueError(f"Expected {expected_patients} values for {key} {metric}, found {len(values)}")
        grouped.append(values)
        labels.append(str(model_spec["display_label"]))
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        whis=(0, 100),
        medianprops={"color": "black", "linewidth": 1.0},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 1.8, "alpha": 0.45},
    )

    for patch, face, hatch in zip(box["boxes"], grayscale, hatches):
        patch.set_facecolor(face)
        patch.set_hatch(hatch)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_yscale(str(spec["yscale"]))
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    if "ylim" in spec:
        ax.set_ylim(*spec["ylim"])
    elif str(spec["yscale"]) == "log":
        positive = np.concatenate([vals[vals > 0] for vals in grouped])
        lower = max(float(positive.min()) * 0.45, 1e-8)
        upper = float(max(vals.max() for vals in grouped)) * 1.9
        ax.set_ylim(lower, upper)
    else:
        upper = float(max(vals.max() for vals in grouped))
        ax.set_ylim(0.0, upper * 1.18 if upper > 0 else 1.0)

    ax.text(
        -0.04,
        1.08,
        f"({spec['panel']})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="black",
        clip_on=False,
    )

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)


def build_figure(df: pd.DataFrame, output_dir: Path, expected_patients: int) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.8))
    axes = axes.flatten()

    for ax, spec in zip(axes, PANEL_SPECS):
        plot_panel(ax, df, spec, expected_patients)

    figure.tight_layout(pad=1.0, w_pad=1.7, h_pad=1.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "figure2_architecture_comparison_final.png"
    pdf_path = output_dir / "figure2_architecture_comparison_final.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    args = parse_args()
    expected_patients = len(load_test_patient_ids(args.splits))
    if not args.model_key:
        cached_boxplot_df = load_existing_boxplot_metrics(args.metrics_output, expected_patients)
        if cached_boxplot_df is not None:
            base_metrics_df = cached_boxplot_df
        else:
            base_metrics_df = load_or_compute_metrics(args.metrics_output, args.splits, args.data_dir, args.model_key)
    else:
        base_metrics_df = load_or_compute_metrics(args.metrics_output, args.splits, args.data_dir, args.model_key)

    metrics_df = base_metrics_df.copy()
    print_console_summary(metrics_df)
    if set(metrics_df["model_key"].unique()) == {str(spec["key"]) for spec in MODEL_SPECS}:
        print(f"Saved {args.metrics_output}")
        build_figure(metrics_df, args.output_dir, expected_patients)
    else:
        cached_keys = sorted(metrics_df["model_key"].unique())
        print(f"Saved per-model cache entries for: {', '.join(cached_keys)}")


if __name__ == "__main__":
    main()
