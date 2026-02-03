#!/usr/bin/env python3
"""
Extract latent representations from a trained DoseAE model checkpoint.

This script supports two single-patient paths:
1) H5-based extraction (fastest): uses an existing H5 file with patches.
2) CT+Dose extraction: runs preprocessing (segmentation + patching) or
   optionally reuses a cached per-patient H5 if `--use-cache` is passed.

Segmentation caching:
- Use `--segmentation-cache-dir` to store per-patient TotalSegmentator masks.
- If the folder contains cached masks, segmentation is skipped.
- If you want CPU segmentation to avoid GPU OOM, pass `--segmentation-device cpu`.

Examples
--------
1) Extract using an existing H5 (consolidated train.h5 or per-patient cache):

    CUDA_VISIBLE_DEVICES=0 python scripts/extract_doseae_latents.py \\
      --config optuna_runs/patch_search/config_YYYYMMDD_HHMMSS_trialNN.json \\
      --weights /path/to/lung_best_model.pth \\
      --h5-path /path/to/processed_patches/train.h5 \\
      --patient-id xx \\
      --output-dir outputs/latents \\
      --device cuda

   If `--h5-path` points to a per-patient cache (e.g. .../patient_cache/<PATIENT>.h5),
   the script will use the embedded patient_id and `--patient-id` is optional.

2) Extract using CT + Dose (recompute from scratch by default):

    CUDA_VISIBLE_DEVICES=0 python scripts/extract_doseae_latents.py \\
      --config optuna_runs/patch_search/config_YYYYMMDD_HHMMSS_trialNN.json \\
      --weights /path/to/lung_best_model.pth \\
      --ct-path /path/to/CT.nrrd \\
      --dose-path /path/to/DOSE.nrrd \\
      --output-dir outputs/latents \\
      --device cuda \\
      --segmentation-cache-dir outputs/latents/segmentation/<PATIENT_ID>

   To reuse existing per-patient patch cache (skip recompute), add `--use-cache`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import yaml
import h5py
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.loaders import create_dataset, collate_medical_batch
from entities.lung.datasets.dataset import _compute_fused_channel, _resolve_input_channels
from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor
from models import get_model


LOGGER = logging.getLogger("doseae.latent_extraction")

DEFAULT_PREPROCESS_CONFIG: Dict[str, Any] = {
    "dataset": {
        "entity": "lung",
        "dataset_type": "patches",
    },
    "lung": {
        "patch_size": 50,
        "overlap_percentage": 20,
        "patch_workers": 1,
        "dose_threshold": 0.1,
        "mask_acceptance": 0.7,
        "lobe_composite_threshold": 0.1,
        "max_patches_per_lobe": 400,
    },
    "preprocessing": {
        "voxel_spacing": [0.5, 0.5, 0.5],
        "resize_to": [128, 128, 128],
        "ct_preprocessing": {
            "method": "lung_window",
            "window_center": -600,
            "window_width": 1500,
            "clip_range": [-1500, 300],
        },
    },
    "output": {
        "cache_dir": "/data/pgsal/NSCLC-Cetuximab_AE_cache",
    },
    "augmentation": {
        "enable_synthetic_dose": False,
        "synthetic_doses_per_patient": 0,
        "random_seed": None,
        "opentps": {
            "core_path": "external/OpenTPS/opentps_core",
            "min_beams": 3,
            "max_beams": 5,
            "gantry_range_deg": [0.0, 360.0],
            "couch_range_deg": [-10.0, 10.0],
            "beamlet_spacing_mm": 5.0,
            "target_margin_mm": 5.0,
            "optimizer_max_iterations": 120,
            "prescription_gy": None,
            "prescription_scale_range": [0.9, 1.1],
            "target_isodose_range": [0.75, 0.95],
            "min_target_voxels": 750,
            "target_mask_dilation_mm": 2.0,
            "ccc_batch_size": 24,
            "workspace_dir": None,
        },
    },
    "segmentation": {
        "device": "gpu",
        "max_concurrent": 1,
        "allow_cpu_fallback": True,
        "roi_subset": [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
            "aorta",
            "trachea",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the model config (JSON or YAML).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Alias for --checkpoint.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Dataset split(s) to process (e.g., train val test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "latents",
        help="Directory to store latent outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for extraction (defaults to config training batch size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers (defaults to config dataset.num_workers).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per split (for quick runs).",
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention weights if the model returns them.",
    )
    parser.add_argument(
        "--latent-size",
        default=None,
        help="Select a specific latent projection size (int) or 'all' to save every size.",
    )
    parser.add_argument(
        "--ct-path",
        default=None,
        help="CT volume path for single-patient extraction (NRRD).",
    )
    parser.add_argument(
        "--dose-path",
        default=None,
        help="Dose volume path for single-patient extraction (NRRD).",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Patient ID for single-patient extraction.",
    )
    parser.add_argument(
        "--prescribed-dose",
        type=float,
        default=None,
        help="Optional prescribed dose for normalization.",
    )
    parser.add_argument(
        "--mask-acceptance",
        type=float,
        default=None,
        help="Override lung mask acceptance ratio for patch extraction.",
    )
    parser.add_argument(
        "--lobe-composite-threshold",
        type=float,
        default=None,
        help="Override lobe composite threshold for patch extraction.",
    )
    parser.add_argument(
        "--max-patches-per-lobe",
        type=int,
        default=None,
        help="Override max patches per lobe for patch extraction.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Path to preprocessed images directory for a patient (skip re-segmentation).",
    )
    parser.add_argument(
        "--preprocessed-split",
        default="train",
        help="Split name for preprocessed cache lookup (default: train).",
    )
    parser.add_argument(
        "--segmentation-cache-dir",
        default=None,
        help="Directory to cache per-patient segmentations (skips reruns if present).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use existing per-patient cache if available (default: false).",
    )
    parser.add_argument(
        "--segmentation-device",
        default=None,
        help="Override segmentation device (e.g., gpu, gpu:1, cpu).",
    )
    parser.add_argument(
        "--h5-path",
        default=None,
        help="Path to processed H5 patches file (skip preprocessing and use cached patches).",
    )

    # Optional preprocessing hook
    parser.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Run scripts/preprocess.py before extraction.",
    )
    parser.add_argument(
        "--preprocess-config",
        default=None,
        help="Preprocess config YAML (defaults to config['preprocessing_config']).",
    )
    parser.add_argument(
        "--splits-json",
        default=None,
        help="Splits JSON file for preprocessing (required if --run-preprocess).",
    )
    parser.add_argument(
        "--preprocess-output",
        default=None,
        help="Output directory for preprocessing (defaults inferred from dataset config).",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Worker count for preprocessing.",
    )
    parser.add_argument(
        "--preprocess-n-patients",
        type=int,
        default=None,
        help="Number of patients to preprocess (defaults to full split size).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        with path.open("r") as handle:
            return json.load(handle)
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_preprocess_config(_config: Dict[str, Any], _args: argparse.Namespace) -> Dict[str, Any]:
    return dict(DEFAULT_PREPROCESS_CONFIG)

def infer_patient_id_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    match = re.search(r"\b(\d{10})\b", str(path))
    if match:
        return match.group(1)
    return None


def find_patient_cache(cache_root: Path, patient_id: str) -> Optional[Path]:
    for split_name in ("train", "val", "test"):
        candidate = cache_root / "processed_patches" / split_name / "patient_cache" / f"{patient_id}.h5"
        if candidate.exists():
            return candidate
    return None


def load_splits(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r") as handle:
        return json.load(handle)


def resolve_data_dir(config: Dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    return (
        dataset_cfg.get("data_root")
        or dataset_cfg.get("base_dir")
        or config.get("data", {}).get("root_dir")
        or "."
    )


def resolve_input_channels(config: Dict[str, Any]) -> List[str]:
    dataset_cfg = config.get("dataset", {})
    config_channels = dataset_cfg.get("input_channels") or dataset_cfg.get("modalities")
    config_mode = dataset_cfg.get("input_mode") or config.get("model", {}).get("input_mode")
    return _resolve_input_channels(
        config_channels=config_channels,
        config_mode=config_mode,
        default=["dose", "ct"],
    )


def resolve_h5_path(config: Dict[str, Any], split: str) -> Path:
    dataset_cfg = config.get("dataset", {})
    data_h5 = dataset_cfg.get("data_h5") or dataset_cfg.get("h5_path")
    base_dir = Path(resolve_data_dir(config))

    if isinstance(data_h5, dict):
        candidate = data_h5.get(split)
    else:
        candidate = data_h5

    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    dataset_type = str(dataset_cfg.get("dataset_type", "patches")).lower()
    subdir = "processed_patches" if dataset_type.startswith("patch") else "processed_images"
    return (base_dir / subdir / f"{split}.h5").resolve()


def infer_preprocess_output(config: Dict[str, Any], splits: Sequence[str]) -> Optional[Path]:
    for split in splits:
        h5_path = resolve_h5_path(config, split)
        if h5_path.parts:
            return h5_path.parent.parent
    return None


def run_preprocess(
    preprocess_config: Path,
    splits_json: Path,
    output_dir: Path,
    experiment_type: str,
    workers: int,
    n_patients: Optional[int],
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess.py"),
        "--config",
        str(preprocess_config),
        "--splits",
        str(splits_json),
        "--output",
        str(output_dir),
        "--experiment_type",
        experiment_type,
        "--workers",
        str(workers),
    ]
    if n_patients is not None:
        cmd.extend(["--n_patients", str(n_patients)])
    LOGGER.info("Running preprocessing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_model_config_consistency(config: Dict[str, Any], dataset) -> None:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.setdefault("model", {})

    dataset_is_2d = dataset_cfg.get("is_2d")
    model_is_2d = model_cfg.get("is_2d")
    is_2d = bool(dataset_is_2d if dataset_is_2d is not None else model_is_2d)
    model_cfg["is_2d"] = is_2d

    inferred_channels = getattr(dataset, "input_channels", None)
    if isinstance(inferred_channels, (list, tuple)):
        inferred_channels = len(inferred_channels)
    if isinstance(inferred_channels, int) and inferred_channels > 0:
        model_cfg.setdefault("in_channels", inferred_channels)
    model_cfg.setdefault("output_channels", 1)


def ensure_model_config_for_channels(config: Dict[str, Any], input_channels: Sequence[str]) -> None:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.setdefault("model", {})
    model_cfg["is_2d"] = bool(dataset_cfg.get("is_2d", False))
    model_cfg["in_channels"] = len(input_channels)
    model_cfg.setdefault("output_channels", 1)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: Dict[str, torch.Tensor]

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    if all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys when loading checkpoint: %s", unexpected)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    device_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            device_batch[key] = value.to(device)
        else:
            device_batch[key] = value
    return device_batch


def _value_at_index(value: Any, idx: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value[idx]
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _to_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def build_metadata(batch: Dict[str, Any], idx: int, fallback_keys: Sequence[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    raw_meta = batch.get("metadata")
    if isinstance(raw_meta, list) and idx < len(raw_meta) and isinstance(raw_meta[idx], dict):
        meta.update(raw_meta[idx])

    for key in fallback_keys:
        if key in meta:
            continue
        value = batch.get(key)
        if value is None:
            continue
        meta[key] = _to_serializable(_value_at_index(value, idx))

    return _to_serializable(meta)


def select_input_tensor(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    for key in ("input", "dose", "ct", "dose_patches", "ct_patches"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _normalize_latent_size(latent_size: Optional[str]) -> Optional[str]:
    if latent_size is None:
        return None
    value = str(latent_size).strip().lower()
    if value == "all":
        return "all"
    if value.isdigit():
        return value
    return value


def extract_latent_from_outputs(
    outputs: Any,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    latent_size: Optional[str],
    save_attention: bool,
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
    latent_size = _normalize_latent_size(latent_size)
    attention = None

    if isinstance(outputs, dict):
        latent = outputs.get("latent")
        if save_attention and "attention_weights" in outputs:
            weights = outputs.get("attention_weights")
            if torch.is_tensor(weights):
                attention = weights.detach().cpu()
        if latent is None and hasattr(model, "get_latent_representation"):
            latent = model.get_latent_representation(input_tensor)
        if latent is not None:
            return latent, attention

    if isinstance(outputs, (tuple, list)):
        if save_attention and len(outputs) > 1 and torch.is_tensor(outputs[1]):
            attention = outputs[1].detach().cpu()
        for item in outputs:
            if isinstance(item, dict):
                multi_head = item
                if latent_size in (None, ""):
                    raise ValueError(
                        "Model returned multiple projection heads; pass --latent-size <dim> "
                        "or --latent-size all to select which embedding to save."
                    )
                if latent_size == "all":
                    return multi_head, attention
                if latent_size in multi_head:
                    return multi_head[latent_size], attention
                if str(latent_size) in multi_head:
                    return multi_head[str(latent_size)], attention
        # Fallback to a latent tensor in tuple
        for item in outputs:
            if torch.is_tensor(item):
                return item, attention

    if hasattr(model, "get_latent_representation"):
        latent = model.get_latent_representation(input_tensor)
        return latent, attention

    raise ValueError("Model did not return latent representations.")


def assemble_input_tensor_batch(
    channels: Sequence[str],
    ct_tensor: torch.Tensor,
    dose_tensor: torch.Tensor,
    fusion_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    available: Dict[str, torch.Tensor] = {
        "ct": ct_tensor,
        "dose": dose_tensor,
    }
    if "fused" in channels:
        available["fused"] = _compute_fused_channel(ct_tensor, dose_tensor, fusion_cfg)
    tensors = [available[ch] for ch in channels]
    return torch.cat(tensors, dim=1)


def infer_patient_id(ct_path: Path) -> str:
    for part in ct_path.parts:
        if part.isdigit() and len(part) >= 8:
            return part
    return ct_path.stem


def load_preprocessed_patient(
    preprocessed_dir: Path, patient_id: str
) -> Tuple[sitk.Image, sitk.Image, Dict[str, sitk.Image]]:
    ct_path = preprocessed_dir / f"{patient_id}_ct_processed.nrrd"
    dose_path = preprocessed_dir / f"{patient_id}_dose_processed.nrrd"
    if not ct_path.exists() or not dose_path.exists():
        raise FileNotFoundError(f"Missing preprocessed CT/dose at {preprocessed_dir}")

    ct_image = sitk.ReadImage(str(ct_path))
    dose_image = sitk.ReadImage(str(dose_path))

    mask_suffixes = {
        "left_upper_lobe": "left_upper_lobe_mask",
        "left_lower_lobe": "left_lower_lobe_mask",
        "right_upper_lobe": "right_upper_lobe_mask",
        "right_middle_lobe": "right_middle_lobe_mask",
        "right_lower_lobe": "right_lower_lobe_mask",
        "left_lung": "left_lung_mask",
        "right_lung": "right_lung_mask",
        "ipsi_lung": "ipsi_lung_mask",
        "contra_lung": "contra_lung_mask",
        "aorta": "aorta_mask",
        "trachea": "trachea_mask",
    }

    lung_masks: Dict[str, sitk.Image] = {}
    for key, suffix in mask_suffixes.items():
        mask_path = preprocessed_dir / f"{patient_id}_{suffix}.nrrd"
        if mask_path.exists():
            lung_masks[key] = sitk.ReadImage(str(mask_path))

    return ct_image, dose_image, lung_masks


def load_h5_patient_indices(
    h5_path: Path, patient_id: str
) -> Tuple[List[int], List[Dict[str, Any]]]:
    with h5py.File(h5_path, "r") as f:
        metadata = None
        if "patch_metadata" in f:
            metadata_raw = f["patch_metadata"][()]
            try:
                metadata = json.loads(metadata_raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"Failed to decode patch_metadata from {h5_path}: {exc}") from exc
        elif "metadata" in f:
            metadata_raw = f["metadata"][()]
            try:
                metadata = json.loads(metadata_raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"Failed to decode metadata from {h5_path}: {exc}") from exc
        else:
            raise KeyError("H5 file missing patch_metadata or metadata datasets.")

        file_patient_id = None
        try:
            file_patient_id = f.attrs.get("patient_id")
            if isinstance(file_patient_id, (bytes, bytearray)):
                file_patient_id = file_patient_id.decode("utf-8")
        except Exception:
            file_patient_id = None

    indices: List[int] = []
    entries: List[Dict[str, Any]] = []
    if file_patient_id and file_patient_id == patient_id:
        for idx, entry in enumerate(metadata):
            indices.append(entry.get("patch_id", idx))
            entries.append(entry)
        return indices, entries

    for idx, entry in enumerate(metadata):
        if entry.get("patient_id") == patient_id or file_patient_id is None:
            indices.append(entry.get("patch_id", idx))
            entries.append(entry)

    return indices, entries


def resolve_patient_id_for_h5(h5_path: Path, provided: Optional[str]) -> str:
    if provided:
        return provided
    try:
        with h5py.File(h5_path, "r") as f:
            file_patient_id = f.attrs.get("patient_id")
            if isinstance(file_patient_id, (bytes, bytearray)):
                file_patient_id = file_patient_id.decode("utf-8")
            if file_patient_id:
                return str(file_patient_id)
    except Exception:
        pass
    raise ValueError("--patient-id is required when using a multi-patient H5.")


def extract_latents_from_h5_patient(
    model: torch.nn.Module,
    h5_path: Path,
    patient_id: str,
    input_channels: Sequence[str],
    fusion_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    limit: Optional[int],
    save_attention: bool,
    latent_size: Optional[str],
    model_in_channels: Optional[int],
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], List[Dict[str, Any]], Optional[List[torch.Tensor]]]:
    indices, metadata = load_h5_patient_indices(h5_path, patient_id)
    if not indices:
        raise ValueError(f"No patches found for patient {patient_id} in {h5_path}")

    if limit is not None:
        indices = indices[:limit]
        metadata = metadata[:limit]

    latents: List[torch.Tensor] = []
    latents_by_size: Dict[str, List[torch.Tensor]] = {}
    attention: Optional[List[torch.Tensor]] = [] if save_attention else None

    with h5py.File(h5_path, "r") as f:
        ct_patches = f.get("ct_patches")
        dose_patches = f.get("dose_patches")
        spatial_coords = f.get("spatial_coords")

        if ct_patches is None or dose_patches is None:
            raise ValueError("H5 file missing ct_patches or dose_patches datasets.")

        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                ct_batch = np.asarray(ct_patches[batch_idx], dtype=np.float32)
                dose_batch = np.asarray(dose_patches[batch_idx], dtype=np.float32)

                ct_tensor = torch.from_numpy(ct_batch).float().unsqueeze(1).to(device)
                dose_tensor = torch.from_numpy(dose_batch).float().unsqueeze(1).to(device)
                input_tensor = assemble_input_tensor_batch(input_channels, ct_tensor, dose_tensor, fusion_cfg)
                if model_in_channels and input_tensor.shape[1] != model_in_channels:
                    if model_in_channels == 3:
                        input_tensor = assemble_input_tensor_batch(
                            ["dose", "ct", "fused"], ct_tensor, dose_tensor, fusion_cfg
                        )
                    elif model_in_channels == 2:
                        input_tensor = assemble_input_tensor_batch(
                            ["dose", "ct"], ct_tensor, dose_tensor, fusion_cfg
                        )
                    elif model_in_channels == 1:
                        input_tensor = assemble_input_tensor_batch(
                            ["dose"], ct_tensor, dose_tensor, fusion_cfg
                        )
                if model_in_channels and input_tensor.shape[1] != model_in_channels:
                    raise ValueError(
                        f"Input channel mismatch: model expects {model_in_channels}, "
                        f"but input has {input_tensor.shape[1]} channels. "
                        f"Check config.dataset.input_channels and model.in_channels."
                    )

                payload: Dict[str, torch.Tensor] = {
                    "ct": ct_tensor,
                    "dose": dose_tensor,
                }
                if spatial_coords is not None:
                    coords = np.asarray(spatial_coords[batch_idx], dtype=np.float32)
                    payload["spatial_coords"] = torch.from_numpy(coords).to(device)

                outputs = model(input_tensor, **payload)
                latent, attn = extract_latent_from_outputs(
                    outputs=outputs,
                    model=model,
                    input_tensor=input_tensor,
                    latent_size=latent_size,
                    save_attention=save_attention,
                )
                if attn is not None and attention is not None:
                    attention.append(attn)

                if isinstance(latent, dict):
                    for key, value in latent.items():
                        if torch.is_tensor(value):
                            latents_by_size.setdefault(str(key), []).append(value.detach().cpu())
                else:
                    latents.append(latent.detach().cpu())

    if latents_by_size:
        return {k: torch.cat(v, dim=0) for k, v in latents_by_size.items()}, metadata, attention
    if not latents:
        raise ValueError("No latents extracted from H5 data.")
    return torch.cat(latents, dim=0), metadata, attention


def flatten_patch_entries(patches_data: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for side_type in ("ipsilateral", "contralateral"):
        side_dict = patches_data.get(side_type, {})
        for lobe_name, patches in side_dict.items():
            for patch in patches:
                entry = dict(patch)
                entry.setdefault("patient_id", patient_id)
                entry.setdefault("side_type", side_type)
                entry.setdefault("lobe_name", lobe_name)
                entries.append(entry)
    return entries


def build_patch_batch(
    entries: Sequence[Dict[str, Any]],
    ct_array: np.ndarray,
    dose_array: np.ndarray,
    patch_size: int,
    input_channels: Sequence[str],
    fusion_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    input_tensors: List[torch.Tensor] = []
    ct_tensors: List[torch.Tensor] = []
    dose_tensors: List[torch.Tensor] = []
    spatial_coords: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []

    for entry in entries:
        coords = entry.get("start_coords") or entry.get("coordinates")
        if coords is None:
            raise ValueError("Patch entry missing start coordinates.")
        z, y, x = [int(v) for v in coords]

        ct_patch = ct_array[z:z + patch_size, y:y + patch_size, x:x + patch_size]
        dose_patch = dose_array[z:z + patch_size, y:y + patch_size, x:x + patch_size]
        if ct_patch.shape != (patch_size, patch_size, patch_size):
            continue
        if dose_patch.shape != (patch_size, patch_size, patch_size):
            continue

        ct_tensor = torch.from_numpy(ct_patch).float().unsqueeze(0)
        dose_tensor = torch.from_numpy(dose_patch).float().unsqueeze(0)
        input_tensor, _ = _assemble_input_tensor(
            input_channels, ct_tensor, dose_tensor, fusion_cfg
        )

        input_tensors.append(input_tensor)
        ct_tensors.append(ct_tensor)
        dose_tensors.append(dose_tensor)

        spatial = entry.get("spatial_coord") or entry.get("start_coords") or [0, 0, 0]
        spatial_coords.append([float(v) for v in spatial])
        metadata.append(_to_serializable(entry))

    if not input_tensors:
        raise ValueError("No valid patches assembled for latent extraction.")

    input_batch = torch.stack(input_tensors).to(device)
    batch_payload: Dict[str, torch.Tensor] = {
        "ct": torch.stack(ct_tensors).to(device),
        "dose": torch.stack(dose_tensors).to(device),
    }
    if spatial_coords:
        batch_payload["spatial_coords"] = torch.tensor(
            spatial_coords, dtype=torch.float32, device=device
        )
    return input_batch, batch_payload, metadata


def extract_latents_from_patch_entries(
    model: torch.nn.Module,
    patch_entries: Sequence[Dict[str, Any]],
    ct_array: np.ndarray,
    dose_array: np.ndarray,
    patch_size: int,
    input_channels: Sequence[str],
    fusion_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    limit: Optional[int],
    save_attention: bool,
    latent_size: Optional[str],
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], List[Dict[str, Any]], Optional[List[torch.Tensor]]]:
    latents: List[torch.Tensor] = []
    latents_by_size: Dict[str, List[torch.Tensor]] = {}
    metadata: List[Dict[str, Any]] = []
    attention: Optional[List[torch.Tensor]] = [] if save_attention else None

    total = len(patch_entries)
    if limit is not None:
        total = min(total, limit)

    processed = 0
    model.eval()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = patch_entries[start:end]

            input_batch, payload, meta = build_patch_batch(
                chunk,
                ct_array,
                dose_array,
                patch_size,
                input_channels,
                fusion_cfg,
                device,
            )
            outputs = model(input_batch, **payload)
            latent, attn = extract_latent_from_outputs(
                outputs=outputs,
                model=model,
                input_tensor=input_batch,
                latent_size=latent_size,
                save_attention=save_attention,
            )
            if attn is not None and attention is not None:
                attention.append(attn)

            if isinstance(latent, dict):
                for key, value in latent.items():
                    if not torch.is_tensor(value):
                        continue
                    latents_by_size.setdefault(str(key), []).append(value.detach().cpu())
            else:
                latents.append(latent.detach().cpu())
            metadata.extend(meta)
            processed += len(meta)

    if not latents and not latents_by_size:
        raise ValueError("No latents extracted; dataset may be empty.")

    if latents_by_size:
        return {k: torch.cat(v, dim=0) for k, v in latents_by_size.items()}, metadata, attention
    return torch.cat(latents, dim=0), metadata, attention


def extract_latents_from_patient(
    config: Dict[str, Any],
    preproc_config: Dict[str, Any],
    ct_path: Path,
    dose_path: Path,
    patient_id: str,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    limit: Optional[int],
    save_attention: bool,
    latent_size: Optional[str],
    preprocess_output: Path,
    prescribed_dose: Optional[float],
    preprocessed_dir: Optional[Path],
    mask_acceptance: Optional[float],
    lobe_composite_threshold: Optional[float],
    max_patches_per_lobe: Optional[int],
    segmentation_cache_dir: Optional[str],
) -> None:
    preprocessor = LungPreprocessor(preproc_config, str(preprocess_output))
    if segmentation_cache_dir:
        preprocessor.segmentation_cache_dir = segmentation_cache_dir
    else:
        preprocessor.segmentation_cache_dir = str(output_dir / "segmentation" / patient_id)
    if mask_acceptance is not None:
        preprocessor.lobe_acceptance = float(mask_acceptance)
    if lobe_composite_threshold is not None:
        preprocessor.lobe_composite_threshold = float(lobe_composite_threshold)
    if max_patches_per_lobe is not None:
        preprocessor.max_patches_per_lobe = int(max_patches_per_lobe)
    if preprocessed_dir is not None:
        ct_image, dose_image, lung_masks = load_preprocessed_patient(preprocessed_dir, patient_id)
        if "ipsi_lung" not in lung_masks or "contra_lung" not in lung_masks:
            if "left_lung" in lung_masks and "right_lung" in lung_masks:
                ipsi, contra = preprocessor._determine_ipsi_contra_lungs(
                    lung_masks["left_lung"], lung_masks["right_lung"], dose_image
                )
                lung_masks.setdefault("ipsi_lung", ipsi)
                lung_masks.setdefault("contra_lung", contra)
        patches_data = preprocessor.extract_patches(ct_image, dose_image, lung_masks, patient_id)
        result = {
            "patches_data": patches_data,
            "ct_image": ct_image,
            "dose_image": dose_image,
        }
    else:
        result = preprocessor.process_patient(
            patient_id=patient_id,
            ct_path=str(ct_path),
            dose_path=str(dose_path),
            prescribed_dose=prescribed_dose,
            split_name="single",
            experiment_type="patch",
            retain_patch_arrays=True,
            save_to_cache=False,
            save_visualizations=False,
        )
        patches_data = result.get("patches_data", {})
    patch_entries = flatten_patch_entries(patches_data, patient_id)
    if not patch_entries:
        raise ValueError("No patches extracted for patient; check preprocessing output.")

    ct_array = patches_data.get("ct_array")
    dose_array = patches_data.get("dose_array")
    if ct_array is None or dose_array is None:
        ct_array = sitk.GetArrayFromImage(result["ct_image"]).astype(np.float32)
        dose_array = sitk.GetArrayFromImage(result["dose_image"]).astype(np.float32)
    else:
        ct_array = np.asarray(ct_array, dtype=np.float32)
        dose_array = np.asarray(dose_array, dtype=np.float32)

    input_channels = resolve_input_channels(config)
    ensure_model_config_for_channels(config, input_channels)

    model = get_model(config)
    model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    fusion_cfg = config.get("dataset", {}).get("fusion", {})
    patch_size = int(preproc_config.get("lung", {}).get("patch_size", 50))

    LOGGER.info("Extracting latents from %d patches for patient %s", len(patch_entries), patient_id)
    latents, metadata, attention = extract_latents_from_patch_entries(
        model=model,
        patch_entries=patch_entries,
        ct_array=ct_array,
        dose_array=dose_array,
        patch_size=patch_size,
        input_channels=input_channels,
        fusion_cfg=fusion_cfg,
        device=device,
        batch_size=batch_size,
        limit=limit,
        save_attention=save_attention,
        latent_size=latent_size,
    )

    trial_tag = f"trial{config.get('trial_num')}" if "trial_num" in config else checkpoint_path.stem
    output_payload: Dict[str, Any] = {
        "embeddings": latents,
        "metadata": metadata,
        "encoder": f"doseae_{trial_tag}",
        "patient_id": patient_id,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path),
        "ct_path": str(ct_path),
        "dose_path": str(dose_path),
    }
    if isinstance(latents, dict):
        output_payload["latent_sizes"] = sorted(latents.keys())
    if attention:
        output_payload["attention_weights"] = attention

    output_path = output_dir / f"doseae_{trial_tag}_{patient_id}_latents.pt"
    torch.save(output_payload, output_path)
    LOGGER.info("Saved latents to %s", output_path)


def extract_latents_for_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit: Optional[int],
    save_attention: bool,
    latent_size: Optional[str],
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], List[Dict[str, Any]], Optional[List[torch.Tensor]]]:
    latents: List[torch.Tensor] = []
    latents_by_size: Dict[str, List[torch.Tensor]] = {}
    metadata: List[Dict[str, Any]] = []
    attention: Optional[List[torch.Tensor]] = [] if save_attention else None

    fallback_keys = [
        "patient_id",
        "index",
        "side_type",
        "lobe_name",
        "anatomical_side",
        "lobe_index",
        "side_type_index",
        "anatomical_side_index",
        "slice_index",
        "patch_index",
        "spatial_coords",
    ]

    processed = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if limit is not None and processed >= limit:
                break

            batch = move_batch_to_device(batch, device)
            input_tensor = select_input_tensor(batch)
            if input_tensor is None:
                raise ValueError("No suitable input tensor found in batch.")

            outputs = model(input_tensor, **batch)
            latent, attn = extract_latent_from_outputs(
                outputs=outputs,
                model=model,
                input_tensor=input_tensor,
                latent_size=latent_size,
                save_attention=save_attention,
            )
            if attn is not None and attention is not None:
                attention.append(attn)

            if isinstance(latent, dict):
                first_tensor = None
                for value in latent.values():
                    if torch.is_tensor(value):
                        first_tensor = value
                        break
                if first_tensor is None:
                    raise ValueError("Model returned projection heads without tensors.")

                batch_size = first_tensor.shape[0]
                keep = batch_size
                if limit is not None and processed + batch_size > limit:
                    keep = max(0, limit - processed)

                for key, value in latent.items():
                    if not torch.is_tensor(value):
                        continue
                    value = value.detach().cpu()[:keep]
                    latents_by_size.setdefault(str(key), []).append(value)
            else:
                latent = latent.detach().cpu()
                batch_size = latent.shape[0]
                keep = batch_size
                if limit is not None and processed + batch_size > limit:
                    keep = max(0, limit - processed)
                    latent = latent[:keep]

                latents.append(latent)

            for idx in range(keep):
                metadata.append(build_metadata(batch, idx, fallback_keys))

            processed += keep

    if not latents and not latents_by_size:
        raise ValueError("No latents extracted; dataset may be empty.")

    if latents_by_size:
        return {k: torch.cat(v, dim=0) for k, v in latents_by_size.items()}, metadata, attention
    return torch.cat(latents, dim=0), metadata, attention


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.checkpoint is None and args.weights is not None:
        args.checkpoint = args.weights

    dataset_cfg = config.get("dataset", {})
    dataset_type = str(dataset_cfg.get("dataset_type", "patches")).lower()
    experiment_type = "patch" if dataset_type.startswith("patch") else "image"

    single_patient_mode = bool(args.ct_path or args.dose_path or args.h5_path)
    if single_patient_mode:
        if args.h5_path is None and (not args.ct_path or not args.dose_path):
            raise ValueError("Both --ct-path and --dose-path are required unless --h5-path is provided.")

        preproc_config = resolve_preprocess_config(config, args)
        if args.segmentation_device:
            preproc_config.setdefault("segmentation", {})["device"] = args.segmentation_device
        ct_path = Path(args.ct_path) if args.ct_path else None
        dose_path = Path(args.dose_path) if args.dose_path else None
        patient_id = args.patient_id or (infer_patient_id(ct_path) if ct_path else None)
        if not patient_id:
            patient_id = infer_patient_id_from_path(str(ct_path) if ct_path else None)

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
        if checkpoint_path is None:
            trial_num = config.get("trial_num")
            if trial_num is not None:
                candidate = PROJECT_ROOT / f"best_model_trial_{trial_num}.pth"
                if candidate.exists():
                    checkpoint_path = candidate
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("Checkpoint not found; please provide --checkpoint.")

        batch_size = args.batch_size
        if batch_size is None:
            batch_size = int(config.get("training", {}).get("batch_size", 1) or 1)

        preprocess_output = Path(args.preprocess_output) if args.preprocess_output else output_dir / "preprocess_cache"
        preprocess_output.mkdir(parents=True, exist_ok=True)

        if args.use_cache and not args.h5_path and ct_path and dose_path and patient_id:
            cache_root = preproc_config.get("output", {}).get("cache_dir")
            if cache_root:
                cache_path = find_patient_cache(Path(cache_root), patient_id)
                if cache_path:
                    LOGGER.info("Using cached patch H5 for patient %s: %s", patient_id, cache_path)
                    args.h5_path = str(cache_path)

        if args.h5_path:
            patient_id = resolve_patient_id_for_h5(Path(args.h5_path), patient_id)
            input_channels = resolve_input_channels(config)
            ensure_model_config_for_channels(config, input_channels)
            model = get_model(config)
            model.to(device)
            load_checkpoint(model, checkpoint_path, device)
            fusion_cfg = config.get("dataset", {}).get("fusion", {})
            model_in_channels = config.get("model", {}).get("in_channels")
            latents, metadata, attention = extract_latents_from_h5_patient(
                model=model,
                h5_path=Path(args.h5_path),
                patient_id=patient_id,
                input_channels=input_channels,
                fusion_cfg=fusion_cfg,
                device=device,
                batch_size=batch_size,
                limit=args.limit,
                save_attention=args.save_attention,
                latent_size=args.latent_size,
                model_in_channels=model_in_channels,
            )

            trial_tag = f"trial{config.get('trial_num')}" if "trial_num" in config else checkpoint_path.stem
            output_payload: Dict[str, Any] = {
                "embeddings": latents,
                "metadata": metadata,
                "encoder": f"doseae_{trial_tag}",
                "patient_id": patient_id,
                "checkpoint": str(checkpoint_path),
                "config_path": str(config_path),
                "h5_path": str(args.h5_path),
            }
            if isinstance(latents, dict):
                output_payload["latent_sizes"] = sorted(latents.keys())
            if attention:
                output_payload["attention_weights"] = attention

            output_path = output_dir / f"doseae_{trial_tag}_{patient_id}_latents.pt"
            torch.save(output_payload, output_path)
            LOGGER.info("Saved latents to %s", output_path)
        else:
            extract_latents_from_patient(
                config=config,
                preproc_config=preproc_config,
                ct_path=ct_path,
                dose_path=dose_path,
                patient_id=patient_id,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                output_dir=output_dir,
                batch_size=batch_size,
                device=device,
                limit=args.limit,
                save_attention=args.save_attention,
                latent_size=args.latent_size,
                preprocess_output=preprocess_output,
                prescribed_dose=args.prescribed_dose,
                preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None,
                mask_acceptance=args.mask_acceptance,
                lobe_composite_threshold=args.lobe_composite_threshold,
                max_patches_per_lobe=args.max_patches_per_lobe,
                segmentation_cache_dir=args.segmentation_cache_dir,
            )
        return

    if args.run_preprocess:
        if args.splits_json is None:
            raise ValueError("--splits-json is required when --run-preprocess is set.")
        preprocess_cfg = resolve_preprocess_config(config, args)
        preprocess_cfg_path = Path(args.preprocess_config) if args.preprocess_config else None
        splits_json_path = Path(args.splits_json)
        output_dir = (
            Path(args.preprocess_output)
            if args.preprocess_output
            else infer_preprocess_output(config, args.splits)
        )
        if output_dir is None:
            raise ValueError("Could not infer preprocess output dir; set --preprocess-output.")

        n_patients = args.preprocess_n_patients
        if n_patients is None:
            splits_data = load_splits(splits_json_path)
            n_patients = max((len(v) for v in splits_data.values() if isinstance(v, list)), default=0)

        run_preprocess(
            preprocess_config=preprocess_cfg_path or (PROJECT_ROOT / "config" / "new_pipeline_config.yaml"),
            splits_json=splits_json_path,
            output_dir=output_dir,
            experiment_type=experiment_type,
            workers=args.preprocess_workers,
            n_patients=n_patients,
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(config)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(dataset_cfg.get("num_workers", 0) or 0)
    pin_memory = bool(dataset_cfg.get("pin_memory", False))

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        trial_num = config.get("trial_num")
        if trial_num is not None:
            candidate = PROJECT_ROOT / f"best_model_trial_{trial_num}.pth"
            if candidate.exists():
                checkpoint_path = candidate

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found; please provide --checkpoint.")

    for split in args.splits:
        dataset = create_dataset(
            config=config,
            entity_type=config.get("entity", "lung"),
            split=split,
            data_dir=data_dir,
            transform=None,
            apply_transforms=False,
        )
        ensure_model_config_consistency(config, dataset)

        model = get_model(config)
        model.to(device)
        load_checkpoint(model, checkpoint_path, device)

        batch_size = args.batch_size
        if batch_size is None:
            batch_size = int(config.get("training", {}).get("batch_size", 1) or 1)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None) or collate_medical_batch,
        )

        LOGGER.info("Extracting latents for split '%s' (%d samples)", split, len(dataset))
        latents, metadata, attention = extract_latents_for_split(
            model,
            loader,
            device,
            limit=args.limit,
            save_attention=args.save_attention,
            latent_size=args.latent_size,
        )

        trial_tag = None
        if "trial_num" in config:
            trial_tag = f"trial{config['trial_num']}"
        if trial_tag is None:
            trial_tag = checkpoint_path.stem

        output_payload: Dict[str, Any] = {
            "embeddings": latents,
            "metadata": metadata,
            "encoder": f"doseae_{trial_tag}",
            "split": split,
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
        }
        if isinstance(latents, dict):
            output_payload["latent_sizes"] = sorted(latents.keys())
        if attention:
            output_payload["attention_weights"] = attention

        output_path = output_dir / f"doseae_{trial_tag}_{split}_latents.pt"
        torch.save(output_payload, output_path)
        LOGGER.info("Saved latents to %s", output_path)


if __name__ == "__main__":
    main()
