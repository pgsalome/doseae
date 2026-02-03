#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the test split and push metrics to Weights & Biases.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import copy

import torch
from torch.utils.data import DataLoader
import yaml

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.loaders import create_dataset, _LimitedDataset, collate_medical_batch
from entities.lung.datasets.dataset import PatchDataset
from models import get_model
from core.training.trainer import Trainer
from utils.clinical_metrics import ClinicalMetricsCalculator
from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor, STANDARD_LOBE_NAMES
import numpy as np
import copy


def prepare_wandb_metadata(config: Dict[str, Any], entity_type: str) -> None:
    wandb_cfg = config.setdefault("wandb", {})
    if not wandb_cfg.get("use_wandb", False):
        return

    model_cfg = config.get("model", {})
    hyper_cfg = config.get("hyperparameters", {})
    training_cfg = config.get("training", {})

    project_default = f"doseae_{entity_type}"
    wandb_cfg.setdefault("project_name", project_default)

    model_type = model_cfg.get("type", "unknown")
    base_filters = model_cfg.get("base_filters", "")
    latent_dim = model_cfg.get("latent_dim", "")
    learning_rate = hyper_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-4))
    batch_size = hyper_cfg.get("batch_size", training_cfg.get("batch_size", 1))
    optimizer = training_cfg.get("optimizer", "adam")
    trial_number = wandb_cfg.get("trial_number")

    if wandb_cfg.get("run_name"):
        run_name = wandb_cfg["run_name"]
    else:
        if isinstance(learning_rate, (int, float)):
            lr_str = f"{learning_rate:.0e}"
        else:
            lr_str = str(learning_rate)
        prefix = f"{entity_type}_{model_type}"
        suffix = f"f{base_filters}_l{latent_dim}_lr{lr_str}_bs{batch_size}_{optimizer}"
        if trial_number is not None:
            run_name = f"{prefix}_trial_{int(trial_number):02d}_{suffix}"
        else:
            run_name = f"{prefix}_{suffix}"
        wandb_cfg["run_name"] = run_name

    tags = set(wandb_cfg.get("tags", []))
    tags.add(entity_type)
    tags.add(model_type)
    if base_filters:
        tags.add(f"filters_{base_filters}")
    if latent_dim:
        tags.add(f"latent_{latent_dim}")
    tags.add(f"optimizer_{optimizer}")
    wandb_cfg["tags"] = sorted(tags)
    wandb_cfg.setdefault("group", f"{entity_type}_training")
    wandb_cfg.setdefault("watch", {"log": "all", "log_freq": 100})


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def load_splits(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def summarize_values(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_volume_metrics(
    calculator: ClinicalMetricsCalculator,
    target_volume: np.ndarray,
    recon_volume: np.ndarray,
    mask: Optional[np.ndarray],
) -> Dict[str, float]:
    if mask is not None:
        bool_mask = mask.astype(bool)
        if bool_mask.sum() == 0:
            bool_mask = None
    else:
        bool_mask = None

    if bool_mask is not None:
        target = target_volume[bool_mask]
        recon = recon_volume[bool_mask]
    else:
        target = target_volume
        recon = recon_volume

    metrics = {
        "mse": float(np.mean((recon - target) ** 2)),
        "mae": float(np.mean(np.abs(recon - target))),
    }

    try:
        dvh_ref = calculator.calculate_dvh_metrics(target_volume, bool_mask)
        dvh_eval = calculator.calculate_dvh_metrics(recon_volume, bool_mask)
        for key, ref_val in dvh_ref.items():
            eval_val = dvh_eval.get(key)
            if eval_val is None:
                continue
            metrics[f"{key}_ref"] = float(ref_val)
            metrics[f"{key}_eval"] = float(eval_val)
            diff = abs(ref_val - eval_val)
            metrics[f"{key}_diff"] = float(diff)
            if ref_val != 0:
                metrics[f"{key}_rel_diff"] = float(diff / ref_val * 100.0)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).debug("DVH computation failed: %s", exc)

    return metrics


def convert_masks_to_numpy(lung_masks: Dict[str, "sitk.Image"]) -> Dict[str, np.ndarray]:
    import SimpleITK as sitk

    masks_np: Dict[str, np.ndarray] = {}
    for name, mask in lung_masks.items():
        masks_np[name] = sitk.GetArrayFromImage(mask).astype(bool)
    return masks_np


def build_patient_bbox(dataset, indices: List[int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    patch_shape = dataset.patch_shape
    min_coords = [10 ** 9, 10 ** 9, 10 ** 9]
    max_coords = [0, 0, 0]
    for idx in indices:
        meta = dataset._metadata_for_index(idx)
        z, y, x = [int(v) for v in meta['start_coords']]
        if z < min_coords[0]:
            min_coords[0] = z
        if y < min_coords[1]:
            min_coords[1] = y
        if x < min_coords[2]:
            min_coords[2] = x
        if z > max_coords[0]:
            max_coords[0] = z
        if y > max_coords[1]:
            max_coords[1] = y
        if x > max_coords[2]:
            max_coords[2] = x
    min_coords = tuple(int(v) for v in min_coords)
    volume_shape = tuple(int(max_coords[i] - min_coords[i] + patch_shape[i]) for i in range(3))
    return min_coords, volume_shape


def reconstruct_from_cached_dataset(
    trainer: Trainer,
    dataset,
    indices: List[int],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    min_coords: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not indices:
        return (
            np.zeros(volume_shape, dtype=np.float32),
            np.zeros(volume_shape, dtype=np.float32),
            np.zeros(volume_shape, dtype=np.uint16),
        )

    trainer.model.eval()

    recon_volume = np.zeros(volume_shape, dtype=np.float32)
    gt_volume = np.zeros(volume_shape, dtype=np.float32)
    vote_volume = np.zeros(volume_shape, dtype=np.uint16)

    def offset_coords(coords):
        return tuple(int(coords[i] - min_coords[i]) for i in range(3))

    total = len(indices)
    for start in range(0, total, batch_size):
        chunk = indices[start:start + batch_size]
        samples = [dataset[idx] for idx in chunk]
        inputs = torch.stack([sample['input'] for sample in samples]).to(trainer.device)

        with torch.no_grad():
            raw_outputs = trainer.model(inputs)
            outputs = trainer._standardize_outputs(raw_outputs)
        preds = outputs['reconstruction'].detach().cpu().numpy()
        targets = torch.stack([sample['dose'] for sample in samples]).numpy()

        for sample, pred_patch, target_patch in zip(samples, preds, targets):
            z, y, x = offset_coords(sample['metadata']['start_coords'])
            dz, dy, dx = patch_shape
            recon_volume[z:z + dz, y:y + dy, x:x + dx] += pred_patch[0]
            gt_volume[z:z + dz, y:y + dy, x:x + dx] += target_patch[0]
            vote_volume[z:z + dz, y:y + dy, x:x + dx] += 1

    nonzero_mask = vote_volume > 0
    recon_volume[nonzero_mask] /= vote_volume[nonzero_mask].astype(np.float32)
    gt_volume[nonzero_mask] /= vote_volume[nonzero_mask].astype(np.float32)

    return recon_volume, gt_volume, vote_volume


def save_mask_cache(cache_path: Path, mask_dict: Dict[str, np.ndarray], min_coords: Tuple[int, int, int], volume_shape: Tuple[int, int, int]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {name: arr.astype(np.uint8) for name, arr in mask_dict.items()}
    save_payload['_min_coords'] = np.array(min_coords, dtype=np.int32)
    save_payload['_volume_shape'] = np.array(volume_shape, dtype=np.int32)
    np.savez_compressed(cache_path, **save_payload)


def load_mask_cache(cache_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[int, int, int], Tuple[int, int, int]]:
    data = np.load(cache_path)
    min_coords = tuple(int(v) for v in data['_min_coords'])
    volume_shape = tuple(int(v) for v in data['_volume_shape'])
    mask_dict = {key: data[key].astype(bool) for key in data.files if not key.startswith('_')}
    return mask_dict, min_coords, volume_shape


def resolve_cache_root(config: Dict[str, Any]) -> Path:
    dataset_cfg = config.get('dataset', {})
    data_h5 = dataset_cfg.get('data_h5')
    path: Optional[Path] = None
    if isinstance(data_h5, dict):
        for key in ('train', 'val', 'test'):
            candidate = data_h5.get(key)
            if candidate:
                path = Path(candidate)
                break
    elif isinstance(data_h5, str):
        path = Path(data_h5)
    if path is None:
        raise ValueError("Unable to resolve data_h5 path for cache root")
    if path.suffix:
        return path.parent.parent
    return path.parent


def get_or_compute_masks(
    patient: Dict[str, Any],
    config: Dict[str, Any],
    results_dir: Path,
    mask_cache_dir: Path,
    processed_images_dir: Path,
    min_coords: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    preprocessor_holder: Dict[str, Optional[LungPreprocessor]],
) -> Dict[str, np.ndarray]:
    patient_id = patient['patient_id']
    cache_path = mask_cache_dir / f"{patient_id}.npz"

    if cache_path.exists():
        mask_dict, cached_min, cached_shape = load_mask_cache(cache_path)
        if cached_min == min_coords and cached_shape == volume_shape:
            return mask_dict

    def crop_mask(arr: np.ndarray) -> np.ndarray:
        z0, y0, x0 = min_coords
        dz, dy, dx = volume_shape
        cropped = np.zeros(volume_shape, dtype=bool)
        z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
        arr_z, arr_y, arr_x = arr.shape
        z_start = max(z0, 0)
        y_start = max(y0, 0)
        x_start = max(x0, 0)
        z_end = min(z1, arr_z)
        y_end = min(y1, arr_y)
        x_end = min(x1, arr_x)
        if z_end <= z_start or y_end <= y_start or x_end <= x_start:
            return cropped
        cropped[z_start - z0:z_end - z0, y_start - y0:y_end - y0, x_start - x0:x_end - x0] = arr[z_start:z_end, y_start:y_end, x_start:x_end]
        return cropped

    import SimpleITK as sitk
    mask_dict: Dict[str, np.ndarray] = {}
    patient_dir = processed_images_dir / patient_id
    if patient_dir.exists():
        mask_names = {
            'left_upper_lobe',
            'left_lower_lobe',
            'right_upper_lobe',
            'right_middle_lobe',
            'right_lower_lobe',
            'left_lung',
            'right_lung',
            'ipsi_lung',
            'contra_lung',
        }
        for name in mask_names:
            path = patient_dir / f"{patient_id}_{name}_mask.nrrd"
            if not path.exists():
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool)
            mask_dict[name] = crop_mask(arr)

    if mask_dict:
        save_mask_cache(cache_path, mask_dict, min_coords, volume_shape)
        return mask_dict

    # Fallback: regenerate via preprocessor
    if preprocessor_holder.get('instance') is None:
        preprocessor_holder['instance'] = LungPreprocessor(config, str(results_dir))
    preprocessor = preprocessor_holder['instance']
    result = preprocessor.process_patient(
        patient_id=patient_id,
        ct_path=patient['ct_path'],
        dose_path=patient['dose_path'],
        prescribed_dose=patient.get('prescribed_dose'),
        split_name='test',
        experiment_type='image',
        retain_patch_arrays=False,
        save_to_cache=False,
        save_visualizations=False,
    )

    lung_masks = convert_masks_to_numpy(result['lung_masks'])
    regenerated: Dict[str, np.ndarray] = {}
    for name, arr in lung_masks.items():
        regenerated[name] = crop_mask(arr)

    if regenerated:
        save_mask_cache(cache_path, regenerated, min_coords, volume_shape)
    return regenerated


def build_test_loader(
    config: Dict[str, Any],
    entity: str,
    data_dir: Path,
    disable_test_mode: bool,
) -> DataLoader:
    dataset_cfg = config.get("dataset", {})
    test_mode_enabled = bool(dataset_cfg.get("test_mode", False)) and not disable_test_mode
    test_dataset = create_dataset(
        config,
        entity,
        "test",
        str(data_dir),
        transform=None,
        apply_transforms=False,
    )

    batch_size = int(config.get("training", {}).get("batch_size", 1) or 1)
    if test_mode_enabled:
        limit_batches = int(dataset_cfg.get("n_test_samples", dataset_cfg.get("n_test_batches", 20)))
        if limit_batches > 0:
            limit_samples = limit_batches * batch_size
            logging.getLogger(__name__).info(
                "Test mode enabled: restricting test dataset to %d batches (%d samples)",
                limit_batches,
                limit_samples,
            )
            test_dataset = _LimitedDataset(test_dataset, limit_samples)

    num_workers = dataset_cfg.get("num_workers", config.get("data", {}).get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=getattr(test_dataset, "collate_fn", None) or collate_medical_batch,
    )


def aggregate_full_volume_metrics(
    trainer: Trainer,
    config: Dict[str, Any],
    entity: str,
    data_dir: Path,
    splits_path: Path,
    batch_size: int,
    patient_limit: Optional[int] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)

    splits = load_splits(splits_path)
    test_patients = splits.get("test", [])
    if patient_limit is not None:
        if patient_limit <= 0:
            logger.info("Patient limit set to %s; skipping full-volume aggregation", patient_limit)
            return {}
        test_patients = test_patients[:patient_limit]

    if not test_patients:
        logger.warning("No test patients found in %s; skipping full-volume aggregation", splits_path)
        return {}

    output_cfg = config.setdefault("output", {})
    results_dir = Path(output_cfg.get("results_dir", "./output")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_root = resolve_cache_root(config)
    processed_images_dir = cache_root / 'processed_images' / 'test'

    config_for_dataset = copy.deepcopy(config)
    dataset_cfg = config_for_dataset.setdefault('dataset', {})
    dataset_cfg['test_mode'] = False
    dataset_cfg.pop('n_test_samples', None)
    dataset_cfg.pop('n_test_batches', None)
    dataset_cfg['num_workers'] = 0
    dataset_cfg['pin_memory'] = False

    dataset_for_patches = create_dataset(
        config_for_dataset,
        entity,
        'test',
        str(data_dir),
        transform=None,
        apply_transforms=False,
    )
    if not isinstance(dataset_for_patches, PatchDataset):
        logger.warning("Cached dataset is not a PatchDataset; skipping full-volume aggregation")
        return {}

    clinical_cfg = dict(config.get("clinical_metrics") or {})
    if "spacing" not in clinical_cfg:
        spacing = config.get("preprocessing", {}).get("voxel_spacing")
        if spacing is None:
            spacing = [0.5, 0.5, 0.5]
        clinical_cfg["spacing"] = spacing
    calculator = ClinicalMetricsCalculator({"clinical_metrics": clinical_cfg})

    patch_shape = tuple(int(v) for v in dataset_for_patches.patch_shape)
    mask_cache_dir = results_dir / "mask_cache"
    preprocessor_holder: Dict[str, Optional[LungPreprocessor]] = {'instance': None}

    aggregated_overall: Dict[str, list] = defaultdict(list)
    aggregated_lobe: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    aggregated_side: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_patient_results: List[Dict[str, Any]] = []

    trainer.model.eval()

    inference_batch_size = max(1, min(batch_size, 8))

    for patient in test_patients:
        patient_id = patient.get("patient_id")
        if not patient_id:
            continue

        indices = dataset_for_patches.get_patient_indices(patient_id, restrict_to_active=True)
        if not indices:
            logger.warning("No cached patches found for patient %s", patient_id)
            continue

        min_coords, volume_shape = build_patient_bbox(dataset_for_patches, indices)
        logger.info(
            "Reconstructing cached volume for patient %s (shape=%s)",
            patient_id,
            volume_shape,
        )

        recon, gt, votes = reconstruct_from_cached_dataset(
            trainer,
            dataset_for_patches,
            indices,
            patch_shape,
            inference_batch_size,
            min_coords,
            volume_shape,
        )

        if np.count_nonzero(votes) == 0:
            logger.warning("No voxels reconstructed for patient %s", patient_id)
            continue

        mask_dict = get_or_compute_masks(
            patient,
            config,
            results_dir,
            mask_cache_dir,
            processed_images_dir,
            min_coords,
            volume_shape,
            preprocessor_holder,
        )

        mask_volume = votes > 0
        patient_overall = compute_volume_metrics(calculator, gt, recon, mask_volume)
        for metric, value in patient_overall.items():
            aggregated_overall[metric].append(float(value))

        lobe_results: Dict[str, Dict[str, float]] = {}
        for lobe_name in STANDARD_LOBE_NAMES.values():
            mask_arr = mask_dict.get(lobe_name)
            if mask_arr is None:
                continue
            lobe_metrics = compute_volume_metrics(calculator, gt, recon, mask_arr)
            lobe_results[lobe_name] = lobe_metrics
            for metric, value in lobe_metrics.items():
                aggregated_lobe[lobe_name][metric].append(float(value))

        side_mapping = {
            'ipsi_lung': 'ipsilateral',
            'contra_lung': 'contralateral',
            'left_lung': 'left_lung',
            'right_lung': 'right_lung',
        }
        side_results: Dict[str, Dict[str, float]] = {}
        for key, label in side_mapping.items():
            mask_arr = mask_dict.get(key)
            if mask_arr is None:
                continue
            side_metrics = compute_volume_metrics(calculator, gt, recon, mask_arr)
            side_results[label] = side_metrics
            for metric, value in side_metrics.items():
                aggregated_side[label][metric].append(float(value))

        per_patient_results.append(
            {
                'patient_id': patient_id,
                'overall': patient_overall,
                'per_lobe': lobe_results,
                'per_side': side_results,
            }
        )

    def summarise_group(group: Dict[str, list]) -> Dict[str, Dict[str, float]]:
        return {metric: summarize_values(values) for metric, values in group.items() if values}

    overall_summary = summarise_group(aggregated_overall)
    per_lobe_summary = {
        lobe: summarise_group(metrics) for lobe, metrics in aggregated_lobe.items() if metrics
    }
    per_side_summary = {
        side: summarise_group(metrics) for side, metrics in aggregated_side.items() if metrics
    }

    return {
        'patients': per_patient_results,
        'overall': overall_summary,
        'per_lobe': per_lobe_summary,
        'per_side': per_side_summary,
    }


def create_full_volume_table(summary: Dict[str, Any]):
    if wandb is None:
        return None

    columns = ["segmentation", "metric", "mean", "std", "min", "max"]
    table = wandb.Table(columns=columns)

    def add_rows(segment: str, metrics: Dict[str, Dict[str, float]]):
        if not metrics:
            return
        for metric, stats in metrics.items():
            metric_lower = metric.lower()
            if not (metric_lower.startswith("gamma") or metric_lower.startswith("d")):
                continue
            table.add_data(
                segment,
                metric,
                stats.get("mean"),
                stats.get("std"),
                stats.get("min"),
                stats.get("max"),
            )

    add_rows("overall", summary.get("overall", {}))
    for lobe, metrics in summary.get("per_lobe", {}).items():
        add_rows(f"lobe:{lobe}", metrics)
    for side, metrics in summary.get("per_side", {}).items():
        add_rows(f"side:{side}", metrics)

    return table


def evaluate(
    config_path: Path,
    entity: str,
    data_dir: Path,
    checkpoint_path: Optional[Path],
    run_id: Optional[str],
    output_dir: Optional[Path],
    disable_test_mode: bool,
    splits_path: Path,
    patient_limit: Optional[int],
) -> Dict[str, Any]:
    config = load_config(config_path)
    prepare_wandb_metadata(config, entity)

    wandb_cfg = config.setdefault("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("use_wandb", False))
    wandb_cfg_snapshot = dict(wandb_cfg)
    if wandb_enabled:
        wandb_cfg["use_wandb"] = False

    output_cfg = config.setdefault("output", {})
    if output_dir is not None:
        output_cfg["results_dir"] = str(output_dir)
        output_cfg.setdefault("model_dir", str(output_dir / "models"))
        output_cfg.setdefault("log_dir", str(output_dir / "logs"))

    results_path = Path(output_cfg.get("results_dir", "./output")).resolve()
    model_path = Path(output_cfg.get("model_dir", results_path / "models")).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    batch_size = int(config.get("training", {}).get("batch_size", 1) or 1)

    test_loader = build_test_loader(config, entity, data_dir, disable_test_mode)

    def _unwrap_dataset(ds: Any) -> Any:
        current = ds
        visited = set()
        while hasattr(current, "base_dataset") and id(current) not in visited:
            visited.add(id(current))
            current = getattr(current, "base_dataset")
        return current

    base_dataset = _unwrap_dataset(test_loader.dataset)
    inferred_channels = getattr(base_dataset, "input_channels", None)
    if isinstance(inferred_channels, (list, tuple)):
        inferred_channels = len(inferred_channels)
    if isinstance(inferred_channels, int) and inferred_channels > 0:
        config.setdefault("model", {})["in_channels"] = inferred_channels
    config.setdefault("model", {}).setdefault("output_channels", 1)

    model = get_model(config)
    trainer = Trainer(model, config, entity)

    if wandb_enabled:
        config["wandb"].update(wandb_cfg_snapshot)

    if checkpoint_path is None:
        checkpoint_path = model_path / f"{entity}_best_model.pth"
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    trainer.load_checkpoint(str(checkpoint_path))

    logging.getLogger(__name__).info("Evaluating checkpoint %s", checkpoint_path)
    metrics = trainer.evaluate_testset(test_loader)

    inference_batch_size = max(1, min(batch_size, 8))

    full_volume_summary = aggregate_full_volume_metrics(
        trainer,
        config,
        entity,
        data_dir,
        splits_path,
        inference_batch_size,
        patient_limit=patient_limit,
    )

    if full_volume_summary:
        fv_path = results_path / "full_volume_summary.json"
        with fv_path.open("w") as handle:
            json.dump(full_volume_summary, handle, indent=2)
        logging.getLogger(__name__).info("Saved full-volume summary to %s", fv_path)

    metrics["full_volume_summary"] = full_volume_summary

    metrics_path = results_path / "test_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    logging.getLogger(__name__).info("Saved test metrics to %s", metrics_path)

    if wandb_enabled and wandb is not None:
        wandb_cfg = config["wandb"]
        init_kwargs: Dict[str, Any] = {
            "project": wandb_cfg.get("project_name"),
            "entity": wandb_cfg.get("entity"),
            "config": config,
        }
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        if wandb_cfg.get("name"):
            init_kwargs["name"] = wandb_cfg["name"]
        if wandb_cfg.get("group"):
            init_kwargs["group"] = wandb_cfg["group"]
        if wandb_cfg.get("tags"):
            init_kwargs["tags"] = wandb_cfg["tags"]

        run = wandb.init(**init_kwargs)
        if run is not None:
            run.config.update(config, allow_val_change=True)

        wandb_log: Dict[str, Any] = {}

        overall_stats = metrics.get("overall", {})
        if overall_stats:
            table = wandb.Table(columns=["metric", "mean", "std", "min", "max"])
            for metric in ["mse", "mae"]:
                summary = overall_stats.get(metric)
                if not summary:
                    continue
                table.add_data(
                    metric,
                    summary.get("mean"),
                    summary.get("std"),
                    summary.get("min"),
                    summary.get("max"),
                )
            if len(table.data) > 0:
                wandb_log["test/overall_table"] = table

        if full_volume_summary:
            fv_table = create_full_volume_table(full_volume_summary)
            if fv_table is not None and len(fv_table.data) > 0:
                wandb_log["test/full_volume_metrics"] = fv_table

        wandb.log(wandb_log)
        wandb.finish()

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    parser.add_argument("--config", required=True, help="Path to configuration YAML.")
    parser.add_argument("--entity", required=True, choices=["lung", "hnc"], help="Entity type.")
    parser.add_argument("--data_dir", required=True, help="Root directory with processed data.")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to config output path).")
    parser.add_argument("--run_id", help="Existing WandB run id to resume for logging.")
    parser.add_argument("--output_dir", help="Override results directory.")
    parser.add_argument("--log_level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--splits", required=True, help="JSON file containing train/val/test patient splits.")
    parser.add_argument(
        "--disable_test_mode",
        action="store_true",
        help="Ignore dataset.test_mode and evaluate on the full test split.",
    )
    parser.add_argument(
        "--patient_limit",
        type=int,
        default=None,
        help="Optional limit on number of test patients for full-volume aggregation (for quick diagnostics).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    evaluate(
        config_path=Path(args.config),
        entity=args.entity,
        data_dir=Path(args.data_dir),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        run_id=args.run_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        disable_test_mode=args.disable_test_mode,
        splits_path=Path(args.splits),
        patient_limit=args.patient_limit,
    )


if __name__ == "__main__":
    main()
