#!/usr/bin/env python3
"""
Full-volume inference and evaluation for DoseAE.

Reconstructs complete dose volumes from autoencoder patch predictions,
computes reconstruction / clinical metrics, and logs results to WandB.
Supports re-extracting patches directly from the original CT/Dose using
the preprocessing pipeline, or consuming the cached HDF5 patch dataset.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import SimpleITK as sitk

# Make project modules importable
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.append(str(PROJECT_ROOT))

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

from datasets.loaders import create_dataset
from entities.lung.datasets.dataset import PatchDataset
from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor
from scripts.train import (
    create_model,
    load_config,
    prepare_wandb_metadata,
)
from core.training.trainer import Trainer
from utils.clinical_metrics import ClinicalMetricsCalculator

LOGGER = logging.getLogger("doseae.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-volume inference and evaluation for DoseAE")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", help="Path to trained checkpoint (.pth). Defaults to best checkpoint.")
    parser.add_argument("--data_dir", required=True, help="Root directory containing preprocessed HDF5 caches")
    parser.add_argument("--splits", required=True, help="JSON file with train/val/test patient splits")
    parser.add_argument("--entity", choices=["lung", "hnc"], default="lung", help="Entity name (default: lung)")
    parser.add_argument("--output_dir", help="Directory to store inference outputs (metrics, reconstructions)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference (patch level)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of test patients")
    parser.add_argument("--reextract", action="store_true", help="Re-extract patches from original CT/Dose on the fly")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--device", default=None, help="Torch device override (e.g., cuda:0)")
    return parser.parse_args()


def load_splits(path: Path) -> Dict[str, List[Dict]]:
    with path.open("r") as f:
        return json.load(f)


def ensure_output_dir(config: Dict[str, any], cli_output: Optional[str]) -> Path:
    output_cfg = config.setdefault("output", {})
    if cli_output:
        output_cfg["results_dir"] = cli_output
    results_dir = Path(output_cfg.get("results_dir", "./output"))
    model_dir = Path(output_cfg.get("model_dir", results_dir / "models"))
    log_dir = Path(output_cfg.get("log_dir", results_dir / "logs"))
    for path in {results_dir, model_dir, log_dir}:
        path.mkdir(parents=True, exist_ok=True)
    output_cfg["results_dir"] = str(results_dir.resolve())
    output_cfg["model_dir"] = str(model_dir.resolve())
    output_cfg["log_dir"] = str(log_dir.resolve())
    return results_dir


def infer_volume_shape(start_coords: List[List[int]], patch_size: int) -> Tuple[int, int, int]:
    max_coords = np.max(np.array(start_coords), axis=0)
    return tuple(int(c) + patch_size for c in max_coords)


def load_patient_indices(dataset: PatchDataset, patient_id: str) -> List[int]:
    indices = dataset.get_patient_indices(patient_id, restrict_to_active=True)
    if not indices:
        LOGGER.warning("No patches found for patient %s in cached dataset", patient_id)
    return indices


def batch_iterable(items: List[int], batch_size: int) -> List[List[int]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def reconstruct_from_cached(
    trainer: Trainer,
    dataset: PatchDataset,
    patient_id: str,
    indices: List[int],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, any]]]:
    metadata = []
    start_coords = []
    patch_size = dataset.patch_shape[0]
    all_start = []

    for idx in indices:
        meta = dataset._metadata_for_index(idx)  # type: ignore[attr-defined]
        metadata.append(meta)
        all_start.append(meta["start_coords"])

    if not all_start:
        return (
            np.zeros((1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.float32),
            metadata,
        )

    volume_shape = infer_volume_shape(all_start, patch_size)
    recon_volume = np.zeros(volume_shape, dtype=np.float32)
    gt_volume = np.zeros(volume_shape, dtype=np.float32)
    vote_volume = np.zeros(volume_shape, dtype=np.float32)

    trainer.model.eval()

    for chunk in batch_iterable(indices, batch_size):
        samples = [dataset[idx] for idx in chunk]
        inputs = torch.stack([sample["input"] for sample in samples]).to(trainer.device)
        with torch.no_grad():
            raw_outputs = trainer.model(inputs)
            outputs = trainer._standardize_outputs(raw_outputs)
        preds = outputs["reconstruction"].detach().cpu().numpy()

        for sample, pred_patch in zip(samples, preds):
            gt_patch = sample["dose"].numpy()
            coords = sample["metadata"]["start_coords"]
            z, y, x = coords
            recon_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += pred_patch[0]
            gt_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += gt_patch[0]
            vote_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += 1.0

    vote_volume[vote_volume == 0] = 1.0
    recon_volume /= vote_volume
    gt_volume /= vote_volume
    return recon_volume, gt_volume, vote_volume, metadata


def gather_patches_from_preprocessor(
    patient_result: Dict[str, any],
    patch_size: int,
) -> Tuple[List[Dict[str, any]], np.ndarray, np.ndarray]:
    patch_entries: List[Dict[str, any]] = []
    patch_data = patient_result.get("patches_data", {})
    patch_id = 0
    for side_type in ["ipsilateral", "contralateral"]:
        side_dict = patch_data.get(side_type, {})
        for lobe_name, patches in side_dict.items():
            for patch in patches:
                entry = dict(patch)
                entry["side_type"] = side_type
                entry["lobe_name"] = lobe_name
                entry["patch_id"] = patch_id
                patch_id += 1
                patch_entries.append(entry)
    ct_volume = patch_data.get("ct_array")
    dose_volume = patch_data.get("dose_array")
    if ct_volume is None or dose_volume is None:
        ct_volume = np.asarray(sitk.GetArrayFromImage(patient_result["ct_image"]), dtype=np.float32)
        dose_volume = np.asarray(sitk.GetArrayFromImage(patient_result["dose_image"]), dtype=np.float32)
    return patch_entries, ct_volume, dose_volume


def reconstruct_from_reextract(
    trainer: Trainer,
    patient_result: Dict[str, any],
    batch_size: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, any]]]:
    patch_entries, ct_volume, dose_volume = gather_patches_from_preprocessor(patient_result, patch_size)
    recon_volume = np.zeros_like(dose_volume, dtype=np.float32)
    gt_volume = np.zeros_like(dose_volume, dtype=np.float32)
    vote_volume = np.zeros_like(dose_volume, dtype=np.float32)

    patches = []
    coords = []
    meta = []
    for entry in patch_entries:
        z, y, x = entry["start_coords"]
        patch = dose_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size]
        patches.append(patch)
        coords.append((z, y, x))
        meta.append(entry)

    if not patches:
        return recon_volume, gt_volume, vote_volume, meta

    trainer.model.eval()
    all_inputs = []
    for patch in patches:
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        all_inputs.append(tensor)
    inputs_tensor = torch.cat(all_inputs, dim=0).to(trainer.device)

    preds = []
    for i in range(0, inputs_tensor.shape[0], batch_size):
        batch = inputs_tensor[i:i + batch_size]
        with torch.no_grad():
            raw_outputs = trainer.model(batch)
            outputs = trainer._standardize_outputs(raw_outputs)
        preds.append(outputs["reconstruction"].detach().cpu().numpy())
    preds_np = np.concatenate(preds, axis=0)

    for (z, y, x), pred_patch, gt_patch in zip(coords, preds_np, patches):
        recon_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += pred_patch[0]
        gt_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += gt_patch
        vote_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size] += 1.0

    vote_volume[vote_volume == 0] = 1.0
    recon_volume /= vote_volume
    gt_volume /= vote_volume
    return recon_volume, gt_volume, vote_volume, meta


def summarize(values: List[float]) -> Dict[str, float]:
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

    diff = recon_volume - target_volume
    if bool_mask is not None:
        diff = diff[bool_mask]
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
        comparison = calculator.compare_dose_distributions(target_volume, recon_volume, bool_mask)
        for key, value in comparison.items():
            if key == "gamma_map":
                continue
            if np.isscalar(value):
                metrics[key] = float(value)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Gamma/DVH computation failed: %s", exc)

    return metrics


def evaluate_patient(
    trainer: Trainer,
    calculator: ClinicalMetricsCalculator,
    patient_id: str,
    *,
    cached_dataset: Optional[PatchDataset],
    cached_indices: Optional[List[int]],
    patient_result: Optional[Dict[str, any]],
    batch_size: int,
    patch_size: int,
    lobe_masks: Optional[Dict[str, np.ndarray]] = None,
    side_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, any]:
    if patient_result is not None:
        recon, gt, votes, metadata = reconstruct_from_reextract(trainer, patient_result, batch_size, patch_size)
    elif cached_dataset is not None and cached_indices is not None:
        recon, gt, votes, metadata = reconstruct_from_cached(trainer, cached_dataset, patient_id, cached_indices, batch_size)
    else:
        raise ValueError("Either cached dataset or patient_result must be provided")

    mask = (votes > 0).astype(np.uint8)
    overall = compute_volume_metrics(calculator, gt, recon, mask)

    per_lobe = {}
    if lobe_masks:
        for lobe_name, lobe_mask in lobe_masks.items():
            per_lobe[lobe_name] = compute_volume_metrics(calculator, gt, recon, lobe_mask.astype(bool))

    per_side = {}
    if side_masks:
        for side_name, side_mask in side_masks.items():
            per_side[side_name] = compute_volume_metrics(calculator, gt, recon, side_mask.astype(bool))

    return {
        "patient_id": patient_id,
        "overall": overall,
        "per_lobe": per_lobe,
        "per_side": per_side,
        "num_patches": len(metadata),
    }


def convert_masks_to_numpy(lung_masks: Dict[str, "sitk.Image"]) -> Dict[str, np.ndarray]:
    import SimpleITK as sitk

    masks_np: Dict[str, np.ndarray] = {}
    for name, mask in lung_masks.items():
        masks_np[name] = sitk.GetArrayFromImage(mask).astype(bool)
    return masks_np


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dir(config, args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    LOGGER.info("Starting full-volume inference")

    splits = load_splits(Path(args.splits))
    test_patients = splits.get("test", [])
    if args.limit:
        test_patients = test_patients[: args.limit]
    patient_records = {p["patient_id"]: p for p in test_patients}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare WandB if enabled
    if not args.no_wandb and wandb is not None:
        prepare_wandb_metadata(config, args.entity)
        wandb_cfg = config.get("wandb", {})
        wandb_cfg.setdefault("job_type", "inference")
        wandb_run = wandb.init(
            project=wandb_cfg.get("project_name"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", f"inference_{args.entity}"),
            group=wandb_cfg.get("group", f"{args.entity}_inference"),
            config=config,
        )
    else:
        wandb_run = None

    LOGGER.info("Creating model and trainer")
    model = create_model(config, args.entity)
    trainer = Trainer(model, config, args.entity)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    elif trainer.best_checkpoint_path is None:
        default_checkpoint = Path(config["output"]["model_dir"]) / f"{args.entity}_best_model.pth"
        if default_checkpoint.exists():
            trainer.load_checkpoint(str(default_checkpoint))
        else:
            LOGGER.warning("No checkpoint provided and default best checkpoint not found; using current weights")

    batch_size = args.batch_size or int(config.get("training", {}).get("batch_size", 1))
    LOGGER.info("Inference batch size: %d", batch_size)

    clinical_cfg = config.get("clinical_metrics", {}) or {}
    if "spacing" not in clinical_cfg:
        spacing = config.get("preprocessing", {}).get("voxel_spacing", [0.5, 0.5, 0.5])
        clinical_cfg["spacing"] = spacing
    calculator = ClinicalMetricsCalculator({"clinical_metrics": clinical_cfg})

    cached_dataset = None
    cached_h5_dataset = None
    patient_ids_from_cache: List[str] = []
    if not args.reextract:
        LOGGER.info("Loading cached test patch dataset")
        cached_dataset = create_dataset(config, args.entity, "test", args.data_dir, transform=None, apply_transforms=False)
        if isinstance(cached_dataset, PatchDataset):
            patient_ids_from_cache = cached_dataset.get_patient_ids()
        else:
            raise ValueError("Expected PatchDataset for cached patches")

    if args.reextract:
        LOGGER.info("Re-extracting patches from original CT/Dose volumes")
        preprocessor = LungPreprocessor(config, config["output"]["results_dir"])
    else:
        preprocessor = None

    per_patient_results = []
    if args.reextract:
        patch_size_inference = int(config.get("lung", {}).get("patch_size", 50))
    else:
        patch_size_inference = int(cached_dataset.patch_shape[0]) if cached_dataset is not None else int(config.get("lung", {}).get("patch_size", 50))

    for patient in test_patients:
        patient_id = patient["patient_id"]
        LOGGER.info("Processing patient %s", patient_id)

        lobe_masks = None
        side_masks = None
        patient_result = None
        cached_indices = None

        if args.reextract:
            if preprocessor is None:
                raise RuntimeError("Preprocessor not initialised")
            patient_result = preprocessor.process_patient(
                patient_id=patient_id,
                ct_path=patient["ct_path"],
                dose_path=patient["dose_path"],
                prescribed_dose=patient.get("prescribed_dose"),
                split_name="test",
                experiment_type="patch",
                retain_patch_arrays=True,
                save_to_cache=False,
                save_visualizations=False,
            )
            if "lung_masks" in patient_result:
                mask_arrays = convert_masks_to_numpy(patient_result["lung_masks"])
                lobe_masks = {
                    k: mask_arrays[k]
                    for k in ["left_upper_lobe", "left_lower_lobe", "right_upper_lobe", "right_middle_lobe", "right_lower_lobe"]
                    if k in mask_arrays
                }
                side_masks = {}
                if "ipsi_lung" in mask_arrays:
                    side_masks["ipsilateral"] = mask_arrays["ipsi_lung"]
                if "contra_lung" in mask_arrays:
                    side_masks["contralateral"] = mask_arrays["contra_lung"]
        else:
            if cached_dataset is None:
                raise RuntimeError("Cached dataset not available")
            cached_indices = load_patient_indices(cached_dataset, patient_id)

        result = evaluate_patient(
            trainer,
            calculator,
            patient_id,
            cached_dataset=cached_dataset if not args.reextract else None,
            cached_indices=cached_indices,
            patient_result=patient_result,
            batch_size=batch_size,
            patch_size=patch_size_inference,
            lobe_masks=lobe_masks,
            side_masks=side_masks,
        )
        per_patient_results.append(result)

        if wandb_run is not None:
            log_payload = {
                "patient_id": patient_id,
                "test/overall_mse": result["overall"].get("mse"),
                "test/overall_mae": result["overall"].get("mae"),
                "test/overall_gamma_pass_rate": result["overall"].get("gamma_pass_rate"),
            }
            wandb.log(log_payload)

    overall_summary: Dict[str, List[float]] = defaultdict(list)
    for res in per_patient_results:
        for metric, value in res["overall"].items():
            if isinstance(value, (int, float)):
                overall_summary[metric].append(value)
    aggregated = {metric: summarize(values) for metric, values in overall_summary.items()}

    output_dir = Path(config["output"]["results_dir"])
    metrics_path = output_dir / "full_volume_test_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "patients": per_patient_results,
                "aggregate": aggregated,
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved full-volume metrics to %s", metrics_path)

    if wandb_run is not None:
        wandb_payload = {}
        for metric, summary in aggregated.items():
            safe_metric = metric.replace(" ", "_")
            for stat_name, value in summary.items():
                wandb_payload[f"test_summary/{safe_metric}_{stat_name}"] = value
        wandb.log(wandb_payload)
        wandb.finish()


if __name__ == "__main__":
    main()
