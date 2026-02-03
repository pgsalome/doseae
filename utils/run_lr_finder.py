#!/usr/bin/env python3
"""
Learning-rate finder for DoseAE training.

Loads a training configuration, samples a subset of patients, and runs the
LR range test using torch-lr-finder. Produces a PNG plot with loss vs LR
and prints the LR that achieved the minimum loss as well as the LR suggested
by the finder (steepest gradient heuristic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder, TrainDataLoaderIter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.loaders import collate_medical_batch, create_dataset
from core.training.trainer import Trainer

# Dynamically load scripts/train.py as a module
SPEC = (
    (PROJECT_ROOT / "scripts" / "train.py")
    .resolve()
    .as_posix()
)
MODULE_SPEC = None
TRAIN_MODULE = None
if Path(SPEC).exists():
    import importlib.util

    MODULE_SPEC = importlib.util.spec_from_file_location("doseae_train", SPEC)
    TRAIN_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
    MODULE_SPEC.loader.exec_module(TRAIN_MODULE)
else:  # pragma: no cover
    raise FileNotFoundError("scripts/train.py not found; cannot run LR finder.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LR range test for DoseAE")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--data_dir", required=True, help="Root directory with preprocessed data")
    parser.add_argument(
        "--splits",
        required=True,
        help="JSON file containing train/val/test patient splits",
    )
    parser.add_argument(
        "--entity",
        default="lung",
        choices=["lung", "hnc"],
        help="Entity name (default: lung)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=8,
        help="Number of training patients to sample for the LR finder (default: 8)",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=30,
        help="Number of mini-batches for the LR sweep (default: 30)",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate to test (default: 1e-2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (defaults to config['training']['batch_size'])",
    )
    parser.add_argument(
        "--output",
        default="lr_finder.png",
        help="Path to save the LR finder plot (default: lr_finder.png)",
    )
    return parser.parse_args()


def load_train_subset(dataset, patient_ids: List[str]) -> None:
    indices: List[int] = []
    for pid in patient_ids:
        indices.extend(dataset.get_patient_indices(pid, restrict_to_active=True))
    dataset._indices = np.array(indices, dtype=dataset._indices.dtype)  # type: ignore[attr-defined]


class DoseTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch):
        return batch["input"].float(), batch["dose"].float()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    config.setdefault("wandb", {})["use_wandb"] = False

    train_dataset = create_dataset(
        config,
        args.entity,
        "train",
        args.data_dir,
        transform=None,
        apply_transforms=False,
    )
    config["model"]["in_channels"] = len(getattr(train_dataset, "input_channels", [1]))

    # Load splits and sample subset
    splits = json.load(open(args.splits))
    train_patients = [p["patient_id"] for p in splits["train"]]
    subset = train_patients[: args.subset_size]
    load_train_subset(train_dataset, subset)

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=getattr(train_dataset, "collate_fn", None) or collate_medical_batch,
    )

    model = TRAIN_MODULE.create_model(config, args.entity)
    trainer = Trainer(model, config, args.entity)
    optimizer = trainer.optimizer

    def mse_dict(outputs, labels):
        return F.mse_loss(outputs["reconstruction"], labels)

    train_iter = DoseTrainIter(train_loader)
    lr_finder = LRFinder(trainer.model, optimizer, mse_dict, device=trainer.device)
    lr_finder.range_test(train_iter, end_lr=args.end_lr, num_iter=args.num_iter)

    plot_obj = lr_finder.plot()
    if hasattr(plot_obj, "savefig"):
        fig = plot_obj
    elif hasattr(plot_obj, "figure"):
        fig = plot_obj.figure
    elif isinstance(plot_obj, tuple):
        fig = plot_obj[0]
    else:
        raise RuntimeError("Unexpected object returned from lr_finder.plot()")
    fig.savefig(args.output)

    losses = np.array(lr_finder.history["loss"])
    lrs = np.array(lr_finder.history["lr"])
    min_lr = lrs[np.argmin(losses)]
    print(f"LR finder plot saved to: {args.output}")
    print(f"Minimum-loss LR: {min_lr:.3e}")
    if lr_finder._best_lr is not None:
        print(f"Suggested LR (steepest descent): {lr_finder._best_lr:.3e}")

    lr_finder.reset()


if __name__ == "__main__":
    main()

