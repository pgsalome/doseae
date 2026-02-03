#!/usr/bin/env python3
"""Probe MedicalNet and cOOpD encoders on cached CT patches."""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class ProbeResult:
    name: str
    embeddings: np.ndarray
    dose_max: np.ndarray
    stats: Dict[str, float]
    probe_metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=["medicalnet", "coopd", "both"],
        default="medicalnet",
        help="Which encoder to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to probe.",
    )
    parser.add_argument(
        "--h5-path",
        default="/data/pgsal/NSCLC-Cetuximab_AE_cache/processed_patches/train.h5",
        help="Path to the cached patches HDF5 file.",
    )
    parser.add_argument(
        "--weights-path",
        default="external/MedicalNet/pretrain/resnet_10_23dataset.pth",
        help="Path to MedicalNet ResNet10 pretrained weights.",
    )
    parser.add_argument(
        "--coopd-embeddings",
        default=None,
        help="Path to precomputed cOOpD embeddings (.pt). Defaults to cOOpD/embeddings/{split}_patch_embeddings.pt.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1024,
        help="Number of patches to sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for MedicalNet inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run MedicalNet on.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold used to label high-dose patches.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.25,
        help="Quantile threshold used to label low-dose patches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchSubset(Dataset):
    """Stream CT patches directly from the HDF5 cache."""

    def __init__(self, h5_path: Path, indices: Sequence[int]):
        self.h5_path = h5_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.file = h5py.File(self.h5_path, "r")
        self._ct = self.file["ct_patches"]
        self._dose = self.file["dose_patches"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_idx = int(self.indices[item])
        ct_patch = self._ct[patch_idx]
        dose_patch = self._dose[patch_idx]
        ct = torch.from_numpy(ct_patch).unsqueeze(0).float() - 0.5
        dose_max = float(np.max(dose_patch))
        return ct, torch.tensor(dose_max, dtype=torch.float32)

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None  # type: ignore[assignment]

    def __del__(self):
        self.close()


class MedicalNetEncoder(nn.Module):
    """Feature extractor mirroring MedicalNet ResNet10 up to global pooling."""

    def __init__(self, weights_path: Path, device: torch.device):
        super().__init__()

        medicalnet_root = Path(__file__).resolve().parents[1] / "external" / "MedicalNet"
        models_path = medicalnet_root / "models" / "resnet.py"
        if not models_path.exists():
            raise FileNotFoundError(f"MedicalNet resnet definition not found at {models_path}")

        import importlib.util

        spec = importlib.util.spec_from_file_location("medicalnet_resnet", models_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load MedicalNet resnet module from {models_path}")
        resnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet)  # type: ignore[arg-type]

        self.backbone = resnet.resnet10(
            sample_input_D=64,
            sample_input_H=64,
            sample_input_W=64,
            shortcut_type="B",
            no_cuda=device.type == "cpu",
            num_seg_classes=1,
        )

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = {
            k.replace("module.", "", 1): v for k, v in checkpoint["state_dict"].items()
        }
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in MedicalNet checkpoint: {unexpected}")
        if missing:
            print(f"[MedicalNet] Missing keys ignored (decoder heads): {missing}")

        self.backbone.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.nn.functional.adaptive_avg_pool3d(x, output_size=1)
        return torch.flatten(x, 1)


def get_total_patches_from_h5(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["ct_patches"].shape[0])


def get_total_patches_from_coopd(path: Path) -> int:
    state = torch.load(path, map_location="cpu")
    embeddings = state.get("embeddings")
    if embeddings is None:
        raise KeyError(f"'embeddings' key not found in {path}")
    return int(embeddings.shape[0])


def choose_indices(total: int, sample_count: int, seed: int) -> np.ndarray:
    sample_count = min(sample_count, total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=sample_count, replace=False).astype(np.int64))


def gather_medicalnet_embeddings(
    h5_path: Path,
    weights_path: Path,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = PatchSubset(h5_path, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    encoder = MedicalNetEncoder(weights_path, device)
    encoder.eval()

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for ct_batch, dose_batch in loader:
            ct_batch = ct_batch.to(device, non_blocking=True)
            outputs = encoder(ct_batch)
            feats.append(outputs.cpu().numpy())
            labels.append(dose_batch.numpy())

    dataset.close()
    embeddings = np.concatenate(feats, axis=0)
    dose_max = np.concatenate(labels, axis=0)
    return embeddings, dose_max


def gather_coopd_embeddings(path: Path, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    state = torch.load(path, map_location="cpu")
    embeddings_tensor = state.get("embeddings")
    metadata: Iterable[Dict[str, object]] = state.get("metadata", [])
    if embeddings_tensor is None or not isinstance(metadata, Iterable):
        raise KeyError(f"cOOpD embeddings file {path} is missing required keys.")

    idx_list = [int(i) for i in indices]
    embeddings = embeddings_tensor[idx_list].cpu().numpy()
    dose_max = np.array(
        [float(metadata[i]["dose_max"]) for i in idx_list],
        dtype=np.float32,
    )
    return embeddings, dose_max


def summarise_embeddings(embeddings: np.ndarray) -> Dict[str, float]:
    per_dim_var = embeddings.var(axis=0)
    stats = {
        "dimension": float(embeddings.shape[1]),
        "per_dim_var_mean": float(per_dim_var.mean()),
        "per_dim_var_min": float(per_dim_var.min()),
        "per_dim_var_max": float(per_dim_var.max()),
        "overall_var": float(embeddings.var()),
    }
    return stats


def probe_linear_separation(
    embeddings: np.ndarray,
    dose_max: np.ndarray,
    high_q: float,
    low_q: float,
    seed: int,
) -> Dict[str, float]:
    if embeddings.shape[0] < 10:
        return {"error": float("nan")}

    high_thr = float(np.quantile(dose_max, high_q))
    low_thr = float(np.quantile(dose_max, low_q))

    if not np.isfinite(high_thr) or not np.isfinite(low_thr) or high_thr <= low_thr:
        return {"error": float("nan")}

    mask_high = dose_max >= high_thr
    mask_low = dose_max <= low_thr
    keep_mask = mask_high | mask_low

    if keep_mask.sum() < 8 or not (mask_high.any() and mask_low.any()):
        return {"error": float("nan")}

    labels = np.where(mask_high[keep_mask], 1, 0)
    features = embeddings[keep_mask]
    doses = dose_max[keep_mask]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    try:
        X_train, X_test, y_train, y_test, dose_train, dose_test = train_test_split(
            features_scaled,
            labels,
            doses,
            test_size=0.25,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        return {"error": float("nan")}

    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "sampled_pairs": float(features.shape[0]),
        "high_threshold": high_thr,
        "low_threshold": low_thr,
        "train_mean_high": float(dose_train[y_train == 1].mean()),
        "train_mean_low": float(dose_train[y_train == 0].mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def run_probe(
    name: str,
    embeddings: np.ndarray,
    dose_max: np.ndarray,
    args: argparse.Namespace,
) -> ProbeResult:
    stats = summarise_embeddings(embeddings)
    probe_metrics = probe_linear_separation(
        embeddings,
        dose_max,
        high_q=args.high_quantile,
        low_q=args.low_quantile,
        seed=args.seed,
    )
    return ProbeResult(name=name, embeddings=embeddings, dose_max=dose_max, stats=stats, probe_metrics=probe_metrics)


def print_result(result: ProbeResult) -> None:
    stats = result.stats
    probe = result.probe_metrics
    print(f"\n=== {result.name} encoder ===")
    print(f"Embedding shape: {result.embeddings.shape}")
    print(
        "Variance stats: "
        f"overall={stats['overall_var']:.6f}, "
        f"per-dim mean={stats['per_dim_var_mean']:.6f}, "
        f"min={stats['per_dim_var_min']:.6f}, "
        f"max={stats['per_dim_var_max']:.6f}"
    )
    if "error" in probe:
        print("Probe: insufficient separable samples to train logistic baseline.")
    else:
        print(
            f"Probe thresholds: low <= {probe['low_threshold']:.3f}, high >= {probe['high_threshold']:.3f}"
        )
        print(
            f"Probe train dose means: low={probe['train_mean_low']:.3f}, high={probe['train_mean_high']:.3f}"
        )
        print(
            f"Probe metrics: accuracy={probe['accuracy']:.3f}, ROC-AUC={probe['roc_auc']:.3f} "
            f"(samples kept={probe['sampled_pairs']:.0f})"
        )


def resolve_default_paths(args: argparse.Namespace) -> None:
    split = args.split
    if args.h5_path.endswith("train.h5") and split != "train":
        args.h5_path = args.h5_path.replace("train", split)
    elif args.h5_path.endswith("val.h5") and split != "val":
        args.h5_path = args.h5_path.replace("val", split)
    elif args.h5_path.endswith("test.h5") and split != "test":
        args.h5_path = args.h5_path.replace("test", split)

    if args.coopd_embeddings is None:
        args.coopd_embeddings = f"cOOpD/embeddings/{split}_patch_embeddings.pt"


def main() -> None:
    args = parse_args()
    resolve_default_paths(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    encoder_choice = args.encoder

    total_h5: Optional[int] = None
    total_coopd: Optional[int] = None

    if encoder_choice in ("medicalnet", "both"):
        h5_path = Path(args.h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 patches not found at {h5_path}")
        total_h5 = get_total_patches_from_h5(h5_path)

    if encoder_choice in ("coopd", "both"):
        coopd_path = Path(args.coopd_embeddings)
        if not coopd_path.exists():
            raise FileNotFoundError(f"cOOpD embeddings not found at {coopd_path}")
        total_coopd = get_total_patches_from_coopd(coopd_path)

    if encoder_choice == "medicalnet":
        indices = choose_indices(total_h5, args.sample_count, args.seed)  # type: ignore[arg-type]
    elif encoder_choice == "coopd":
        indices = choose_indices(total_coopd, args.sample_count, args.seed)  # type: ignore[arg-type]
    else:
        assert total_h5 is not None and total_coopd is not None
        total = min(total_h5, total_coopd)
        indices = choose_indices(total, args.sample_count, args.seed)

    results: List[ProbeResult] = []

    if encoder_choice in ("medicalnet", "both"):
        embeddings, dose_max = gather_medicalnet_embeddings(
            h5_path=Path(args.h5_path),
            weights_path=Path(args.weights_path),
            indices=indices,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(run_probe("MedicalNet", embeddings, dose_max, args))

    if encoder_choice in ("coopd", "both"):
        embeddings, dose_max = gather_coopd_embeddings(
            path=Path(args.coopd_embeddings),
            indices=indices,
        )
        results.append(run_probe("cOOpD", embeddings, dose_max, args))

    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()
