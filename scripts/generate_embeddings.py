#!/usr/bin/env python3
"""Generate patch embeddings for MedicalNet and cOOpD encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import h5py
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=["medicalnet", "coopd"],
        required=True,
        help="Which encoder to use for embedding extraction.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        help="Dataset splits to process (expects <data_root>/<split>.h5).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/pgsal/NSCLC-Cetuximab_AE_cache/processed_patches"),
        help="Directory containing processed_patches/<split>.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where embedding tensors will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction.",
    )

    # MedicalNet specific
    parser.add_argument(
        "--medicalnet-weights",
        type=Path,
        default=PROJECT_ROOT / "external/MedicalNet/pretrain/resnet_10_23dataset.pth",
        help="Path to MedicalNet ResNet10 pretrained weights.",
    )

    # cOOpD specific
    parser.add_argument(
        "--coopd-checkpoint",
        type=Path,
        default=Path("/data/pgsal/epoch=99-step=785799.ckpt"),
        help="Path to trained cOOpD Lightning checkpoint.",
    )
    parser.add_argument(
        "--coopd-bn-train",
        action="store_true",
        help="Keep BatchNorm layers in train() mode when extracting cOOpD embeddings (recommended).",
    )

    parser.add_argument(
        "--modality",
        choices=["ct", "dose"],
        default="ct",
        help="Patch modality stored in the HDF5 caches.",
    )
    return parser.parse_args()


def load_metadata(handle: h5py.File) -> List[Dict]:
    if "patch_metadata" not in handle:
        return []
    raw = handle["patch_metadata"][()]
    if isinstance(raw, bytes):
        text = raw.decode()
    else:
        text = raw.tobytes().decode()
    return json.loads(text)


def build_medicalnet(weights_path: Path, device: torch.device) -> torch.nn.Module:
    import importlib.util

    models_path = PROJECT_ROOT / "external" / "MedicalNet" / "models" / "resnet.py"
    if not models_path.exists():
        raise FileNotFoundError(f"MedicalNet resnet definition not found at {models_path}")

    spec = importlib.util.spec_from_file_location("medicalnet_resnet", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MedicalNet module from {models_path}")
    resnet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resnet)  # type: ignore[arg-type]

    backbone = resnet.resnet10(
        sample_input_D=64,
        sample_input_H=64,
        sample_input_W=64,
        shortcut_type="B",
        no_cuda=device.type == "cpu",
        num_seg_classes=1,
    )

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
    }
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in MedicalNet checkpoint: {unexpected}")
    if missing:
        print(f"[MedicalNet] Missing keys ignored (decoder head): {missing}")

    class MedicalNetFeatureExtractor(torch.nn.Module):
        def __init__(self, net: torch.nn.Module):
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)
            x = F.adaptive_avg_pool3d(x, output_size=1)
            return torch.flatten(x, 1)

    encoder = MedicalNetFeatureExtractor(backbone)
    encoder.to(device)
    encoder.eval()
    return encoder


def build_coopd_encoder(checkpoint_path: Path, device: torch.device, bn_train: bool) -> torch.nn.Module:
    coopd_path = str(PROJECT_ROOT / "cOOpD")
    sys_path_added = False
    if coopd_path not in sys.path:
        sys.path.insert(0, coopd_path)
        sys_path_added = True

    try:
        from models.base import ResNet_Encoder  # type: ignore
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint  # type: ignore
            torch.serialization.add_safe_globals([ModelCheckpoint])
        except Exception:
            pass
    finally:
        if sys_path_added:
            sys.path.remove(coopd_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"'state_dict' missing in checkpoint {checkpoint_path}")
    hparams = checkpoint.get("hyper_parameters", {})

    channels_in = int(hparams.get("input_shape", (1,))[0])
    base_model = hparams.get("model_type", "resnet18")

    encoder = ResNet_Encoder(base_model=base_model, channels_in=channels_in, cifar_stem=True)
    encoder_state = {
        key.replace("model.encoder.", ""): tensor
        for key, tensor in state_dict.items()
        if key.startswith("model.encoder.")
    }
    missing = []
    unexpected = []
    try:
        load_res = encoder.load_state_dict(encoder_state, strict=False)
        if isinstance(load_res, tuple):
            missing, unexpected = load_res
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load encoder weights: {exc}") from exc

    if unexpected:
        print(f"[cOOpD] Unexpected keys ignored: {unexpected}")
    if missing:
        print(f"[cOOpD] Missing keys when loading encoder: {missing}")

    encoder.to(device)
    if bn_train:
        encoder.train()
    else:
        encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def extract_embeddings(
    encoder_name: str,
    encoder: torch.nn.Module,
    h5_path: Path,
    modality: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    dataset_name = f"{modality}_patches"
    with h5py.File(h5_path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"Dataset '{dataset_name}' not found in {h5_path}")
        patches_ds = handle[dataset_name]
        metadata = load_metadata(handle)
        total = int(patches_ds.shape[0])

        embeddings: Optional[torch.Tensor] = None
        start_idx = 0
        while start_idx < total:
            end_idx = min(start_idx + batch_size, total)
            batch_np = patches_ds[start_idx:end_idx]
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            batch = batch.unsqueeze(1)
            if encoder_name == "medicalnet":
                batch = batch - 0.5

            with torch.no_grad():
                outputs = encoder(batch)
            outputs = outputs.detach().cpu()

            if embeddings is None:
                embeddings = torch.empty((total, outputs.shape[1]), dtype=torch.float32)
            embeddings[start_idx:end_idx] = outputs

            start_idx = end_idx

    if embeddings is None:
        raise RuntimeError(f"No embeddings computed for {h5_path}")

    return {
        "embeddings": embeddings,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder == "medicalnet":
        encoder = build_medicalnet(args.medicalnet_weights, device=device)
    else:
        encoder = build_coopd_encoder(
            args.coopd_checkpoint,
            device=device,
            bn_train=args.coopd_bn_train,
        )

    for split in args.splits:
        h5_path = args.data_root / f"{split}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 cache not found at {h5_path}")

        print(f"[{args.encoder}] Processing split '{split}' from {h5_path}")
        result = extract_embeddings(
            encoder_name=args.encoder,
            encoder=encoder,
            h5_path=h5_path,
            modality=args.modality,
            batch_size=args.batch_size,
            device=device,
        )

        output_path = args.output_dir / f"{args.encoder}_{split}_patch_embeddings.pt"
        torch.save(
            {
                "encoder": args.encoder,
                "split": split,
                "data_root": str(args.data_root),
                "modality": args.modality,
                "embeddings": result["embeddings"],
                "metadata": result["metadata"],
                "config": {
                    "device_used": str(device),
                    "batch_size": args.batch_size,
                    "medicalnet_weights": str(args.medicalnet_weights),
                    "coopd_checkpoint": str(args.coopd_checkpoint),
                    "coopd_bn_train": bool(args.coopd_bn_train),
                },
            },
            output_path,
        )
        print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
