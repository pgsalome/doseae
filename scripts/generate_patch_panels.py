#!/usr/bin/env python3
"""
Generate per-patch visualization panels that combine global CT context and
dose overlays for every cached patch.

Example:
    python scripts/generate_patch_panels.py \
        --config config/new_pipeline_config.yaml \
        --splits data/full_splits_auto.json \
        --cache-root /data/pgsal/NSCLC-Cetuximab_AE_cache/processed_patches \
        --split train \
        --patient-id 0617660380 \
        --max-patches 50

For each patch we save a 1×2 panel:
  * Left  : resampled CT slice with a yellow rectangle locating the patch.
  * Right : patch-centered CT slice with a dose overlay.
Panels are written to processed_patches/<split>/<patient_id>/png_patches_loc/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import SimpleITK as sitk
import yaml

from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-patch CT/dose visualization panels.")
    parser.add_argument("--config", required=True, type=Path, help="Path to preprocessing config YAML")
    parser.add_argument("--splits", required=True, type=Path, help="Path to splits JSON")
    parser.add_argument("--cache-root", required=True, type=Path,
                        help="Root directory that contains processed_patches/<split>/patient_cache")
    parser.add_argument("--split", default="train", help="Dataset split to process (train/val/test)")
    parser.add_argument("--patient-id", action="append",
                        help="Patient ID(s) to visualize (repeat flag for multiple). If omitted, process all.")
    parser.add_argument("--limit", type=int, help="Limit number of patients processed")
    parser.add_argument("--max-patches", type=int,
                        help="Cap the number of patches visualized per patient")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip patches whose PNG already exists")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def load_splits(splits_path: Path) -> Dict[str, List[Dict]]:
    with splits_path.open("r") as handle:
        return json.load(handle)


def prepare_ct_array(preprocessor: LungPreprocessor, ct_path: Path) -> np.ndarray:
    ct_image = sitk.ReadImage(str(ct_path))
    ct_image = preprocessor.resample_to_common_spacing(ct_image, preprocessor.target_spacing)
    ct_image = preprocessor.apply_ct_preprocessing(ct_image)
    return sitk.GetArrayFromImage(ct_image).astype(np.float32)


def iter_patients(entries: Sequence[Dict], selected_ids: Optional[Iterable[str]] = None):
    if not selected_ids:
        yield from entries
        return
    wanted = set(selected_ids)
    for item in entries:
        if item.get("patient_id") in wanted:
            yield item


def load_cache(cache_path: Path):
    with h5py.File(cache_path, "r") as h5_file:
        metadata = json.loads(h5_file["metadata"][()].decode("utf-8"))
        ct_patches = h5_file["ct_patches"][()]
        dose_patches = h5_file["dose_patches"][()]
    return metadata, ct_patches, dose_patches


def plot_panel(ct_volume: np.ndarray,
               ct_patch: np.ndarray,
               dose_patch: np.ndarray,
               start_coords: Sequence[int],
               lobe_name: str,
               organ_ratio: float,
               output_path: Path) -> None:
    z_start, y_start, x_start = [int(v) for v in start_coords]
    depth, height, width = ct_patch.shape

    slice_index = int(z_start + depth // 2)
    slice_index = max(0, min(ct_volume.shape[0] - 1, slice_index))
    base_slice = ct_volume[slice_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(base_slice, cmap="gray")
    rect = patches.Rectangle(
        (x_start, y_start),
        width,
        height,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"{lobe_name} @ slice {slice_index}\nratio={organ_ratio:.2f}")
    axes[0].axis("off")

    center_idx = depth // 2
    ct_slice = ct_patch[center_idx]
    dose_slice = dose_patch[center_idx]
    axes[1].imshow(ct_slice, cmap="gray")
    im = axes[1].imshow(dose_slice, cmap="hot", alpha=0.6)
    axes[1].set_title("Patch dose overlay")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


def visualize_patient(preprocessor: LungPreprocessor,
                      ct_path: Path,
                      cache_path: Path,
                      output_dir: Path,
                      max_patches: Optional[int],
                      skip_existing: bool) -> int:
    if not ct_path.exists():
        print(f"[WARN] Missing CT for {cache_path.stem}: {ct_path}")
        return 0
    if not cache_path.exists():
        print(f"[WARN] Cache file missing: {cache_path}")
        return 0

    metadata, ct_patches, dose_patches = load_cache(cache_path)
    if not metadata:
        print(f"[WARN] No metadata entries in {cache_path}")
        return 0

    ct_volume = prepare_ct_array(preprocessor, ct_path)

    count = 0
    for entry in metadata:
        patch_id = int(entry.get("patch_id", -1))
        if patch_id < 0 or patch_id >= ct_patches.shape[0]:
            continue
        filename = f"patch_{patch_id:05d}.png"
        panel_path = output_dir / filename
        if skip_existing and panel_path.exists():
            count += 1
            continue
        ct_patch = np.asarray(ct_patches[patch_id], dtype=np.float32)
        dose_patch = np.asarray(dose_patches[patch_id], dtype=np.float32)
        depth, height, width = ct_patch.shape

        coords = entry.get("start_coords")
        if coords is None:
            coords = entry.get("coordinates")
        if coords is None and "center_coords" in entry:
            cx, cy, cz = entry["center_coords"]
            coords = (
                int(round(cz - depth / 2)),
                int(round(cy - height / 2)),
                int(round(cx - width / 2)),
            )
        if coords is None:
            coords = (0, 0, 0)
        coords = [int(v) for v in coords]
        coords[0] = max(0, min(coords[0], ct_volume.shape[0] - depth))
        coords[1] = max(0, min(coords[1], ct_volume.shape[1] - height))
        coords[2] = max(0, min(coords[2], ct_volume.shape[2] - width))

        lobe_name = entry.get("lobe_name", "unknown_lobe")
        organ_ratio = float(entry.get("organ_ratio", 0.0))

        plot_panel(ct_volume, ct_patch, dose_patch, coords, lobe_name, organ_ratio, panel_path)
        count += 1

        if max_patches is not None and count >= max_patches:
            break

    return count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    splits = load_splits(args.splits)

    if args.split not in splits:
        raise SystemExit(f"Split '{args.split}' not found in {args.splits}")

    entries = list(iter_patients(splits[args.split], args.patient_id))
    if not entries:
        raise SystemExit("No patients match the requested criteria.")

    if args.limit is not None:
        entries = entries[: args.limit]

    preprocessor = LungPreprocessor(config, output_dir=Path(".") / "tmp_patch_panels")
    processed = 0
    total_panels = 0

    for entry in entries:
        patient_id = entry.get("patient_id")
        if not patient_id:
            continue

        ct_path = Path(entry.get("ct_path", ""))
        cache_path = args.cache_root / args.split / "patient_cache" / f"{patient_id}.h5"
        output_dir = args.cache_root / args.split / patient_id / "png_patches_loc"

        try:
            panel_count = visualize_patient(
                preprocessor,
                ct_path,
                cache_path,
                output_dir,
                args.max_patches,
                args.skip_existing,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] Failed to visualize {patient_id}: {exc}")
            continue

        print(f"{patient_id}: wrote {panel_count} patch panel(s)")
        total_panels += panel_count
        processed += 1

    print(f"Completed {processed} patient(s); generated {total_panels} panel(s).")


if __name__ == "__main__":
    main()
