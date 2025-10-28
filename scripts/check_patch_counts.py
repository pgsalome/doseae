#!/usr/bin/env python3
"""
Utility script to inspect patient patch-cache files and report patch counts.

Example:
    python scripts/check_patch_counts.py /data/pgsal/NSCLC-Cetuximab_AE_cache/processed_patches/train/patient_cache
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple


def iter_h5_paths(root: Path) -> Iterable[Path]:
    """Yield .h5 files under the given root (non-recursive by default)."""
    if root.is_file() and root.suffix == ".h5":
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    for path in sorted(root.glob("*.h5")):
        if path.is_file():
            yield path


def read_patch_count(h5_path: Path) -> Tuple[int, int]:
    """Return patch count and optional attribute for an HDF5 file."""
    import h5py

    with h5py.File(h5_path, "r") as h5_file:
        if "ct_patches" not in h5_file:
            raise KeyError(f"{h5_path} missing 'ct_patches' dataset")
        dataset = h5_file["ct_patches"]
        shape = dataset.shape
        if not shape:
            raise ValueError(f"{h5_path} has empty shape for 'ct_patches': {shape}")
        patch_count = int(shape[0])
        attr_count = int(h5_file.attrs.get("patch_count", patch_count))
    return patch_count, attr_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Print patch counts for cache HDF5 files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="HDF5 files or directories that contain patient_cache .h5 files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning for .h5 files",
    )
    args = parser.parse_args()

    total_files = 0
    total_patches = 0
    exit_code = 0

    for input_path in args.paths:
        root = Path(input_path).expanduser()
        try:
            if args.recursive and root.is_dir():
                h5_paths = sorted(p for p in root.rglob("*.h5") if p.is_file())
            else:
                h5_paths = list(iter_h5_paths(root))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] {root}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        if not h5_paths:
            print(f"[WARN] {root}: no .h5 files found", file=sys.stderr)
            continue

        for h5_path in h5_paths:
            try:
                patch_count, attr_count = read_patch_count(h5_path)
                print(f"{h5_path}: {patch_count} patches (attr patch_count={attr_count})")
                total_files += 1
                total_patches += patch_count
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERROR] {h5_path}: {exc}", file=sys.stderr)
                exit_code = 1

    if total_files:
        print(f"---\nScanned {total_files} file(s); total patches: {total_patches}")
    else:
        print("No HDF5 files processed.", file=sys.stderr)
        exit_code = max(exit_code, 1)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

