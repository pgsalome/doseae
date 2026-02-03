#!/usr/bin/env python3
"""Generate UMAP and t-SNE plots from saved patch embeddings."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import umap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-files",
        nargs="+",
        required=True,
        help="Paths to *.pt embedding files produced by generate_embeddings.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store the generated plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Max number of patches to subsample for plotting (per file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = torch.load(path, map_location="cpu")
    embeddings = data["embeddings"].numpy()

    metadata = data.get("metadata", [])
    if metadata:
        dose = np.array([float(item.get("dose_max", 0.0)) for item in metadata], dtype=np.float32)
    else:
        dose = np.zeros(embeddings.shape[0], dtype=np.float32)
    info = {
        "encoder": data.get("encoder", path.stem),
        "split": data.get("split", "unknown"),
    }
    return embeddings, dose, info


def subsample(embeddings: np.ndarray, dose: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    total = embeddings.shape[0]
    if total <= sample_size:
        return embeddings, dose
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=sample_size, replace=False)
    return embeddings[indices], dose[indices]


def plot_embedding(
    coords: np.ndarray,
    dose: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=dose, s=6, cmap=cmap, alpha=0.85)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Dose (max patch value)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for file_path_str in args.embedding_files:
        file_path = Path(file_path_str)
        embeddings, dose, info = load_embeddings(file_path)
        embeddings, dose = subsample(embeddings, dose, args.sample_size, args.seed)

        encoder = info["encoder"]
        split = info["split"]

        reducer = umap.UMAP(random_state=args.seed)
        umap_coords = reducer.fit_transform(embeddings)
        umap_title = f"{encoder} – {split} – UMAP (n={embeddings.shape[0]})"
        umap_path = args.output_dir / f"{encoder}_{split}_umap.png"
        plot_embedding(umap_coords, dose, umap_title, umap_path)

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            random_state=args.seed,
            max_iter=1500,
            learning_rate="auto",
        )
        tsne_coords = tsne.fit_transform(embeddings)
        tsne_title = f"{encoder} – {split} – t-SNE (n={embeddings.shape[0]})"
        tsne_path = args.output_dir / f"{encoder}_{split}_tsne.png"
        plot_embedding(tsne_coords, dose, tsne_title, tsne_path)

        print(f"Saved plots for {encoder} {split} to {umap_path} and {tsne_path}")


if __name__ == "__main__":
    main()
