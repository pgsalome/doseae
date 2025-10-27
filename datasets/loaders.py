"""
Shared dataset construction utilities used by training, inference, and tuning.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset

from entities.lung.datasets import ImageH5Dataset, PatchDataset
from entities.hnc.datasets.hnc_dataset import HNCDataset, HNCPatchDataset


def build_transform_pipeline(transform_cfg: Dict[str, Any]):
    """Create a callable data augmentation pipeline from configuration."""
    if not transform_cfg or not transform_cfg.get('augment', False):
        return None

    rotate_enabled = transform_cfg.get('rotate', False)
    flip_enabled = transform_cfg.get('flip', False)
    brightness_enabled = transform_cfg.get('adjust_brightness', False)
    noise_enabled = transform_cfg.get('random_noise', False)
    random_crop_enabled = transform_cfg.get('random_crop', False)

    if random_crop_enabled:
        logging.getLogger(__name__).warning("random_crop transform not implemented; ignoring.")

    def _transform(sample: dict) -> dict:
        if not isinstance(sample, dict):
            return sample

        transformed = sample.copy()

        tensor_keys = [
            key for key in ('input', 'image', 'ct', 'dose', 'fused')
            if key in transformed and torch.is_tensor(transformed[key])
        ]
        if not tensor_keys:
            return transformed

        base_tensor = transformed[tensor_keys[0]]
        spatial_dims = list(range(1, base_tensor.dim())) if base_tensor.dim() > 1 else []

        # Determine shared random operations
        k_rot = random.randint(0, 3) if rotate_enabled and base_tensor.dim() >= 3 else 0
        dims_to_flip: List[int] = []
        if flip_enabled and spatial_dims:
            dims_to_flip = [dim for dim in spatial_dims if random.random() < 0.5]

        brightness_delta = random.uniform(-0.1, 0.1) if brightness_enabled else 0.0
        noise_std = 0.05 if noise_enabled else 0.0

        def _apply_ops(tensor: torch.Tensor) -> torch.Tensor:
            result = tensor.clone()
            if dims_to_flip:
                result = torch.flip(result, dims=dims_to_flip)
            if k_rot and result.dim() >= 3:
                result = torch.rot90(result, k_rot, dims=(-2, -1))
            return result

        for key in tensor_keys:
            transformed[key] = _apply_ops(transformed[key])

        if 'input' in transformed and torch.is_tensor(transformed['input']):
            tensor = transformed['input'].clone()
            orig_min = transformed['input'].min()
            orig_max = transformed['input'].max()
            if brightness_enabled:
                tensor = torch.clamp(tensor + brightness_delta, min=orig_min, max=orig_max)
            if noise_enabled:
                tensor = tensor + torch.randn_like(tensor) * noise_std
            transformed['input'] = tensor

        return transformed

    return _transform


def create_dataset(
    config: Dict[str, Any],
    entity_type: str,
    split: str,
    data_dir: str,
    *,
    transform=None,
    apply_transforms: Optional[bool] = None,
):
    """Create dataset based on entity type."""
    if entity_type == 'lung':
        dataset_cfg = config.get('dataset', {})
        data_cfg = config.get('data', {})

        dataset_type = dataset_cfg.get('dataset_type')
        if not dataset_type:
            dataset_type = 'patches' if data_cfg.get('use_patch_dataset', False) else 'images'
        dataset_type = str(dataset_type).lower()

        base_dir = Path(data_dir)

        def _resolve_h5_path(default_subdir: str) -> Path:
            h5_spec = dataset_cfg.get('data_h5') or dataset_cfg.get('h5_path')
            if isinstance(h5_spec, dict):
                candidate = h5_spec.get(split)
            else:
                candidate = h5_spec
            if candidate:
                candidate = candidate.format(split=split)
                candidate_path = Path(candidate)
                if not candidate_path.is_absolute():
                    candidate_path = base_dir / candidate_path
            else:
                candidate_path = base_dir / default_subdir / f"{split}.h5"
            return candidate_path.resolve()

        if dataset_type.startswith('patch'):
            h5_path = _resolve_h5_path('processed_patches')
            kwargs = {'config': config}
            if transform is not None:
                kwargs['transform'] = transform
            if apply_transforms is not None:
                kwargs['apply_transforms'] = apply_transforms
            return PatchDataset(h5_path, **kwargs)
        else:
            h5_path = _resolve_h5_path('processed_images')
            kwargs = {'config': config}
            if transform is not None:
                kwargs['transform'] = transform
            if apply_transforms is not None:
                kwargs['apply_transforms'] = apply_transforms
            return ImageH5Dataset(h5_path, **kwargs)

    if entity_type == 'hnc':
        if config.get('data', {}).get('use_patch_dataset', False):
            return HNCPatchDataset(data_dir, config, split)
        return HNCDataset(data_dir, config, split)

    raise ValueError(f"Unknown entity type: {entity_type}")


class _LimitedDataset(Dataset):
    """Lightweight wrapper that limits a dataset to the first N samples."""

    def __init__(self, base_dataset: Dataset, limit: int):
        self.base_dataset = base_dataset
        self.limit = max(0, min(len(base_dataset), int(limit)))
        self.collate_fn = getattr(base_dataset, 'collate_fn', None)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        if index >= self.limit:
            raise IndexError(index)
        return self.base_dataset[index]

    def __getattr__(self, item):
        return getattr(self.base_dataset, item)


class _SliceDataset(Dataset):
    """View a volumetric dataset as a collection of 2D slices."""

    def __init__(self, base_dataset: Dataset):
        if len(base_dataset) == 0:
            raise ValueError("Base dataset is empty; cannot construct slice view.")

        self.base_dataset = base_dataset
        self.collate_fn = getattr(base_dataset, 'collate_fn', None)

        sample = base_dataset[0]
        if 'input' not in sample or not torch.is_tensor(sample['input']):
            raise ValueError("Dataset sample must contain tensor entry 'input' for slice extraction.")

        input_tensor = sample['input']
        if input_tensor.ndim < 3:
            raise ValueError("Input tensor is not volumetric; cannot create slice view.")

        self.depth = input_tensor.shape[1] if input_tensor.ndim >= 4 else 1
        self.input_channels = getattr(base_dataset, 'input_channels', input_tensor.shape[0])
        self._length = len(base_dataset) * self.depth

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        patch_idx, slice_idx = divmod(index, self.depth)
        sample = self.base_dataset[patch_idx]
        slice_sample: Dict[str, Any] = {}

        for key, value in sample.items():
            if torch.is_tensor(value):
                if value.ndim == 4:
                    slice_sample[key] = value[:, slice_idx, :, :]
                else:
                    slice_sample[key] = value
            else:
                slice_sample[key] = value

        slice_sample['slice_index'] = slice_idx
        slice_sample['patch_index'] = patch_idx

        metadata = slice_sample.get('metadata')
        if isinstance(metadata, dict):
            enriched = dict(metadata)
            enriched['slice_index'] = slice_idx
            slice_sample['metadata'] = enriched

        return slice_sample


def create_data_loaders(
    config: Dict[str, Any],
    entity_type: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    dataset_cfg = config.get('dataset', {})
    if entity_type is None:
        entity_type = config.get('entity', 'lung')
    if data_dir is None:
        data_dir = (
            dataset_cfg.get('data_root')
            or dataset_cfg.get('base_dir')
            or config.get('data', {}).get('root_dir')
            or '.'
        )
    data_dir = str(data_dir)
    model_cfg = config.setdefault('model', {})
    transforms_cfg = config.get('transforms', {})

    dataset_is_2d = dataset_cfg.get('is_2d')
    model_is_2d = model_cfg.get('is_2d')
    is_2d = bool(dataset_is_2d if dataset_is_2d is not None else model_is_2d)
    model_cfg['is_2d'] = is_2d

    transform = build_transform_pipeline(transforms_cfg) if transforms_cfg else None
    apply_transforms = dataset_cfg.get('apply_transforms') if dataset_cfg.get('apply_transforms') is not None else None

    train_dataset = create_dataset(
        config,
        entity_type,
        'train',
        data_dir,
        transform=transform,
        apply_transforms=apply_transforms,
    )
    val_dataset = create_dataset(
        config,
        entity_type,
        'val',
        data_dir,
        transform=None,
        apply_transforms=False,
    )

    if is_2d:
        train_dataset = _SliceDataset(train_dataset)
        val_dataset = _SliceDataset(val_dataset)

    inferred_channels = getattr(train_dataset, 'input_channels', None)
    if isinstance(inferred_channels, (list, tuple)):
        inferred_channels = len(inferred_channels)
    if isinstance(inferred_channels, int) and inferred_channels > 0:
        model_cfg.setdefault('in_channels', inferred_channels)
    model_cfg.setdefault('output_channels', 1)

    if dataset_cfg.get('test_mode', False):
        limit = int(dataset_cfg.get('n_test_samples', 20))
        if limit > 0:
            logging.getLogger(__name__).info(
                "Test mode enabled: restricting datasets to %d samples", limit
            )
            train_dataset = _LimitedDataset(train_dataset, limit)
            val_dataset = _LimitedDataset(val_dataset, limit)

    if dataset_cfg.get('limit_train_samples'):
        train_dataset = _LimitedDataset(train_dataset, dataset_cfg['limit_train_samples'])
    if dataset_cfg.get('limit_val_samples'):
        val_dataset = _LimitedDataset(val_dataset, dataset_cfg['limit_val_samples'])

    batch_size = config.get('training', {}).get('batch_size', 1)
    num_workers = dataset_cfg.get('num_workers', config.get('data', {}).get('num_workers', 0))
    pin_memory = bool(dataset_cfg.get('pin_memory', False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=getattr(train_dataset, 'collate_fn', None),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=getattr(val_dataset, 'collate_fn', None),
    )

    return train_loader, val_loader


__all__ = [
    "build_transform_pipeline",
    "create_dataset",
    "create_data_loaders",
]
