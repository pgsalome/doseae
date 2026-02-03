"""
Shared dataset construction utilities used by training, inference, and tuning.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn.functional as F
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
    anatomy_cfg = transform_cfg.get('anatomy_deformation', {}) or {}
    anatomy_enabled = bool(anatomy_cfg.get('enabled', False))
    anatomy_prob = float(anatomy_cfg.get('probability', 0.5))
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

        def _generate_anatomy_theta(sample_shape: torch.Size) -> Optional[torch.Tensor]:
            if not anatomy_enabled or random.random() >= anatomy_prob:
                return None

            if len(sample_shape) < 4:
                return None  # Expecting (C, D, H, W)

            _, depth, height, width = sample_shape[-4], sample_shape[-3], sample_shape[-2], sample_shape[-1]

            max_rot = float(anatomy_cfg.get('max_rotation_deg', 5.0))
            max_trans = float(anatomy_cfg.get('max_translation_voxels', 2.0))
            scale_range = anatomy_cfg.get('scale_range', [0.95, 1.05])
            if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
                scale_range = [0.95, 1.05]

            rx = math.radians(random.uniform(-max_rot, max_rot))
            ry = math.radians(random.uniform(-max_rot, max_rot))
            rz = math.radians(random.uniform(-max_rot, max_rot))

            sx = random.uniform(scale_range[0], scale_range[1])
            sy = random.uniform(scale_range[0], scale_range[1])
            sz = random.uniform(scale_range[0], scale_range[1])

            tx = random.uniform(-max_trans, max_trans)
            ty = random.uniform(-max_trans, max_trans)
            tz = random.uniform(-max_trans, max_trans)

            cx = math.cos(rx)
            sx_sin = math.sin(rx)
            cy = math.cos(ry)
            sy_sin = math.sin(ry)
            cz = math.cos(rz)
            sz_sin = math.sin(rz)

            rot_x = torch.tensor([[1, 0, 0],
                                  [0, cx, -sx_sin],
                                  [0, sx_sin, cx]], dtype=torch.float32)
            rot_y = torch.tensor([[cy, 0, sy_sin],
                                  [0, 1, 0],
                                  [-sy_sin, 0, cy]], dtype=torch.float32)
            rot_z = torch.tensor([[cz, -sz_sin, 0],
                                  [sz_sin, cz, 0],
                                  [0, 0, 1]], dtype=torch.float32)

            rot = rot_z @ rot_y @ rot_x
            scale_matrix = torch.diag(torch.tensor([sx, sy, sz], dtype=torch.float32))
            affine = rot @ scale_matrix

            theta = torch.zeros((3, 4), dtype=torch.float32)
            theta[:, :3] = affine
            # Normalize translations to [-1, 1]
            theta[0, 3] = 2.0 * tx / max(width - 1, 1)
            theta[1, 3] = 2.0 * ty / max(height - 1, 1)
            theta[2, 3] = 2.0 * tz / max(depth - 1, 1)

            return theta

        def _apply_anatomy(tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            if tensor.dim() < 4:
                return tensor
            device = tensor.device
            dtype = tensor.dtype
            theta = theta.to(device=device, dtype=torch.float32)

            input_tensor = tensor.unsqueeze(0)
            grid = F.affine_grid(theta.unsqueeze(0), size=input_tensor.shape, align_corners=True)
            warped = F.grid_sample(
                input_tensor.float(),
                grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=True,
            )
            return warped.squeeze(0).to(dtype=dtype)

        def _apply_ops(tensor: torch.Tensor, theta: Optional[torch.Tensor]) -> torch.Tensor:
            result = tensor.clone()
            if dims_to_flip:
                result = torch.flip(result, dims=dims_to_flip)
            if k_rot and result.dim() >= 3:
                result = torch.rot90(result, k_rot, dims=(-2, -1))
            if theta is not None:
                result = _apply_anatomy(result, theta)
            return result

        anatomy_theta = None
        if anatomy_enabled:
            anatomy_theta = _generate_anatomy_theta(transformed[tensor_keys[0]].shape)

        for key in tensor_keys:
            transformed[key] = _apply_ops(transformed[key], anatomy_theta)

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


def collate_medical_batch(batch: Sequence[dict]) -> dict:
    """Custom collate function that tolerates mixed metadata types."""
    if not batch:
        return {}

    collated: Dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        first = values[0]

        if torch.is_tensor(first):
            try:
                collated[key] = torch.stack(values)
            except Exception:
                collated[key] = values
        elif isinstance(first, (int, float, bool)):
            try:
                collated[key] = torch.tensor(values)
            except Exception:
                collated[key] = values
        elif isinstance(first, (list, tuple)):
            collated[key] = list(values)
        elif isinstance(first, dict):
            collated[key] = values
        else:
            collated[key] = values

    return collated


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

    batch_size = int(config.get('training', {}).get('batch_size', 1) or 1)
    if dataset_cfg.get('test_mode', False):
        limit_batches = int(dataset_cfg.get('n_test_samples', dataset_cfg.get('n_test_batches', 20)))
        if limit_batches > 0:
            limit_samples = limit_batches * batch_size
            logging.getLogger(__name__).info(
                "Test mode enabled: restricting datasets to %d batches (%d samples)",
                limit_batches,
                limit_samples,
            )
            train_dataset = _LimitedDataset(train_dataset, limit_samples)
            val_dataset = _LimitedDataset(val_dataset, limit_samples)

    if dataset_cfg.get('limit_train_samples'):
        train_dataset = _LimitedDataset(train_dataset, dataset_cfg['limit_train_samples'])
    if dataset_cfg.get('limit_val_samples'):
        val_dataset = _LimitedDataset(val_dataset, dataset_cfg['limit_val_samples'])

    num_workers = dataset_cfg.get('num_workers', config.get('data', {}).get('num_workers', 0))
    pin_memory = bool(dataset_cfg.get('pin_memory', False))
    collate_fn = getattr(train_dataset, 'collate_fn', None) or collate_medical_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=getattr(val_dataset, 'collate_fn', None) or collate_medical_batch,
    )

    return train_loader, val_loader


__all__ = [
    "build_transform_pipeline",
    "create_dataset",
    "create_data_loaders",
    "collate_medical_batch",
]
