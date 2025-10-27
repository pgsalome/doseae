"""Legacy compatibility wrapper for older experiment scripts."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional, Tuple

from .loaders import create_data_loaders as _create_data_loaders


def create_data_loaders(
    config_path: str,
    mapping_file_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[object, object, Optional[object]]:
    """Legacy helper that mirrors the old cached_dataset.create_data_loaders API.

    Parameters match the historical signature but most arguments are ignored in
    the new pipeline. The function now loads the configuration from
    ``config_path`` and delegates to :func:`datasets.loaders.create_data_loaders`.
    A third return value (test loader) is kept for backwards compatibility and
    is currently ``None``.
    """
    config_path = Path(config_path)
    with config_path.open('r') as f:
        config = yaml.safe_load(f)

    entity_type = config.get('entity', 'lung')

    dataset_cfg = config.get('dataset', {})
    data_dir = dataset_cfg.get('data_root') or dataset_cfg.get('base_dir')
    if not data_dir:
        data_dir = cache_dir or config_path.parent
    data_dir = str(data_dir)

    train_loader, val_loader = _create_data_loaders(config, entity_type, data_dir)
    return train_loader, val_loader, None
