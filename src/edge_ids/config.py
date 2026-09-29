# Config loading

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

# Load a YAML config file and validate structure
def load_config(path: str| Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError('Config file must contain a YAML mapping/object.')
    return config

# create configured runtime directories when they do not exist
def ensure_directories(config: dict[str, Any], root: str | Path = '.') -> None:
    root = Path(root)
    paths = config.get('paths', {})
    for key in ('data_dir', 'model_dir', 'log_dir', 'result_dir'):
        directory = root / paths.get(key, key)
        directory.mkdir(parents=True, exist_ok=True)
