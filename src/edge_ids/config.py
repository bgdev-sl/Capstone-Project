"""Configuration loading helpers.

The prototype deliberately keeps configuration external to the code so the same
software can be used for CICIoT2023, ToN-IoT, or another compatible flow dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate the top-level structure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping/object.")

    return config


def ensure_directories(config: dict[str, Any], root: str | Path = ".") -> None:
    """Create configured runtime directories when they do not exist."""
    root = Path(root)
    paths = config.get("paths", {})
    for key in ("data_dir", "model_dir", "log_dir", "result_dir"):
        directory = root / paths.get(key, key)
        directory.mkdir(parents=True, exist_ok=True)
