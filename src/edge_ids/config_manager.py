# Configuration update control

from __future__ import annotations
from pathlib import Path 
from typing import Any 
import yaml 

ALLOWED_UPDATES = {
    'security.alert_confidence_threshold',
    'feature_reduction.k',
    'data.test_size'
}

# Update config if an explicitly whitelisted option
def update_config_value(path: str | Path, key: str, value: Any) -> dict[str, Any]:

    if key not in ALLOWED_UPDATES:
        raise ValueError(f"Configuration key '{key}' is not editable through the dashboard.")

    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    section, name = key.split('.', 1)
    config.setdefault(section, {})[name] = value

    # Simple rollback point
    backup = path.with_suffix(path.suffix + '.bak')
    backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return config 

