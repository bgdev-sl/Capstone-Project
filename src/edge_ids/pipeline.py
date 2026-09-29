# Tying the architecture modules together

from __future__ import annotations
from pathlib import Path
from typing import Any 
import pandas as pd
from .alerts import AlertManager
from .config import ensure_directories
from .data import load_dataset
from .inference import InferenceEngine
from .registry import ModelRegistry

# Coordinate registry, integrity verification, inference, logging, and config
class EdgeIDSApplication:
    def __init__(self, config: dict[str, Any], root: str | Path = '.'):
        self.config = config
        self.root = Path(root)
        ensure_directories(config, root=self.root)
        paths = config.get('paths', {})
        self.model_registry = ModelRegistry(self.root / paths.get('model_dir', 'models'))
        security = config.get('security', {})
        self.alert_manager = AlertManager(
            self.root / paths.get('log_dir', 'logs'),
            confidence_threshold=security.get('alert_confidence_threshold', 0.80),
            rotation_mb=security.get('log_rotation_mb', 5),
            backup_count=security.get('log_backup_count', 3)
        )

    def available_models(self) -> list[dict[str, Any]]:
        return self.model_registry.list_models()

    def load_model(self, model_version: str) -> InferenceEngine:
        bundle, _manifest = self.model_registry.load(model_version, verify_integrity=True)
        return InferenceEngine(bundle, model_version, self.alert_manager)

    # Predict using a CSV containing same features as training set
    def predict_csv(self, model_version: str, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = pd.read_csv(path)
        label_column = self.config.get('data', {}).get('label_column')
        if label_column and label_column in frame.columns:
            frame = frame.drop(columns=[label_column])
        for column in self.config.get('data', {}).get('drop_columns', []):
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        return self.load_model(model_version).predict(frame)
