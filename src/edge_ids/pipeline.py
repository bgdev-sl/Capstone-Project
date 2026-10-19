"""High-level application service tying the architecture modules together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .alerts import AlertManager
from .config import ensure_directories
from .data import load_dataset
from .inference import InferenceEngine
from .registry import ModelRegistry


class EdgeIDSApplication:
    """Coordinate registry, integrity verification, inference, logging, and config."""

    def __init__(self, config: dict[str, Any], root: str | Path = "."):
        self.config = config
        self.root = Path(root)
        ensure_directories(config, root=self.root)
        paths = config.get("paths", {})
        self.model_registry = ModelRegistry(self.root / paths.get("model_dir", "models"))
        security = config.get("security", {})
        self.alert_manager = AlertManager(
            self.root / paths.get("log_dir", "logs"),
            confidence_threshold=security.get("alert_confidence_threshold", 0.80),
            rotation_mb=security.get("log_rotation_mb", 5),
            backup_count=security.get("log_backup_count", 3),
        )

    def available_models(self) -> list[dict[str, Any]]:
        return self.model_registry.list_models()

    def load_model(self, model_version: str) -> InferenceEngine:
        bundle, _manifest = self.model_registry.load(model_version, verify_integrity=True)
        return InferenceEngine(bundle, model_version, self.alert_manager)

    def predict_csv(self, model_version: str, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Predict using a CSV containing the raw feature columns used at training time."""
        return self.predict_frame(model_version, pd.read_csv(path))

    def predict_frame(self, model_version: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run inference on an in-memory frame, shared by the UI and HTTP API.

        The fitted ColumnTransformer selects only the training-time feature columns,
        but known target/metadata columns are removed up front so they cannot leak
        into returned predictions or downstream logs.
        """
        from .data import LABEL_CANDIDATES

        if frame.empty:
            raise ValueError("Cannot perform inference on an empty dataset.")
        frame = frame.copy()
        label_column = self.config.get("data", {}).get("label_column")
        possible_labels = set(LABEL_CANDIDATES)
        if label_column:
            possible_labels.add(label_column)
        removable = [column for column in frame.columns if column in possible_labels]
        if removable:
            frame = frame.drop(columns=removable)
        for column in self.config.get("data", {}).get("drop_columns", []):
            if column in frame.columns:
                frame = frame.drop(columns=[column])

        if frame.empty or len(frame.columns) == 0:
            raise ValueError("No usable feature columns were supplied for inference.")
        return self.load_model(model_version).predict(frame)
