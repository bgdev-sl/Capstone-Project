# ML interface for batch inference and resource measurement

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any 
import pandas as pd
from .alerts import AlertManager
from .evaluation import safe_confidence
from .monitoring import ResourceMonitor

# Run a registered model against flow features
class InferenceEngine:
    def __init__(self, bundle: dict[str, Any], model_version: str, alert_manager: AlertManager | None = None):
        self.model = bundle['model']
        self.features = bundle['features']
        self.model_version = model_version
        self.alert_manager = alert_manager

    # Predict benign/malicious labels and return measurements
    def predict(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if frame.empty:
            raise ValueError('Cannot perform inference on an empty DataFrame.')

        monitor = ResourceMonitor()

        def _run():
            transformed = self.features.transform(frame)
            prediction = self.model.predict(transformed)
            confidence = safe_confidence(self.model, transformed, prediction)
            return prediction, confidence

        (prediction, confidence), resources = monitor.measure(_run, sample_count=len(frame))

        timestamp = datetime.now(timezone.utc).isoformat()
        result = frame.copy()
        result['prediction'] = prediction.astype(int)
        result['class'] = result['prediction'].map({0: 'benign', 1: 'malicious'})
        result['confidence'] = confidence
        result['model_version'] = self.model_version
        result['timestamp_utc'] = timestamp

        if self.alert_manager:
            self.alert_manager.inspect_predictions(result.to_dict(orient='records'))

        return result, resources.as_dict()
