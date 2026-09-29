# Alert generation and logging

from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable 

# Serialize log records as one JSON object
class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, 'payload', None) 
        if payload is None:
            payload = {'message': record.getMessage()}
        payload = {'level': record.levelname, 'logger': record.name, **payload}
        return json.dumps(payload, ensure_ascii=False, default=str)

# Generate alerts for malicious predictions
class AlertManager:
    def __init__(self, log_dir: str | Path, confidence_threshold: float = 0.80, rotation_mb: int = 5, backup_count: int = 3):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger('edge_ids.alerts')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = RotatingFileHandler(
                log_dir / 'alerts.jsonl',
                maxBytes=rotation_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
            handler.setFormatter(JsonLineFormatter())
            self.logger.addHandler(handler)

    def emit(self, alert: dict) -> None:
        self.logger.info('alert', extra={'payload': alert})

    # Create alert records
    def inspect_predictions(self, predictions: Iterable[dict]) -> list[dict]:
        alerts = []
        for item in predictions:
            malicious = int(item.get('prediction', 0)) == 1
            confidence = float(item.get('confidence', 0.0))
            if malicious and confidence >= self.confidence_threshold:
                alert = {
                    'type': 'intrusion_detected',
                    'confidence': confidence,
                    'model_version': item.get('model_version'),
                    'timestamp_utc': item.get('timestamp_utc')
                }
                self.emit(alert)
                alerts.append(alert)
        return alerts
    