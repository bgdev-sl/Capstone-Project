# Model training workflow shared by CLI scripts and the dashboard

from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import DatasetSplit, split_dataset
from .evaluation import classification_metrics, safe_confidence
from .models import build_model
from .monitoring import ResourceMonitor
from .preprocessing import FeaturePipeline
from .registry import ModelRegistry

# Train one model and save a versioned bundle in the model registry
def train_one(
    model_name: str,
    X,
    y,
    config: dict[str, Any],
    model_version: str,
    feature_reduction: bool,
    feature_k: int,
    registry: ModelRegistry
) -> dict[str, Any]:
    random_state = config.get('project', {}).get('random_state', 42)
    split: DatasetSplit = split_dataset(
        X,
        y,
        test_size=config.get('data', {}).get('test_size', 0.20),
        random_state=random_state,
    )

    features = FeaturePipeline(
        feature_reduction_enabled=feature_reduction,
        k=feature_k,
    )
    X_train = features.fit_transform(split.X_train, split.y_train)

    model = build_model(model_name, config, random_state=random_state)

    train_monitor = ResourceMonitor()
    _, train_resources = train_monitor.measure(
        lambda: model.fit(X_train, split.y_train),
        sample_count=len(split.X_train)
    )

    # Warm the full preprocessing+prediction path, then repeat the same batch
    # to reduce timer/CPU-counter quantization effects for very fast models.
    # This is warmed software-pipeline latency, not cold-start or network latency.
    _preprocess_and_predict(model, features, split.X_test)
    inference_repeats = int(config.get('evaluation', {}).get('inference_repeats', 20))
    if inference_repeats < 1:
        raise ValueError('evaluation.inference_repeats must be at least 1')
    infer_monitor = ResourceMonitor()
    (predictions, confidence), inference_resources = infer_monitor.measure(
        lambda: _predict_repeated(model, features, split.X_test, inference_repeats),
        sample_count=len(split.X_test) * inference_repeats
    )

    metrics = classification_metrics(split.y_test, predictions)
    metadata = {
        'model_name': model_name,
        'model_version': model_version,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'feature_reduction': feature_reduction,
        'feature_k': feature_k if feature_reduction else None,
        'output_feature_count': features.output_feature_count,
        'train_rows': len(split.X_train),
        'test_rows': len(split.X_test),
        'inference_repeats': inference_repeats,
        'metrics': metrics,
        'train_resource_metrics': asdict(train_resources),
        'inference_resource_metrics': asdict(inference_resources)
    }

    bundle = {
        'model': model,
        'features': features,
        'metadata': metadata,
    }

    manifest = registry.save(model_version, bundle, metadata)
    return {'manifest': manifest, **metadata, 'confidence_mean': float(confidence.mean())}

def _predict(model, transformed_X):
    predictions = model.predict(transformed_X)
    confidence = safe_confidence(model, transformed_X, predictions)
    return predictions, confidence

# Apply the fitted feature transform and classify one raw feature batch
def _preprocess_and_predict(model, features, raw_X):
    transformed = features.transform(raw_X)
    return _predict(model, transformed)

# Repeat full inference and return final batch's outputs for metrics
def _predict_repeated(model, features, raw_X, repeats: int):
    result = None
    for _ in range(repeats):
        result = _preprocess_and_predict(model, features, raw_X)
    return result
