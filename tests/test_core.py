# Unit tests for the prototype's core modules.

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.alerts import AlertManager
from edge_ids.auth import authenticate, create_user_store
from edge_ids.config_manager import update_config_value
from edge_ids.data import infer_label_column, load_dataset, normalize_binary_labels, split_dataset
from edge_ids.evaluation import classification_metrics, safe_confidence
from edge_ids.integrity import sha256_file, verify_sha256
from edge_ids.models import SUPPORTED_MODELS, build_model
from edge_ids.preprocessing import FeaturePipeline
from edge_ids.registry import ModelRegistry
from edge_ids.validation import OutputValidationError, validate_benchmark_frame, validate_benchmark_csv


def sample_frame(rows: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            'bytes': np.arange(rows) * 10 + 100,
            'packets': np.arange(rows) + 1,
            'protocol': ['TCP', 'UDP', 'TCP', 'UDP'] * (rows // 4) + ['TCP'] * (rows % 4),
        }
    )
    y = pd.Series([0, 1] * (rows // 2) + ([0] if rows % 2 else []), name='label')
    return X, y


def test_normalize_binary_labels_string_and_numeric():
    labels = pd.Series(['Benign', 'DDoS', 'Scanning'])
    assert normalize_binary_labels(labels).tolist() == [0, 1, 1]

    numeric = pd.Series([0, 1, 1, 0])
    assert normalize_binary_labels(numeric).tolist() == [0, 1, 1, 0]


def test_infer_label_column_prefers_configured_name():
    frame = pd.DataFrame({'target': [0, 1], 'label': [1, 0]})
    assert infer_label_column(frame, 'target') == 'target'


def test_infer_label_column_rejects_missing_configured_column():
    frame = pd.DataFrame({'label': [0, 1]})
    with pytest.raises(ValueError, match='was not found'):
        infer_label_column(frame, 'attack')


def test_load_dataset_drops_metadata_and_normalizes_labels(tmp_path):
    path = tmp_path / 'flows.csv'
    pd.DataFrame(
        {
            'Label': ['Benign', 'DDoS', 'Benign'],
            'Flow_ID': [1, 2, 3],
            'bytes': [100, 200, 150],
        }
    ).to_csv(path, index=False)

    X, y, label = load_dataset(path, drop_columns=['Flow_ID'])
    assert label == 'Label'
    assert list(X.columns) == ['bytes']
    assert y.tolist() == [0, 1, 0]


def test_split_dataset_is_stratified_and_reproducible():
    X, y = sample_frame(40)
    first = split_dataset(X, y, test_size=0.25, random_state=42)
    second = split_dataset(X, y, test_size=0.25, random_state=42)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    assert first.y_train.mean() == pytest.approx(0.5)


def test_feature_pipeline_requires_fit_before_transform():
    pipeline = FeaturePipeline()
    X, _ = sample_frame()
    with pytest.raises(RuntimeError, match='has not been fitted'):
        pipeline.transform(X)


def test_feature_pipeline_reduces_features():
    X, y = sample_frame(30)
    pipeline = FeaturePipeline(feature_reduction_enabled=True, k=2)
    transformed = pipeline.fit_transform(X, y)
    assert transformed.shape[1] == 2
    assert pipeline.output_feature_count == 2


def test_feature_pipeline_rejects_non_positive_k():
    with pytest.raises(ValueError, match='greater than zero'):
        FeaturePipeline(feature_reduction_enabled=True, k=0)


def test_all_model_factories_are_supported():
    config = {
        'models': {
            'decision_tree': {},
            'random_forest': {'n_estimators': 5, 'n_jobs': 1},
            'xgboost': {'n_estimators': 5},
            'mlp': {'max_iter': 5, 'batch_size': 8},
        }
    }
    for name in SUPPORTED_MODELS:
        model = build_model(name, config, random_state=42)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


def test_classification_metrics_contains_required_fields():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics['accuracy'] == pytest.approx(0.75)
    assert metrics['precision'] == pytest.approx(2 / 3)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['f1'] == pytest.approx(0.8)
    assert metrics['confusion_matrix'] == [[1, 1], [0, 2]]


def test_safe_confidence_returns_probability_maxima():
    class Dummy:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.8, 0.2]])

    confidence = safe_confidence(Dummy(), np.zeros((2, 1)), np.array([1, 0]))
    assert confidence.tolist() == [0.9, 0.8]


def test_integrity_hash_and_tamper_detection(tmp_path):
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'original')
    digest = sha256_file(path)
    assert verify_sha256(path, digest)

    path.write_bytes(b'tampered')
    assert not verify_sha256(path, digest)


def test_model_registry_round_trip_and_manifest(tmp_path):
    registry = ModelRegistry(tmp_path / 'models')
    bundle = {'model': 'dummy', 'metadata': {'test': True}}
    manifest = registry.save('demo_v1', bundle, {'model_name': 'demo'})

    assert manifest['model_version'] == 'demo_v1'
    loaded, loaded_manifest = registry.load('demo_v1')
    assert loaded == bundle
    assert loaded_manifest['sha256'] == manifest['sha256']


def test_model_registry_detects_tampered_artifact(tmp_path):
    registry = ModelRegistry(tmp_path / 'models')
    registry.save('demo_v1', {'model': 'dummy'}, {})
    artifact = tmp_path / 'models' / 'demo_v1.joblib'
    artifact.write_bytes(artifact.read_bytes() + b'tamper')

    with pytest.raises(RuntimeError, match='Integrity verification failed'):
        registry.load('demo_v1')


def test_authentication_and_rbac_store(tmp_path):
    user_file = tmp_path / 'users.json'
    create_user_store(user_file, 'admin', 'correct-password', 'ADMIN')

    assert authenticate(user_file, 'admin', 'correct-password') == {
        'username': 'admin',
        'role': 'ADMIN',
    }
    assert authenticate(user_file, 'admin', 'wrong-password') is None
    assert authenticate(user_file, 'missing', 'correct-password') is None


def test_authentication_rejects_invalid_role(tmp_path):
    with pytest.raises(ValueError, match='Role must be ADMIN or ANALYST'):
        create_user_store(tmp_path / 'users.json', 'user', 'password', 'operator')


def test_alert_managers_keep_log_destinations_isolated(tmp_path):
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    first = AlertManager(first_dir, confidence_threshold=0.5)
    second = AlertManager(second_dir, confidence_threshold=0.5)

    first.inspect_predictions([{'prediction': 1, 'confidence': 0.9, 'model_version': 'v1'}])
    second.inspect_predictions([{'prediction': 1, 'confidence': 0.9, 'model_version': 'v2'}])

    assert (first_dir / 'alerts.jsonl').exists()
    assert (second_dir / 'alerts.jsonl').exists()
    assert 'v1' in (first_dir / 'alerts.jsonl').read_text(encoding='utf-8')
    assert 'v2' in (second_dir / 'alerts.jsonl').read_text(encoding='utf-8')


def test_alert_manager_only_alerts_confident_malicious_predictions(tmp_path):
    manager = AlertManager(tmp_path, confidence_threshold=0.80)
    alerts = manager.inspect_predictions(
        [
            {'prediction': 1, 'confidence': 0.95, 'model_version': 'v1', 'timestamp_utc': 't1'},
            {'prediction': 1, 'confidence': 0.50, 'model_version': 'v1', 'timestamp_utc': 't2'},
            {'prediction': 0, 'confidence': 0.99, 'model_version': 'v1', 'timestamp_utc': 't3'},
        ]
    )

    assert len(alerts) == 1
    payload = json.loads((tmp_path / 'alerts.jsonl').read_text(encoding='utf-8').strip())
    assert payload['type'] == 'intrusion_detected'
    assert payload['confidence'] == pytest.approx(0.95)


def test_config_manager_whitelists_updates(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(
        'security:\n  alert_confidence_threshold: 0.8\nfeature_reduction:\n  k: 64\n',
        encoding='utf-8',
    )
    config = update_config_value(path, 'feature_reduction.k', 10)
    assert config['feature_reduction']['k'] == 10
    assert path.with_suffix('.yaml.bak').exists()

    with pytest.raises(ValueError, match='not editable'):
        update_config_value(path, 'paths.model_dir', 'elsewhere')


def test_benchmark_output_validation_accepts_valid_frame():
    frame = pd.DataFrame(
        {
            'model': ['decision_tree'],
            'feature_reduction': [False],
            'model_version': ['decision_tree_v1_full'],
            'accuracy': [0.9],
            'precision': [0.8],
            'recall': [0.85],
            'f1': [0.82],
            'latency_ms_per_sample': [0.1],
            'cpu_percent': [20.0],
            'rss_after_mb': [150.0],
            'rss_delta_mb': [1.0],
            'output_feature_count': [19],
            'train_rows': [100],
            'test_rows': [25],
            'model_size_bytes': [5000],
        }
    )
    validate_benchmark_frame(frame)


def test_benchmark_output_validation_rejects_bad_metric():
    frame = pd.DataFrame(
        {
            'model': ['decision_tree'],
            'feature_reduction': [False],
            'model_version': ['v1'],
            'accuracy': [1.1],
            'precision': [0.8],
            'recall': [0.85],
            'f1': [0.82],
            'latency_ms_per_sample': [0.1],
            'cpu_percent': [20.0],
            'rss_after_mb': [150.0],
            'rss_delta_mb': [1.0],
            'output_feature_count': [19],
            'train_rows': [100],
            'test_rows': [25],
            'model_size_bytes': [5000],
        }
    )
    with pytest.raises(OutputValidationError, match='between 0 and 1'):
        validate_benchmark_frame(frame)


def test_benchmark_csv_validation_reads_existing_result():
    path = ROOT / 'results' / 'demo_benchmark.csv'
    if not path.exists():
        pytest.skip('Demo benchmark file is not present')
    frame = validate_benchmark_csv(path)
    assert not frame.empty
