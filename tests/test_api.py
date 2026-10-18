# HTTP-level integration tests for the FastAPI adapter and shared services

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.api import create_app
from edge_ids.auth import create_user_store
from edge_ids.preprocessing import FeaturePipeline
from edge_ids.registry import ModelRegistry

# Build isolated API with two roles and small test-only model
def _prepared_app(tmp_path: Path) -> TestClient:
    (tmp_path / 'configs').mkdir(parents=True)
    config_path = tmp_path / 'configs' / 'config.yaml'
    config_path.write_text(
        """project:\n  random_state: 42\npaths:\n  data_dir: data\n  model_dir: models\n  log_dir: logs\n  result_dir: results\n  user_file: configs/users.json\ndata:\n  label_column: Label\n  drop_columns: [Flow_ID, Timestamp]\n  test_size: 0.2\nfeature_reduction:\n  enabled: false\n  k: 2\nsecurity:\n  alert_confidence_threshold: 0.5\n  log_rotation_mb: 1\n  log_backup_count: 1\n""",
        encoding='utf-8'
    )
    create_user_store(tmp_path / 'configs/users.json', 'admin', 'admin-pass', 'ADMIN')
    create_user_store(tmp_path / 'configs/users.json', 'analyst', 'analyst-pass', 'ANALYST')

    X = pd.DataFrame({'byte_count': np.arange(40), 'packet_count': np.arange(40) * 2})
    y = pd.Series([0] * 20 + [1] * 20)
    feature_pipeline = FeaturePipeline().fit(X, y)
    transformed = feature_pipeline.transform(X)
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(transformed, y)
    registry = ModelRegistry(tmp_path / 'models')
    registry.save(
        'decision_tree_test_v1',
        {'model': model, 'features': feature_pipeline, 'metadata': {'model_name': 'decision_tree'}},
        {'model_name': 'decision_tree', 'output_feature_count': 2}
    )
    return TestClient(create_app(root=tmp_path, config_path=config_path))

def test_health_endpoint_is_public(tmp_path):
    client = _prepared_app(tmp_path)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_models_endpoint_requires_authentication_and_lists_registry(tmp_path):
    client = _prepared_app(tmp_path)
    assert client.get('/api/v1/models').status_code == 401
    response = client.get('/api/v1/models', auth=('analyst', 'analyst-pass'))
    assert response.status_code == 200
    assert response.json()['models'][0]['model_version'] == 'decision_tree_test_v1'

def test_predict_api_returns_predictions_and_resources_and_drops_label(tmp_path):
    client = _prepared_app(tmp_path)
    response = client.post(
        '/api/v1/predict',
        auth=('analyst', 'analyst-pass'),
        json={
            'model_version': 'decision_tree_test_v1',
            'records': [
                {'byte_count': 1, 'packet_count': 2, 'Label': 'DDoS'},
                {'byte_count': 35, 'packet_count': 70, 'Label': 'Benign'}
            ]
        }
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['count'] == 2
    assert len(body['predictions']) == 2
    assert 'Label' not in body['predictions'][0]
    assert 'byte_count' not in body['predictions'][0]
    assert 'packet_count' not in body['predictions'][0]
    assert {'prediction', 'class', 'confidence', 'model_version', 'timestamp_utc'} <= set(body['predictions'][0])
    assert body['resource_metrics']['latency_ms_per_sample'] >= 0
    assert body['resource_metrics']['throughput_samples_per_second'] >= 0

def test_predict_api_returns_only_classification_fields_not_input_data(tmp_path):
    client = _prepared_app(tmp_path)
    response = client.post(
        '/api/v1/predict',
        auth=('analyst', 'analyst-pass'),
        json={
            'model_version': 'decision_tree_test_v1',
            'records': [
                {
                    'byte_count': 1,
                    'packet_count': 2,
                    'Label': 'Benign',
                    'Flow_ID': 'private-flow-id',
                    'Timestamp': '2026-10-15T12:00:00Z'
                }
            ]
        }
    )
    assert response.status_code == 200, response.text
    prediction = response.json()['predictions'][0]
    assert set(prediction) == {
        'prediction', 'class', 'confidence', 'model_version', 'timestamp_utc'
    }
    assert 'private-flow-id' not in response.text
    assert 'byte_count' not in response.text

def test_admin_config_requires_admin_and_validates_values(tmp_path):
    client = _prepared_app(tmp_path)
    payload = {'key': 'security.alert_confidence_threshold', 'value': 0.7}
    denied = client.patch('/api/v1/admin/config', auth=('analyst', 'analyst-pass'), json=payload)
    assert denied.status_code == 403

    updated = client.patch('/api/v1/admin/config', auth=('admin', 'admin-pass'), json=payload)
    assert updated.status_code == 200, updated.text
    assert updated.json()['backup_created'] is True
    assert (tmp_path / 'configs/config.yaml.bak').exists()

    invalid = client.patch(
        '/api/v1/admin/config',
        auth=('admin', 'admin-pass'),
        json={'key': 'paths.model_dir', 'value': 'outside'}
    )
    assert invalid.status_code == 422
