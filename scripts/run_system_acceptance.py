#!/usr/bin/env python3

# Run a bounded live-API acceptance suite against an already-running Edge IDS.

# Required environment variables:
#   EDGE_IDS_ACCEPTANCE_ANALYST_USER / EDGE_IDS_ACCEPTANCE_ANALYST_PASSWORD
#   EDGE_IDS_ACCEPTANCE_ADMIN_USER / EDGE_IDS_ACCEPTANCE_ADMIN_PASSWORD
# Optional:
#   EDGE_IDS_ACCEPTANCE_URL (default http://127.0.0.1:8000)
#   EDGE_IDS_ACCEPTANCE_THRESHOLD (default 0.81)

# This script is for an isolated test instance only. It will submit one approved configuration update using the threshold supplied in the environment.

from __future__ import annotations
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.getenv('EDGE_IDS_ACCEPTANCE_URL', 'http://127.0.0.1:8000').rstrip('/')
ANALYST_USER = os.getenv('EDGE_IDS_ACCEPTANCE_ANALYST_USER')
ANALYST_PASSWORD = os.getenv('EDGE_IDS_ACCEPTANCE_ANALYST_PASSWORD')
ADMIN_USER = os.getenv('EDGE_IDS_ACCEPTANCE_ADMIN_USER')
ADMIN_PASSWORD = os.getenv('EDGE_IDS_ACCEPTANCE_ADMIN_PASSWORD')
THRESHOLD = float(os.getenv('EDGE_IDS_ACCEPTANCE_THRESHOLD', '0.81'))
ROOT = Path(__file__).resolve().parents[1]

results: list[tuple[str, bool, int, int]] = []

def send(
    name: str,
    path: str,
    *,
    method: str = 'GET',
    payload: dict[str, Any] | None = None,
    credentials: tuple[str, str] | None = None,
    expected_status: int = 200,
) -> tuple[dict[str, Any], str]:
    headers = {'Content-Type': 'application/json'}
    if credentials:
        token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode()).decode()
        headers['Authorization'] = f'Basic {token}'
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f'{BASE_URL}{path}', data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status, text = response.status, response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        status, text = exc.code, exc.read().decode('utf-8')
    except Exception as exc:
        print(f'ERROR: {name}: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
    passed = status == expected_status
    results.append((name, passed, status, expected_status))
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {}
    return parsed, text

def verify(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)

def main() -> int:
    if not all([ANALYST_USER, ANALYST_PASSWORD, ADMIN_USER, ADMIN_PASSWORD]):
        raise SystemExit(
            'Set EDGE_IDS_ACCEPTANCE_ANALYST_USER/PASSWORD and '
            'EDGE_IDS_ACCEPTANCE_ADMIN_USER/PASSWORD before running this script.'
        )
    if not 0 <= THRESHOLD <= 1:
        raise SystemExit('EDGE_IDS_ACCEPTANCE_THRESHOLD must be between 0 and 1.')

    request_file = ROOT / 'examples' / 'api_predict_request.json'
    example = json.loads(request_file.read_text(encoding='utf-8'))

    send('Health endpoint', '/health')
    send('Models endpoint rejects missing credentials', '/api/v1/models', expected_status=401)
    send(
        'Models endpoint rejects incorrect password',
        '/api/v1/models',
        credentials=(ANALYST_USER, 'intentionally-wrong-password'),
        expected_status=401
    )
    models, _ = send(
        'Analyst can list model registry',
        '/api/v1/models',
        credentials=(ANALYST_USER, ANALYST_PASSWORD)
    )
    verify(len(models.get('models', [])) >= 4, 'Expected at least four registered model versions')
    send(
        'Analyst can retrieve alerts',
        '/api/v1/alerts',
        credentials=(ANALYST_USER, ANALYST_PASSWORD)
    )

    prediction, raw_prediction = send(
        'Prediction API returns data-minimal result and metrics',
        '/api/v1/predict',
        method='POST',
        payload=example,
        credentials=(ANALYST_USER, ANALYST_PASSWORD)
    )
    verify(prediction.get('count') == len(example['records']), 'Prediction count mismatch')
    verify(len(prediction.get('predictions', [])) == len(example['records']), 'Prediction list mismatch')
    metric_names = prediction.get('resource_metrics', {})
    verify('latency_ms_per_sample' in metric_names, 'Inference latency metric missing')
    output = prediction['predictions'][0]
    verify(
        {'prediction', 'class', 'confidence', 'model_version', 'timestamp_utc'}.issubset(output),
        'Required prediction result fields missing'
    )
    raw_keys = set(example['records'][0])
    verify(not raw_keys.intersection(output), 'Prediction response echoes input feature fields')
    for key in raw_keys:
        if key in {'Flow_Duration', 'Packet_Length_Mean', 'Protocol_Type'}:
            verify(key not in raw_prediction, f'Prediction response contains feature name {key}')

    send(
        'Unknown model version returns 404',
        '/api/v1/predict',
        method='POST',
        payload={'model_version': 'not_a_real_model', 'records': example['records']},
        credentials=(ANALYST_USER, ANALYST_PASSWORD),
        expected_status=404
    )
    send(
        'Empty prediction batch returns 422',
        '/api/v1/predict',
        method='POST',
        payload={'model_version': example['model_version'], 'records': []},
        credentials=(ANALYST_USER, ANALYST_PASSWORD),
        expected_status=422
    )
    send(
        'Analyst cannot modify configuration',
        '/api/v1/admin/config',
        method='PATCH',
        payload={'key': 'security.alert_confidence_threshold', 'value': THRESHOLD},
        credentials=(ANALYST_USER, ANALYST_PASSWORD),
        expected_status=403
    )
    send(
        'Admin cannot modify a non-whitelisted setting',
        '/api/v1/admin/config',
        method='PATCH',
        payload={'key': 'paths.model_dir', 'value': 'unexpected'},
        credentials=(ADMIN_USER, ADMIN_PASSWORD),
        expected_status=422
    )
    admin_response, _ = send(
        'Admin can update an approved setting',
        '/api/v1/admin/config',
        method='PATCH',
        payload={'key': 'security.alert_confidence_threshold', "value": THRESHOLD},
        credentials=(ADMIN_USER, ADMIN_PASSWORD)
    )
    verify(admin_response.get('backup_created') is True, 'Configuration backup was not reported')

    print('LIVE SYSTEM ACCEPTANCE TESTS')
    print(f'Base URL: {BASE_URL}')
    print('Case | Result | Actual HTTP | Expected HTTP')
    for name, passed, actual, expected in results:
        print(f"{name} | {'PASS' if passed else 'FAIL'} | {actual} | {expected}")
    passed_count = sum(item[1] for item in results)
    print(f'TOTAL: {passed_count}/{len(results)} checks passed')
    return 0 if passed_count == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
