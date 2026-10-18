# FastAPI adapter exposing the shared Edge IDS application service.
# This REST layer is intended for a local, isolated research environment. It uses
# HTTP Basic authentication against the prototype's local user store; place it
# behind TLS and an appropriate identity gateway before any wider deployment.

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from .auth import authenticate
from .config import load_config
from .config_manager import update_config_value
from .pipeline import EdgeIDSApplication

def _default_root() -> Path:
    configured = os.getenv('EDGE_IDS_ROOT')
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[2]

# Batch prediction request
class PredictionRequest(BaseModel):
    model_version: str = Field(min_length=1, max_length=100)
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)

# Single, whitelisted configuration update
class ConfigUpdateRequest(BaseModel):
    key: str
    value: Any

# Create an app instance; isolated paths for tests and deployment
def create_app(root: str | Path | None = None, config_path: str | Path | None = None) -> FastAPI:
    app_root = Path(root or _default_root()).resolve()
    selected_config = Path(
        config_path or os.getenv('EDGE_IDS_CONFIG', str(app_root / 'configs' / 'config.yaml'))
    ).resolve()
    try:
        config = load_config(selected_config)
    except FileNotFoundError:
        raise

    service = EdgeIDSApplication(config, app_root)
    paths = config.get('paths', {})
    user_file = app_root / paths.get('user_file', 'configs/users.json')
    log_file = app_root / paths.get('log_dir', 'logs') / 'alerts.jsonl'

    api = FastAPI(
        title='Edge IDS Research API',
        version='1.0.0',
        description='Experimental API for integrity-checked ML-based intrusion detection.',
    )
    api.state.root = app_root
    api.state.config_path = selected_config
    api.state.config = config
    api.state.ids_service = service
    api.state.user_file = user_file
    api.state.log_file = log_file
    basic_security = HTTPBasic()

    def current_user(
        request: Request,
        credentials: HTTPBasicCredentials = Depends(basic_security),
    ) -> dict[str, str]:
        identity = authenticate(request.app.state.user_file, credentials.username, credentials.password)
        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid credentials.',
                headers={'WWW-Authenticate': 'Basic'},
            )
        return identity

    def administrator(identity: dict[str, str] = Depends(current_user)) -> dict[str, str]:
        if identity['role'] != 'ADMIN':
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='ADMIN role required.')
        return identity

    # Liveness endpoint; does not disclose model or user info
    @api.get('/health', tags=['system'])
    def health() -> dict[str, Any]:
        return {'status': 'ok', 'service': 'edge-ids-api', 'research_prototype': True}

    @api.get('/api/v1/models', tags=['models'])
    def list_models(identity: dict[str, str] = Depends(current_user)) -> dict[str, Any]:
        return {'models': api.state.ids_service.available_models()}

    # Classify one bounded batch and return predictions with resource consumption
    @api.post('/api/v1/predict', tags=['inference'])
    def predict(
        payload: PredictionRequest,
        identity: dict[str, str] = Depends(current_user),
    ) -> dict[str, Any]:
        import pandas as pd

        try:
            predictions, resources = api.state.ids_service.predict_frame(
                payload.model_version, pd.DataFrame(payload.records)
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            # Includes a rejected model artifact whose SHA-256 does not match.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Keep the REST response data-minimal: callers receive classification results and model metadata, not a second copy of potentially sensitive network-flow input features.
        # The local dashboard may still show its in-memory input rows alongside predictions for the analyst workflow.
        response_columns = [
            'prediction', 'class', 'confidence', 'model_version', 'timestamp_utc'
        ]
        safe_predictions = predictions.loc[:, response_columns]
        return {
            'model_version': payload.model_version,
            'count': len(safe_predictions),
            'resource_metrics': resources,
            'predictions': safe_predictions.to_dict(orient='records'),
        }

    @api.get('/api/v1/alerts', tags=['alerts'])
    def read_alerts(
        limit: int = Query(default=20, ge=1, le=200),
        identity: dict[str, str] = Depends(current_user),
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        if api.state.log_file.exists():
            with api.state.log_file.open('r', encoding='utf-8') as handle:
                for line in handle:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Ignore a partially written/corrupt line rather than failing the endpoint.
                        continue
        return {'alerts': records[-limit:], 'count': min(len(records), limit)}

    # Update a whitelisted setting, take backup, and refresh service
    @api.patch('/api/v1/admin/config', tags=['administration'])
    def update_config(
        payload: ConfigUpdateRequest,
        identity: dict[str, str] = Depends(administrator),
    ) -> dict[str, Any]:
        allowed_ranges = {
            'security.alert_confidence_threshold': (0.0, 1.0),
            'feature_reduction.k': (1, 100_000),
            'data.test_size': (0.05, 0.50),
        }
        if payload.key not in allowed_ranges:
            raise HTTPException(status_code=422, detail='Configuration key is not editable.')
        low, high = allowed_ranges[payload.key]
        try:
            value = float(payload.value) if payload.key != 'feature_reduction.k' else int(payload.value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail='Configuration value has an invalid type.') from exc
        if not low <= value <= high:
            raise HTTPException(status_code=422, detail=f'Value must be between {low} and {high}.')

        try:
            new_config = update_config_value(api.state.config_path, payload.key, value)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        api.state.config = new_config
        api.state.ids_service = EdgeIDSApplication(new_config, api.state.root)
        return {'updated': payload.key, 'value': value, 'backup_created': True}

    return api

app = create_app()
