# Versioned model storage and integrity verification

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path 
from typing import Any 
import joblib 
from .integrity import sha256_file, verify_sha256

# Store models, version metadata, and SHA-256 integrity info
class ModelRegistry: 
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model_version: str, bundle: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        model_path =self.model_dir / f'{model_version}.joblib'
        manifest_path = self.model_dir / f'{model_version}.json'

        joblib.dump(bundle, model_path, compress=3)
        digest = sha256_file(model_path)

        manifest = {
            'model_version': model_version,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'artifact': model_path.name,
            'sha256': digest,
            'artifact_size_bytes': model_path.stat().st_size,
            'metadata': metadata
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest 
    def list_models(self) -> list[dict[str, Any]]:
        manifests = []
        for path in sorted(self.model_dir.glob('*.json')):
            manifests.append(json.loads(path.read_text(encoding='utf-8')))
        return manifests

    def load(self, model_version: str, verify_integrity: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest_path = self.model_dir / f'{model_version}.json'
        model_path = self.model_dir / f'{model_version}.joblib'
        if not manifest_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Model version '{model_version}' is not registered.")

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if verify_integrity and not verify_sha256(model_path, manifest['sha256']):
            raise RuntimeError(f"Integrity verification failed for model '{model_version}'.")

        return joblib.load(model_path), manifest 