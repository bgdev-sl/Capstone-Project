# Model and artifact integrity support 

from __future__ import annotations
import hashlib
from pathlib import Path

# Compute SHA-256 for a file
def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

# Verify file against a stored SHA-256 digest
def verify_sha256(path: str | Path, expected_hash: str) -> bool:
    return sha256_file(path).lower() == expected_hash.lower()

