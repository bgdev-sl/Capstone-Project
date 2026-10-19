"""Model and artifact integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file without loading the whole file into memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_hash: str) -> bool:
    """Verify a file against a stored SHA-256 digest."""
    return sha256_file(path).lower() == expected_hash.lower()
