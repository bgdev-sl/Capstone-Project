# Minimal local authenication/RBAC
# Deomonstrating architecture without production ready identity provider

from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any 

ITERATIONS = 200_000

def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)
    return f'pbkdf2_sha256${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}'

def _verify_password(password: str, stored: str) -> bool:
    scheme, iterations, salt_b64, digest_b64 = stored.split('$')
    if scheme != 'pbkdf2_sha256':
        return False
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(digest_b64)
    actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, int(iterations))
    return hmac.compare_digest(actual, expected)

# Create or update local account
def create_user_store(path: str | Path, username: str, password: str, role: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    users: dict[str, Any] = {}
    if path.exists():
        users = json.loads(path.read_text(encoding='utf-8'))

    role = role.upper()
    if role not in {'ADMIN', 'ANALYST'}:
        raise ValueError('Role must be ADMIN or ANALYST')

    users[username] = {'password_hash': _hash_password(password), 'role': role}
    path.write_text(json.dumps(users, indent=2), encoding='utf-8')

# Authenticate username/password
def authenticate(path: str | Path, username: str, password: str) -> dict [str, str] | None:
    path = Path(path)
    if not path.exists():
        return None
    users = json.loads(path.read_text(encoding='utf-8'))
    record = users.get(username)
    if not record:
        return None 
    if not _verify_password(password, record['password_hash']):
        return None
    return {'username': username, 'role': record['role']}
