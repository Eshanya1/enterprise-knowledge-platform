"""
API-key auth for the FastAPI service. Simpler than JWT here since this is
a service-to-service API (internal tools querying the knowledge platform),
not a user-facing login flow -- API keys are the right tool for that.
Keys are stored as SHA-256 hashes, never in plaintext.
"""
import os
import hashlib
import secrets
from typing import Optional
from fastapi import Header, HTTPException

# In production, load key hashes from a database/secrets manager. For this
# project, a single key is provisioned via env var at startup so the demo
# runs without extra infra.
_DEFAULT_KEY = os.environ.get("API_KEY") or secrets.token_urlsafe(24)
_VALID_KEY_HASHES = {hashlib.sha256(_DEFAULT_KEY.encode()).hexdigest()}

if not os.environ.get("API_KEY"):
    print(f"[auth] No API_KEY set -- generated one for this run: {_DEFAULT_KEY}")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    # Explicit None-check first so a *missing* header returns 401 (an auth
    # failure) rather than FastAPI's default 422 (a validation failure) --
    # caught this distinction while writing the test suite; a client
    # can't tell "you're not authorized" from "you sent a malformed
    # request" if both come back as 422.
    if not x_api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    if key_hash not in _VALID_KEY_HASHES:
        raise HTTPException(status_code=401, detail="invalid API key")
    return x_api_key
