"""API-key authentication for brain-api.

Enforcement policy
------------------
* If ``settings.api_key`` is empty **or** equal to the sentinel ``"change-me"``,
  the dependency is a **no-op** (open for local dev / CI with no key configured).
* If ``settings.api_key`` is any other non-empty value, every request to ``/api/*``
  **must** supply the matching key via either:
    - ``Authorization: Bearer <key>``
    - ``X-API-Key: <key>``
  Anything else → ``HTTP 401``.

JWT upgrade seam
----------------
Swap this file's ``require_api_key`` implementation when bearer-JWT is added (Wave 6+).
The factory pattern here (``require_api_key(settings) -> Depends``) keeps the FastAPI
router wiring unchanged: all callers do ``Depends(require_api_key(settings))`` and the
internal verify logic is entirely local to this module.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from morgan_brain.config import Settings

_SENTINEL = "change-me"

# FastAPI security scheme objects — reused across factory calls.
_bearer_scheme = HTTPBearer(auto_error=False)
_apikey_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(settings: Settings) -> Callable[..., Coroutine[Any, Any, None]]:
    """Return a FastAPI dependency that enforces the configured API key.

    When the key is empty or the default sentinel (``"change-me"``), enforcement is
    skipped so local-dev / zero-config deployments work out of the box.
    """
    enforced = bool(settings.api_key) and settings.api_key != _SENTINEL

    async def _check(
        bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
        x_api_key: str | None = Security(_apikey_header),
    ) -> None:
        if not enforced:
            return

        provided: str | None = None
        if bearer is not None:
            provided = bearer.credentials
        elif x_api_key:
            provided = x_api_key

        if not provided or provided != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _check
