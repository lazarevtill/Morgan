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

The open case is only reachable from this machine. Two independent controls, because one of
them is not enough:

* ``security/network.py::assert_safe_bind`` refuses to *start* the documented entry points on
  a non-loopback bind with no key. It reads a setting, so it cannot see a socket opened some
  other way.
* This dependency refuses any unauthenticated request whose *peer* is not loopback. That is a
  fact about the connection, so it holds no matter how the ASGI app was started -- including
  ``uvicorn morgan_brain.apps.brain_api.app:app --host 0.0.0.0``, which imports ``app``
  directly and never runs the entry point.

That module also owns the sentinel this file used to define for itself.

JWT upgrade seam
----------------
Swap this file's ``require_api_key`` implementation when bearer-JWT is added (Wave 6+).
The factory pattern here (``require_api_key(settings) -> Depends``) keeps the FastAPI
router wiring unchanged: all callers do ``Depends(require_api_key(settings))`` and the
internal verify logic is entirely local to this module.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from morgan_brain.config import Settings
from morgan_brain.security.network import api_key_is_configured, unauthenticated_peer_allowed

log = structlog.get_logger("brain_api.auth")

# FastAPI security scheme objects — reused across factory calls.
_bearer_scheme = HTTPBearer(auto_error=False)
_apikey_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(settings: Settings) -> Callable[..., Coroutine[Any, Any, None]]:
    """Return a FastAPI dependency that enforces the configured API key.

    When the key is empty or the default sentinel (``"change-me"``), enforcement is
    skipped so local-dev / zero-config deployments work out of the box.
    """
    enforced = api_key_is_configured(settings.api_key)

    async def _check(
        request: Request,
        bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
        x_api_key: str | None = Security(_apikey_header),
    ) -> None:
        if not enforced:
            # Open mode is confined to this machine, checked against the real peer rather than
            # against MORGAN_API_HOST -- importing this ASGI app under `uvicorn --host 0.0.0.0`
            # binds every interface while that setting still reads 127.0.0.1.
            if unauthenticated_peer_allowed(request.client.host if request.client else None):
                return
            log.warning(
                "unauthenticated_remote_request_refused",
                path=request.url.path,
                peer=request.client.host if request.client else None,
                remedy="set MORGAN_API_KEY to serve /api/* beyond loopback",
            )
            # Deliberately the same opaque detail as a bad key: a caller learns that it is
            # unauthorized, not that this deployment has no key configured.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

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
