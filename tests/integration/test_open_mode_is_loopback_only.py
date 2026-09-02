"""Open mode (no API key configured) must never serve a remote peer.

The startup guard in ``security/network.py`` reads ``MORGAN_API_HOST``, and a setting has no
causal relationship to the socket a server binds. ``uvicorn morgan_brain.apps.brain_api.app:app
--host 0.0.0.0`` imports the ASGI app directly, never runs the entry point, and binds every
interface while that setting still reads ``127.0.0.1`` -- which served ``/api/profile`` to an
unauthenticated caller on a non-loopback address. These tests pin the per-request control that
closes it, on both surfaces, by driving each app with a spoofed peer address.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.config import Settings
from morgan_brain.security.network import UNSET_API_KEY_SENTINEL

REMOTE = ("172.23.27.215", 51234)
LOOPBACK = ("127.0.0.1", 51234)


def _app(api_key: str) -> FastAPI:
    """A minimal app wired with the real dependency -- the guard under test, nothing else."""
    settings = Settings(llm_model="x", llm_fast_model="x", api_key=api_key)
    app = FastAPI()

    @app.get("/api/probe", dependencies=[Depends(require_api_key(settings))])
    async def probe() -> dict[str, bool]:
        return {"served": True}

    return app


@pytest.mark.parametrize("unset_key", ["", UNSET_API_KEY_SENTINEL])
def test_open_mode_refuses_a_remote_peer(unset_key: str) -> None:
    with TestClient(_app(unset_key), client=REMOTE) as client:
        response = client.get("/api/probe")
    assert response.status_code == 401, response.text
    # The refusal must not disclose that this deployment has no key configured.
    assert "MORGAN_API_KEY" not in response.text


@pytest.mark.parametrize("unset_key", ["", UNSET_API_KEY_SENTINEL])
def test_open_mode_still_serves_loopback(unset_key: str) -> None:
    """The zero-config path a fresh clone depends on."""
    with TestClient(_app(unset_key), client=LOOPBACK) as client:
        response = client.get("/api/probe")
    assert response.status_code == 200, response.text


def test_a_real_key_serves_a_remote_peer_that_presents_it() -> None:
    with TestClient(_app("a-real-key"), client=REMOTE) as client:
        assert client.get("/api/probe").status_code == 401
        authorized = client.get("/api/probe", headers={"Authorization": "Bearer a-real-key"})
    assert authorized.status_code == 200, authorized.text


def test_mcp_open_mode_refuses_a_remote_peer() -> None:
    """The MCP HTTP surface exposes forget(); the same rule has to hold there."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Route

    from morgan_brain.ports.mcp_server import _BearerAuthMiddleware

    async def _tool(request: object) -> _JSONResponse:
        return _JSONResponse({"served": True})

    settings = Settings(llm_model="x", llm_fast_model="x", api_key=UNSET_API_KEY_SENTINEL)
    app = Starlette(routes=[Route("/mcp", _tool)])
    app.add_middleware(_BearerAuthMiddleware, settings=settings)

    with TestClient(app, client=REMOTE) as remote:
        assert remote.get("/mcp").status_code == 401
    with TestClient(app, client=LOOPBACK) as local:
        assert local.get("/mcp").status_code == 200


def test_mcp_enforces_the_bearer_token_when_a_key_is_set() -> None:
    """Mutation-proofing: setting ``_enforced = False`` used to leave the whole suite green."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Route

    from morgan_brain.ports.mcp_server import _BearerAuthMiddleware

    async def _tool(request: object) -> _JSONResponse:
        return _JSONResponse({"served": True})

    settings = Settings(llm_model="x", llm_fast_model="x", api_key="a-real-key")
    app = Starlette(routes=[Route("/mcp", _tool)])
    app.add_middleware(_BearerAuthMiddleware, settings=settings)

    # Loopback must not be a way around the token once a key exists.
    with TestClient(app, client=LOOPBACK) as client:
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.get("/mcp", headers={"Authorization": "Bearer a-real-key"})
    assert ok.status_code == 200, ok.text
