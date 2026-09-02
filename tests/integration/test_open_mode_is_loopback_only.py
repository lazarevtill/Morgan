"""Open mode (no API key configured) must never serve a remote peer.

The startup guard reads a setting, and a setting has no causal relationship to the socket a
server binds: an ASGI app imported directly and served on 0.0.0.0 never runs the entry
point. These tests pin the per-request control that closes that, by driving the MCP app
with a spoofed peer address.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from morgan_brain.config import Settings
from morgan_brain.network import UNSET_API_KEY_SENTINEL

REMOTE = ("172.23.27.215", 51234)
LOOPBACK = ("127.0.0.1", 51234)


def test_mcp_open_mode_refuses_a_remote_peer() -> None:
    """The MCP HTTP surface exposes forget(), so open mode must be loopback-only."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Route

    from morgan_brain.mcp_server import _BearerAuthMiddleware

    async def _tool(request: object) -> _JSONResponse:
        return _JSONResponse({"served": True})

    settings = Settings(llm_model="x", api_key=UNSET_API_KEY_SENTINEL)
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

    from morgan_brain.mcp_server import _BearerAuthMiddleware

    async def _tool(request: object) -> _JSONResponse:
        return _JSONResponse({"served": True})

    settings = Settings(llm_model="x", api_key="a-real-key")
    app = Starlette(routes=[Route("/mcp", _tool)])
    app.add_middleware(_BearerAuthMiddleware, settings=settings)

    # Loopback must not be a way around the token once a key exists.
    with TestClient(app, client=LOOPBACK) as client:
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.get("/mcp", headers={"Authorization": "Bearer a-real-key"})
    assert ok.status_code == 200, ok.text
