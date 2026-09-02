"""The MCP HTTP entry point consults the bind guard before a socket exists.

``tests/unit/security/test_network_bind_guard.py`` proves the guard decides correctly. These
tests prove it is actually *reached* -- the failure mode being a correct guard nobody calls.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from morgan_brain import mcp_server
from morgan_brain.config import get_settings
from morgan_brain.network import UNSET_API_KEY_SENTINEL


@pytest.fixture(autouse=True)
def _fresh_settings() -> Any:
    """``get_settings`` is ``lru_cache``d; these tests change the environment under it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_mcp_http_refuses_a_public_bind_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = mcp_server.build_server()

    # Stub the server loop. Without it, a regression in the guard does not fail this test --
    # it binds the port and blocks, so CI hangs instead of reporting.
    async def _explode(self: object) -> None:
        raise AssertionError("uvicorn.Server.serve was reached")

    monkeypatch.setattr("uvicorn.Server.serve", _explode)

    with pytest.raises(SystemExit, match="morgan-mcp"):
        asyncio.run(server.run_http_async("0.0.0.0", 8090))


def test_mcp_stdio_is_unaffected_by_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdio has no socket, so no key is required -- this is how Claude Code runs it."""
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = mcp_server.build_server()

    ran: dict[str, bool] = {}

    async def _fake_stdio() -> None:
        ran["yes"] = True

    monkeypatch.setattr(server.mcp, "run_stdio_async", _fake_stdio)
    asyncio.run(server.run_stdio_async())

    assert ran == {"yes": True}


def test_mcp_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old default was 0.0.0.0 with authentication off whenever no key was set."""
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")

    recorded: dict[str, Any] = {}

    async def _record(self: Any, host: str, port: int) -> None:
        recorded["host"] = host
        recorded["port"] = port

    monkeypatch.setattr(mcp_server.MorganMcpServer, "run_http_async", _record)
    assert mcp_server.main(["--transport", "http"]) == 0
    assert recorded["host"] == "127.0.0.1"
