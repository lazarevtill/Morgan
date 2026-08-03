"""Both entry points consult the bind guard before a socket exists.

``tests/unit/security/test_network_bind_guard.py`` proves the guard decides correctly. These
tests prove it is actually *reached* -- the failure mode being a correct guard nobody calls.
Each asserts the refusal happens before the server starts, by making the server callable
explode if it is ever invoked.
"""

from __future__ import annotations

import asyncio
from typing import Any, NoReturn

import pytest

from morgan_brain.config import get_settings
from morgan_brain.ports import mcp_server
from morgan_brain.security.network import UNSET_API_KEY_SENTINEL


@pytest.fixture(autouse=True)
def _fresh_settings() -> Any:
    """``get_settings`` is ``lru_cache``d; these tests change the environment under it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_brain_api_refuses_a_public_bind_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from morgan_brain.apps.brain_api import __main__ as entrypoint

    monkeypatch.setenv("MORGAN_API_HOST", "0.0.0.0")
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)

    def _explode(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(f"uvicorn.run was called: {args} {kwargs}")

    monkeypatch.setattr(entrypoint.uvicorn, "run", _explode)

    with pytest.raises(SystemExit, match="MORGAN_API_KEY"):
        entrypoint.main()


def test_brain_api_serves_loopback_with_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-config path stays working -- the guard must not have broken a fresh clone."""
    from morgan_brain.apps.brain_api import __main__ as entrypoint

    monkeypatch.setenv("MORGAN_API_HOST", "127.0.0.1")
    monkeypatch.setenv("MORGAN_API_PORT", "8080")
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)

    called: dict[str, Any] = {}

    def _record(app: str, **kwargs: Any) -> None:
        called["app"] = app
        called.update(kwargs)

    monkeypatch.setattr(entrypoint.uvicorn, "run", _record)
    entrypoint.main()

    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8080


def test_brain_api_bind_host_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing hardcoded: an overlay address plus a key is the real deployment, and it runs."""
    from morgan_brain.apps.brain_api import __main__ as entrypoint

    monkeypatch.setenv("MORGAN_API_HOST", "100.64.0.7")
    monkeypatch.setenv("MORGAN_API_PORT", "9999")
    monkeypatch.setenv("MORGAN_API_KEY", "a-real-key")

    called: dict[str, Any] = {}
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda app, **kw: called.update(kw))
    entrypoint.main()

    assert called["host"] == "100.64.0.7"
    assert called["port"] == 9999


def test_mcp_http_refuses_a_public_bind_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORGAN_API_KEY", UNSET_API_KEY_SENTINEL)
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = mcp_server.build_server()

    # Stub the server loop like the brain-api siblings above do. Without it, a regression in
    # the guard does not fail this test -- it binds the port and blocks, so CI hangs instead
    # of reporting, which is the worst of both outcomes. Verified: with `assert_safe_bind`
    # neutered, this test times out rather than failing.
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
