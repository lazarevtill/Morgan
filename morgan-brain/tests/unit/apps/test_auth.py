"""Unit tests for API-key auth dependency (commit 1).

Policy under test:
* api_key empty or "change-me" → no enforcement (open).
* api_key set to any other value → enforce; missing/wrong → 401; correct → 200.
* /health never requires a key regardless.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(api_key: str) -> tuple[FastAPI, TestClient]:
    """Build a minimal app with a single /api/probe route protected by auth."""
    settings = Settings(api_key=api_key)
    app = FastAPI()
    _auth = Depends(require_api_key(settings))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/probe", dependencies=[_auth])
    async def probe() -> dict[str, str]:
        return {"ok": "true"}

    # Explicit loopback peer: open mode (no key configured) is refused for a
    # non-loopback peer, and TestClient's default peer is the literal "testclient".
    return app, TestClient(app, raise_server_exceptions=True, client=("127.0.0.1", 50000))


# ---------------------------------------------------------------------------
# Tests: key enforcement OFF (empty / sentinel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_key", ["", "change-me"])
def test_open_when_key_not_configured(api_key: str) -> None:
    _, client = _make_app(api_key)
    resp = client.get("/api/probe")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: key enforcement ON
# ---------------------------------------------------------------------------

KEY = "super-secret-test-key"


def test_missing_key_returns_401() -> None:
    _, client = _make_app(KEY)
    resp = client.get("/api/probe")
    assert resp.status_code == 401


def test_wrong_bearer_returns_401() -> None:
    _, client = _make_app(KEY)
    resp = client.get("/api/probe", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_wrong_x_api_key_returns_401() -> None:
    _, client = _make_app(KEY)
    resp = client.get("/api/probe", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_correct_bearer_passes() -> None:
    _, client = _make_app(KEY)
    resp = client.get("/api/probe", headers={"Authorization": f"Bearer {KEY}"})
    assert resp.status_code == 200


def test_correct_x_api_key_passes() -> None:
    _, client = _make_app(KEY)
    resp = client.get("/api/probe", headers={"X-API-Key": KEY})
    assert resp.status_code == 200


def test_health_needs_no_key() -> None:
    """The /health route is NOT protected — must be reachable without any key."""
    _, client = _make_app(KEY)
    resp = client.get("/health")
    assert resp.status_code == 200
