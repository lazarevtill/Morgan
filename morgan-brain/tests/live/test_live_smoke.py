"""Live smoke tests — require real external services.

ALL tests in this file are marked ``@pytest.mark.live`` and are SKIPPED BY DEFAULT.
Run them only when Ollama + Qdrant + Redis are available:

    python -m pytest --live tests/live/test_live_smoke.py -v

These tests are a manual confidence check before deployment — NOT CI gates.
They build the production app stack and run a minimal /api/chat turn, asserting
a non-empty, non-error reply from the real Ollama model.

Prerequisites
-------------
- Ollama running at MORGAN_LLM_ENDPOINT (default: http://localhost:11434/v1)
  with MORGAN_LLM_MODEL loaded (default: qwen2.5:7b)
- Qdrant running at MORGAN_QDRANT_URL (default: http://localhost:6333)
  if MORGAN_VECTOR_BACKEND=qdrant; otherwise not required.
- Redis running at MORGAN_REDIS_URL (default: redis://localhost:6379/0)
  if MORGAN_EVENT_BUS=redis; otherwise not required.
- MORGAN_API_KEY set to a non-sentinel value, or left at default ("change-me")
  to bypass auth enforcement in smoke mode.

Usage
-----
    # Minimal: just Ollama + in-memory vector
    MORGAN_LLM_MODEL=qwen2.5:7b \\
    MORGAN_VECTOR_BACKEND=memory \\
    MORGAN_EVENT_BUS=inproc \\
    python -m pytest --live tests/live/test_live_smoke.py -v

The suite is intentionally minimal — one or two asserts per test.
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Live marker — all tests skipped without --live
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_live_app():  # type: ignore[return]
    """Build the production FastAPI app using the real create_app() factory.

    Reads configuration from environment variables (MORGAN_* prefix).
    Requires live Ollama (and optionally Qdrant/Redis) to be reachable.
    """
    # Invalidate the lru_cache so the live test reads fresh env-vars.
    from morgan_brain.config import get_settings

    get_settings.cache_clear()

    from morgan_brain.apps.brain_api.app import create_app

    return create_app()


def _live_auth_headers() -> dict[str, str]:
    """Return auth headers using MORGAN_API_KEY, or empty dict if key is sentinel."""
    key = os.environ.get("MORGAN_API_KEY", "change-me")
    if key and key != "change-me":
        return {"Authorization": f"Bearer {key}"}
    return {}


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_health_endpoint() -> None:
    """GET /health returns 200 {status: ok} against a real app stack."""
    app = _build_live_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    assert "version" in body


@pytest.mark.asyncio
async def test_live_chat_returns_nonempty_reply() -> None:
    """POST /api/chat returns a non-empty response from the real LLM.

    This is the primary smoke: proves Ollama is reachable, the prompt reaches the
    model, and the model returns text — the minimum bar for a working deployment.
    """
    app = _build_live_app()
    headers = _live_auth_headers()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0,  # generous: LLM first-token latency can be high on cold start
    ) as c:
        resp = await c.post(
            "/api/chat",
            json={"message": "Hello. Reply with exactly: OK", "session_id": "smoke-1"},
            headers=headers,
        )

    assert resp.status_code == 200, f"Non-200 response: {resp.status_code} {resp.text}"
    body = resp.json()
    assert "response" in body
    assert len(body["response"].strip()) > 0, "Live LLM returned an empty response"
    assert "turn_id" in body
    assert len(body["turn_id"]) > 0
    assert "model_used" in body


@pytest.mark.asyncio
async def test_live_chat_stream_ends_with_done() -> None:
    """POST /api/chat/stream returns SSE lines ending with [DONE] against a real LLM."""
    app = _build_live_app()
    headers = _live_auth_headers()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=120.0,
    ) as c:
        async with c.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "Say hi", "session_id": "smoke-stream"},
            headers=headers,
        ) as resp:
            assert resp.status_code == 200
            raw = await resp.aread()

    text = raw.decode()
    lines = [ln for ln in text.splitlines() if ln.startswith("data:")]
    assert len(lines) >= 1, "No data: lines in live SSE response"
    assert lines[-1] == "data: [DONE]", f"Live SSE did not end with [DONE]: {lines[-3:]}"


@pytest.mark.asyncio
async def test_live_two_turn_history() -> None:
    """Two turns in the same session — the second turn reply references the first.

    This is a soft smoke: we only assert non-empty replies and unique turn_ids.
    Asserting content would be fragile against non-deterministic LLM outputs.
    """
    app = _build_live_app()
    headers = _live_auth_headers()
    session_id = "smoke-history"

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0,
    ) as c:
        r1 = await c.post(
            "/api/chat",
            json={"message": "My name is SmokeTestUser.", "session_id": session_id},
            headers=headers,
        )
        assert r1.status_code == 200
        turn_id_1 = r1.json()["turn_id"]

        r2 = await c.post(
            "/api/chat",
            json={"message": "What is my name?", "session_id": session_id},
            headers=headers,
        )
        assert r2.status_code == 200
        turn_id_2 = r2.json()["turn_id"]

    # Each turn has a distinct turn_id.
    assert turn_id_1 != turn_id_2
    # Both replies are non-empty.
    assert len(r1.json()["response"].strip()) > 0
    assert len(r2.json()["response"].strip()) > 0
