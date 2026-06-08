"""Unit tests for Orchestrator.stream_turn + SSE /api/chat/stream (commit 2).

All tests are deterministic: no network, no filesystem.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.apps.brain_api.app import ChatRequest
from morgan_brain.composition import build_orchestrator_for_test
from morgan_brain.config import Settings

_CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


# ---------------------------------------------------------------------------
# Orchestrator.stream_turn — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_turn_yields_reply_text() -> None:
    orch, _ = build_orchestrator_for_test(reply="Hello streaming!", clock=_CLOCK)
    chunks = [chunk async for chunk in orch.stream_turn(user_id="u1", text="hi")]
    assert chunks == ["Hello streaming!"]


@pytest.mark.asyncio
async def test_stream_turn_yields_non_empty_chunks() -> None:
    orch, _ = build_orchestrator_for_test(reply="chunk1", clock=_CLOCK)
    chunks = [chunk async for chunk in orch.stream_turn(user_id="u1", text="test")]
    assert all(isinstance(c, str) and c for c in chunks)


@pytest.mark.asyncio
async def test_stream_turn_stores_turn_in_memory() -> None:
    """After stream_turn completes, RESPONSE_GENERATED fires and the turn is stored."""
    orch, mem = build_orchestrator_for_test(reply="Streaming reply", clock=_CLOCK)
    async for _ in orch.stream_turn(user_id="u1", text="My name is Alice", session_id="s1"):
        pass  # consume entire stream

    hits = await mem.recall_raw(user_id="u1", text="Alice")
    assert any("Alice" in h.content for h in hits)


@pytest.mark.asyncio
async def test_stream_turn_is_user_scoped() -> None:
    orch, mem = build_orchestrator_for_test(reply="ok", clock=_CLOCK)
    async for _ in orch.stream_turn(user_id="u1", text="secret for u1", session_id="s1"):
        pass

    other = await mem.recall_raw(user_id="u2", text="secret")
    assert other == []


# ---------------------------------------------------------------------------
# SSE endpoint — /api/chat/stream
# ---------------------------------------------------------------------------


def _sse_app(api_key: str = "") -> tuple[FastAPI, Settings, TestClient]:
    """Return a test app with the SSE endpoint and a known test reply."""
    from morgan_brain.composition import build_orchestrator_for_test
    from morgan_brain.config import Settings

    settings = Settings(api_key=api_key, llm_model="test-model", llm_fast_model="test-model")

    # Build a real orchestrator backed by fakes.
    orch, _ = build_orchestrator_for_test(reply="SSE token", clock=_CLOCK)

    app = FastAPI()
    _auth = Depends(require_api_key(settings))

    @app.post("/api/chat/stream", dependencies=[_auth])
    async def chat_stream(req: ChatRequest):  # type: ignore[return]
        import json
        from typing import AsyncIterator
        from fastapi.responses import StreamingResponse

        async def _event_stream() -> AsyncIterator[str]:
            async for delta in orch.stream_turn(
                user_id=req.user_id or settings.owner_user_id,
                text=req.message,
                session_id=req.session_id,
            ):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    return app, settings, TestClient(app, raise_server_exceptions=True)


def test_sse_endpoint_streams_data_lines() -> None:
    _, _, client = _sse_app()
    resp = client.post("/api/chat/stream", json={"message": "hello"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
    assert len(lines) >= 1


def test_sse_endpoint_ends_with_done() -> None:
    _, _, client = _sse_app()
    resp = client.post("/api/chat/stream", json={"message": "hello"})
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
    assert data_lines[-1] == "data: [DONE]"


def test_sse_endpoint_delta_json_well_formed() -> None:
    _, _, client = _sse_app()
    resp = client.post("/api/chat/stream", json={"message": "hello"})
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
    token_lines = [ln for ln in data_lines if ln != "data: [DONE]"]
    for line in token_lines:
        payload = json.loads(line[len("data: "):])  # noqa: FURB184
        assert "delta" in payload
        assert isinstance(payload["delta"], str)


def test_sse_endpoint_enforces_auth() -> None:
    KEY = "test-stream-key"
    _, _, client = _sse_app(api_key=KEY)
    resp = client.post("/api/chat/stream", json={"message": "hello"})
    assert resp.status_code == 401


def test_sse_endpoint_passes_with_correct_key() -> None:
    KEY = "test-stream-key"
    _, _, client = _sse_app(api_key=KEY)
    resp = client.post(
        "/api/chat/stream",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert resp.status_code == 200
