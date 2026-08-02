"""E2E test: POST /api/chat → turn_id → POST /api/feedback → SignalStore row.

This consolidates and extends the existing test_feedback_api.py into a comprehensive
E2E suite. Uses httpx AsyncClient with ASGITransport over a minimal FastAPI app built
from composition components (fakes, no live services).

Auth: MORGAN_API_KEY set in test settings; Authorization: Bearer header required.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from morgan_brain.apps.brain_api.app import ChatRequest, ChatResponse
from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.apps.brain_api.routes import build_router
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.learning.signals import Thumb
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731
API_KEY = "e2e-feedback-key"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _build_app() -> tuple[FastAPI, object]:
    """Build a minimal FastAPI app with /api/chat + /api/feedback wired via fakes."""
    settings = Settings(api_key=API_KEY, llm_model="test-model", llm_fast_model="test-model")
    fake_client = FakeChatClient(reply="test reply")
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    history_store = SessionHistoryStore()
    orch, _, signal_store, recorder, executor, skills, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )

    app = FastAPI()
    _auth = Depends(require_api_key(settings))

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[_auth])
    async def chat(req: ChatRequest) -> ChatResponse:
        uid = req.user_id or settings.owner_user_id
        hist = history_store.recent(req.session_id or "default")
        result, turn_id = await orch.handle_turn_with_id(
            user_id=uid,
            project=req.project,
            text=req.message,
            session_id=req.session_id,
            history=hist,
        )
        return ChatResponse(response=result.text, model_used=result.model_used, turn_id=turn_id)

    app.include_router(
        build_router(
            orchestrator=orch,
            signal_recorder=recorder,
            executor=executor,
            skills=skills,
            learner=learner,
            settings=settings,
        )
    )

    return app, signal_store


# ---------------------------------------------------------------------------
# Tests using httpx AsyncClient (deterministic, no live services)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_thumb_up_via_async_client() -> None:
    """POST /api/chat → turn_id; POST /api/feedback thumb=up → 200 {ok: true}."""
    app, signal_store = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # Step 1: chat to get a turn_id.
        resp = await c.post(
            "/api/chat",
            json={"message": "hello", "project": "default", "session_id": "s1"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "turn_id" in body
        assert "model_used" in body
        turn_id = body["turn_id"]
        assert len(turn_id) > 0

        # Step 2: send thumb-up feedback.
        fb = await c.post(
            "/api/feedback",
            json={"turn_id": turn_id, "project": "default", "kind": "thumb", "thumb": "up"},
            headers=AUTH_HEADERS,
        )
        assert fb.status_code == 200
        assert fb.json() == {"ok": True}


@pytest.mark.asyncio
async def test_feedback_thumb_up_persisted_in_signal_store() -> None:
    """After thumb=up feedback, SignalStore must have that turn_id with thumb=UP."""
    app, signal_store = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/chat",
            json={"message": "ping", "project": "default"},
            headers=AUTH_HEADERS,
        )
        turn_id = resp.json()["turn_id"]

        await c.post(
            "/api/feedback",
            json={"turn_id": turn_id, "project": "default", "kind": "thumb", "thumb": "up"},
            headers=AUTH_HEADERS,
        )

    signals = await signal_store.for_user("owner")  # type: ignore[attr-defined]
    matching = [s for s in signals if s.turn_id == turn_id]
    assert len(matching) == 1
    assert matching[0].thumb is Thumb.UP


@pytest.mark.asyncio
async def test_feedback_missing_key_returns_401() -> None:
    """Feedback without Authorization header returns 401 when key is enforced."""
    app, _ = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/feedback",
            json={"turn_id": "x", "project": "default", "kind": "thumb", "thumb": "up"},
            # No auth header
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_feedback_edit_updates_signal() -> None:
    """kind='edit' with edited_reply stores a user edit signal."""
    app, signal_store = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/chat",
            json={"message": "help", "project": "default"},
            headers=AUTH_HEADERS,
        )
        turn_id = resp.json()["turn_id"]

        fb = await c.post(
            "/api/feedback",
            json={
                "turn_id": turn_id,
                "project": "default",
                "kind": "edit",
                "edited_reply": "better answer",
            },
            headers=AUTH_HEADERS,
        )
        assert fb.status_code == 200
        assert fb.json()["ok"] is True

    signals = await signal_store.for_user("owner")  # type: ignore[attr-defined]
    matching = [s for s in signals if s.turn_id == turn_id]
    assert len(matching) == 1
    assert matching[0].user_edit == "better answer"


@pytest.mark.asyncio
async def test_feedback_retry_updates_signal() -> None:
    """kind='retry' marks the turn as retried in the signal store."""
    app, signal_store = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/chat",
            json={"message": "retry me", "project": "default"},
            headers=AUTH_HEADERS,
        )
        turn_id = resp.json()["turn_id"]

        await c.post(
            "/api/feedback",
            json={"turn_id": turn_id, "project": "default", "kind": "retry"},
            headers=AUTH_HEADERS,
        )

    signals = await signal_store.for_user("owner")  # type: ignore[attr-defined]
    matching = [s for s in signals if s.turn_id == turn_id]
    assert len(matching) == 1
    assert matching[0].retried is True
