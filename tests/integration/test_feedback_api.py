"""Integration tests for POST /api/feedback (commit 3).

Flow:
  1. POST /api/chat → capture turn_id from response body.
  2. POST /api/feedback {kind: "thumb", thumb: "up"} with that turn_id.
  3. Assert HTTP 200 {ok: true}.
  4. Assert the SignalStore row for that turn_id now has thumb=UP.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from morgan_brain.apps.brain_api.app import ChatRequest, ChatResponse
from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.apps.brain_api.routes import build_router
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.history import SessionHistoryStore as _HSS
from morgan_brain.learning.signals import Thumb
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731
API_KEY = "test-feedback-key"


def _build_test_app() -> tuple[FastAPI, TestClient, object]:  # (app, client, signal_store)
    settings = Settings(api_key=API_KEY, llm_model="test-model", llm_fast_model="test-model")
    fake_client = FakeChatClient(reply="ok response")
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
    history_store = _HSS()
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
        user_id = req.user_id or settings.owner_user_id
        history = history_store.recent(req.session_id or "default", project="default")
        result, turn_id = await orch.handle_turn_with_id(
            user_id=user_id,
            project=req.project,
            text=req.message,
            session_id=req.session_id,
            history=history,
        )
        return ChatResponse(response=result.text, model_used=result.model_used, turn_id=turn_id)

    # Mount the feedback + read router
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

    client = TestClient(app, raise_server_exceptions=True)
    return app, client, signal_store


_AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feedback_thumb_up_returns_ok() -> None:
    _, client, _signal_store = _build_test_app()

    # Step 1: post a chat turn to get a turn_id
    resp = client.post(
        "/api/chat",
        json={"message": "hello", "project": "default", "session_id": "s1"},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    turn_id = resp.json()["turn_id"]
    assert len(turn_id) > 0

    # Step 2: send thumb-up feedback
    fb_resp = client.post(
        "/api/feedback",
        json={"turn_id": turn_id, "project": "default", "kind": "thumb", "thumb": "up"},
        headers=_AUTH_HEADERS,
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_feedback_thumb_up_updates_signal_store() -> None:
    """After sending thumb=up, the SignalStore row must have thumb=UP."""
    _, client, signal_store = _build_test_app()

    resp = client.post(
        "/api/chat",
        json={"message": "ping", "project": "default", "session_id": "s1"},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200
    turn_id = resp.json()["turn_id"]

    # Send feedback
    client.post(
        "/api/feedback",
        json={"turn_id": turn_id, "project": "default", "kind": "thumb", "thumb": "up"},
        headers=_AUTH_HEADERS,
    )

    # Assert the signal row now has thumb=UP
    signals = await signal_store.for_user("owner")
    matching = [s for s in signals if s.turn_id == turn_id]
    assert len(matching) == 1
    assert matching[0].thumb is Thumb.UP


def test_feedback_edit_returns_ok() -> None:
    _, client, _ = _build_test_app()
    resp = client.post(
        "/api/chat", json={"message": "hello", "project": "default"}, headers=_AUTH_HEADERS
    )
    turn_id = resp.json()["turn_id"]

    fb_resp = client.post(
        "/api/feedback",
        json={
            "turn_id": turn_id,
            "project": "default",
            "kind": "edit",
            "edited_reply": "better reply",
        },
        headers=_AUTH_HEADERS,
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["ok"] is True


def test_feedback_retry_returns_ok() -> None:
    _, client, _ = _build_test_app()
    resp = client.post(
        "/api/chat", json={"message": "hello", "project": "default"}, headers=_AUTH_HEADERS
    )
    turn_id = resp.json()["turn_id"]

    fb_resp = client.post(
        "/api/feedback",
        json={"turn_id": turn_id, "project": "default", "kind": "retry"},
        headers=_AUTH_HEADERS,
    )
    assert fb_resp.status_code == 200


def test_feedback_requires_auth() -> None:
    _, client, _ = _build_test_app()
    resp = client.post(
        "/api/feedback",
        json={"turn_id": "x", "project": "default", "kind": "thumb", "thumb": "up"},
        # No auth header
    )
    assert resp.status_code == 401


def test_feedback_invalid_kind_returns_422() -> None:
    _, client, _ = _build_test_app()
    resp = client.post(
        "/api/feedback",
        json={"turn_id": "x", "project": "default", "kind": "unknown"},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 422
