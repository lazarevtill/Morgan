"""Integration tests for GET /api/tools, GET /api/skills, GET /api/profile (commit 3)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_brain.apps.brain_api.routes import build_router
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731
API_KEY = "test-read-key"
_AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _build_test_app() -> TestClient:
    settings = Settings(api_key=API_KEY, llm_model="test-model", llm_fast_model="test-model")
    fake_client = FakeChatClient(reply="hi")
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
    orch, _, _signal_store, recorder, executor, skills, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )
    app = FastAPI()
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
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /api/tools
# ---------------------------------------------------------------------------


def test_get_tools_returns_list() -> None:
    client = _build_test_app()
    resp = client.get("/api/tools", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_tools_includes_builtin_names() -> None:
    client = _build_test_app()
    resp = client.get("/api/tools", headers=_AUTH_HEADERS)
    names = {t["name"] for t in resp.json()}
    # The four built-in tools from _build_tool_executor
    assert "calculator" in names
    assert "current_time" in names
    assert "memory_search" in names


def test_get_tools_requires_auth() -> None:
    client = _build_test_app()
    resp = client.get("/api/tools")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/skills
# ---------------------------------------------------------------------------


def test_get_skills_returns_list() -> None:
    client = _build_test_app()
    resp = client.get("/api/skills", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # Each skill has name and triggers
    for skill in data:
        assert "name" in skill
        assert "triggers" in skill


# ---------------------------------------------------------------------------
# GET /api/profile
# ---------------------------------------------------------------------------


def test_get_profile_returns_user_model() -> None:
    client = _build_test_app()
    resp = client.get("/api/profile", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "user_id" in data
    assert data["user_id"] == "owner"  # default owner_user_id


def test_get_profile_custom_user_id() -> None:
    client = _build_test_app()
    resp = client.get("/api/profile?user_id=alice", headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"


def test_get_profile_requires_auth() -> None:
    client = _build_test_app()
    resp = client.get("/api/profile")
    assert resp.status_code == 401
