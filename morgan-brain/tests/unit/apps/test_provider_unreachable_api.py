"""brain-api over a model server that is down -- through the real ``create_app``.

A gateway whose upstream is gone answers 502 with the endpoint in the body, on every route
that needs the model. The stream is the delicate one: its status is fixed the moment the
first byte goes out, so a failure before the first token is a 502 and a failure after it is
an in-band error event before the terminal sentinel -- never an empty 200 that looks like an
answer with no words in it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from morgan_brain.apps.brain_api.app import create_app
from morgan_brain.composition import AppContext, _assemble
from morgan_brain.config import Settings
from morgan_brain.interfaces.llm import ProviderUnreachable
from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec

_CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731
_ENDPOINT = "http://gpu-box.example:8081/v1"


class _DownClient:
    """Refuses every call; optionally after *tokens_before_failure* streamed tokens."""

    def __init__(self, tokens_before_failure: int = 0) -> None:
        self._tokens = tokens_before_failure

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        raise ProviderUnreachable(_ENDPOINT, "connection refused")

    async def astream(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]:
        for i in range(self._tokens):
            yield StreamDelta(kind="text_delta", text=f"tok{i} ")
        raise ProviderUnreachable(_ENDPOINT, "connection reset")


def _app(tokens_before_failure: int = 0) -> Any:
    settings = Settings(llm_model="m", llm_fast_model="m", embedding_backend="hash")
    reg = CapabilityRegistry.from_seed({"fake/m": {"context_window": 8192}})
    router = RoleRouter(
        reg=reg, bindings={"strong": [Binding("fake", "m", _DownClient(tokens_before_failure))]}
    )
    history = SessionHistoryStore()
    orch, mem, signal_store, recorder, executor, skills, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=_CLOCK,
        temporal_path=":memory:",
        history_store=history,
    )
    ctx = AppContext(
        orchestrator=orch,
        signal_store=signal_store,
        signal_recorder=recorder,
        executor=executor,
        skills=skills,
        learner=learner,
        prompt_registry=LocalPromptRegistry(),
        bus=orch._bus,
        vectors=mem._vectors,
        history_store=history,
    )
    return create_app(settings, ctx)


async def _post(app: Any, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, json={"message": "hi", "project": "p"})


async def test_chat_is_a_502_that_names_the_endpoint() -> None:
    resp = await _post(_app(), "/api/chat")
    assert resp.status_code == 502
    body = resp.json()
    assert body["endpoint"] == _ENDPOINT
    assert _ENDPOINT in body["error"]


async def test_stream_that_never_starts_is_a_502_not_an_empty_200() -> None:
    resp = await _post(_app(), "/api/chat/stream")
    assert resp.status_code == 502
    assert resp.json()["endpoint"] == _ENDPOINT


async def test_stream_that_dies_mid_way_reports_it_in_band_then_ends() -> None:
    resp = await _post(_app(tokens_before_failure=2), "/api/chat/stream")
    assert resp.status_code == 200
    events = [line[len("data: ") :] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]
    assert [p.get("delta") for p in payloads[:2]] == ["tok0 ", "tok1 "]
    assert payloads[-1]["endpoint"] == _ENDPOINT
    assert _ENDPOINT in payloads[-1]["error"]


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
async def test_health_still_answers_while_the_model_is_down(path: str) -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/health")).status_code == 200
        assert (await c.post(path, json={"message": "hi", "project": "p"})).status_code == 502
