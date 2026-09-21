"""``recall.done`` is emitted once per recall, to stderr, never stdout.

Phase 1a's availability trigger and the owner's own review read this line, not memory: it must
carry enough to tell a slow recall from an unused one, and which language the query was in,
without a stray ``print()`` corrupting the ``--json`` and MCP stdio protocols that also run
through this process's stdout.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from structlog.testing import capture_logs

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryKind, MemoryQuery
from morgan_brain.providers.wire import ProviderRefused, ProviderUnreachable

_CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731

#: An orthogonal pair of unit vectors: cosine 0.5 between the query and every background
#: memory, the same setup ``recall/test_floor_in_recall.py`` uses to force a decline.
_QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]
_FLAT = [0.5, 0.866, 0.0, 0.0]
_QUERY_TEXT = "unrelated question"

_ENDPOINT = "http://model.invalid/v1"
_SETTING = "MORGAN_EMBEDDING_ENDPOINT"


def _module():
    return build_memory_module(
        open_db(":memory:"), embedder=FakeEmbedder(dim=16), dim=16, clock=_CLOCK
    )


class _PlacedEmbedder:
    """Puts the query where the test says, so the floor's decision is chosen, not hashed."""

    async def embed(self, text: str) -> list[float]:
        return _QUERY_VECTOR if text == _QUERY_TEXT else _FLAT

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class _FailingEmbedder:
    """An embedder whose every call raises *exc* -- for proving ``recall.done`` still fires
    (and still names what went wrong) when the query's own embedding never came back."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def embed(self, text: str) -> list[float]:
        raise self._exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise self._exc


def _module_with_failing_embedder(exc: Exception):
    return build_memory_module(
        open_db(":memory:"), embedder=_FailingEmbedder(exc), dim=4, clock=_CLOCK
    )


async def test_recall_done_carries_its_four_fields(capsys):
    module = _module()
    await module.store(
        Memory(user_id="u1", project="p", kind=MemoryKind.EPISODIC, content="катер в порту")
    )

    with capture_logs() as logs:
        await module.recall(MemoryQuery(user_id="u1", project="p", text="где катер", top_k=5))

    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    entry = done[0]
    assert isinstance(entry["embed_latency_ms"], int | float)
    assert entry["embed_latency_ms"] >= 0
    assert entry["embed_outcome"] == "ok"
    assert entry["degraded"] is None
    assert entry["reason"] == "no_floor"  # this module has no floor configured
    assert entry["query_language"] == "ru"


async def test_recall_done_is_emitted_once_on_an_empty_project(capsys):
    module = _module()

    with capture_logs() as logs:
        found = await module.recall(
            MemoryQuery(user_id="u1", project="empty", text="anything", top_k=5)
        )

    assert found.memories == []
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["query_language"] == "en"
    assert done[0]["embed_outcome"] == "ok"
    assert done[0]["reason"] == "empty"


async def test_recall_done_is_emitted_once_even_when_the_floor_declines(capsys):
    module = build_memory_module(
        open_db(":memory:"),
        embedder=_PlacedEmbedder(),
        dim=4,
        clock=_CLOCK,
        floor_margin=0.1,
    )
    for n in range(6):
        await module.store(
            Memory(
                user_id="u1", project="p", kind=MemoryKind.EPISODIC, content=f"background note {n}"
            )
        )

    with capture_logs() as logs:
        found = await module.recall(
            MemoryQuery(user_id="u1", project="p", text=_QUERY_TEXT, top_k=5)
        )

    assert found.memories == []  # the floor declined: proof this hit the early return
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["embed_outcome"] == "ok"
    assert done[0]["reason"] == "declined"


async def test_recall_done_logs_unreachable_and_still_raises(capsys):
    exc = ProviderUnreachable(_ENDPOINT, "connect failed", setting=_SETTING, outcome="unreachable")
    module = _module_with_failing_embedder(exc)

    with capture_logs() as logs, pytest.raises(ProviderUnreachable) as raised:
        await module.recall(MemoryQuery(user_id="u1", project="p", text="a question", top_k=5))

    assert raised.value is exc
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["embed_outcome"] == "unreachable"
    assert done[0]["degraded"] is None
    assert done[0]["reason"] is None
    assert done[0]["query_language"] == "en"
    assert isinstance(done[0]["embed_latency_ms"], int | float)


async def test_recall_done_logs_slow_and_still_raises(capsys):
    exc = ProviderUnreachable(_ENDPOINT, "answered too slowly", setting=_SETTING, outcome="slow")
    module = _module_with_failing_embedder(exc)

    with capture_logs() as logs, pytest.raises(ProviderUnreachable) as raised:
        await module.recall(MemoryQuery(user_id="u1", project="p", text="a question", top_k=5))

    assert raised.value is exc
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["embed_outcome"] == "slow"


async def test_recall_done_logs_refused_and_still_raises(capsys):
    exc = ProviderRefused(_ENDPOINT, 400, _SETTING)
    module = _module_with_failing_embedder(exc)

    with capture_logs() as logs, pytest.raises(ProviderRefused) as raised:
        await module.recall(MemoryQuery(user_id="u1", project="p", text="a question", top_k=5))

    assert raised.value is exc
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["embed_outcome"] == "refused"


async def test_recall_done_logs_error_on_an_unclassified_embed_failure(capsys):
    """Neither `ProviderRefused` nor `ProviderUnreachable`: the embedder's own bug, not the
    model server's. `recall.done` must still fire, naming it `"error"` rather than silently
    skipping the line."""
    exc = RuntimeError("the embedder itself misbehaved")
    module = _module_with_failing_embedder(exc)

    with capture_logs() as logs, pytest.raises(RuntimeError) as raised:
        await module.recall(MemoryQuery(user_id="u1", project="p", text="a question", top_k=5))

    assert raised.value is exc
    assert capsys.readouterr().out == ""
    done = [entry for entry in logs if entry["event"] == "recall.done"]
    assert len(done) == 1
    assert done[0]["embed_outcome"] == "error"
