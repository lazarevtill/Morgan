"""Byte ceilings include complete request data; no live endpoints."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource
from morgan_brain.providers.request_budget import StructuredRequestTooLarge
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage
from tests.fakes import FakeChatClient


class Answer(BaseModel):
    answer: str


def schema(mode):
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": Answer.model_json_schema()},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def measured(messages, mode):
    request = {"model": "fake", "messages": [m.to_openai() for m in messages]}
    if schema(mode):
        request["response_format"] = schema(mode)
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())


@pytest.mark.parametrize("text", ['EN "quoted"\nsource', "RU источник «данные»\nстрока"])
@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompted"])
async def test_exact_utf8_boundary_includes_overhead(text, mode):
    messages = [ChatMessage(role="user", content=text)]
    probe = FakeChatClient(replies=['{"answer":"ok"}'])
    await generate_structured(probe, messages, model="fake", schema=Answer, json_mode=mode)
    ceiling = measured(probe.last_messages, mode)
    assert ceiling > len(text.encode())
    exact = FakeChatClient(replies=['{"answer":"ok"}'])
    result = await generate_structured(
        exact, messages, model="fake", schema=Answer, json_mode=mode, request_byte_limit=ceiling
    )
    assert result.answer == "ok" and exact.calls == 1
    rejected = FakeChatClient(replies=['{"answer":"ok"}'])
    with pytest.raises(StructuredRequestTooLarge) as caught:
        await generate_structured(
            rejected,
            messages,
            model="fake",
            schema=Answer,
            json_mode=mode,
            request_byte_limit=ceiling - 1,
        )
    error = caught.value
    assert (error.measured_bytes, error.limit_bytes, error.attempt) == (ceiling, ceiling - 1, 1)
    assert "No generation attempted for this request" in str(error)
    assert rejected.calls == 0 and messages == [ChatMessage(role="user", content=text)]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_limit_before_client(limit):
    fake = FakeChatClient(replies=['{"answer":"ok"}'])
    with pytest.raises(ValueError, match="positive integer"):
        await generate_structured(fake, [], model="fake", schema=Answer, request_byte_limit=limit)
    assert fake.calls == 0


async def test_reask_refused_before_second_generation():
    messages = [ChatMessage(role="user", content="source")]
    first_size = measured(messages, "json_schema")
    fake = FakeChatClient(replies=["invalid", '{"answer":"ok"}'])
    with pytest.raises(StructuredRequestTooLarge) as caught:
        await generate_structured(
            fake, messages, model="fake", schema=Answer, request_byte_limit=first_size
        )
    assert caught.value.attempt == 2 and caught.value.measured_bytes > first_size
    assert fake.calls == 1


async def test_fake_embedding_gate_refusal_no_generation_no_writes_and_default_unchanged():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    conn = open_db(":memory:")
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=lambda: now)
    gate = MemoryGate(module)
    await gate.store(
        Memory(
            id="public-source",
            created_at=now,
            user_id="u",
            project="p",
            content="public синтетический источник",
            source=MemorySource.USER_STATED,
            author_id="person:u",
        )
    )
    before = conn.serialize()
    fake = FakeChatClient(replies=['{"ops":[]}'])
    worker = MemoryConsolidator(
        gate=gate, client=fake, model="fake", clock=lambda: now, request_byte_limit=1
    )
    with pytest.raises(StructuredRequestTooLarge, match="caller limit is 1 bytes"):
        await worker.consolidate("u", project="p")
    assert fake.calls == 0 and conn.serialize() == before
    default = MemoryConsolidator(gate=gate, client=fake, model="fake", clock=lambda: now)
    assert await default.consolidate("u", project="p") == []
    assert fake.calls == 1 and conn.serialize() == before
    conn.close()


async def test_checked_embedding_recall_can_record_fingerprint_before_request_refusal():
    """The guard prevents proposal/apply, not preceding recall metadata writes."""
    from morgan_brain.config import Settings
    from morgan_brain.memory.checked_embedder import CheckedEmbedder
    from morgan_brain.memory.store import spaces

    now = datetime(2026, 1, 1, tzinfo=UTC)
    conn = open_db(":memory:")
    try:
        initial = MemoryGate(
            build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=lambda: now)
        )
        await initial.store(
            Memory(
                id="public-source",
                created_at=now,
                user_id="u",
                project="p",
                content="public synthetic source",
                source=MemorySource.USER_STATED,
                author_id="person:u",
            )
        )
        spaces.register(
            conn, model="public-offline", dims=16, table_name="vec_items", clock=lambda: now
        )
        checked = CheckedEmbedder(
            FakeEmbedder(dim=16),
            conn=conn,
            settings=Settings(embedding_model="public-offline", embedding_dim=16),
            endpoint="public-offline-control",
            setting="MORGAN_EMBEDDING_ENDPOINT",
            clock=lambda: now,
        )
        gate = MemoryGate(build_memory_module(conn, embedder=checked, dim=16, clock=lambda: now))
        assert spaces.active(conn).fingerprint is None
        before = conn.serialize()
        fake = FakeChatClient(replies=['{"ops":[]}'])
        worker = MemoryConsolidator(
            gate=gate, client=fake, model="fake", clock=lambda: now, request_byte_limit=1
        )
        with pytest.raises(StructuredRequestTooLarge):
            await worker.consolidate("u", project="p")
        assert fake.calls == 0
        assert spaces.active(conn).fingerprint is not None
        assert conn.serialize() != before
        assert await gate.current_facts(user_id="u", project="p") == []
    finally:
        conn.close()
