"""Public synthetic Gate/NOOP demo: PYTHONPATH=. python examples/consolidation_byte_limit_noop.py.

In-memory SQLite, fake embeddings and a fixed local response; no network or model.
The one-byte ceiling is a deliberate rejection control, not a recommended policy.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource, TemporalFact
from morgan_brain.providers.request_budget import StructuredRequestTooLarge
from morgan_brain.providers.wire import ChatMessage, ChatResult, ToolSpec


class LocalNoop:
    """A deterministic substitute for generation, with no external capability."""

    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[ChatMessage] = []

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.messages = messages
        return ChatResult(
            text='{"ops":[{"op":"NOOP","subject":"user","predicate":"permission"}]}', model=model
        )


async def main() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    conn = open_db(":memory:")
    try:
        gate = MemoryGate(
            build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=lambda: now)
        )
        for identity, content, parents in [
            ("public-original", "user may publish project report", []),
            ("public-revision", "user may not publish project report", ["public-original"]),
        ]:
            await gate.store(
                Memory(
                    id=identity,
                    user_id="synthetic",
                    project="public",
                    created_at=now,
                    content=content,
                    source=MemorySource.USER_STATED,
                    author_id="person:synthetic",
                    revises_event_ids=parents,
                )
            )
        await gate.upsert_fact(
            TemporalFact(
                user_id="synthetic",
                project="public",
                created_at=now,
                subject="user",
                predicate="permission",
                object="may publish project report",
                source=MemorySource.USER_STATED,
            )
        )
        before = conn.serialize()
        local = LocalNoop()
        bounded = MemoryConsolidator(
            gate=gate, client=local, model="local-noop", clock=lambda: now, request_byte_limit=1
        )
        try:
            await bounded.consolidate("synthetic", project="public")
        except StructuredRequestTooLarge as error:
            refusal = {
                "reason": str(error),
                "measured_bytes": error.measured_bytes,
                "limit_bytes": error.limit_bytes,
                "attempt": error.attempt,
            }
        else:
            raise RuntimeError("Expected explicit size refusal")
        if local.calls != 0 or conn.serialize() != before:
            raise RuntimeError("Rejected initial request changed state or called local generator")
        compatible = MemoryConsolidator(
            gate=gate, client=local, model="local-noop", clock=lambda: now
        )
        effects = await compatible.consolidate("synthetic", project="public")
        prompt = local.messages[-1].content
        if (
            effects
            or local.calls != 1
            or conn.serialize() != before
            or '"id": "public-revision"' not in prompt
        ):
            raise RuntimeError("Default NOOP contract failed")
        print(
            json.dumps(
                {
                    "initial_refusal": refusal,
                    "default_local_noop_calls": local.calls,
                    "db_unchanged": True,
                    "revision_delivered": True,
                    "provider_calls": 0,
                    "token_count": None,
                },
                indent=2,
            )
        )
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
