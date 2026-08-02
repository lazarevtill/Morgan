"""GAP-1 regression: the streaming path must apply the learned champion preprompt.

``stream_turn`` previously had no ``system_override`` parameter, so ``/api/chat/stream``
silently ran on the base prompt — the eval-gated champion (the entire visible output of
the self-learning loop) never reached a streamed turn. These tests pin the fix.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731
CHAMPION = "CHAMPION-PREPROMPT-XYZ: always answer like a pirate."


def _router(fake: FakeChatClient) -> RoleRouter:
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    return RoleRouter(reg=reg, bindings={"strong": [Binding("fake", "test-model", fake)]})


@pytest.mark.asyncio
async def test_stream_turn_applies_champion_system_override() -> None:
    fake = FakeChatClient(replies=["arr"])
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=_router(fake),
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
    )

    async for _ in orch.stream_turn(
        user_id="u1", project="default", text="hello", system_override=CHAMPION
    ):
        pass

    full_prompt = " ".join(m.content for m in fake.last_messages)
    assert CHAMPION in full_prompt, (
        f"champion preprompt missing from streamed prompt: {fake.last_messages}"
    )


@pytest.mark.asyncio
async def test_stream_turn_without_override_has_no_champion() -> None:
    fake = FakeChatClient(replies=["hi"])
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=_router(fake),
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
    )

    async for _ in orch.stream_turn(user_id="u1", project="default", text="hello"):
        pass

    full_prompt = " ".join(m.content for m in fake.last_messages)
    assert CHAMPION not in full_prompt
