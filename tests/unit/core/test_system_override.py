"""Tests for system_override in ReasoningRequest / build_messages / Orchestrator.

Asserts:
- build_messages with system_override prepends the override before _BASE_SYSTEM.
- handle_turn with system_override produces system message that contains override text.
- handle_turn with system_override="" is identical to the old behaviour (no regression).
- handle_turn_with_id also threads system_override correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.models.memory import Memory, MemoryKind
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.context.builder import _BASE_SYSTEM, build_messages

CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731


# ---------------------------------------------------------------------------
# Unit tests for build_messages
# ---------------------------------------------------------------------------


def _request(**kw: object) -> ReasoningRequest:
    base: dict[str, object] = {
        "user_id": "u1",
        "project": "p",
        "perception": FusedPerception(text="test query"),
        "personalization": PersonalizedContext(system_fragment=""),
        "memories": [],
        "history": [],
        "skill_prompt": "",
    }
    base.update(kw)
    return ReasoningRequest(**base)  # type: ignore[arg-type]


def test_system_override_prepended_before_base_system() -> None:
    """system_override text appears before _BASE_SYSTEM in the system message."""
    req = _request(system_override="CANDIDATE CHAMPION: be extra concise.")
    msgs = build_messages(req)
    system_content = msgs[0].content
    assert "CANDIDATE CHAMPION" in system_content
    assert _BASE_SYSTEM in system_content
    # Override must come BEFORE base system
    assert system_content.index("CANDIDATE CHAMPION") < system_content.index(_BASE_SYSTEM[:20])


def test_no_system_override_produces_same_as_before() -> None:
    """Omitting system_override (default '') does not alter the system message."""
    req_no_override = _request()
    req_empty = _request(system_override="")
    assert build_messages(req_no_override)[0].content == build_messages(req_empty)[0].content


def test_system_override_with_memories_and_personalization() -> None:
    """system_override co-exists with other system parts."""
    req = _request(
        system_override="OVERRIDE",
        personalization=PersonalizedContext(system_fragment="terse"),
        memories=[Memory(user_id="u1", kind=MemoryKind.SEMANTIC, content="lives in Berlin")],
    )
    msgs = build_messages(req)
    sys_content = msgs[0].content
    assert "OVERRIDE" in sys_content
    assert "terse" in sys_content
    assert "Berlin" in sys_content


# ---------------------------------------------------------------------------
# Integration tests for Orchestrator.handle_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_turn_system_override_in_system_message() -> None:
    """handle_turn with system_override causes override text to appear in the
    system message sent to the fake client."""
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.config import Settings
    from morgan_brain.providers.adapters.fake import FakeChatClient

    override_text = "CHAMPION_OVERRIDE_SENTINEL"
    fake_client = FakeChatClient(reply="answer")
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    # Rebuild with a client we can inspect
    orch2, _ = _build_orch_with_inspectable_client(override_text, fake_client, settings, bus)

    result = await orch2.handle_turn(
        user_id="u1",
        project="default",
        text="hello",
        system_override=override_text,
    )
    assert result.text == "answer"
    # The override must appear in the system message (index 0)
    system_msg = fake_client.last_messages[0]
    assert system_msg.role == "system"
    assert override_text in system_msg.content


def _build_orch_with_inspectable_client(
    override_text: str,
    fake_client: object,
    settings: object,
    bus: object,
) -> tuple[object, object]:
    """Helper — builds a minimal Orchestrator whose FakeChatClient we can inspect."""
    from morgan_brain.composition import _assemble
    from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
    from morgan_brain.providers.capability import CapabilityRegistry
    from morgan_brain.providers.router import Binding, RoleRouter

    fc = fake_client  # type: ignore[assignment]
    s = settings  # type: ignore[assignment]

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
        bindings={"strong": [Binding("fake", "test-model", fc)]},
    )
    return _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=s,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,  # type: ignore[arg-type]
    )[:2]


@pytest.mark.asyncio
async def test_handle_turn_no_override_no_sentinel() -> None:
    """Default (no system_override) does NOT add sentinel to system message."""
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.config import Settings
    from morgan_brain.providers.adapters.fake import FakeChatClient

    sentinel = "CHAMPION_OVERRIDE_SENTINEL"
    fake_client = FakeChatClient(reply="answer")
    s = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _ = _build_orch_with_inspectable_client(sentinel, fake_client, s, bus)

    result = await orch.handle_turn(  # type: ignore[union-attr]
        user_id="u1", project="default", text="hello"
    )
    assert result.text == "answer"
    system_msg = fake_client.last_messages[0]
    assert sentinel not in system_msg.content


@pytest.mark.asyncio
async def test_handle_turn_with_id_system_override() -> None:
    """handle_turn_with_id also threads system_override into the system message."""
    from morgan_brain.bus.inproc import InProcessBus
    from morgan_brain.config import Settings
    from morgan_brain.providers.adapters.fake import FakeChatClient

    sentinel = "OVERRIDE_FOR_TURN_WITH_ID"
    fake_client = FakeChatClient(reply="ok")
    s = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _ = _build_orch_with_inspectable_client(sentinel, fake_client, s, bus)

    result, turn_id = await orch.handle_turn_with_id(  # type: ignore[union-attr]
        user_id="u1",
        project="default",
        text="hello",
        system_override=sentinel,
    )
    assert result.text == "ok"
    assert len(turn_id) > 0
    system_msg = fake_client.last_messages[0]
    assert sentinel in system_msg.content
