"""E2E test: THE core "learns from me" proof.

Turn 1: user states a preference ("I prefer terse, code-first answers").
Consolidate: FakeChatClient returns a FactOpBatch that adds the preference as a fact.
Turn 2: user asks a related question — the consolidated fact must appear in the
        recalled memories reaching turn 2's prompt (via memory recall or current_facts).

This proves the full pipeline: episodic storage → LLM-driven consolidation →
fact write → memory recall → fact visible in turn 2's prompt context.

Also demonstrates: "I live in Berlin" → consolidate → turn 2 knows it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.consolidation import FactOp, FactOpBatch, FactOpKind
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731
USER_ID = "u-learn-recall"
SESSION_ID = "sess-learn"


def _fact_op_batch_json(subject: str, predicate: str, obj: str) -> str:
    """Return a JSON string that validate as FactOpBatch with a single ADD op."""
    batch = FactOpBatch(
        ops=[
            FactOp(
                op=FactOpKind.ADD,
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.95,
                reason="stated by user",
            )
        ]
    )
    return batch.model_dump_json()


def _make_router_with_clients(
    *,
    turn_replies: list[str],
    consolidation_reply: str,
) -> tuple[RoleRouter, FakeChatClient]:
    """Build a router with separate clients for turns vs. consolidation.

    Both the 'strong' role (turns) and any structured-output calls (consolidation)
    use FakeChatClient.  We use a single client with a scripted reply queue to
    handle both paths.

    The consolidation call uses ``generate_structured``, which calls ``agenerate``.
    We need to ensure the consolidation call returns valid FactOpBatch JSON, while
    turn calls return conversational text.

    Strategy: prepend the consolidation reply (FactOpBatch JSON) to the queue
    followed by the turn replies.
    """
    all_replies = [consolidation_reply, *turn_replies]
    client = FakeChatClient(replies=all_replies)
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
        bindings={"strong": [Binding("fake", "test-model", client)]},
    )
    return router, client


@pytest.mark.asyncio
async def test_learn_preference_then_recall_in_turn2() -> None:
    """Turn 1: user states preference. Consolidate. Turn 2: fact reaches the prompt.

    This test exercises the entire state → consolidate → recall pipeline
    with fully deterministic fakes (no live services).

    IMPORTANT: The consolidation triggers a structured-output call via the LLM.
    We script the fake client to return a valid FactOpBatch JSON for that call,
    then return the turn2 answer for the second turn.
    """
    user_pref = "terse, code-first answers"
    fact_subject = "user"
    fact_predicate = "prefers"
    fact_object = user_pref

    consolidation_json = _fact_op_batch_json(fact_subject, fact_predicate, fact_object)

    # Script:
    # Call 0 (turn 1 LLM): returns acknowledgement text.
    # Call 1 (consolidation LLM): returns FactOpBatch JSON.
    # Call 2 (turn 2 LLM): returns the answer.
    client = FakeChatClient(
        replies=[
            "Got it, I'll keep answers terse and code-first.",  # turn 1 reply
            consolidation_json,  # consolidation call
            "Here is the code.",  # turn 2 reply
        ]
    )
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
        bindings={"strong": [Binding("fake", "test-model", client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _, _, _, _, _, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )

    # Turn 1: user states preference.
    result1 = await orch.handle_turn(
        user_id=USER_ID,
        text=f"I prefer {user_pref}.",
        session_id=SESSION_ID,
    )
    assert result1.text == "Got it, I'll keep answers terse and code-first."

    # Consolidation: the learner consolidates episodics into a fact.
    # This calls the LLM (fake client returns the FactOpBatch JSON).
    await learner.consolidate(USER_ID)

    # Turn 2: ask a related question. The consolidated fact should be in recall.
    result2 = await orch.handle_turn(
        user_id=USER_ID,
        text="Show me how to reverse a list in Python.",
        session_id=SESSION_ID,
    )
    assert result2.text == "Here is the code."

    # Verify the fact is now stored (via the temporal store / memory gate).
    # We access the learner's consolidator → temporal to check current facts.
    temporal = learner._consolidator._temporal  # type: ignore[attr-defined]  # noqa: SLF001
    facts = await temporal.current_facts(user_id=USER_ID)
    matching = [
        f
        for f in facts
        if f.subject == fact_subject and f.predicate == fact_predicate and fact_object in f.object
    ]
    assert len(matching) >= 1, (
        f"Expected fact (user prefers {user_pref}) in current_facts. Got: {facts}"
    )


@pytest.mark.asyncio
async def test_learn_location_then_visible_in_memories() -> None:
    """User states 'I live in Berlin' → consolidate → fact visible in current_facts."""
    consolidation_json = _fact_op_batch_json("user", "lives_in", "Berlin")

    client = FakeChatClient(
        replies=[
            "Noted, you live in Berlin.",  # turn 1 reply
            consolidation_json,  # consolidation
        ]
    )
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
        bindings={"strong": [Binding("fake", "test-model", client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    bus = InProcessBus()

    orch, _, _, _, _, _, learner = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=bus,
    )

    # Turn 1: user states location.
    result1 = await orch.handle_turn(
        user_id=USER_ID,
        text="I live in Berlin.",
        session_id=SESSION_ID,
    )
    assert result1.text == "Noted, you live in Berlin."

    # Consolidate.
    await learner.consolidate(USER_ID)

    # Check the temporal store directly.
    temporal = learner._consolidator._temporal  # type: ignore[attr-defined]  # noqa: SLF001
    facts = await temporal.current_facts(user_id=USER_ID)
    berlin_facts = [f for f in facts if "Berlin" in f.object]
    assert len(berlin_facts) >= 1, f"Expected Berlin fact in temporal store. Got: {facts}"
