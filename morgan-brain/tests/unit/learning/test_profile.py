"""Unit tests for UserProfileBuilder and CIPHER learn-from-edits.

Phase 2C — profile.build() derives UserModel from facts; render_md() stays under char cap;
preference_delta_from_edit + apply_edit_delta merge preferences deterministically.
All tests use fakes — no network, no LLM, deterministic clock.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from morgan_brain.learning.profile import UserProfileBuilder, apply_edit_delta, preference_delta_from_edit
from morgan_brain.models.memory import MemorySource, TemporalFact
from morgan_brain.models.user import CommunicationPrefs, RelationshipStage, UserModel
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
USER = "u1"


def _make_gate(facts: list[TemporalFact] | None = None) -> tuple[MemoryGate, SqliteTemporalStore]:
    temporal = SqliteTemporalStore(":memory:")
    mm = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=temporal,
        clock=lambda: T0,
    )
    gate = MemoryGate(mm)
    return gate, temporal


async def _seed_fact(gate: MemoryGate, predicate: str, object_: str, confidence: float = 1.0) -> None:
    await gate.upsert_fact(
        TemporalFact(
            user_id=USER,
            subject="user",
            predicate=predicate,
            object=object_,
            source=MemorySource.USER_STATED,
            confidence=confidence,
            created_at=T0,
        )
    )


def _make_router(replies: list[str]) -> tuple[RoleRouter, CapabilityRegistry]:
    fake_client = FakeChatClient(replies=replies)
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
    return router, reg


# ---------------------------------------------------------------------------
# build() — comm_prefs from facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_terse_pref_from_fact() -> None:
    """A 'prefers→terse' fact should produce comm_prefs.length == 'terse'."""
    gate, _ = _make_gate()
    await _seed_fact(gate, "prefers", "terse")

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert um.comm_prefs.length == "terse"


@pytest.mark.asyncio
async def test_build_comm_tone_from_fact() -> None:
    """A 'comm_tone→formal' fact sets comm_prefs.formality."""
    gate, _ = _make_gate()
    await _seed_fact(gate, "comm_tone", "formal")

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert um.comm_prefs.formality == "formal"


@pytest.mark.asyncio
async def test_build_topic_fact_populates_topics() -> None:
    """Facts with predicate 'topic' or 'interest_in' should populate topics_of_interest."""
    gate, _ = _make_gate()
    await _seed_fact(gate, "topic", "python", confidence=0.9)

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert "python" in um.topics_of_interest


@pytest.mark.asyncio
async def test_build_interest_in_fact_populates_topics() -> None:
    """Facts with predicate 'interest_in' should also populate topics_of_interest."""
    gate, _ = _make_gate()
    await _seed_fact(gate, "interest_in", "machine learning", confidence=0.8)

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert "machine learning" in um.topics_of_interest


@pytest.mark.asyncio
async def test_build_relationship_stage_new_with_no_facts() -> None:
    """Zero facts → NEW stage."""
    gate, _ = _make_gate()
    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert um.relationship_stage is RelationshipStage.NEW


@pytest.mark.asyncio
async def test_build_relationship_stage_scales_with_fact_count() -> None:
    """5+ facts → at least ACQUAINTED."""
    gate, _ = _make_gate()
    for i in range(6):
        await _seed_fact(gate, f"fact_{i}", f"val_{i}")

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert um.relationship_stage is not RelationshipStage.NEW


@pytest.mark.asyncio
async def test_build_trusted_with_many_facts() -> None:
    """50+ facts → TRUSTED stage."""
    gate, _ = _make_gate()
    for i in range(55):
        await _seed_fact(gate, f"fact_{i}", f"val_{i}")

    router, reg = _make_router([])
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router,
        capability_registry=reg,
        clock=lambda: T0,
    )
    um = await builder.build(USER)
    assert um.relationship_stage is RelationshipStage.TRUSTED


# ---------------------------------------------------------------------------
# render_md() — format & char cap
# ---------------------------------------------------------------------------


def test_render_md_under_char_cap() -> None:
    """render_md output must be under 1200 characters."""
    from morgan_brain.learning.profile import render_md

    um = UserModel(
        user_id=USER,
        comm_prefs=CommunicationPrefs(length="terse", formality="formal"),
        topics_of_interest={"python": 0.9, "ml": 0.8},
        relationship_stage=RelationshipStage.FAMILIAR,
    )
    md = render_md(um)
    assert len(md) <= 1200


def test_render_md_has_stable_and_dynamic_sections() -> None:
    """render_md must contain STABLE and DYNAMIC section markers."""
    from morgan_brain.learning.profile import render_md

    um = UserModel(user_id=USER)
    md = render_md(um)
    assert "STABLE" in md
    assert "DYNAMIC" in md


def test_render_md_truncates_lowest_confidence_first() -> None:
    """When traits exceed budget, lowest-confidence traits should be omitted first."""
    from morgan_brain.learning.profile import render_md
    from morgan_brain.models.user import Trait

    # Build a model with many low-conf traits (i/20 capped at 1.0 to stay valid)
    traits = [Trait(name=f"trait_{i}", value="x", confidence=min(1.0, i / 30.0)) for i in range(30)]
    um = UserModel(user_id=USER, traits=traits)
    md = render_md(um)
    # Lowest-conf traits (trait_0 = conf=0.0) should not appear if we're over budget
    # High-conf trait should appear
    assert "trait_29" in md  # highest confidence
    assert len(md) <= 1200


# ---------------------------------------------------------------------------
# preference_delta_from_edit — CIPHER
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preference_delta_from_edit_calls_llm_and_returns_delta() -> None:
    """preference_delta_from_edit should call the fake LLM and return the delta string."""
    delta_payload = json.dumps({"delta": "prefers concise, code-first"})
    _, reg = _make_router([delta_payload])
    fake_client = FakeChatClient(reply=delta_payload)
    reg2 = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router2 = RoleRouter(
        reg=reg2,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    gate, _ = _make_gate()
    builder = UserProfileBuilder(
        gate=gate,
        signals=None,  # type: ignore[arg-type]
        router=router2,
        capability_registry=reg2,
        clock=lambda: T0,
    )

    original = "Here is a detailed explanation of the algorithm with extensive background..."
    edited = "Use code. Skip the intro."
    delta = await preference_delta_from_edit(builder, USER, original, edited)
    assert isinstance(delta, str)
    assert len(delta) > 0


# ---------------------------------------------------------------------------
# apply_edit_delta — deterministic merge
# ---------------------------------------------------------------------------


def test_apply_edit_delta_concise_sets_terse() -> None:
    """'concise' keyword in delta → comm_prefs.length = 'terse'."""
    um = UserModel(user_id=USER)
    result = apply_edit_delta(um, "prefers concise answers")
    assert result.comm_prefs.length == "terse"


def test_apply_edit_delta_code_sets_code_first() -> None:
    """'code' keyword in delta → comm_prefs.code_vs_prose = 'code_first'."""
    um = UserModel(user_id=USER)
    result = apply_edit_delta(um, "code-first responses preferred")
    assert result.comm_prefs.code_vs_prose == "code_first"


def test_apply_edit_delta_formal_sets_formality() -> None:
    """'formal' keyword in delta → comm_prefs.formality = 'formal'."""
    um = UserModel(user_id=USER)
    result = apply_edit_delta(um, "prefers formal tone")
    assert result.comm_prefs.formality == "formal"


def test_apply_edit_delta_thorough_sets_length() -> None:
    """'thorough' or 'detailed' keyword → comm_prefs.length = 'thorough'."""
    um = UserModel(user_id=USER)
    result = apply_edit_delta(um, "wants thorough explanations")
    assert result.comm_prefs.length == "thorough"


def test_apply_edit_delta_no_hedging_adds_trait() -> None:
    """'no hedging' keyword in delta → a trait is added."""
    um = UserModel(user_id=USER)
    result = apply_edit_delta(um, "no hedging, be direct")
    trait_names = {t.name for t in result.traits}
    assert "no_hedging" in trait_names


def test_apply_edit_delta_is_deterministic() -> None:
    """Same delta applied twice produces identical result."""
    um = UserModel(user_id=USER)
    r1 = apply_edit_delta(um, "concise code-first")
    r2 = apply_edit_delta(um, "concise code-first")
    assert r1.comm_prefs == r2.comm_prefs
    assert r1.traits == r2.traits
