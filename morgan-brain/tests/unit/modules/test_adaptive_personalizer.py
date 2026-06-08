"""Unit tests for AdaptivePersonalizer (Phase 2C).

Verifies:
- Selects traits relevant to the current turn (text/intent/entity overlap).
- Skips traits with no relevance to the turn (irrelevant low-relevance traits NOT selected).
- Respects budget (caps trait count).
- Guardrail: skips low-confidence traits.
- Sets tone from comm_prefs.
- Sets proactive_threshold from relationship_stage (NEW→high; TRUSTED→low).
- Stateless: no writes occur.
"""
from __future__ import annotations

import pytest

from morgan_brain.models.base import Entity
from morgan_brain.models.perception import FusedPerception, Intent
from morgan_brain.models.user import CommunicationPrefs, RelationshipStage, Trait, UserModel
from morgan_brain.modules.personalization.adaptive import AdaptivePersonalizer


def _make_um(
    *,
    traits: list[Trait] | None = None,
    comm_prefs: CommunicationPrefs | None = None,
    stage: RelationshipStage = RelationshipStage.NEW,
) -> UserModel:
    return UserModel(
        user_id="u1",
        traits=traits or [],
        comm_prefs=comm_prefs or CommunicationPrefs(),
        relationship_stage=stage,
    )


def _make_perception(text: str, intent: str = "chat", entities: list[str] | None = None) -> FusedPerception:
    ents = [Entity(name=e) for e in (entities or [])]
    return FusedPerception(
        text=text,
        intent=Intent(name=intent, confidence=0.9),
        entities=ents,
    )


# ---------------------------------------------------------------------------
# Trait selection — relevance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selects_trait_matching_query() -> None:
    """A trait whose name/value overlaps with the query text should be selected."""
    traits = [
        Trait(name="python_expert", value="python", confidence=0.9),
        Trait(name="unrelated_cooking", value="cooking", confidence=0.9),
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("tell me about python decorators")

    p = AdaptivePersonalizer(budget=0.5)
    ctx = await p.build(user_model=um, perception=perc)
    assert "python_expert" in ctx.selected_traits


@pytest.mark.asyncio
async def test_skips_irrelevant_low_relevance_trait() -> None:
    """A trait with no overlap with the query should NOT be selected."""
    traits = [
        Trait(name="cooking_hobby", value="cooking", confidence=0.9),
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("explain how git rebase works")

    p = AdaptivePersonalizer(budget=0.5)
    ctx = await p.build(user_model=um, perception=perc)
    assert "cooking_hobby" not in ctx.selected_traits


@pytest.mark.asyncio
async def test_selects_via_entity_overlap() -> None:
    """A trait whose value overlaps with perception entities should be selected."""
    traits = [
        Trait(name="typescript_pref", value="typescript", confidence=0.9),
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("how to type this function?", entities=["typescript"])

    p = AdaptivePersonalizer(budget=0.5)
    ctx = await p.build(user_model=um, perception=perc)
    assert "typescript_pref" in ctx.selected_traits


# ---------------------------------------------------------------------------
# Confidence guardrail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_skips_low_confidence_trait() -> None:
    """Traits below confidence floor should NOT be selected even if text-relevant."""
    traits = [
        Trait(name="python_expert", value="python", confidence=0.05),
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("python question")

    p = AdaptivePersonalizer(budget=0.5)
    ctx = await p.build(user_model=um, perception=perc)
    assert "python_expert" not in ctx.selected_traits


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_caps_trait_count() -> None:
    """With a very small budget, only a small number of traits are selected."""
    # Create many relevant traits
    traits = [
        Trait(name=f"python_{i}", value="python", confidence=0.9) for i in range(20)
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("python programming help")

    p = AdaptivePersonalizer(budget=0.05)  # very small budget → max ~1 trait
    ctx = await p.build(user_model=um, perception=perc)
    assert len(ctx.selected_traits) <= max(1, round(0.05 * 20))


@pytest.mark.asyncio
async def test_budget_default_selects_reasonable_count() -> None:
    """Default budget (0.15) should select at most round(0.15*20)=3 traits."""
    traits = [
        Trait(name=f"python_{i}", value="python", confidence=0.9) for i in range(20)
    ]
    um = _make_um(traits=traits)
    perc = _make_perception("python programming")

    p = AdaptivePersonalizer()
    ctx = await p.build(user_model=um, perception=perc)
    assert len(ctx.selected_traits) <= max(1, round(0.15 * 20))


# ---------------------------------------------------------------------------
# Tone + proactive_threshold from model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tone_set_from_comm_prefs() -> None:
    """tone should come from user_model.comm_prefs.tone."""
    um = _make_um(comm_prefs=CommunicationPrefs(tone="warm"))
    ctx = await AdaptivePersonalizer().build(
        user_model=um, perception=_make_perception("hello")
    )
    assert ctx.tone == "warm"


@pytest.mark.asyncio
async def test_proactive_threshold_high_for_new_user() -> None:
    """NEW stage → proactive_threshold should be high (less proactive)."""
    um = _make_um(stage=RelationshipStage.NEW)
    ctx = await AdaptivePersonalizer().build(
        user_model=um, perception=_make_perception("hi")
    )
    assert ctx.proactive_threshold >= 0.7


@pytest.mark.asyncio
async def test_proactive_threshold_low_for_trusted_user() -> None:
    """TRUSTED stage → proactive_threshold should be lower (more proactive)."""
    um = _make_um(stage=RelationshipStage.TRUSTED)
    ctx = await AdaptivePersonalizer().build(
        user_model=um, perception=_make_perception("hi")
    )
    assert ctx.proactive_threshold <= 0.4


# ---------------------------------------------------------------------------
# system_fragment content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_fragment_includes_selected_trait_info() -> None:
    """system_fragment should mention selected trait information."""
    traits = [Trait(name="python_expert", value="python", confidence=0.9)]
    um = _make_um(traits=traits)
    perc = _make_perception("python question")

    p = AdaptivePersonalizer()
    ctx = await p.build(user_model=um, perception=perc)
    # Fragment should mention the trait or its value
    assert "python" in ctx.system_fragment.lower()


@pytest.mark.asyncio
async def test_system_fragment_includes_comm_prefs() -> None:
    """system_fragment should include comm_prefs even when no traits selected."""
    um = _make_um(comm_prefs=CommunicationPrefs(length="terse"))
    perc = _make_perception("hello")

    p = AdaptivePersonalizer()
    ctx = await p.build(user_model=um, perception=perc)
    assert "terse" in ctx.system_fragment


@pytest.mark.asyncio
async def test_stateless_no_side_effects() -> None:
    """build() should not modify the user_model or write anything."""
    um = _make_um(traits=[Trait(name="t1", value="python", confidence=0.9)])
    original_traits = list(um.traits)
    perc = _make_perception("python question")

    p = AdaptivePersonalizer()
    await p.build(user_model=um, perception=perc)
    assert um.traits == original_traits


# ---------------------------------------------------------------------------
# Personalizer Protocol conformance
# ---------------------------------------------------------------------------


def test_adaptive_personalizer_satisfies_protocol() -> None:
    from morgan_brain.interfaces.personalization import Personalizer

    assert isinstance(AdaptivePersonalizer(), Personalizer)
