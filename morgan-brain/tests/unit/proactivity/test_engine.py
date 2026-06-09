"""Unit tests for ProactivityEngine (Phase 4, commit 3).

All tests are deterministic:
  - InProcessBus captures emitted events.
  - Injected fake clock.
  - No LLM calls, no network.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.models.user import BehavioralPattern, RelationshipStage, UserModel
from morgan_brain.proactivity.consent import ConsentGate, ConsentRule
from morgan_brain.proactivity.engine import ProactiveSuggestion, ProactivityEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 0, 0, 0)

NEW = RelationshipStage.NEW
ACQUAINTED = RelationshipStage.ACQUAINTED
FAMILIAR = RelationshipStage.FAMILIAR
TRUSTED = RelationshipStage.TRUSTED


def _make_engine(
    rules: list[ConsentRule],
) -> tuple[ProactivityEngine, list[Event]]:
    """Return (engine, collected_events)."""
    bus = InProcessBus()
    collected: list[Event] = []

    async def _collect(event: Event) -> None:
        collected.append(event)

    bus.subscribe(EventType.PROACTIVE_SUGGESTION, _collect)

    gate = ConsentGate(rules=rules)
    engine = ProactivityEngine(gate=gate, bus=bus, clock=lambda: T0)
    return engine, collected


def _user(stage: RelationshipStage, patterns: list[BehavioralPattern] | None = None) -> UserModel:
    return UserModel(
        user_id="alice",
        relationship_stage=stage,
        behavioral_patterns=patterns or [],
    )


# ---------------------------------------------------------------------------
# maybe_suggest: default-deny
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_rules_no_suggestions() -> None:
    engine, events = _make_engine(rules=[])
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(TRUSTED),
        candidates=[ProactiveSuggestion(kind="reminder", message="Hey!")],
    )
    assert result == []
    assert events == []


@pytest.mark.asyncio
async def test_new_stage_user_gets_nothing_even_with_matching_rule() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(NEW),
        candidates=[ProactiveSuggestion(kind="reminder", message="Hey!")],
    )
    assert result == []
    assert events == []


@pytest.mark.asyncio
async def test_familiar_user_allowed_by_familiar_rule() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(FAMILIAR),
        candidates=[ProactiveSuggestion(kind="reminder", message="Check in?")],
    )
    assert len(result) == 1
    assert result[0].kind == "reminder"


@pytest.mark.asyncio
async def test_trusted_user_gets_suggestion() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="suggestion", min_stage=TRUSTED)])
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(TRUSTED),
        candidates=[ProactiveSuggestion(kind="suggestion", message="Plan your week?")],
    )
    assert len(result) == 1
    assert len(events) == 1
    assert events[0].type == EventType.PROACTIVE_SUGGESTION
    assert events[0].user_id == "alice"
    assert events[0].payload["kind"] == "suggestion"


@pytest.mark.asyncio
async def test_allowed_suggestion_payload_contains_message_and_evidence() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="summary", min_stage=ACQUAINTED)])
    candidate = ProactiveSuggestion(
        kind="summary",
        message="Here's your weekly summary.",
        evidence=["weekly review pattern"],
    )
    await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(ACQUAINTED),
        candidates=[candidate],
    )
    assert events[0].payload["message"] == "Here's your weekly summary."
    assert events[0].payload["evidence"] == ["weekly review pattern"]


@pytest.mark.asyncio
async def test_mixed_candidates_only_allowed_ones_returned() -> None:
    """Some kinds allowed, some denied → only allowed ones returned and published."""
    engine, events = _make_engine(
        rules=[
            ConsentRule(kind="reminder", min_stage=FAMILIAR),
            ConsentRule(kind="summary", min_stage=TRUSTED),
        ]
    )
    candidates = [
        ProactiveSuggestion(kind="reminder", message="Reminder!"),  # allowed (FAMILIAR >= FAMILIAR)
        ProactiveSuggestion(kind="summary", message="Summary!"),  # denied (FAMILIAR < TRUSTED)
        ProactiveSuggestion(kind="unknown", message="Unknown!"),  # denied (no rule)
    ]
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(FAMILIAR),
        candidates=candidates,
    )
    assert len(result) == 1
    assert result[0].kind == "reminder"
    assert len(events) == 1


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="reminder", min_stage=NEW)])
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(TRUSTED),
        candidates=[],
    )
    assert result == []
    assert events == []


@pytest.mark.asyncio
async def test_multiple_suggestions_all_published() -> None:
    engine, events = _make_engine(rules=[ConsentRule(kind="reminder", min_stage=NEW)])
    candidates = [ProactiveSuggestion(kind="reminder", message=f"msg{i}") for i in range(3)]
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=_user(TRUSTED),
        candidates=candidates,
    )
    assert len(result) == 3
    assert len(events) == 3


# ---------------------------------------------------------------------------
# derive_from_patterns
# ---------------------------------------------------------------------------


def test_derive_from_patterns_empty_when_no_patterns() -> None:
    engine, _ = _make_engine(rules=[])
    user = _user(TRUSTED, patterns=[])
    result = engine.derive_from_patterns(user)
    assert result == []


def test_derive_from_patterns_maps_weekly_plan_pattern() -> None:
    """A 'plan' keyword in description → suggestion kind."""
    engine, _ = _make_engine(rules=[])
    pattern = BehavioralPattern(description="Sunday planning session", cue="Sunday 10:00")
    user = _user(TRUSTED, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    assert len(candidates) == 1
    assert candidates[0].kind == "suggestion"
    assert "plan" in candidates[0].message.lower() or "plan" in candidates[0].evidence[0].lower()


def test_derive_from_patterns_maps_morning_pattern() -> None:
    """A 'morning' keyword in description → reminder kind."""
    engine, _ = _make_engine(rules=[])
    pattern = BehavioralPattern(description="Morning check-in routine", cue="07:30")
    user = _user(TRUSTED, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    assert candidates[0].kind == "reminder"


def test_derive_from_patterns_includes_evidence_from_pattern() -> None:
    engine, _ = _make_engine(rules=[])
    pattern = BehavioralPattern(description="Weekly summary review", cue="Friday 17:00")
    user = _user(TRUSTED, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    assert pattern.description in candidates[0].evidence


def test_derive_from_patterns_unknown_pattern_gets_default_suggestion() -> None:
    engine, _ = _make_engine(rules=[])
    pattern = BehavioralPattern(description="Unusual behavior with no keyword match XYZ")
    user = _user(TRUSTED, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    assert candidates[0].kind == "suggestion"


def test_derive_from_patterns_multiple_patterns() -> None:
    engine, _ = _make_engine(rules=[])
    patterns = [
        BehavioralPattern(description="Morning walk routine"),
        BehavioralPattern(description="Weekly goal review"),
        BehavioralPattern(description="Daily summary check"),
    ]
    user = _user(TRUSTED, patterns=patterns)
    candidates = engine.derive_from_patterns(user)
    assert len(candidates) == 3


# ---------------------------------------------------------------------------
# End-to-end: patterns → derive → maybe_suggest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trusted_user_with_plan_pattern_gets_suggestion_event() -> None:
    """Full pipeline: pattern → derive → gate → event published."""
    engine, events = _make_engine(rules=[ConsentRule(kind="suggestion", min_stage=FAMILIAR)])
    pattern = BehavioralPattern(description="Sunday planning session", cue="Sunday 10:00")
    user = _user(TRUSTED, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=user,
        candidates=candidates,
    )
    assert len(result) >= 1
    assert any(e.type == EventType.PROACTIVE_SUGGESTION for e in events)


@pytest.mark.asyncio
async def test_new_stage_user_with_patterns_gets_no_events() -> None:
    """A NEW user never receives suggestions regardless of patterns."""
    engine, events = _make_engine(rules=[ConsentRule(kind="suggestion", min_stage=FAMILIAR)])
    pattern = BehavioralPattern(description="Sunday planning session", cue="Sunday 10:00")
    user = _user(NEW, patterns=[pattern])
    candidates = engine.derive_from_patterns(user)
    result = await engine.maybe_suggest(
        user_id="alice",
        user_model=user,
        candidates=candidates,
    )
    assert result == []
    assert events == []
