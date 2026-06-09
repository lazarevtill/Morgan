"""Unit tests for ConsentGate + ConsentRule (Phase 4, commit 3).

All tests are deterministic and have no external dependencies.
"""

from __future__ import annotations


from morgan_brain.models.user import RelationshipStage
from morgan_brain.proactivity.consent import ConsentGate, ConsentRule

# Shorthand aliases for readability
NEW = RelationshipStage.NEW
ACQUAINTED = RelationshipStage.ACQUAINTED
FAMILIAR = RelationshipStage.FAMILIAR
TRUSTED = RelationshipStage.TRUSTED


# ---------------------------------------------------------------------------
# Default-deny
# ---------------------------------------------------------------------------


def test_empty_gate_denies_everything() -> None:
    gate = ConsentGate(rules=[])
    assert gate.allows("reminder", TRUSTED) is False
    assert gate.allows("summary", NEW) is False
    assert gate.allows("suggestion", FAMILIAR) is False


def test_no_matching_rule_denies() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    assert gate.allows("summary", TRUSTED) is False


# ---------------------------------------------------------------------------
# Rule matching: enabled + stage
# ---------------------------------------------------------------------------


def test_allows_when_stage_exactly_meets_min_stage() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    assert gate.allows("reminder", FAMILIAR) is True


def test_allows_when_stage_exceeds_min_stage() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    assert gate.allows("reminder", TRUSTED) is True


def test_denies_when_stage_below_min_stage() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="reminder", min_stage=FAMILIAR)])
    assert gate.allows("reminder", NEW) is False
    assert gate.allows("reminder", ACQUAINTED) is False


# ---------------------------------------------------------------------------
# NEW stage always denied (unless rule allows NEW explicitly)
# ---------------------------------------------------------------------------


def test_new_stage_user_denied_with_familiar_rule() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="suggestion", min_stage=FAMILIAR)])
    assert gate.allows("suggestion", NEW) is False


def test_new_stage_user_allowed_only_when_rule_min_stage_is_new() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="welcome", min_stage=NEW)])
    assert gate.allows("welcome", NEW) is True


# ---------------------------------------------------------------------------
# enabled=False disables a rule
# ---------------------------------------------------------------------------


def test_disabled_rule_denies_even_at_trusted() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="reminder", min_stage=NEW, enabled=False)])
    assert gate.allows("reminder", TRUSTED) is False


def test_enabled_flag_default_is_true() -> None:
    rule = ConsentRule(kind="test", min_stage=NEW)
    assert rule.enabled is True


# ---------------------------------------------------------------------------
# RelationshipStage ordering (smoke-test the ordinal helper)
# ---------------------------------------------------------------------------


def test_stage_ordering_new_acquainted_familiar_trusted() -> None:
    """Verify our ordinal helper produces NEW < ACQUAINTED < FAMILIAR < TRUSTED."""
    from morgan_brain.proactivity.consent import _stage_gte

    assert _stage_gte(TRUSTED, NEW) is True
    assert _stage_gte(TRUSTED, TRUSTED) is True
    assert _stage_gte(FAMILIAR, ACQUAINTED) is True
    assert _stage_gte(ACQUAINTED, NEW) is True
    assert _stage_gte(NEW, ACQUAINTED) is False
    assert _stage_gte(NEW, FAMILIAR) is False
    assert _stage_gte(ACQUAINTED, FAMILIAR) is False


# ---------------------------------------------------------------------------
# Multiple rules — first match wins
# ---------------------------------------------------------------------------


def test_first_matching_rule_wins_disabled() -> None:
    """When first matching rule is disabled, it returns False (not checked further)."""
    gate = ConsentGate(
        rules=[
            ConsentRule(kind="reminder", min_stage=NEW, enabled=False),
            ConsentRule(kind="reminder", min_stage=NEW, enabled=True),
        ]
    )
    # First rule is disabled → deny; second rule is not consulted.
    assert gate.allows("reminder", TRUSTED) is False


def test_different_kinds_independent() -> None:
    gate = ConsentGate(
        rules=[
            ConsentRule(kind="reminder", min_stage=FAMILIAR),
            ConsentRule(kind="summary", min_stage=TRUSTED),
        ]
    )
    assert gate.allows("reminder", FAMILIAR) is True
    assert gate.allows("summary", FAMILIAR) is False
    assert gate.allows("summary", TRUSTED) is True


# ---------------------------------------------------------------------------
# add_rule
# ---------------------------------------------------------------------------


def test_add_rule_appends_and_is_consulted() -> None:
    gate = ConsentGate(rules=[])
    gate.add_rule(ConsentRule(kind="ping", min_stage=NEW))
    assert gate.allows("ping", NEW) is True


def test_add_rule_is_lower_priority_than_existing() -> None:
    gate = ConsentGate(rules=[ConsentRule(kind="ping", min_stage=FAMILIAR, enabled=False)])
    gate.add_rule(ConsentRule(kind="ping", min_stage=NEW, enabled=True))
    # First rule (disabled) wins → deny.
    assert gate.allows("ping", TRUSTED) is False
