"""Consent gate for proactive suggestions.

Design principle: default-deny.

A suggestion of a given *kind* is allowed only when:
1. A :class:`ConsentRule` with ``kind == suggestion_kind`` exists in the gate.
2. That rule has ``enabled=True``.
3. The user's ``relationship_stage >= rule.min_stage`` (ordinal comparison).

Rule lookup uses the first matching rule (by registration order).  If no rule
matches, the suggestion is denied.

RelationshipStage ordering
--------------------------
The enum values are ``"new" < "acquainted" < "familiar" < "trusted"`` in the
*semantic* sense, but string lexicographic order does not preserve this
(``"new" > "acquainted"`` lexicographically).  We therefore use an explicit
ordinal mapping:

    NEW=0, ACQUAINTED=1, FAMILIAR=2, TRUSTED=3
"""
from __future__ import annotations

from pydantic import BaseModel

from morgan_brain.models.user import RelationshipStage

# Explicit ordinal for RelationshipStage (string comparison is unreliable).
_STAGE_ORDER: dict[RelationshipStage, int] = {
    RelationshipStage.NEW: 0,
    RelationshipStage.ACQUAINTED: 1,
    RelationshipStage.FAMILIAR: 2,
    RelationshipStage.TRUSTED: 3,
}


def _stage_gte(a: RelationshipStage, b: RelationshipStage) -> bool:
    """Return True if *a* >= *b* in semantic relationship order."""
    return _STAGE_ORDER[a] >= _STAGE_ORDER[b]


class ConsentRule(BaseModel):
    """A single consent grant for a suggestion kind.

    Parameters
    ----------
    kind:
        The suggestion kind this rule governs (e.g. ``"reminder"``,
        ``"summary"``, ``"suggestion"``).
    min_stage:
        Minimum :class:`RelationshipStage` required for this rule to fire.
        Default: :attr:`RelationshipStage.FAMILIAR`.
    enabled:
        Whether this rule is active (can be toggled off without removing it).
    """

    kind: str
    min_stage: RelationshipStage = RelationshipStage.FAMILIAR
    enabled: bool = True


class ConsentGate:
    """Evaluates whether a suggestion kind is allowed for a given stage.

    Parameters
    ----------
    rules:
        Ordered list of :class:`ConsentRule` objects.  Lookup stops at the
        first rule whose ``kind`` matches.
    """

    def __init__(self, *, rules: list[ConsentRule]) -> None:
        self._rules = list(rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allows(self, kind: str, stage: RelationshipStage) -> bool:
        """Return True iff a matching, enabled rule exists AND stage >= min_stage.

        Default-deny: returns False when no rule matches.
        """
        for rule in self._rules:
            if rule.kind != kind:
                continue
            # First matching rule wins.
            if not rule.enabled:
                return False
            return _stage_gte(stage, rule.min_stage)
        # No matching rule → deny.
        return False

    def add_rule(self, rule: ConsentRule) -> None:
        """Append a rule (lower priority than existing rules of the same kind)."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[ConsentRule]:
        return list(self._rules)
