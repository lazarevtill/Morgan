"""ProactivityEngine — derive and publish consent-gated suggestions.

Design principles
-----------------
* default-deny: every suggestion kind must be explicitly granted by a
  :class:`ConsentRule` in the :class:`ConsentGate`.
* Delivery stays out-of-scope here (Phase 5); the engine only publishes
  ``EventType.PROACTIVE_SUGGESTION`` events on the bus.
* ``derive_from_patterns`` provides a lightweight, LLM-free mapping from
  :class:`BehavioralPattern` objects to :class:`ProactiveSuggestion` candidates.
  Callers can optionally enrich the candidates before passing them to
  ``maybe_suggest``.

Event schema
------------
``Event.payload`` for ``PROACTIVE_SUGGESTION``::

    {
        "kind": "<suggestion kind>",
        "message": "<suggestion text>",
        "evidence": ["...", ...],
    }
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, Field

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.models.user import BehavioralPattern, RelationshipStage, UserModel
from morgan_brain.proactivity.consent import ConsentGate

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class ProactiveSuggestion(BaseModel):
    """A single proactive suggestion candidate.

    Parameters
    ----------
    kind:
        Suggestion kind; matched against :class:`ConsentRule` kinds in the gate.
    message:
        Human-readable suggestion text.
    evidence:
        Optional list of evidence strings (e.g. pattern descriptions, cue labels)
        that justify the suggestion.
    """

    kind: str
    message: str
    evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern → suggestion heuristics
# ---------------------------------------------------------------------------

# Keyword → (kind, message template) mapping.
# The heuristic is intentionally simple — the full NLP pipeline is Phase 5.
_PATTERN_HEURISTICS: list[tuple[str, str, str]] = [
    # (keyword_in_description_lower, kind, message_template)
    ("plan", "suggestion", "Want me to help plan that for you?"),
    ("week", "suggestion", "Would you like me to summarise your week or plan ahead?"),
    ("morning", "reminder", "Shall I set up a morning check-in for you?"),
    ("remind", "reminder", "Should I create a reminder for that?"),
    ("summary", "summary", "Want a quick summary of recent activity?"),
    ("review", "summary", "Would you like a review of what we've covered?"),
    ("daily", "reminder", "Would a daily nudge help you stay on track?"),
    ("goal", "suggestion", "Would you like help tracking or adjusting that goal?"),
]

_DEFAULT_KIND = "suggestion"
_DEFAULT_MESSAGE = "Is there anything I can help you with right now?"


def _pattern_to_suggestion(pattern: BehavioralPattern) -> ProactiveSuggestion:
    """Map one :class:`BehavioralPattern` to a :class:`ProactiveSuggestion`.

    Uses keyword matching on ``pattern.description`` (lowercased).  Falls back
    to a generic suggestion when no heuristic matches.
    """
    desc_lower = pattern.description.lower()
    for keyword, kind, message in _PATTERN_HEURISTICS:
        if keyword in desc_lower:
            return ProactiveSuggestion(
                kind=kind,
                message=message,
                evidence=[pattern.description, pattern.cue] if pattern.cue else [pattern.description],
            )
    return ProactiveSuggestion(
        kind=_DEFAULT_KIND,
        message=_DEFAULT_MESSAGE,
        evidence=[pattern.description],
    )


# ---------------------------------------------------------------------------
# ProactivityEngine
# ---------------------------------------------------------------------------


class ProactivityEngine:
    """Filters candidate suggestions through the consent gate and publishes events.

    Parameters
    ----------
    gate:
        The :class:`ConsentGate` that decides which suggestion kinds are allowed.
    bus:
        The :class:`EventBus` to publish ``PROACTIVE_SUGGESTION`` events to.
    clock:
        Zero-argument callable returning the current :class:`datetime`.
    """

    def __init__(
        self,
        *,
        gate: ConsentGate,
        bus: EventBus,
        clock: Callable[[], datetime],
    ) -> None:
        self._gate = gate
        self._bus = bus
        self._clock = clock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def maybe_suggest(
        self,
        *,
        user_id: str,
        user_model: UserModel,
        candidates: list[ProactiveSuggestion],
    ) -> list[ProactiveSuggestion]:
        """Filter *candidates* through the consent gate and publish allowed ones.

        For each candidate that passes the gate:
        1. A ``PROACTIVE_SUGGESTION`` event is published on the bus.
        2. The suggestion is included in the returned list.

        Suggestions that are denied by the gate are silently dropped.

        Parameters
        ----------
        user_id:
            Embedded in emitted events.
        user_model:
            The current :class:`UserModel`; ``relationship_stage`` is used for
            consent gate evaluation.
        candidates:
            Candidate :class:`ProactiveSuggestion` objects to filter.

        Returns
        -------
        list[ProactiveSuggestion]
            The subset of *candidates* that were allowed and published.
        """
        allowed: list[ProactiveSuggestion] = []
        stage = user_model.relationship_stage

        for candidate in candidates:
            if not self._gate.allows(candidate.kind, stage):
                continue
            # Publish the event.
            event = Event(
                type=EventType.PROACTIVE_SUGGESTION,
                user_id=user_id,
                payload={
                    "kind": candidate.kind,
                    "message": candidate.message,
                    "evidence": candidate.evidence,
                },
            )
            await self._bus.publish(event)
            allowed.append(candidate)

        return allowed

    def derive_from_patterns(
        self,
        user_model: UserModel,
    ) -> list[ProactiveSuggestion]:
        """Turn ``user_model.behavioral_patterns`` into suggestion candidates.

        This is a lightweight, LLM-free pass that maps each pattern to a
        suggestion using keyword heuristics.  The returned candidates should be
        passed to :meth:`maybe_suggest` for gate filtering and event publishing.

        Parameters
        ----------
        user_model:
            The :class:`UserModel` whose ``behavioral_patterns`` are inspected.

        Returns
        -------
        list[ProactiveSuggestion]
            One candidate per behavioral pattern.  May be empty when the model
            has no patterns.
        """
        return [_pattern_to_suggestion(p) for p in user_model.behavioral_patterns]
