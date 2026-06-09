"""Proactivity package — consent-gated proactive suggestions.

Design principle: default-deny.  A suggestion is only sent when an explicit
:class:`ConsentRule` allows it **and** the user's :class:`RelationshipStage`
meets the minimum required by that rule.
"""

from morgan_brain.proactivity.consent import ConsentGate, ConsentRule
from morgan_brain.proactivity.engine import ProactiveSuggestion, ProactivityEngine

__all__ = [
    "ConsentGate",
    "ConsentRule",
    "ProactiveSuggestion",
    "ProactivityEngine",
]
