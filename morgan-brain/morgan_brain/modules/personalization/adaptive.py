"""Phase 2C — AdaptivePersonalizer.

Implements the ``Personalizer`` Protocol.  Reads ``UserModel`` + ``FusedPerception``,
selects the traits relevant to THIS turn, and composes a ``PersonalizedContext``.

Stateless: it reads user_model + perception, writes nothing.

Trait selection algorithm:
1. Score each trait by text/entity/intent overlap with the perception.
2. Filter out traits below the confidence floor (anti-sycophancy guardrail).
3. Pick top-K within the budget (budget ~ fraction of context; mapped to
   max(1, round(budget * 20)) traits).
4. Compose system_fragment from comm_prefs + selected traits.
5. Set tone from comm_prefs, proactive_threshold from relationship_stage.

Anti-sycophancy / over-personalization guardrails:
- Traits below confidence floor (default 0.2) are never injected.
- Traits with zero text/entity relevance are never injected.
- Correctness is independent of the user model: the fragment is advisory only,
  never overrides factual grounding from memory recall.
"""
from __future__ import annotations

import re

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import RelationshipStage, Trait, UserModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum confidence for a trait to be eligible for injection.
_CONFIDENCE_FLOOR = 0.2

# Proactive-threshold mapping: NEW→less proactive (higher threshold), TRUSTED→more.
_PROACTIVE_THRESHOLD: dict[RelationshipStage, float] = {
    RelationshipStage.NEW: 0.9,
    RelationshipStage.ACQUAINTED: 0.6,
    RelationshipStage.FAMILIAR: 0.4,
    RelationshipStage.TRUSTED: 0.2,
}


# ---------------------------------------------------------------------------
# AdaptivePersonalizer
# ---------------------------------------------------------------------------


class AdaptivePersonalizer:
    """Budget-aware, turn-relevant trait selector implementing ``Personalizer``.

    Parameters
    ----------
    profile_builder:
        Optional ``UserProfileBuilder``.  Not used in the hot-path ``build`` call
        (the orchestrator already passes a pre-built ``UserModel``), but stored for
        future lazy-build scenarios.
    budget:
        Fraction of the context window allocated to personalisation (default 0.15).
        Mapped to ``max(1, round(budget * 20))`` traits.
    """

    def __init__(
        self,
        *,
        profile_builder: object = None,
        budget: float = 0.15,
    ) -> None:
        self._profile_builder = profile_builder
        self._budget = budget

    @property
    def _max_traits(self) -> int:
        return max(1, round(self._budget * 20))

    async def build(
        self, *, user_model: UserModel, perception: FusedPerception
    ) -> PersonalizedContext:
        """Select relevant traits and compose a ``PersonalizedContext`` for this turn."""
        selected = self._select_traits(user_model, perception)
        fragment = self._compose_fragment(user_model, selected)
        tone = user_model.comm_prefs.tone
        threshold = _PROACTIVE_THRESHOLD.get(
            user_model.relationship_stage,
            _PROACTIVE_THRESHOLD[RelationshipStage.NEW],
        )
        return PersonalizedContext(
            system_fragment=fragment,
            selected_traits=[t.name for t in selected],
            tone=tone,
            proactive_threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Trait selection
    # ------------------------------------------------------------------

    def _select_traits(self, user_model: UserModel, perception: FusedPerception) -> list[Trait]:
        """Return up to ``_max_traits`` eligible traits, scored by relevance."""
        # Build the turn context: lower-case tokens from text + entity names
        query_tokens = _tokenize(perception.text)
        entity_tokens: set[str] = set()
        for entity in perception.entities:
            entity_tokens |= _tokenize(entity.name)
        intent_token = perception.intent.name.lower()
        all_turn_tokens = query_tokens | entity_tokens | {intent_token}

        scored: list[tuple[float, Trait]] = []
        for trait in user_model.traits:
            # Guardrail: skip low-confidence traits
            if trait.confidence < _CONFIDENCE_FLOOR:
                continue

            # Score by token overlap
            score = _relevance_score(trait, all_turn_tokens, query_tokens)
            if score > 0.0:
                scored.append((score, trait))

        # Sort by relevance desc, then confidence desc as tiebreaker
        scored.sort(key=lambda sv: (sv[0], sv[1].confidence), reverse=True)
        return [t for _, t in scored[: self._max_traits]]

    # ------------------------------------------------------------------
    # Fragment composition
    # ------------------------------------------------------------------

    def _compose_fragment(self, user_model: UserModel, selected: list[Trait]) -> str:
        """Compose the system fragment from comm_prefs + selected traits."""
        parts: list[str] = []

        # comm_prefs
        prefs = user_model.comm_prefs
        pref_bits: list[str] = []
        if prefs.length != "balanced":
            pref_bits.append(f"length={prefs.length}")
        if prefs.tone != "neutral":
            pref_bits.append(f"tone={prefs.tone}")
        if prefs.formality != "neutral":
            pref_bits.append(f"formality={prefs.formality}")
        if prefs.code_vs_prose != "balanced":
            pref_bits.append(f"code_vs_prose={prefs.code_vs_prose}")
        if pref_bits:
            parts.append("User prefs: " + ", ".join(pref_bits))

        # selected traits
        for trait in selected:
            parts.append(f"  - {trait.name}: {trait.value}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase alpha-numeric tokens from *text*."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _relevance_score(trait: Trait, all_tokens: set[str], query_tokens: set[str]) -> float:
    """Score a trait's relevance to the current turn.

    Scoring strategy:
    - Tokenise the trait's name and value.
    - Overlap with query tokens → higher weight (direct mention).
    - Overlap with all_tokens (includes entities) → base weight.
    """
    trait_tokens = _tokenize(trait.name) | _tokenize(trait.value)
    if not trait_tokens:
        return 0.0

    query_overlap = len(trait_tokens & query_tokens)
    all_overlap = len(trait_tokens & all_tokens)

    # Weighted score: direct query match counts more
    raw = query_overlap * 2.0 + all_overlap
    if raw == 0.0:
        return 0.0

    # Normalise by trait token count so short traits aren't unduly penalised
    return raw / len(trait_tokens)
