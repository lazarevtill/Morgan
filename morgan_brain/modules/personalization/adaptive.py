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
from morgan_brain.models.memory import DEFAULT_PROJECT
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import RelationshipStage, Trait, UserModel
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.modules.personalization.persona_graph import (
    PersonaGraph,
    PersonaKind,
)
from morgan_brain.modules.personalization.persona_graph import (
    PersonaNode as PersonaNodeLike,
)

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
    persona_graph:
        Optional ``PersonaGraph`` (VoiceMem's right brain). When wired, the turn also
        activates persona nodes -- intrinsic dispositions matched from the turn's words,
        and attitudes anchored to the entities the turn made relevant. Read-only: this
        class never records an observation, which is the cold path's job.
    semantic_index:
        Optional ``SemanticIndex`` (the left brain). When wired, the anchor set is the
        turn's entities expanded one hop through it, which is the joint retrieval of
        eq. (5) -- the right brain expands over the entity set the left brain activated,
        not only over the literal entities in the text.
    """

    def __init__(
        self,
        *,
        profile_builder: object = None,
        budget: float = 0.15,
        persona_graph: PersonaGraph | None = None,
        semantic_index: SemanticIndex | None = None,
    ) -> None:
        self._profile_builder = profile_builder
        self._budget = budget
        self._persona = persona_graph
        self._semantic = semantic_index

    @property
    def _max_traits(self) -> int:
        return max(1, round(self._budget * 20))

    async def build(
        self,
        *,
        user_model: UserModel,
        perception: FusedPerception,
        project: str = DEFAULT_PROJECT,
    ) -> PersonalizedContext:
        """Select relevant traits and compose a ``PersonalizedContext`` for this turn."""
        selected = self._select_traits(user_model, perception)
        persona = self._activate_persona(user_model, perception, project)
        fragment = self._compose_fragment(user_model, selected, persona)
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
    # Persona activation (right brain)
    # ------------------------------------------------------------------

    def _activate_persona(
        self, user_model: UserModel, perception: FusedPerception, project: str
    ) -> list[PersonaNodeLike]:
        """Activate persona nodes for this turn. Returns ``[]`` when no graph is wired."""
        if self._persona is None:
            return []
        anchors = {e.name for e in perception.entities if e.name}
        if self._semantic is not None and anchors:
            # Joint retrieval: expand the turn's entities through the left brain first, so
            # an attitude toward something the turn implies but does not name still
            # surfaces. Without the index the anchor set is the literal entities only.
            pool = self._semantic.neighbours(
                sorted(anchors), user_id=user_model.user_id, project=project
            )
            anchors |= pool
        return list(
            self._persona.activate(
                user_id=user_model.user_id,
                project=project,
                terms=sorted(_tokenize(perception.text)),
                entities=anchors,
            )
        )

    # ------------------------------------------------------------------
    # Fragment composition
    # ------------------------------------------------------------------

    def _compose_fragment(
        self,
        user_model: UserModel,
        selected: list[Trait],
        persona: list[PersonaNodeLike] | None = None,
    ) -> str:
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

        # Persona nodes. An attitude is always rendered with its anchor: "impatient" on
        # its own reads as a claim about the person, which is the collapse the persona
        # graph exists to prevent. Dispositions promoted to intrinsic have earned the
        # unanchored form and are rendered without one.
        for node in persona or []:
            if node.kind is PersonaKind.CROSS_ENTITY and node.entity:
                parts.append(f"  - about {node.entity}: {node.description}")
            else:
                parts.append(f"  - {node.description}")

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
