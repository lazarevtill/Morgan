"""Phase 2C — UserProfileBuilder and CIPHER learn-from-edits.

``UserProfileBuilder`` derives a ``UserModel`` from currently-valid ``TemporalFact``s
and high-value ``InteractionSignal``s.  The build is deterministic: heuristics only,
no LLM required for the base profile construction.

``render_md`` renders a compact human-readable profile with a STABLE section
(traits, comm_prefs, topics, interests) and a DYNAMIC section (emotional_baseline,
relationship_stage).  Hard char cap: 1200.  Lowest-confidence traits are truncated first.

``preference_delta_from_edit`` (CIPHER): given an original reply and the user's edited
version, calls the ``strong`` LLM role to produce a one-line natural-language preference
delta (e.g. "prefers concise, code-first, no hedging").

``apply_edit_delta``: merges the preference delta into a ``UserModel`` deterministically
by keyword matching.  The LLM only produces the delta text; the merge is pure/deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from morgan_brain.models.memory import MemorySource, TemporalFact
from morgan_brain.models.user import (
    CommunicationPrefs,
    RelationshipStage,
    Trait,
    UserModel,
)
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage
from morgan_brain.security.memory_gate import MemoryGate

# ---------------------------------------------------------------------------
# Relationship stage thresholds (fact count proxy)
# ---------------------------------------------------------------------------

_STAGE_THRESHOLDS = {
    RelationshipStage.TRUSTED: 50,
    RelationshipStage.FAMILIAR: 20,
    RelationshipStage.ACQUAINTED: 5,
    RelationshipStage.NEW: 0,
}

# Char budget for render_md
_MD_CHAR_BUDGET = 1200


# ---------------------------------------------------------------------------
# CIPHER schema
# ---------------------------------------------------------------------------


class PreferenceDelta(BaseModel):
    """One-line natural-language preference delta produced by the CIPHER LLM call."""

    delta: str


# ---------------------------------------------------------------------------
# UserProfileBuilder
# ---------------------------------------------------------------------------


class UserProfileBuilder:
    """Derives a ``UserModel`` from persisted facts and interaction signals.

    Parameters
    ----------
    gate:
        The ``MemoryGate`` — all fact reads pass through here.
    signals:
        The ``SignalStore`` for high-value interaction signals.  May be ``None``
        in lightweight test setups that only have facts.
    router:
        The ``RoleRouter`` for LLM dispatch (used only in CIPHER calls).
    capability_registry:
        The ``CapabilityRegistry`` used to build ``CapabilityDescriptor``s for
        ``generate_structured``.
    clock:
        Injected callable returning the current :class:`datetime`.
    role:
        LLM role to use for CIPHER calls (default ``"strong"``).
    """

    def __init__(
        self,
        *,
        gate: MemoryGate,
        signals: object,  # SignalStore | None — typed loosely to avoid circular imports
        router: RoleRouter,
        capability_registry: CapabilityRegistry,
        clock: Callable[[], datetime],
        role: str = "strong",
    ) -> None:
        self._gate = gate
        self._signals = signals
        self._router = router
        self._reg = capability_registry
        self._clock = clock
        self._role = role

    async def build(self, user_id: str) -> UserModel:
        """Derive a ``UserModel`` from currently-valid facts and high-value signals.

        Heuristics (deterministic, no LLM):

        - Facts with predicate ``prefers`` → ``comm_prefs.length`` (terse/thorough)
          or a generic ``Trait``.
        - Facts with predicate ``comm_tone`` → ``comm_prefs.formality`` / ``tone``.
        - Facts with predicate ``comm_length`` → ``comm_prefs.length``.
        - Facts with predicate ``comm_code`` → ``comm_prefs.code_vs_prose``.
        - Facts with predicate ``topic`` or ``interest_in`` → ``topics_of_interest``.
        - Fact count is used as a proxy for relationship depth:
          NEW (<5), ACQUAINTED (<20), FAMILIAR (<50), TRUSTED (≥50).
        - ``confidence`` = min(1.0, count / threshold).
        """
        facts: list[TemporalFact] = await self._gate.current_facts(user_id=user_id)

        # Start with defaults
        comm_prefs = CommunicationPrefs()
        traits: list[Trait] = []
        topics: dict[str, float] = {}

        for fact in facts:
            pred = fact.predicate.lower()
            obj = fact.object.lower()

            # -----------------------------------------------------------------
            # Communication preferences from explicit comm_* predicates
            # -----------------------------------------------------------------
            if pred == "comm_tone":
                if obj in ("formal", "informal", "casual", "professional"):
                    comm_prefs = comm_prefs.model_copy(update={"formality": obj})
                else:
                    comm_prefs = comm_prefs.model_copy(update={"tone": obj})
            elif pred == "comm_length":
                if obj in ("terse", "balanced", "thorough"):
                    comm_prefs = comm_prefs.model_copy(update={"length": obj})
            elif pred == "comm_code":
                if obj in ("code_first", "balanced", "prose_first"):
                    comm_prefs = comm_prefs.model_copy(update={"code_vs_prose": obj})

            # -----------------------------------------------------------------
            # Generic "prefers" predicate — map common values
            # -----------------------------------------------------------------
            elif pred == "prefers":
                if obj in ("terse", "concise", "brief", "short"):
                    comm_prefs = comm_prefs.model_copy(update={"length": "terse"})
                elif obj in ("thorough", "detailed", "verbose", "long"):
                    comm_prefs = comm_prefs.model_copy(update={"length": "thorough"})
                elif obj == "formal":
                    comm_prefs = comm_prefs.model_copy(update={"formality": "formal"})
                elif obj in ("informal", "casual"):
                    comm_prefs = comm_prefs.model_copy(update={"formality": "informal"})
                elif obj in ("code", "code_first"):
                    comm_prefs = comm_prefs.model_copy(update={"code_vs_prose": "code_first"})
                elif obj in ("prose", "prose_first"):
                    comm_prefs = comm_prefs.model_copy(update={"code_vs_prose": "prose_first"})
                else:
                    # Generic preference as a trait
                    traits.append(
                        Trait(
                            name=f"prefers_{fact.object.replace(' ', '_')}",
                            value=fact.object,
                            confidence=fact.confidence,
                        )
                    )

            # -----------------------------------------------------------------
            # Topics of interest
            # -----------------------------------------------------------------
            elif pred in ("topic", "interest_in", "interests_in", "interested_in"):
                topics[fact.object] = fact.confidence

            # -----------------------------------------------------------------
            # Any other predicate becomes a generic trait
            # -----------------------------------------------------------------
            else:
                traits.append(
                    Trait(
                        name=f"{fact.predicate}_{fact.object.replace(' ', '_')}",
                        value=fact.object,
                        confidence=fact.confidence,
                    )
                )

        # -----------------------------------------------------------------
        # Relationship stage — fact count proxy
        # -----------------------------------------------------------------
        count = len(facts)
        stage = _stage_from_count(count)
        threshold = _STAGE_THRESHOLDS.get(stage, 5)
        overall_confidence = min(1.0, count / max(1, threshold))

        return UserModel(
            user_id=user_id,
            traits=traits,
            comm_prefs=comm_prefs,
            topics_of_interest=topics,
            relationship_stage=stage,
            confidence=overall_confidence,
        )


def _stage_from_count(count: int) -> RelationshipStage:
    """Map fact count to relationship stage using defined thresholds."""
    if count >= _STAGE_THRESHOLDS[RelationshipStage.TRUSTED]:
        return RelationshipStage.TRUSTED
    if count >= _STAGE_THRESHOLDS[RelationshipStage.FAMILIAR]:
        return RelationshipStage.FAMILIAR
    if count >= _STAGE_THRESHOLDS[RelationshipStage.ACQUAINTED]:
        return RelationshipStage.ACQUAINTED
    return RelationshipStage.NEW


# ---------------------------------------------------------------------------
# render_md — compact human-readable profile
# ---------------------------------------------------------------------------


def render_md(user_model: UserModel) -> str:
    """Render a compact profile markdown string with STABLE + DYNAMIC sections.

    Hard char cap: 1200.  Truncates lowest-confidence traits first.
    Pure/deterministic — no LLM calls.
    """
    lines: list[str] = []

    # --- STABLE section ---
    lines.append("## STABLE")

    # comm_prefs
    prefs = user_model.comm_prefs
    pref_parts = []
    if prefs.length != "balanced":
        pref_parts.append(f"length={prefs.length}")
    if prefs.tone != "neutral":
        pref_parts.append(f"tone={prefs.tone}")
    if prefs.formality != "neutral":
        pref_parts.append(f"formality={prefs.formality}")
    if prefs.code_vs_prose != "balanced":
        pref_parts.append(f"code_vs_prose={prefs.code_vs_prose}")
    if pref_parts:
        lines.append(f"prefs: {', '.join(pref_parts)}")

    # topics
    if user_model.topics_of_interest:
        top_topics = sorted(user_model.topics_of_interest.items(), key=lambda kv: -kv[1])[:5]
        lines.append("topics: " + ", ".join(t for t, _ in top_topics))

    # traits — sorted descending by confidence; we'll truncate when we hit budget
    sorted_traits = sorted(user_model.traits, key=lambda t: -t.confidence)

    # --- DYNAMIC section ---
    dynamic_lines: list[str] = [
        "## DYNAMIC",
        f"stage: {user_model.relationship_stage.value}",
        f"confidence: {user_model.confidence:.2f}",
    ]
    baseline = user_model.emotional_baseline
    dynamic_lines.append(
        f"emotional_baseline: valence={baseline.valence:.2f} arousal={baseline.arousal:.2f}"
    )

    # Assemble without traits first, then add traits within budget
    base = "\n".join(lines + dynamic_lines)
    budget = _MD_CHAR_BUDGET - len(base) - 20  # 20-char margin

    trait_lines: list[str] = []
    for trait in sorted_traits:
        line = f"  - {trait.name}: {trait.value} (conf={trait.confidence:.2f})"
        if budget - len(line) - 1 >= 0:
            trait_lines.append(line)
            budget -= len(line) + 1
        else:
            break

    if trait_lines:
        lines.append("traits:")
        lines.extend(trait_lines)

    full = "\n".join(lines + dynamic_lines)
    # Final safety truncation
    if len(full) > _MD_CHAR_BUDGET:
        full = full[: _MD_CHAR_BUDGET - 3] + "..."
    return full


# ---------------------------------------------------------------------------
# CIPHER — preference_delta_from_edit
# ---------------------------------------------------------------------------


async def preference_delta_from_edit(
    builder: UserProfileBuilder,
    user_id: str,
    original: str,
    edited: str,
) -> str:
    """Distil a one-line preference delta from an edit pair via the LLM.

    Calls the ``strong`` role with a prompt asking for a natural-language delta
    explaining how *edited* differs from *original* in terms of user preferences.
    Returns the delta string (e.g. "prefers concise, code-first, no hedging").

    The LLM call is the only non-deterministic step; ``apply_edit_delta`` is pure.
    """
    try:
        client, model = builder._router.chat_for(builder._role, needs_json_schema=True)
        provider = _provider_for_model(builder._router, model)
    except LookupError:
        client, model = builder._router.chat_for(builder._role)
        provider = _provider_for_model(builder._router, model)

    descriptor = builder._reg.get(provider, model)

    system_msg = ChatMessage(
        role="system",
        content=(
            "You are a preference-extraction engine. "
            "Given an original AI reply and the user's edited version, "
            "produce a ONE-LINE natural-language description of what the user prefers differently. "
            "Focus on style/tone/length/format changes. "
            "Example: 'prefers concise, code-first, no hedging'."
        ),
    )
    user_msg = ChatMessage(
        role="user",
        content=(
            f"ORIGINAL:\n{original}\n\nEDITED:\n{edited}\n\n"
            "Describe the preference delta in one line."
        ),
    )

    result = await generate_structured(
        client,
        [system_msg, user_msg],
        model=model,
        schema=PreferenceDelta,
        descriptor=descriptor,
    )
    return result.delta


# ---------------------------------------------------------------------------
# apply_edit_delta — deterministic keyword-based merge
# ---------------------------------------------------------------------------

# Keyword → field update mapping
_LENGTH_TERSE_KEYWORDS = frozenset({"concise", "terse", "brief", "short", "shorter", "compact"})
_LENGTH_THOROUGH_KEYWORDS = frozenset(
    {"thorough", "detailed", "verbose", "long", "elaborate", "comprehensive"}
)
_CODE_FIRST_KEYWORDS = frozenset({"code", "code-first", "code_first"})
_PROSE_FIRST_KEYWORDS = frozenset({"prose", "prose-first", "prose_first", "narrative"})
_FORMAL_KEYWORDS = frozenset({"formal"})
_INFORMAL_KEYWORDS = frozenset({"informal", "casual"})
_WARM_KEYWORDS = frozenset({"warm", "friendly"})
_NO_HEDGING_KEYWORDS = frozenset({"hedging", "no hedging", "no-hedging"})
_DIRECT_KEYWORDS = frozenset({"direct", "blunt", "assertive"})


def apply_edit_delta(user_model: UserModel, delta: str) -> UserModel:
    """Merge a natural-language preference delta into a ``UserModel`` deterministically.

    The merge is keyword-based — no LLM involved.  Returns a new ``UserModel``
    instance; the original is never mutated.
    """
    delta_lower = delta.lower()
    tokens = set(delta_lower.replace("-", " ").replace("_", " ").split())

    prefs = user_model.comm_prefs
    traits = list(user_model.traits)

    # --- length ---
    if tokens & _LENGTH_TERSE_KEYWORDS or "no hedging" in delta_lower:
        prefs = prefs.model_copy(update={"length": "terse"})
    elif tokens & _LENGTH_THOROUGH_KEYWORDS:
        prefs = prefs.model_copy(update={"length": "thorough"})

    # --- code vs prose ---
    if tokens & _CODE_FIRST_KEYWORDS:
        prefs = prefs.model_copy(update={"code_vs_prose": "code_first"})
    elif tokens & _PROSE_FIRST_KEYWORDS:
        prefs = prefs.model_copy(update={"code_vs_prose": "prose_first"})

    # --- formality ---
    if tokens & _FORMAL_KEYWORDS:
        prefs = prefs.model_copy(update={"formality": "formal"})
    elif tokens & _INFORMAL_KEYWORDS:
        prefs = prefs.model_copy(update={"formality": "informal"})

    # --- tone ---
    if tokens & _WARM_KEYWORDS:
        prefs = prefs.model_copy(update={"tone": "warm"})

    # --- no_hedging trait ---
    if "no hedging" in delta_lower or "no-hedging" in delta_lower:
        existing_names = {t.name for t in traits}
        if "no_hedging" not in existing_names:
            traits.append(Trait(name="no_hedging", value="no hedging", confidence=0.8))

    # --- direct trait ---
    if tokens & _DIRECT_KEYWORDS:
        existing_names = {t.name for t in traits}
        if "direct" not in existing_names:
            traits.append(Trait(name="direct", value="direct communication", confidence=0.8))

    return user_model.model_copy(update={"comm_prefs": prefs, "traits": traits})


def preference_facts_from_delta(
    user_id: str, delta: str, now: datetime, *, project: str
) -> list[TemporalFact]:
    """Map a CIPHER preference delta to agent-inferred ``comm_*`` facts.

    Reuses ``apply_edit_delta``'s keyword detection but emits durable ``TemporalFact``s
    (``source=agent_inferred``) instead of mutating a transient ``UserModel`` — so the
    next ``UserProfileBuilder.build()`` (which reads currently-valid facts) reflects the
    learned preference. This is the persistence half of "edit → future turns change".
    Predicates are chosen to match ``build()``'s mapping: ``comm_length`` →
    ``length``; ``comm_code`` → ``code_vs_prose``; ``comm_tone`` →
    ``formality``/``tone``.
    """
    delta_lower = delta.lower()
    # Word-boundary tokenisation (robust to commas/periods the LLM emits, e.g.
    # "prefers concise, code-first" → {prefers, concise, code, first}).
    tokens = set(re.findall(r"[a-z]+", delta_lower))

    def _fact(predicate: str, obj: str) -> TemporalFact:
        return TemporalFact(
            user_id=user_id,
            project=project,
            subject="user",
            predicate=predicate,
            object=obj,
            source=MemorySource.AGENT_INFERRED,
            confidence=0.7,
            valid_from=now,
            last_confirmed=now,
            created_at=now,
        )

    facts: list[TemporalFact] = []
    if tokens & _LENGTH_TERSE_KEYWORDS or "no hedging" in delta_lower:
        facts.append(_fact("comm_length", "terse"))
    elif tokens & _LENGTH_THOROUGH_KEYWORDS:
        facts.append(_fact("comm_length", "thorough"))

    if tokens & _CODE_FIRST_KEYWORDS:
        facts.append(_fact("comm_code", "code_first"))
    elif tokens & _PROSE_FIRST_KEYWORDS:
        facts.append(_fact("comm_code", "prose_first"))

    if tokens & _FORMAL_KEYWORDS:
        facts.append(_fact("comm_tone", "formal"))
    elif tokens & _INFORMAL_KEYWORDS:
        facts.append(_fact("comm_tone", "casual"))

    if tokens & _WARM_KEYWORDS:
        facts.append(_fact("comm_tone", "warm"))

    return facts


def _provider_for_model(router: RoleRouter, model: str) -> str:
    """Best-effort: find the provider string for a model from registered bindings."""
    for binding_list in router._bindings.values():
        for binding in binding_list:
            if binding.model == model:
                return binding.provider
    return "unknown"
