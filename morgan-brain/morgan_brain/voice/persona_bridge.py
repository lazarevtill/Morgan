"""PersonaPlex persona bridge — maps Morgan's learned state to a voice persona.

``build_voice_persona`` is a **pure function**: no I/O, no randomness, fully
deterministic for a given set of inputs.  The same ``user_id`` always produces the
same ``voice_id``; the same ``UserModel`` + inputs always produce the same
``role_prompt`` (after budget truncation).

## Voice-id selection (deterministic + stable per user)

The catalog is ordered warm→varied (NAT* voices first, then VAR*).  We compute::

    index = sum(ord(c) for c in user_model.user_id) % len(catalog)

so the same user always gets the same slot.  When ``comm_prefs.tone`` is
``"warm"`` or ``"empathetic"`` we additionally *restrict* the catalog to the NAT*
prefix before applying the modulo, ensuring warm/empathetic users land on a
natural-sounding voice.  The restriction is applied only when there is at least
one NAT voice in the catalog; otherwise the full catalog is used.

## role_prompt composition (highest- to lowest-priority)

Priority order (highest survives budget truncation last):
1. **Relationship stage line** (always kept, short, ~60 chars)
2. **Communication prefs** (always kept, ~120 chars max)
3. **Top traits** (truncated from the end when budget is tight)
4. **Champion preprompt header** (first line only; dropped first if over budget)

Deterministic ordering within each section: traits sorted descending by
confidence then alphabetically by name.
"""
from __future__ import annotations

_DEFAULT_CATALOG: list[str] = ["NATF0", "NATF1", "NATM0", "NATM1", "VARF0", "VARM0"]

_STAGE_LINES: dict[str, str] = {
    "new": "You are getting to know them; be welcoming.",
    "acquainted": "You have met before; be friendly and attentive.",
    "familiar": "You know them reasonably well; be comfortable and helpful.",
    "trusted": "You know them well; be familiar and direct.",
}

_MAX_TRAITS: int = 5  # include at most this many traits in the prompt

from morgan_brain.interfaces.voice import VoicePersona
from morgan_brain.models.emotion import EmotionState, EmotionType
from morgan_brain.models.user import RelationshipStage, UserModel


def _nat_prefix(catalog: list[str]) -> list[str]:
    """Return only the NAT* entries from *catalog*, or the full list if none exist."""
    nat = [v for v in catalog if v.upper().startswith("NAT")]
    return nat if nat else catalog


def _is_warm_tone(tone: str) -> bool:
    return tone.lower() in {"warm", "empathetic", "friendly"}


def _select_voice_id(user_model: UserModel, catalog: list[str]) -> str:
    """Stable voice selection: deterministic per user_id, warm-tone biased to NAT*."""
    effective_catalog = (
        _nat_prefix(catalog) if _is_warm_tone(user_model.comm_prefs.tone) else catalog
    )
    idx = sum(ord(c) for c in user_model.user_id) % len(effective_catalog)
    return effective_catalog[idx]


def _prefs_instructions(user_model: UserModel) -> str:
    """Translate comm_prefs to natural-language instructions (deterministic order)."""
    prefs = user_model.comm_prefs
    parts: list[str] = []

    # tone
    tone = prefs.tone.lower()
    if tone == "warm" or tone == "empathetic":
        parts.append("Speak warmly and with empathy.")
    elif tone == "friendly":
        parts.append("Be friendly and approachable.")
    elif tone == "formal":
        parts.append("Maintain a formal, professional tone.")
    elif tone == "casual":
        parts.append("Keep the tone casual and relaxed.")
    # neutral → no explicit instruction (saves budget)

    # length
    length = prefs.length.lower()
    if length == "terse":
        parts.append("Keep replies brief.")
    elif length == "thorough":
        parts.append("Provide thorough, detailed replies.")
    # balanced → no explicit instruction

    # formality
    formality = prefs.formality.lower()
    if formality == "formal":
        parts.append("Use formal language.")
    elif formality == "informal":
        parts.append("Use informal, conversational language.")
    # neutral → no explicit instruction

    # code_vs_prose
    cvp = prefs.code_vs_prose.lower()
    if cvp == "code_first":
        parts.append("Prefer code examples over prose when relevant.")
    elif cvp == "prose_first":
        parts.append("Prefer clear prose explanations over code.")
    # balanced → no explicit instruction

    return " ".join(parts)


def _stage_line(stage: RelationshipStage) -> str:
    return _STAGE_LINES.get(stage.value, _STAGE_LINES["new"])


def _trait_lines(user_model: UserModel, max_traits: int = _MAX_TRAITS) -> str:
    """Top *max_traits* traits sorted by confidence desc, then name asc."""
    sorted_traits = sorted(
        user_model.traits,
        key=lambda t: (-t.confidence, t.name),
    )[:max_traits]
    if not sorted_traits:
        return ""
    lines = [f"- {t.name}: {t.value}" for t in sorted_traits]
    return "Known traits:\n" + "\n".join(lines)


def _emotion_note(emotion: EmotionState | None) -> str:
    """Optional single-line hint for non-neutral emotions."""
    if emotion is None or emotion.primary == EmotionType.NEUTRAL:
        return ""
    label = emotion.primary.value
    return f"The user currently seems {label}; respond accordingly."


def build_voice_persona(
    *,
    user_model: UserModel,
    champion_preprompt: str = "",
    emotion: EmotionState | None = None,
    voice_catalog: list[str] | None = None,
    char_budget: int = 1200,
) -> VoicePersona:
    """Build a PersonaPlex ``VoicePersona`` from Morgan's learned user state.

    Parameters
    ----------
    user_model:
        The owner's current ``UserModel`` (comm prefs, traits, relationship stage).
    champion_preprompt:
        The current GEPA champion system prompt.  Only the first non-blank line is
        included in the voice persona (keeps it compact).
    emotion:
        Current ``EmotionState`` from the perception layer.  Non-neutral emotions
        add a brief adjustment note (low budget cost).
    voice_catalog:
        Ordered list of PersonaPlex voice ids to choose from.  Defaults to
        ``["NATF0","NATF1","NATM0","NATM1","VARF0","VARM0"]``.
    char_budget:
        Maximum character count for ``role_prompt``.  Lowest-priority sections are
        truncated first (champion header → traits → prefs/stage survive).

    Returns
    -------
    VoicePersona
        ``role_prompt`` is budget-constrained; ``voice_id`` is deterministic for
        the given ``user_model.user_id`` and catalog.
    """
    catalog = voice_catalog if voice_catalog is not None else _DEFAULT_CATALOG

    # --- voice_id (pure, deterministic) ---
    voice_id = _select_voice_id(user_model, catalog)

    # --- role_prompt sections (priority order, low→high) ---

    # Priority 4 (lowest) — champion preprompt header (first non-blank line)
    champion_header = ""
    if champion_preprompt:
        first_line = next(
            (ln.strip() for ln in champion_preprompt.splitlines() if ln.strip()), ""
        )
        if first_line:
            champion_header = first_line

    # Priority 3 — traits
    traits_block = _trait_lines(user_model)

    # Optional emotion note (appended after prefs, very short)
    emotion_note = _emotion_note(emotion)

    # Priority 2 — comm prefs (always kept if it fits; trimmed only if prefs+stage > budget)
    prefs_block = _prefs_instructions(user_model)

    # Priority 1 (highest) — relationship stage line (always kept)
    stage_line = _stage_line(user_model.relationship_stage)

    # --- assemble with budget enforcement ---
    # Build from highest priority down; track remaining chars.

    def _join(*parts: str) -> str:
        return "\n\n".join(p for p in parts if p).strip()

    # Mandatory core (prefs + stage + emotion note) — always include
    core_parts = [stage_line]
    if prefs_block:
        core_parts.append(prefs_block)
    if emotion_note:
        core_parts.append(emotion_note)
    core = _join(*core_parts)

    # How much budget remains for optional sections?
    remaining = char_budget - len(core)

    # Add traits if they fit
    traits_section = ""
    if traits_block and remaining > len(traits_block) + 2:
        traits_section = traits_block
        remaining -= len(traits_section) + 2

    # Add champion header if it fits
    champion_section = ""
    if champion_header and remaining > len(champion_header) + 2:
        champion_section = champion_header

    role_prompt = _join(champion_section, traits_section, core)

    # Final hard truncation (should only fire if core itself exceeds budget)
    if len(role_prompt) > char_budget:
        role_prompt = role_prompt[:char_budget].rstrip()

    return VoicePersona(role_prompt=role_prompt, voice_id=voice_id)
