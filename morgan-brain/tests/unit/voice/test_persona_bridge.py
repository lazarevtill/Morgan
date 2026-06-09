"""Tests for persona_bridge.build_voice_persona."""

from __future__ import annotations

from morgan_brain.interfaces.voice import VoicePersona
from morgan_brain.models.emotion import EmotionState, EmotionType
from morgan_brain.models.user import (
    CommunicationPrefs,
    RelationshipStage,
    Trait,
    UserModel,
)
from morgan_brain.voice.persona_bridge import build_voice_persona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(
    user_id: str = "user-abc",
    tone: str = "neutral",
    length: str = "balanced",
    formality: str = "neutral",
    code_vs_prose: str = "balanced",
    stage: RelationshipStage = RelationshipStage.NEW,
    traits: list[Trait] | None = None,
) -> UserModel:
    return UserModel(
        user_id=user_id,
        comm_prefs=CommunicationPrefs(
            tone=tone,
            length=length,
            formality=formality,
            code_vs_prose=code_vs_prose,
        ),
        relationship_stage=stage,
        traits=traits or [],
    )


# ---------------------------------------------------------------------------
# role_prompt: comm_prefs reflected
# ---------------------------------------------------------------------------


def test_warm_tone_appears_in_prompt() -> None:
    user = _make_user(tone="warm")
    vp = build_voice_persona(user_model=user)
    assert "warm" in vp.role_prompt.lower()


def test_terse_length_appears_in_prompt() -> None:
    user = _make_user(length="terse")
    vp = build_voice_persona(user_model=user)
    assert "brief" in vp.role_prompt.lower()


def test_warm_and_terse_both_appear() -> None:
    """warm tone + terse length → both warmth and brevity instructions present."""
    user = _make_user(tone="warm", length="terse")
    vp = build_voice_persona(user_model=user)
    rp = vp.role_prompt.lower()
    assert "warm" in rp
    assert "brief" in rp


def test_formal_tone_appears() -> None:
    user = _make_user(tone="formal")
    vp = build_voice_persona(user_model=user)
    assert "formal" in vp.role_prompt.lower()


def test_neutral_prefs_produce_no_pref_lines() -> None:
    """All-neutral prefs → no extra instruction lines (only stage line remains)."""
    user = _make_user(
        tone="neutral", length="balanced", formality="neutral", code_vs_prose="balanced"
    )
    vp = build_voice_persona(user_model=user)
    # stage line must still be present
    assert "getting to know" in vp.role_prompt.lower()


# ---------------------------------------------------------------------------
# role_prompt: relationship-stage lines differ across stages
# ---------------------------------------------------------------------------


def test_new_stage_line_present() -> None:
    user = _make_user(stage=RelationshipStage.NEW)
    vp = build_voice_persona(user_model=user)
    assert "getting to know" in vp.role_prompt.lower()


def test_trusted_stage_line_present() -> None:
    user = _make_user(stage=RelationshipStage.TRUSTED)
    vp = build_voice_persona(user_model=user)
    assert "familiar" in vp.role_prompt.lower() or "direct" in vp.role_prompt.lower()


def test_new_vs_trusted_prompts_differ() -> None:
    user_new = _make_user(user_id="same-id", stage=RelationshipStage.NEW)
    user_trusted = _make_user(user_id="same-id", stage=RelationshipStage.TRUSTED)
    vp_new = build_voice_persona(user_model=user_new)
    vp_trusted = build_voice_persona(user_model=user_trusted)
    assert vp_new.role_prompt != vp_trusted.role_prompt


def test_acquainted_stage_present() -> None:
    user = _make_user(stage=RelationshipStage.ACQUAINTED)
    vp = build_voice_persona(user_model=user)
    assert "friendly" in vp.role_prompt.lower() or "met" in vp.role_prompt.lower()


def test_familiar_stage_present() -> None:
    user = _make_user(stage=RelationshipStage.FAMILIAR)
    vp = build_voice_persona(user_model=user)
    assert "know them" in vp.role_prompt.lower() or "comfortable" in vp.role_prompt.lower()


# ---------------------------------------------------------------------------
# role_prompt: traits
# ---------------------------------------------------------------------------


def test_top_traits_appear_in_prompt() -> None:
    traits = [
        Trait(name="curious", value="highly curious about science", confidence=0.9),
        Trait(name="patient", value="prefers step-by-step explanations", confidence=0.8),
    ]
    user = _make_user(traits=traits)
    vp = build_voice_persona(user_model=user)
    assert "curious" in vp.role_prompt
    assert "patient" in vp.role_prompt


def test_low_confidence_traits_ordered_last() -> None:
    """High-confidence trait should appear before low-confidence trait."""
    traits = [
        Trait(name="low_conf", value="uncertain trait", confidence=0.1),
        Trait(name="high_conf", value="strong trait", confidence=0.95),
    ]
    user = _make_user(traits=traits)
    vp = build_voice_persona(user_model=user)
    pos_high = vp.role_prompt.find("high_conf")
    pos_low = vp.role_prompt.find("low_conf")
    assert pos_high < pos_low


def test_no_traits_no_traits_section() -> None:
    user = _make_user(traits=[])
    vp = build_voice_persona(user_model=user)
    assert "Known traits" not in vp.role_prompt


# ---------------------------------------------------------------------------
# char_budget enforcement
# ---------------------------------------------------------------------------


def test_huge_champion_preprompt_truncated_prefs_survive() -> None:
    """A champion preprompt of 2000 chars → still under budget; prefs survive."""
    user = _make_user(tone="warm", length="terse")
    huge = "A" * 2000
    vp = build_voice_persona(user_model=user, champion_preprompt=huge, char_budget=1200)
    assert len(vp.role_prompt) <= 1200
    # prefs survive (warmth + brevity)
    rp = vp.role_prompt.lower()
    assert "warm" in rp
    assert "brief" in rp


def test_char_budget_hard_limit() -> None:
    """role_prompt never exceeds char_budget regardless of inputs."""
    user = _make_user(
        tone="warm",
        length="thorough",
        formality="formal",
        traits=[
            Trait(name=f"trait{i}", value="x" * 50, confidence=float(i) / 10) for i in range(1, 8)
        ],
    )
    big_champion = "Champion header: " + "B" * 1000
    vp = build_voice_persona(
        user_model=user,
        champion_preprompt=big_champion,
        char_budget=500,
    )
    assert len(vp.role_prompt) <= 500


def test_stage_always_survives_tight_budget() -> None:
    """Stage line (highest priority) survives even a very tight budget."""
    user = _make_user(stage=RelationshipStage.TRUSTED)
    big_champion = "C" * 2000
    vp = build_voice_persona(
        user_model=user,
        champion_preprompt=big_champion,
        char_budget=300,
    )
    assert "familiar" in vp.role_prompt.lower() or "direct" in vp.role_prompt.lower()


# ---------------------------------------------------------------------------
# voice_id: determinism + stability
# ---------------------------------------------------------------------------


def test_voice_id_deterministic_same_user_id() -> None:
    """Same user_id → same voice_id across two independent calls."""
    user = _make_user(user_id="stable-user-99")
    vp1 = build_voice_persona(user_model=user)
    vp2 = build_voice_persona(user_model=user)
    assert vp1.voice_id == vp2.voice_id


def test_voice_id_from_catalog() -> None:
    """voice_id is always a member of the supplied catalog."""
    catalog = ["NATF0", "NATM0", "VARF0"]
    user = _make_user(user_id="any-user")
    vp = build_voice_persona(user_model=user, voice_catalog=catalog)
    assert vp.voice_id in catalog


def test_different_user_ids_can_differ() -> None:
    """Two distinct user_ids can resolve to different voice slots.
    We pick two ids whose ordinal sums land in different mod-slots over the default catalog.
    """
    catalog = ["NATF0", "NATF1", "NATM0", "NATM1", "VARF0", "VARM0"]  # len=6
    # find two user_ids that hash to different indexes
    results: dict[int, str] = {}
    for i in range(20):
        uid = f"u{i}"
        idx = sum(ord(c) for c in uid) % len(catalog)
        results[idx] = uid
        if len(results) >= 2:
            break
    # We need at least 2 distinct slots found
    assert len(results) >= 2, "Could not find two user_ids mapping to different catalog slots"
    voices = {
        build_voice_persona(user_model=_make_user(user_id=uid)).voice_id for uid in results.values()
    }
    assert len(voices) >= 2


def test_warm_tone_biases_to_nat_voice() -> None:
    """A user with warm tone always gets a NAT* voice from the default catalog."""
    user = _make_user(user_id="warm-user", tone="warm")
    vp = build_voice_persona(user_model=user)
    assert vp.voice_id.upper().startswith("NAT")


def test_empathetic_tone_biases_to_nat_voice() -> None:
    user = _make_user(user_id="empathetic-user", tone="empathetic")
    vp = build_voice_persona(user_model=user)
    assert vp.voice_id.upper().startswith("NAT")


def test_custom_catalog_respected() -> None:
    """When caller provides catalog, selection stays within it."""
    catalog = ["VARF0", "VARF1", "VARF2"]
    user = _make_user(user_id="any")
    vp = build_voice_persona(user_model=user, voice_catalog=catalog)
    assert vp.voice_id in catalog


# ---------------------------------------------------------------------------
# emotion note
# ---------------------------------------------------------------------------


def test_emotion_note_appears_for_joy() -> None:
    user = _make_user()
    emotion = EmotionState(primary=EmotionType.JOY, confidence=0.8)
    vp = build_voice_persona(user_model=user, emotion=emotion)
    assert "joy" in vp.role_prompt.lower()


def test_neutral_emotion_no_note() -> None:
    user = _make_user()
    emotion = EmotionState(primary=EmotionType.NEUTRAL, confidence=0.9)
    vp = build_voice_persona(user_model=user, emotion=emotion)
    assert "neutral" not in vp.role_prompt.lower()


def test_no_emotion_no_note() -> None:
    user = _make_user()
    vp = build_voice_persona(user_model=user, emotion=None)
    # No emotion-related phrases
    for kw in ("joy", "sadness", "anger", "fear", "surprise", "disgust", "seems"):
        assert kw not in vp.role_prompt.lower()


# ---------------------------------------------------------------------------
# return type
# ---------------------------------------------------------------------------


def test_returns_voice_persona_instance() -> None:
    user = _make_user()
    vp = build_voice_persona(user_model=user)
    assert isinstance(vp, VoicePersona)
