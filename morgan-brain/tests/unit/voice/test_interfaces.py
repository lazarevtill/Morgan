"""Tests for the VoiceConversation seam (interfaces/voice.py).

A class that provides the three required coroutine methods satisfies the
``VoiceConversation`` Protocol at runtime — no audio / GPU deps needed.
"""
from __future__ import annotations

import pytest

from morgan_brain.interfaces.voice import VoiceConversation, VoicePersona, VoiceTurn


# ---------------------------------------------------------------------------
# Minimal conforming implementation (defined in test module — not a fake)
# ---------------------------------------------------------------------------


class _MinimalVoice:
    """Bare-minimum class satisfying the VoiceConversation Protocol."""

    async def start(self, persona: VoicePersona) -> None:
        return None

    async def respond(self, user_text: str) -> VoiceTurn:
        return VoiceTurn(user_text=user_text, assistant_text="reply")

    async def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_minimal_class_is_voice_conversation() -> None:
    """A class with the three async methods passes isinstance against the Protocol."""
    obj = _MinimalVoice()
    assert isinstance(obj, VoiceConversation)


def test_missing_method_fails_isinstance() -> None:
    """A class missing any required method does NOT satisfy the Protocol."""

    class _NoStop:
        async def start(self, persona: VoicePersona) -> None: ...
        async def respond(self, user_text: str) -> VoiceTurn:
            return VoiceTurn(user_text=user_text, assistant_text="")

    assert not isinstance(_NoStop(), VoiceConversation)


def test_voice_persona_fields() -> None:
    """VoicePersona stores role_prompt and voice_id."""
    vp = VoicePersona(role_prompt="Be concise.", voice_id="NATF0")
    assert vp.role_prompt == "Be concise."
    assert vp.voice_id == "NATF0"


def test_voice_turn_fields() -> None:
    """VoiceTurn stores user_text and assistant_text."""
    turn = VoiceTurn(user_text="hello", assistant_text="hi there")
    assert turn.user_text == "hello"
    assert turn.assistant_text == "hi there"


@pytest.mark.asyncio
async def test_minimal_voice_start_returns_none() -> None:
    voice = _MinimalVoice()
    result = await voice.start(VoicePersona(role_prompt="p", voice_id="NATM0"))
    assert result is None


@pytest.mark.asyncio
async def test_minimal_voice_respond_returns_turn() -> None:
    voice = _MinimalVoice()
    turn = await voice.respond("what time is it?")
    assert isinstance(turn, VoiceTurn)
    assert turn.user_text == "what time is it?"


@pytest.mark.asyncio
async def test_minimal_voice_stop_returns_none() -> None:
    voice = _MinimalVoice()
    result = await voice.stop()
    assert result is None
