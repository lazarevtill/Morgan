"""Tests for FakeVoiceConversation."""
from __future__ import annotations

import pytest

from morgan_brain.interfaces.voice import VoiceConversation, VoicePersona, VoiceTurn
from morgan_brain.voice.fake import FakeVoiceConversation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona(role: str = "Be helpful.", voice: str = "NATM0") -> VoicePersona:
    return VoicePersona(role_prompt=role, voice_id=voice)


# ---------------------------------------------------------------------------
# isinstance check
# ---------------------------------------------------------------------------


def test_fake_is_voice_conversation() -> None:
    """FakeVoiceConversation satisfies the VoiceConversation Protocol at runtime."""
    assert isinstance(FakeVoiceConversation(), VoiceConversation)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_records_persona() -> None:
    fake = FakeVoiceConversation()
    p = _persona()
    await fake.start(p)
    assert fake.persona is p


@pytest.mark.asyncio
async def test_start_resets_turns_and_stopped() -> None:
    """start() after a previous session clears turns and un-sets stopped."""
    fake = FakeVoiceConversation()
    await fake.start(_persona())
    await fake.respond("hi")
    await fake.stop()
    assert fake.stopped
    assert len(fake.turns) == 1

    # Start a new session
    p2 = _persona(role="Second session", voice="NATF0")
    await fake.start(p2)
    assert fake.persona is p2
    assert fake.turns == []
    assert not fake.stopped


# ---------------------------------------------------------------------------
# respond
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_respond_scripted() -> None:
    fake = FakeVoiceConversation(scripted={"hello": "hi there"})
    await fake.start(_persona())
    turn = await fake.respond("hello")
    assert isinstance(turn, VoiceTurn)
    assert turn.user_text == "hello"
    assert turn.assistant_text == "hi there"


@pytest.mark.asyncio
async def test_respond_default_reply() -> None:
    fake = FakeVoiceConversation(default_reply="fallback")
    await fake.start(_persona())
    turn = await fake.respond("unscripted question")
    assert turn.assistant_text == "fallback"
    assert turn.user_text == "unscripted question"


@pytest.mark.asyncio
async def test_respond_default_reply_is_ok() -> None:
    """Default *default_reply* is 'ok'."""
    fake = FakeVoiceConversation()
    await fake.start(_persona())
    turn = await fake.respond("anything")
    assert turn.assistant_text == "ok"


@pytest.mark.asyncio
async def test_respond_mixed_scripted_and_default() -> None:
    fake = FakeVoiceConversation(scripted={"what is 2+2": "4"})
    await fake.start(_persona())
    t1 = await fake.respond("what is 2+2")
    t2 = await fake.respond("tell me a joke")
    assert t1.assistant_text == "4"
    assert t2.assistant_text == "ok"


# ---------------------------------------------------------------------------
# turn recording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turns_recorded_in_order() -> None:
    fake = FakeVoiceConversation(scripted={"a": "A", "b": "B"})
    await fake.start(_persona())
    await fake.respond("a")
    await fake.respond("b")
    await fake.respond("c")
    assert len(fake.turns) == 3
    assert [t.user_text for t in fake.turns] == ["a", "b", "c"]
    assert [t.assistant_text for t in fake.turns] == ["A", "B", "ok"]


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_sets_flag() -> None:
    fake = FakeVoiceConversation()
    await fake.start(_persona())
    assert not fake.stopped
    await fake.stop()
    assert fake.stopped


@pytest.mark.asyncio
async def test_stop_does_not_clear_turns() -> None:
    """Turns remain accessible after stop (caller may need them)."""
    fake = FakeVoiceConversation()
    await fake.start(_persona())
    await fake.respond("hello")
    await fake.stop()
    assert len(fake.turns) == 1


# ---------------------------------------------------------------------------
# initial state (before start)
# ---------------------------------------------------------------------------


def test_initial_state() -> None:
    fake = FakeVoiceConversation()
    assert fake.persona is None
    assert fake.turns == []
    assert not fake.stopped
