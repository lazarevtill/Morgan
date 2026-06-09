"""FakeVoiceConversation — in-memory VoiceConversation for tests and integration.

Usage example::

    fake = FakeVoiceConversation(scripted={"hello": "hi there"})
    await fake.start(persona)
    turn = await fake.respond("hello")
    assert turn.assistant_text == "hi there"
    assert fake.persona == persona
    await fake.stop()
    assert fake.stopped

``FakeVoiceConversation`` satisfies the ``VoiceConversation`` Protocol at
runtime (``isinstance(fake, VoiceConversation)`` is ``True``).
"""

from __future__ import annotations

from morgan_brain.interfaces.voice import VoiceConversation, VoicePersona, VoiceTurn


class FakeVoiceConversation:
    """Scriptable, state-recording stub for ``VoiceConversation``.

    Parameters
    ----------
    scripted:
        Mapping of ``user_text → assistant_text``.  If the incoming text is in the
        map the scripted reply is returned; otherwise *default_reply* is used.
    default_reply:
        Fallback reply for any ``user_text`` not in *scripted*.
    """

    def __init__(
        self,
        scripted: dict[str, str] | None = None,
        default_reply: str = "ok",
    ) -> None:
        self._scripted: dict[str, str] = scripted or {}
        self._default_reply = default_reply
        self.persona: VoicePersona | None = None
        self.turns: list[VoiceTurn] = []
        self.stopped: bool = False

    async def start(self, persona: VoicePersona) -> None:
        """Record *persona* and reset state for a new session."""
        self.persona = persona
        self.turns = []
        self.stopped = False

    async def respond(self, user_text: str) -> VoiceTurn:
        """Return a scripted or default reply and record the turn."""
        assistant_text = self._scripted.get(user_text, self._default_reply)
        turn = VoiceTurn(user_text=user_text, assistant_text=assistant_text)
        self.turns.append(turn)
        return turn

    async def stop(self) -> None:
        """Set ``stopped = True``."""
        self.stopped = True


# Runtime Protocol check (defence-in-depth: catches API drift at import time).
assert isinstance(FakeVoiceConversation(), VoiceConversation), (
    "FakeVoiceConversation no longer satisfies the VoiceConversation Protocol"
)
