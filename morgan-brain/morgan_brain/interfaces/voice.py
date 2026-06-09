"""VoiceConversation seam — text-level abstraction over a full-duplex speech loop.

The real implementation (PersonaPlex / Moshi-based) lives in ``apps.perception_gpu``
and is deferred to Phase 5-voice.  Everything in ``morgan_brain`` depends only on this
Protocol + the two data models, keeping the brain audio-agnostic and fully testable
without any GPU / audio dependencies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class VoicePersona(BaseModel):
    """PersonaPlex persona: a text role prompt + a pre-packaged voice id."""

    role_prompt: str
    voice_id: str


class VoiceTurn(BaseModel):
    """One exchange in a voice conversation, captured as text."""

    user_text: str
    assistant_text: str


@runtime_checkable
class VoiceConversation(Protocol):
    """Provider-agnostic contract for a full-duplex voice session.

    Callers ``await start(persona)`` once per session, then repeatedly call
    ``await respond(user_text)`` to obtain assistant replies (the concrete
    implementation drives the speech loop and returns transcripts), and finally
    ``await stop()`` to tear down audio I/O.

    The text-level interface is intentional: it lets the brain stay audio-agnostic
    (no numpy / sounddevice / moshi imports) while the GPU service handles all
    codec and audio-stream concerns.
    """

    async def start(self, persona: VoicePersona) -> None:
        """Initialise the voice session with the given persona.

        Must be called before ``respond``.  Safe to call again after ``stop``
        to start a new session.
        """
        ...

    async def respond(self, user_text: str) -> VoiceTurn:
        """Process *user_text* and return the assistant's reply as a VoiceTurn.

        In the real implementation this drives audio encoding/decoding through
        PersonaPlex; here the signature stays text-only so tests never touch
        audio hardware.
        """
        ...

    async def stop(self) -> None:
        """Tear down the voice session and release audio resources."""
        ...
