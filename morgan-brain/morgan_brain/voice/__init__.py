"""Voice sub-package — pure/testable brain-side voice utilities.

Contains:
- ``persona_bridge``: maps a ``UserModel`` + champion preprompt + current
  ``EmotionState`` into a ``VoicePersona`` (PersonaPlex text role prompt + voice id).

GPU audio I/O and the actual PersonaPlex/Moshi serving live in
``apps.perception_gpu`` (Phase 5-voice, deferred).
"""
from morgan_brain.voice.persona_bridge import build_voice_persona

__all__ = ["build_voice_persona"]
