"""perception-gpu — voice/vision perception service (DEFERRED, GPU deployment phase).

The ``interfaces.Perception`` Protocol is already defined and the text-only implementation
(``modules.perception.text.TextPerception``) runs inline in brain-api.  When voice is built,
this service implements the same Protocol: Whisper ASR transcribes the audio frame to text,
Wav2Vec2 extracts prosodic/emotion features, and the results are fused into a ``FusedPerception``
object — which is then returned to brain-api.  Because every downstream module (Orchestrator,
MemoryGate, Reasoner) depends only on the ``Perception`` Protocol and the ``FusedPerception``
model, **zero downstream changes are required**: swapping ``TextPerception`` for this service
is a one-line composition change in ``composition.py``.  DEFERRED — no functional code here.

## Voice option — PersonaPlex (full-duplex, Moshi-based) — Phase 5-voice (deferred)

The voice modality uses **PersonaPlex** (NVIDIA, 7B, built on Kyutai Moshi; Mimi codec) in a
**HYBRID design**:

* **Full-duplex mode (PersonaPlex)** — casual and empathic turns are handled end-to-end by
  PersonaPlex: real-time speech-to-speech with interruption / backchannelling support
  (~0.24 s interruption latency).  PersonaPlex does **not** call Morgan's memory or tools
  during a turn; it consumes a static persona per session.

* **Cascaded mode (ASR → brain → TTS)** — knowledge-heavy, tool-requiring, or memory-recall
  turns fall back to: Whisper ASR (or PersonaPlex's understanding stream) → Morgan brain
  (full memory + tools + skills + personalization) → TTS.  A lightweight intent check picks
  the mode; default is full-duplex for casual/empathic, cascaded for command/question/knowledge.

**Persona conditioning** — before each voice session the brain side calls
``voice.persona_bridge.build_voice_persona(user_model=..., champion_preprompt=..., emotion=...)``
(see ``morgan_brain/voice/persona_bridge.py``) to convert Morgan's *learned* state
(``UserModel`` comm-prefs / traits / relationship stage + current GEPA champion preprompt +
``EmotionState``) into a PersonaPlex ``VoicePersona`` (text role prompt + deterministic voice
id).  This is how the assistant's *learned* persona conditions the voice model each session.
The ``VoiceConversation`` seam (``interfaces/voice.py``) keeps the brain audio-agnostic;
the ``FakeVoiceConversation`` (``voice/fake.py``) is used for all tests.

**Write-back (so voice keeps teaching Morgan)** — every voice turn's transcript (user text +
assistant text) is fed into ``SignalRecorder`` and episodic memory so that consolidation and
optimisation learn from voice interactions exactly as from text.  This closes the
"PersonaPlex doesn't learn" gap: learning stays in Morgan, not in the voice model.

**Deferred work** — the actual PersonaPlex/Moshi serving (Moshi server, audio I/O, voice
embeddings), the hybrid mode-router, and transcript write-back wiring are all deferred to
Phase 5-voice.  They are gated behind the ``[voice]`` optional extra (``moshi`` + audio
deps).

Decision record: ``docs/superpowers/specs/2026-06-09-personaplex-voice-decision.md``
"""
