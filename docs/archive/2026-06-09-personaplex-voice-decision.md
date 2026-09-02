# Decision Record — PersonaPlex as the Phase 5 Voice Layer

**Date:** 2026-06-09 · **Status:** Accepted · **Trigger:** user-provided links to NVIDIA PersonaPlex
(research page, `nvidia/personaplex-7b-v1`, `NVIDIA/personaplex`).

## What PersonaPlex is
A **full-duplex speech↔speech** conversational model (7B, built on Kyutai **Moshi**; Mimi codec).
Natural real-time voice with interruptions/backchanneling (~0.24s interruption latency, 0.95
turn-taking success on FullDuplexBench). Persona = **static, per-conversation**: a *text role prompt*
(role/background/scenario) + a *voice prompt* (audio embedding for vocal style; pre-packaged voices
NATF/NATM/VARF/VARM). **No user-learning, no persona memory, no extraction from conversations**
(stated explicitly). English-only; A100/H100-class GPU; Moshi server / Python API (no
OpenAI-compatible/vLLM surface). License: MIT (code) + NVIDIA Open Model License (weights) +
CC-BY-4.0 (base Moshi) — commercial use permitted.

## Decision
**Adopt PersonaPlex as the option for Morgan's Phase 5 voice modality (currently deferred), in a
HYBRID design — not as a replacement for the text reasoning/learning pipeline.**

Rationale:
- PersonaPlex is a **voice interaction** advance, **not** a personalization/self-learning advance.
  Morgan's "knows me" engine (signals → bi-temporal consolidation → `UserModel`/`USER.md` →
  GEPA champion preprompt, all eval-gated) remains the system that learns the owner. PersonaPlex
  *consumes* a persona; Morgan *produces* one.
- It is the strongest available fit for the one deliberately-deferred piece (full-duplex voice),
  beating the planned cascaded Whisper→LLM→TTS on naturalness/latency.

## Architecture (how it fits the existing seams)
PersonaPlex is a self-contained speech↔speech dialogue model — it does **not** fit behind the text
`ChatClient` (it generates replies itself) nor as a pure ASR behind `Perception`. So it is a
**distinct voice service** in the already-seamed `perception-gpu`, exposed through a new
`VoiceConversation` seam, bridged to Morgan's brain:

1. **Persona bridge (built now, pure/testable):** `morgan_brain/voice/persona_bridge.py` turns
   Morgan's learned state (`UserModel` comm-prefs/traits/relationship_stage + the champion preprompt
   + current emotional signal) into a **PersonaPlex text role prompt + a chosen voice id**. This is
   how the assistant's learned persona conditions the voice model each session.
2. **Hybrid routing (design):**
   - **Full-duplex mode** — PersonaPlex handles casual/empathic voice turns end-to-end (its strength:
     naturalness, latency, backchanneling). It does **not** use Morgan's memory/tools/skills mid-turn.
   - **Cascaded mode** — for knowledge/tool/memory-heavy turns: ASR (Whisper or PersonaPlex's
     understanding stream) → Morgan's brain (memory recall + tools + skills + personalization) → TTS.
   - A lightweight intent/uncertainty check picks the mode; default casual→full-duplex,
     command/question/knowledge→cascaded.
3. **Write-back (so voice keeps teaching Morgan):** every voice turn's transcript (user + assistant)
   is fed into Morgan's `SignalRecorder` + episodic memory, so consolidation/optimization learn from
   voice exactly as from text. This closes the "PersonaPlex doesn't learn" gap by keeping learning in
   Morgan, not the voice model.
4. **Privacy:** voice/transcripts are owner data → same classification + (opt-in) encryption +
   `delete_subject` fan-out. PersonaPlex runs **locally** (GPU), so no egress; if ever remote, the
   transcript path goes through the egress redaction gate.

## Scope now vs deferred
- **Now (this branch, unit-tested, no GPU):** the `VoiceConversation` seam (`interfaces/voice.py`)
  + the `persona_bridge` (UserModel/champion → PersonaPlex persona prompt + voice selection) +
  a `FakeVoiceConversation` for tests. This makes the integration concrete and keeps the brain
  provider/voice-agnostic.
- **Deferred (the GPU service):** the actual PersonaPlex/Moshi serving in `apps/perception_gpu`
  (Moshi server, audio I/O, voice embeddings), the hybrid mode-router, and transcript write-back
  wiring. Tracked as Phase 5-voice. Gated behind a `[voice]` optional extra (`moshi`, audio deps).

## Alternatives considered
- **Cascaded only (Whisper + a TTS):** simpler, reuses the text pipeline fully (memory/tools), but
  robotic / higher latency. Kept as the fallback half of the hybrid.
- **PersonaPlex as the whole assistant:** rejected — it would bypass Morgan's memory/learning/tools
  and can't be eval-gated the same way; it's a voice front-end, not the brain.

## Citations
research.nvidia.com/labs/adlr/personaplex · huggingface.co/nvidia/personaplex-7b-v1 ·
github.com/NVIDIA/personaplex · base: Kyutai Moshi (kyutai/moshiko).
