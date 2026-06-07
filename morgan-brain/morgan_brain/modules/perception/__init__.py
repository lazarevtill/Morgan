"""Perception module — implements ``interfaces.Perception``.

Responsibility: turn raw input into a ``FusedPerception``.
Service: brain-api (text, inline) now; perception-gpu (audio/vision) later.
Phase: 1 (text analyzer) · 5 (Whisper ASR + Wav2Vec2 emotion + prosody sarcasm + vision).

Planned files: text/analyzer.py, audio/* (deferred), vision/* (deferred), fusion/multimodal.py.
"""
