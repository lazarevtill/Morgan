"""Perception module — implements ``interfaces.Perception``.

Responsibility: turn raw input into a ``FusedPerception``.
Service: brain-api — the text analyzer is built and wired inline as the first request-path step.
Audio/vision (Whisper ASR + Wav2Vec2 emotion + prosody sarcasm) remain deferred to perception-gpu
behind the same Protocol (zero downstream change when added).

Files: text/analyzer.py (built). audio/* and vision/* deferred.
"""
