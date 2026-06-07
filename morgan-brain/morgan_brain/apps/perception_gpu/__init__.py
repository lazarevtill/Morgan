"""perception-gpu — DEFERRED.

This service does not exist yet. The ``interfaces.Perception`` Protocol is defined and the text
implementation runs inline in brain-api. When voice/vision is built (design spec Phase 5), this
package implements that same Protocol (Whisper ASR + Wav2Vec2 emotion + prosody sarcasm + vision)
and brain-api routes audio/image inputs here — with zero change to any downstream module.
"""
