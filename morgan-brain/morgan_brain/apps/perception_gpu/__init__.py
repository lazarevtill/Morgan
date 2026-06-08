"""perception-gpu — voice/vision perception service (DEFERRED, GPU deployment phase).

The ``interfaces.Perception`` Protocol is already defined and the text-only implementation
(``modules.perception.text.TextPerception``) runs inline in brain-api.  When voice is built,
this service implements the same Protocol: Whisper ASR transcribes the audio frame to text,
Wav2Vec2 extracts prosodic/emotion features, and the results are fused into a ``FusedPerception``
object — which is then returned to brain-api.  Because every downstream module (Orchestrator,
MemoryGate, Reasoner) depends only on the ``Perception`` Protocol and the ``FusedPerception``
model, **zero downstream changes are required**: swapping ``TextPerception`` for this service
is a one-line composition change in ``composition.py``.  DEFERRED — no functional code here.
"""
