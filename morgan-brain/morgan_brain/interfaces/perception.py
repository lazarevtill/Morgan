"""Perception contract — turns raw input into a modality-agnostic ``FusedPerception``.

The text implementation runs inline in brain-api. A future audio/vision implementation would
satisfy this exact Protocol, so nothing downstream changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from morgan_brain.models.perception import FusedPerception


@runtime_checkable
class Perception(Protocol):
    async def analyze(
        self, *, user_id: str, text: str, audio: bytes | None = None, image: bytes | None = None
    ) -> FusedPerception: ...
