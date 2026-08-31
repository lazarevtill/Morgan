"""Minimal text perception. Implements interfaces.Perception for the text modality.

Phase 1 scope: intent classification by simple heuristics, plus entity extraction
delegated to ``perception/text/entities.py`` -- the same function the cold path uses to
populate ``Memory.entities``, so the traits selected on the hot path and the entities
indexed on the cold path mean the same thing. Emotion/sentiment remain at defaults until
Phase 2; audio/vision are Phase 5.
"""

from __future__ import annotations

import re

from morgan_brain.models.base import Entity
from morgan_brain.models.perception import FusedPerception, Intent, Modality
from morgan_brain.modules.perception.text.entities import extract_entity_names


class TextPerception:
    async def analyze(
        self, *, user_id: str, text: str, audio: bytes | None = None, image: bytes | None = None
    ) -> FusedPerception:
        intent_name = self._classify_intent(text)
        # extract_entity_names already deduplicates and preserves first-appearance order.
        unique = [Entity(name=name) for name in extract_entity_names(text)]
        return FusedPerception(
            text=text,
            intent=Intent(
                name=intent_name, confidence=0.6
            ),  # heuristic-only; calibrated in Phase 2
            entities=unique,
            modalities_used=[Modality.TEXT],
        )

    @staticmethod
    def _classify_intent(text: str) -> str:
        stripped = text.strip()
        if stripped.endswith("?") or re.match(
            r"^(what|when|where|why|how|who|is|are|do|does)\b", stripped, re.IGNORECASE
        ):
            return "question"
        if re.match(r"^(remind|create|add|delete|set|schedule|run)\b", stripped, re.IGNORECASE):
            return "command"
        return "chat"
