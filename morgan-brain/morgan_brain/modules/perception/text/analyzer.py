"""Minimal text perception. Implements interfaces.Perception for the text modality.

Phase 1 scope: intent classification by simple heuristics and capitalized-token entity
extraction. Emotion/sentiment remain at defaults until Phase 2; audio/vision are Phase 5.
"""

from __future__ import annotations

import re

from morgan_brain.models.base import Entity
from morgan_brain.models.perception import FusedPerception, Intent, Modality

_CAP_TOKEN = re.compile(r"\b([A-Z][a-z]{2,})\b")
_STOPWORDS = {
    "The",
    "What",
    "When",
    "Where",
    "Why",
    "How",
    "Remind",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


class TextPerception:
    async def analyze(
        self, *, user_id: str, text: str, audio: bytes | None = None, image: bytes | None = None
    ) -> FusedPerception:
        intent_name = self._classify_intent(text)
        entities = [
            Entity(name=m.group(1))
            for m in _CAP_TOKEN.finditer(text)
            if m.group(1) not in _STOPWORDS
        ]
        seen: set[str] = set()
        unique: list[Entity] = []
        for e in entities:
            if e.name not in seen:
                seen.add(e.name)
                unique.append(e)
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
