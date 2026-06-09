"""Perception output contract. Every perception call — text now, audio/vision later —
returns a ``FusedPerception`` so downstream modules never change when modalities are added."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from morgan_brain.models.base import Entity
from morgan_brain.models.emotion import EmotionState, SentimentScore


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VISION = "vision"


class Intent(BaseModel):
    name: str = "chat"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SarcasmResult(BaseModel):
    detected: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class FusedPerception(BaseModel):
    """Unified, modality-agnostic analysis of one user input."""

    text: str  # original or transcribed
    intent: Intent = Field(default_factory=Intent)
    entities: list[Entity] = Field(default_factory=list)
    emotion: EmotionState = Field(default_factory=EmotionState)
    sentiment: SentimentScore = Field(default_factory=SentimentScore)
    sarcasm: SarcasmResult | None = None  # only when audio is present
    modalities_used: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
