"""Shared emotion vocabulary, used by Perception (detect), Learning (track baseline),
and Personalization (adapt to deltas)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


class EmotionState(BaseModel):
    primary: EmotionType = EmotionType.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SentimentScore(BaseModel):
    """Valence/Arousal/Dominance — richer than pos/neg, supports baseline deltas."""

    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
