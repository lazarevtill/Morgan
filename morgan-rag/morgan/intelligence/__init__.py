"""
Intelligence module for Morgan RAG.

Consolidates emotional awareness, mood tracking, and empathetic response generation
into a cohesive unit.
"""

from .constants import (
    EMOTION_COLORS,
    EMOTION_PATTERNS,
    EMOTION_THRESHOLDS,
    EMOTION_TRANSITIONS,
    EMOTION_VALENCE,
    FORMALITY_INDICATORS,
    INTENSITY_MODIFIERS,
    NEGATION_PATTERNS,
)
from .core.intelligence_engine import (
    EmotionalIntelligenceEngine,
    get_emotional_intelligence_engine,
)

__all__ = [
    # Engine
    "EmotionalIntelligenceEngine",
    "get_emotional_intelligence_engine",
    # Constants
    "EMOTION_PATTERNS",
    "INTENSITY_MODIFIERS",
    "NEGATION_PATTERNS",
    "EMOTION_VALENCE",
    "FORMALITY_INDICATORS",
    "EMOTION_COLORS",
    "EMOTION_TRANSITIONS",
    "EMOTION_THRESHOLDS",
]
