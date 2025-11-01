"""
Emotional intelligence module for Morgan RAG.

Provides emotional awareness, mood tracking, and empathetic response generation
to create meaningful companion relationships with users.
"""

from .intelligence_engine import (
    EmotionalIntelligenceEngine,
    get_emotional_intelligence_engine
)

__all__ = [
    "EmotionalIntelligenceEngine",
    "get_emotional_intelligence_engine"
]