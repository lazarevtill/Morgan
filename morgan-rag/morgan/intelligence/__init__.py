"""
Intelligence module for Morgan RAG.

Consolidates emotional awareness, mood tracking, and empathetic response generation
into a cohesive unit.
"""

from .core.intelligence_engine import (
    EmotionalIntelligenceEngine,
    get_emotional_intelligence_engine,
)

__all__ = ["EmotionalIntelligenceEngine", "get_emotional_intelligence_engine"]
