"""
Emotional intelligence module for Morgan RAG.

Provides emotional awareness, mood tracking, and empathetic response generation
to create meaningful companion relationships with users.
"""

# Note: EmotionalIntelligenceEngine and get_emotional_intelligence_engine
# should be imported directly from .intelligence_engine to avoid circular imports
# Example: from morgan.intelligence.core.intelligence_engine import get_emotional_intelligence_engine

__all__ = ["EmotionalIntelligenceEngine", "get_emotional_intelligence_engine"]


def __getattr__(name):
    """Lazy import to avoid circular dependency with constants.py."""
    if name in ("EmotionalIntelligenceEngine", "get_emotional_intelligence_engine"):
        from .intelligence_engine import (
            EmotionalIntelligenceEngine,
            get_emotional_intelligence_engine,
        )
        if name == "EmotionalIntelligenceEngine":
            return EmotionalIntelligenceEngine
        return get_emotional_intelligence_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
