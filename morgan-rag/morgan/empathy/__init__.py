"""
Advanced empathy engine for Morgan RAG.

Provides empathetic response creation, emotional validation, mirroring and reflection,
crisis detection and support, and emotional tone matching through focused modules.
"""

from .generator import EmpatheticResponseGenerator, get_empathetic_response_generator
from .validator import EmotionalValidator, get_emotional_validator
from .mirror import EmotionalMirror, get_emotional_mirror
from .support import CrisisSupport, get_crisis_support
from .tone import EmotionalToneManager, get_emotional_tone_manager

__all__ = [
    "EmpatheticResponseGenerator",
    "get_empathetic_response_generator",
    "EmotionalValidator",
    "get_emotional_validator",
    "EmotionalMirror",
    "get_emotional_mirror",
    "CrisisSupport",
    "get_crisis_support",
    "EmotionalToneManager",
    "get_emotional_tone_manager"
]