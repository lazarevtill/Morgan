"""
Modular emotion detection system for Morgan RAG.

Provides real-time emotion analysis, mood pattern tracking, emotional state history,
emotion categorization, and intensity measurement through focused, single-responsibility modules.
"""

from .detector import EmotionDetector, get_emotion_detector
from .analyzer import MoodAnalyzer, get_mood_analyzer
from .tracker import EmotionalStateTracker, get_emotional_state_tracker
from .classifier import EmotionClassifier, get_emotion_classifier
from .intensity import IntensityMeasurer, get_intensity_measurer

__all__ = [
    "EmotionDetector",
    "get_emotion_detector",
    "MoodAnalyzer", 
    "get_mood_analyzer",
    "EmotionalStateTracker",
    "get_emotional_state_tracker",
    "EmotionClassifier",
    "get_emotion_classifier",
    "IntensityMeasurer",
    "get_intensity_measurer"
]