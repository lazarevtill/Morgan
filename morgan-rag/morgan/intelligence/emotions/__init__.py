"""
Modular emotion detection system for Morgan RAG.

Provides real-time emotion analysis, mood pattern tracking, emotional state history,
emotion categorization, intensity measurement, and emotion regulation through focused,
single-responsibility modules.
"""

from .analyzer import MoodAnalyzer, get_mood_analyzer
from .classifier import EmotionClassifier, get_emotion_classifier
from .detector import EmotionDetector, get_emotion_detector
from .intensity import IntensityMeasurer, get_intensity_measurer
from .regulator import (
    EmotionRegulator,
    RegulationSession,
    RegulationStrategy,
    get_emotion_regulator,
)
from .tracker import EmotionalStateTracker, get_emotional_state_tracker

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
    "get_intensity_measurer",
    "EmotionRegulator",
    "get_emotion_regulator",
    "RegulationStrategy",
    "RegulationSession",
]
