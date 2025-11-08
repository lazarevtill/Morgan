"""
Emotion detection modules.

11 specialized modules for comprehensive emotion analysis.
"""

from morgan.emotions.modules.aggregator import EmotionAggregator
from morgan.emotions.modules.cache import EmotionCache
from morgan.emotions.modules.classifier import EmotionClassifier
from morgan.emotions.modules.context_analyzer import ContextAnalyzer
from morgan.emotions.modules.history_tracker import EmotionHistoryTracker
from morgan.emotions.modules.intensity import IntensityAnalyzer
from morgan.emotions.modules.multi_emotion import MultiEmotionDetector
from morgan.emotions.modules.pattern_detector import PatternDetector
from morgan.emotions.modules.temporal_analyzer import TemporalAnalyzer
from morgan.emotions.modules.trigger_detector import TriggerDetector

__all__ = [
    "EmotionClassifier",
    "IntensityAnalyzer",
    "PatternDetector",
    "TriggerDetector",
    "EmotionHistoryTracker",
    "ContextAnalyzer",
    "MultiEmotionDetector",
    "TemporalAnalyzer",
    "EmotionCache",
    "EmotionAggregator",
]
