"""
Morgan Emotion Detection System.

A production-quality emotion detection system with 11 specialized modules
for analyzing, classifying, and tracking emotional content in user messages.

Architecture:
- Async-first design for performance (<200ms target)
- Dependency injection for testability
- Clean separation of concerns
- Proper error handling and logging
- Resource cleanup and lifecycle management

Modules:
1. EmotionClassifier - Classifies text into 8 basic emotion types
2. IntensityAnalyzer - Analyzes and adjusts emotion intensities
3. PatternDetector - Detects emotional patterns over time
4. TriggerDetector - Identifies emotional triggers in text
5. EmotionHistoryTracker - Maintains emotional history persistence
6. ContextAnalyzer - Analyzes conversational context
7. MultiEmotionDetector - Handles multiple simultaneous emotions
8. TemporalAnalyzer - Analyzes temporal emotion changes
9. EmotionCache - Caches results for performance
10. EmotionAggregator - Aggregates all results
11. EmotionDetector - Main orchestration and error handling

Usage:
    from morgan.emotions import EmotionDetector, EmotionContext

    # Initialize detector
    detector = EmotionDetector(enable_cache=True, enable_history=True)
    await detector.initialize()

    try:
        # Detect emotions
        result = await detector.detect(
            "I'm so happy today!",
            context=EmotionContext(user_id="user123")
        )

        print(f"Dominant emotion: {result.dominant_emotion}")
        print(f"Valence: {result.valence}")
        print(f"Processing time: {result.processing_time_ms}ms")
    finally:
        await detector.cleanup()
"""

from morgan.emotions.base import AsyncCache, CircuitBreaker, EmotionModule
from morgan.emotions.detector import EmotionDetector
from morgan.emotions.exceptions import (
    EmotionAnalysisError,
    EmotionCacheError,
    EmotionClassificationError,
    EmotionContextError,
    EmotionDetectionError,
    EmotionHistoryError,
    EmotionResourceError,
    EmotionValidationError,
)
from morgan.emotions.modules import (
    ContextAnalyzer,
    EmotionAggregator,
    EmotionCache,
    EmotionClassifier,
    EmotionHistoryTracker,
    IntensityAnalyzer,
    MultiEmotionDetector,
    PatternDetector,
    TemporalAnalyzer,
    TriggerDetector,
)
from morgan.emotions.types import (
    Emotion,
    EmotionContext,
    EmotionIntensity,
    EmotionPattern,
    EmotionResult,
    EmotionTrigger,
    EmotionType,
)
from morgan.emotions.utils import (
    emotions_to_dict,
    filter_emotions_by_type,
    format_colored_output,
    format_emotion_summary,
    get_console_color,
    get_emotion_emoji,
    get_strongest_emotion,
    is_crisis_state,
)

__all__ = [
    # Main detector
    "EmotionDetector",
    # Types
    "Emotion",
    "EmotionType",
    "EmotionIntensity",
    "EmotionResult",
    "EmotionPattern",
    "EmotionTrigger",
    "EmotionContext",
    # Modules
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
    # Base classes
    "EmotionModule",
    "AsyncCache",
    "CircuitBreaker",
    # Exceptions
    "EmotionDetectionError",
    "EmotionClassificationError",
    "EmotionAnalysisError",
    "EmotionValidationError",
    "EmotionCacheError",
    "EmotionHistoryError",
    "EmotionContextError",
    "EmotionResourceError",
    # Utilities
    "format_emotion_summary",
    "format_colored_output",
    "get_console_color",
    "get_emotion_emoji",
    "filter_emotions_by_type",
    "get_strongest_emotion",
    "emotions_to_dict",
    "is_crisis_state",
]

__version__ = "1.0.0"
