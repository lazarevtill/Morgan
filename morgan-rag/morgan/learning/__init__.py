"""
Morgan Learning System.

Production-ready learning system for user behavior analysis, preference learning,
and response adaptation with full async/await, circuit breakers, and fault tolerance.

Main Components:
- LearningEngine: Main orchestrator for all learning operations
- PatternModule: Behavioral pattern detection and analysis
- FeedbackModule: Feedback processing and sentiment analysis
- PreferenceModule: User preference learning and management
- AdaptationModule: Response adaptation with A/B testing
- ConsolidationModule: Knowledge consolidation and meta-learning

Usage:
    from morgan.learning import LearningEngine, FeedbackType, AdaptationStrategy

    # Initialize engine
    engine = LearningEngine()
    await engine.initialize()

    # Process feedback
    feedback = await engine.process_feedback(
        user_id="user123",
        feedback_type=FeedbackType.EXPLICIT_POSITIVE,
        rating=0.9,
        text="Great response!",
    )

    # Adapt response
    adapted, result = await engine.adapt_response(
        user_id="user123",
        base_response={"text": "Hello"},
        strategy=AdaptationStrategy.CONTEXTUAL,
    )

    # Get learning summary
    summary = await engine.get_learning_summary("user123")

    # Cleanup
    await engine.cleanup()
"""

from morgan.learning.engine import LearningEngine
from morgan.learning.exceptions import (
    AdaptationError,
    ConsolidationError,
    FeedbackProcessingError,
    LearningError,
    PatternDetectionError,
    PreferenceLearningError,
)
from morgan.learning.types import (
    AdaptationResult,
    AdaptationStrategy,
    ConsolidationResult,
    FeedbackSignal,
    FeedbackType,
    LearningContext,
    LearningMetrics,
    LearningPattern,
    PatternType,
    PreferenceDimension,
    UserPreference,
)

__all__ = [
    # Main engine
    "LearningEngine",
    # Types
    "AdaptationResult",
    "AdaptationStrategy",
    "ConsolidationResult",
    "FeedbackSignal",
    "FeedbackType",
    "LearningContext",
    "LearningMetrics",
    "LearningPattern",
    "PatternType",
    "PreferenceDimension",
    "UserPreference",
    # Exceptions
    "LearningError",
    "AdaptationError",
    "ConsolidationError",
    "FeedbackProcessingError",
    "PatternDetectionError",
    "PreferenceLearningError",
]

__version__ = "2.0.0"
__author__ = "Morgan Development Team"
