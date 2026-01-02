"""
Specialized error handling for companion features and emotional intelligence in Morgan RAG.

Provides production-grade error handling for emotional processing, relationship building,
memory processing, and companion interactions with graceful degradation and fallback mechanisms.
"""

import functools
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from morgan.exceptions import (
    MorganError,
    EmotionalProcessingError,
    MemoryProcessingError,
    RelationshipTrackingError,
    EmpathyGenerationError,
)
from morgan.utils.error_handling import (
    DegradationLevel,
    get_degradation_manager,
    get_recovery_manager,
)

logger = structlog.get_logger(__name__)


class CompanionFeatureType(Enum):
    """Types of companion features."""

    EMOTIONAL_ANALYSIS = "emotional_analysis"
    RELATIONSHIP_TRACKING = "relationship_tracking"
    MEMORY_PROCESSING = "memory_processing"
    EMPATHY_GENERATION = "empathy_generation"
    PERSONALIZATION = "personalization"
    MILESTONE_DETECTION = "milestone_detection"


# RelationshipTrackingError and EmpathyGenerationError are imported from morgan.exceptions


@dataclass
class CompanionErrorContext:
    """Extended error context for companion operations."""

    feature_type: CompanionFeatureType
    user_id: Optional[str] = None
    interaction_id: Optional[str] = None
    emotion_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    fallback_attempted: bool = False
    fallback_successful: bool = False
    degradation_applied: bool = False


class CompanionFeatureManager:
    """
    Manages companion feature health, fallbacks, and graceful degradation.

    Provides centralized management of companion features with intelligent
    fallback mechanisms and graceful degradation for emotional processing failures.
    """

    def __init__(self):
        self.feature_health: Dict[CompanionFeatureType, bool] = {}
        self.feature_metrics: Dict[str, Dict[str, Any]] = {}
        self.fallback_strategies: Dict[CompanionFeatureType, List[str]] = {}
        self.degradation_thresholds: Dict[CompanionFeatureType, float] = {}

        # Initialize feature health tracking
        self._initialize_feature_tracking()

        # Initialize fallback strategies
        self._initialize_fallback_strategies()

        logger.info("CompanionFeatureManager initialized")

    def _initialize_feature_tracking(self):
        """Initialize feature health tracking."""
        for feature_type in CompanionFeatureType:
            self.feature_health[feature_type] = True
            self.feature_metrics[feature_type.value] = {
                "success_count": 0,
                "error_count": 0,
                "total_duration": 0.0,
                "last_success": None,
                "last_error": None,
                "confidence_scores": [],
            }

        # Set degradation thresholds (error rate %)
        self.degradation_thresholds = {
            CompanionFeatureType.EMOTIONAL_ANALYSIS: 0.15,  # 15% error rate
            CompanionFeatureType.RELATIONSHIP_TRACKING: 0.10,  # 10% error rate
            CompanionFeatureType.MEMORY_PROCESSING: 0.20,  # 20% error rate (less critical)
            CompanionFeatureType.EMPATHY_GENERATION: 0.12,  # 12% error rate
            CompanionFeatureType.PERSONALIZATION: 0.18,  # 18% error rate
            CompanionFeatureType.MILESTONE_DETECTION: 0.25,  # 25% error rate (least critical)
        }

    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for each feature."""
        self.fallback_strategies = {
            CompanionFeatureType.EMOTIONAL_ANALYSIS: [
                "neutral_emotion_fallback",
                "simple_sentiment_analysis",
                "skip_emotional_processing",
            ],
            CompanionFeatureType.RELATIONSHIP_TRACKING: [
                "basic_interaction_counting",
                "simple_engagement_tracking",
                "skip_relationship_updates",
            ],
            CompanionFeatureType.MEMORY_PROCESSING: [
                "simple_keyword_extraction",
                "basic_importance_scoring",
                "skip_memory_processing",
            ],
            CompanionFeatureType.EMPATHY_GENERATION: [
                "template_based_responses",
                "neutral_supportive_responses",
                "skip_empathy_enhancement",
            ],
            CompanionFeatureType.PERSONALIZATION: [
                "basic_user_preferences",
                "generic_adaptive_responses",
                "skip_personalization",
            ],
            CompanionFeatureType.MILESTONE_DETECTION: [
                "simple_milestone_rules",
                "skip_milestone_detection",
            ],
        }

    def record_feature_success(
        self,
        feature_type: CompanionFeatureType,
        operation: str,
        duration: float,
        confidence_score: Optional[float] = None,
    ):
        """Record successful feature operation."""
        feature_key = feature_type.value
        metrics = self.feature_metrics[feature_key]

        metrics["success_count"] += 1
        metrics["total_duration"] += duration
        metrics["last_success"] = datetime.now()

        if confidence_score is not None:
            metrics["confidence_scores"].append(confidence_score)
            # Keep only last 100 confidence scores
            if len(metrics["confidence_scores"]) > 100:
                metrics["confidence_scores"] = metrics["confidence_scores"][-100:]

        # Update feature health
        self._update_feature_health(feature_type)

        logger.debug(
            "Recorded companion feature success",
            feature_type=feature_type.value,
            operation=operation,
            duration=duration,
            confidence_score=confidence_score,
        )

    def record_feature_error(
        self, feature_type: CompanionFeatureType, operation: str, error: Exception
    ):
        """Record feature error and update health status."""
        feature_key = feature_type.value
        metrics = self.feature_metrics[feature_key]

        metrics["error_count"] += 1
        metrics["last_error"] = datetime.now()

        # Update feature health
        self._update_feature_health(feature_type)

        # Check if degradation is needed
        self._check_degradation_threshold(feature_type)

        logger.error(
            "Recorded companion feature error",
            feature_type=feature_type.value,
            operation=operation,
            error=str(error),
        )

    def _update_feature_health(self, feature_type: CompanionFeatureType):
        """Update feature health based on error rate."""
        metrics = self.feature_metrics[feature_type.value]
        total_operations = metrics["success_count"] + metrics["error_count"]

        if total_operations > 0:
            error_rate = metrics["error_count"] / total_operations
            threshold = self.degradation_thresholds[feature_type]

            self.feature_health[feature_type] = error_rate < threshold

    def _check_degradation_threshold(self, feature_type: CompanionFeatureType):
        """Check if feature should be degraded due to high error rate."""
        metrics = self.feature_metrics[feature_type.value]
        total_operations = metrics["success_count"] + metrics["error_count"]

        if total_operations >= 10:  # Only check after sufficient operations
            error_rate = metrics["error_count"] / total_operations
            threshold = self.degradation_thresholds[feature_type]

            if error_rate > threshold:
                logger.warning(
                    f"Companion feature {feature_type.value} error rate ({error_rate:.1%}) "
                    f"exceeds threshold ({threshold:.1%}), applying degradation"
                )

                # Apply feature-specific degradation
                self._apply_feature_degradation(feature_type)

    def _apply_feature_degradation(self, feature_type: CompanionFeatureType):
        """Apply degradation for a specific feature."""
        degradation_manager = get_degradation_manager()

        # Apply appropriate degradation level based on feature criticality
        if feature_type in [
            CompanionFeatureType.EMOTIONAL_ANALYSIS,
            CompanionFeatureType.EMPATHY_GENERATION,
        ]:
            # Critical companion features - apply moderate degradation
            degradation_manager.force_degradation_level(
                DegradationLevel.MODERATE,
                f"companion_feature_failure_{feature_type.value}",
            )
        elif feature_type in [
            CompanionFeatureType.RELATIONSHIP_TRACKING,
            CompanionFeatureType.PERSONALIZATION,
        ]:
            # Important but not critical - apply minimal degradation
            degradation_manager.force_degradation_level(
                DegradationLevel.MINIMAL,
                f"companion_feature_failure_{feature_type.value}",
            )
        # Memory processing and milestone detection are less critical
        # and don't trigger system-wide degradation

    def get_feature_health(self, feature_type: CompanionFeatureType) -> bool:
        """Get current health status of a feature."""
        return self.feature_health.get(feature_type, False)

    def get_fallback_strategies(self, feature_type: CompanionFeatureType) -> List[str]:
        """Get fallback strategies for a feature."""
        return self.fallback_strategies.get(feature_type, [])

    def get_feature_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive feature metrics."""
        return self.feature_metrics.copy()

    def get_average_confidence(self, feature_type: CompanionFeatureType) -> float:
        """Get average confidence score for a feature."""
        metrics = self.feature_metrics[feature_type.value]
        confidence_scores = metrics["confidence_scores"]

        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.0


# Global companion feature manager instance
_companion_feature_manager = CompanionFeatureManager()


def get_companion_feature_manager() -> CompanionFeatureManager:
    """Get global companion feature manager instance."""
    return _companion_feature_manager


def handle_emotional_processing_errors(
    enable_neutral_fallback: bool = True,
    enable_sentiment_fallback: bool = True,
    confidence_threshold: float = 0.5,
):
    """
    Decorator for handling emotional processing errors with intelligent fallbacks.

    Args:
        enable_neutral_fallback: Whether to return neutral emotion on failure
        enable_sentiment_fallback: Whether to use simple sentiment analysis
        confidence_threshold: Minimum confidence threshold for results
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            feature_manager = get_companion_feature_manager()

            # Check if emotional processing is enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("emotional"):
                logger.debug("Emotional processing disabled, returning neutral state")
                return _create_neutral_emotional_state()

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record success
                duration = time.time() - start_time
                confidence = getattr(result, "confidence", None)
                feature_manager.record_feature_success(
                    CompanionFeatureType.EMOTIONAL_ANALYSIS,
                    "emotion_detection",
                    duration,
                    confidence,
                )

                # Check confidence threshold
                if confidence is not None and confidence < confidence_threshold:
                    logger.warning(
                        f"Low confidence emotion detection ({confidence:.2f}), "
                        f"threshold: {confidence_threshold}"
                    )

                    if enable_neutral_fallback:
                        return _create_neutral_emotional_state()

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record error
                feature_manager.record_feature_error(
                    CompanionFeatureType.EMOTIONAL_ANALYSIS, "emotion_detection", e
                )

                # Try fallback strategies
                if enable_sentiment_fallback:
                    try:
                        logger.warning(
                            "Emotional analysis failed, attempting sentiment fallback"
                        )

                        # Simple sentiment analysis fallback
                        text = kwargs.get("text", args[0] if args else "")
                        sentiment_result = _simple_sentiment_analysis(text)

                        if sentiment_result:
                            logger.info("Sentiment analysis fallback successful")
                            return sentiment_result

                    except Exception as sentiment_error:
                        logger.error(
                            f"Sentiment analysis fallback failed: {sentiment_error}"
                        )

                if enable_neutral_fallback:
                    logger.info("Using neutral emotional state fallback")
                    return _create_neutral_emotional_state()

                # Convert to appropriate error type
                raise EmotionalProcessingError(
                    f"Emotional processing failed: {e}",
                    operation="emotion_detection",
                    cause=e,
                )

        return wrapper

    return decorator


def handle_relationship_tracking_errors(
    enable_basic_tracking: bool = True, skip_on_failure: bool = True
):
    """
    Decorator for handling relationship tracking errors.

    Args:
        enable_basic_tracking: Whether to use basic interaction counting
        skip_on_failure: Whether to skip tracking on failure
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            feature_manager = get_companion_feature_manager()

            # Check if companion features are enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("companion"):
                logger.debug(
                    "Companion features disabled, skipping relationship tracking"
                )
                return None

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record success
                duration = time.time() - start_time
                feature_manager.record_feature_success(
                    CompanionFeatureType.RELATIONSHIP_TRACKING,
                    "relationship_update",
                    duration,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record error
                feature_manager.record_feature_error(
                    CompanionFeatureType.RELATIONSHIP_TRACKING, "relationship_update", e
                )

                # Try basic tracking fallback
                if enable_basic_tracking:
                    try:
                        logger.warning(
                            "Relationship tracking failed, using basic interaction counting"
                        )

                        user_id = kwargs.get("user_id", args[0] if args else None)
                        if user_id:
                            # Basic interaction counting
                            basic_result = _basic_interaction_tracking(user_id)
                            logger.info("Basic interaction tracking successful")
                            return basic_result

                    except Exception as basic_error:
                        logger.error(
                            f"Basic interaction tracking failed: {basic_error}"
                        )

                if skip_on_failure:
                    logger.info("Skipping relationship tracking due to failure")
                    return None

                raise RelationshipTrackingError(
                    f"Relationship tracking failed: {e}",
                    user_id=kwargs.get("user_id"),
                    operation="relationship_update",
                    cause=e,
                )

        return wrapper

    return decorator


def handle_memory_processing_errors(
    enable_simple_extraction: bool = True, skip_on_failure: bool = True
):
    """
    Decorator for handling memory processing errors.

    Args:
        enable_simple_extraction: Whether to use simple keyword extraction
        skip_on_failure: Whether to skip memory processing on failure
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            feature_manager = get_companion_feature_manager()

            # Check if memory processing is enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("memory"):
                logger.debug("Memory processing disabled, skipping")
                return []

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record success
                duration = time.time() - start_time
                feature_manager.record_feature_success(
                    CompanionFeatureType.MEMORY_PROCESSING,
                    "memory_extraction",
                    duration,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record error
                feature_manager.record_feature_error(
                    CompanionFeatureType.MEMORY_PROCESSING, "memory_extraction", e
                )

                # Try simple extraction fallback
                if enable_simple_extraction:
                    try:
                        logger.warning(
                            "Memory processing failed, using simple keyword extraction"
                        )

                        conversation_turn = kwargs.get(
                            "conversation_turn", args[0] if args else None
                        )
                        if conversation_turn:
                            simple_memories = _simple_memory_extraction(
                                conversation_turn
                            )
                            logger.info("Simple memory extraction successful")
                            return simple_memories

                    except Exception as simple_error:
                        logger.error(f"Simple memory extraction failed: {simple_error}")

                if skip_on_failure:
                    logger.info("Skipping memory processing due to failure")
                    return []

                raise MemoryProcessingError(
                    f"Memory processing failed: {e}",
                    operation="memory_extraction",
                    cause=e,
                )

        return wrapper

    return decorator


def handle_empathy_generation_errors(
    enable_template_fallback: bool = True, enable_neutral_response: bool = True
):
    """
    Decorator for handling empathy generation errors.

    Args:
        enable_template_fallback: Whether to use template-based responses
        enable_neutral_response: Whether to use neutral supportive responses
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            feature_manager = get_companion_feature_manager()

            # Check if companion features are enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("companion"):
                logger.debug("Companion features disabled, using neutral response")
                return _create_neutral_empathetic_response()

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record success
                duration = time.time() - start_time
                empathy_score = getattr(result, "empathy_level", None)
                feature_manager.record_feature_success(
                    CompanionFeatureType.EMPATHY_GENERATION,
                    "empathy_generation",
                    duration,
                    empathy_score,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record error
                feature_manager.record_feature_error(
                    CompanionFeatureType.EMPATHY_GENERATION, "empathy_generation", e
                )

                # Try template-based fallback
                if enable_template_fallback:
                    try:
                        logger.warning(
                            "Empathy generation failed, using template-based response"
                        )

                        user_emotion = kwargs.get("user_emotion")
                        context = kwargs.get("context", "")

                        template_response = _template_empathetic_response(
                            user_emotion, context
                        )
                        logger.info("Template-based empathy response successful")
                        return template_response

                    except Exception as template_error:
                        logger.error(
                            f"Template-based empathy response failed: {template_error}"
                        )

                if enable_neutral_response:
                    logger.info("Using neutral empathetic response fallback")
                    return _create_neutral_empathetic_response()

                raise EmpathyGenerationError(
                    f"Empathy generation failed: {e}",
                    operation="empathy_generation",
                    cause=e,
                )

        return wrapper

    return decorator


# Fallback implementation functions


def _create_neutral_emotional_state():
    """Create a neutral emotional state for fallback."""
    from dataclasses import dataclass

    @dataclass
    class NeutralEmotionalState:
        primary_emotion: str = "neutral"
        intensity: float = 0.0
        confidence: float = 0.0
        secondary_emotions: list = None
        emotional_indicators: list = None

        def __post_init__(self):
            if self.secondary_emotions is None:
                self.secondary_emotions = []
            if self.emotional_indicators is None:
                self.emotional_indicators = []

    return NeutralEmotionalState()


def _simple_sentiment_analysis(text: str):
    """Simple sentiment analysis fallback."""
    if not text:
        return _create_neutral_emotional_state()

    # Simple keyword-based sentiment analysis
    positive_words = [
        "happy",
        "good",
        "great",
        "excellent",
        "wonderful",
        "amazing",
        "love",
    ]
    negative_words = ["sad", "bad", "terrible", "awful", "hate", "angry", "frustrated"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        emotion = "joy"
        intensity = min(0.8, positive_count * 0.2)
    elif negative_count > positive_count:
        emotion = "sadness"
        intensity = min(0.8, negative_count * 0.2)
    else:
        emotion = "neutral"
        intensity = 0.0

    from dataclasses import dataclass

    @dataclass
    class SimpleSentimentResult:
        primary_emotion: str = emotion
        intensity: float = intensity
        confidence: float = 0.6  # Lower confidence for simple analysis
        secondary_emotions: list = None
        emotional_indicators: list = None

        def __post_init__(self):
            if self.secondary_emotions is None:
                self.secondary_emotions = []
            if self.emotional_indicators is None:
                self.emotional_indicators = []

    return SimpleSentimentResult()


def _basic_interaction_tracking(user_id: str):
    """Basic interaction tracking fallback."""
    return {
        "user_id": user_id,
        "interaction_count": 1,
        "last_interaction": datetime.now(),
        "tracking_method": "basic_fallback",
        "trust_level": 0.5,
        "engagement_score": 0.5,
    }


def _simple_memory_extraction(conversation_turn):
    """Simple memory extraction fallback."""
    # Extract basic keywords and create simple memory
    content = getattr(conversation_turn, "content", str(conversation_turn))

    # Simple keyword extraction
    important_words = []
    for word in content.split():
        if len(word) > 4 and word.isalpha():  # Simple heuristic
            important_words.append(word.lower())

    if important_words:
        return [
            {
                "content": f"User mentioned: {', '.join(important_words[:5])}",
                "importance_score": 0.5,
                "entities": important_words[:3],
                "concepts": [],
                "extraction_method": "simple_fallback",
                "timestamp": datetime.now(),
            }
        ]

    return []


def _template_empathetic_response(user_emotion, context: str):
    """Template-based empathetic response fallback."""
    templates = {
        "joy": "I'm so glad to hear that you're feeling happy! That's wonderful.",
        "sadness": "I understand that you're going through a difficult time. I'm here for you.",
        "anger": "It sounds like you're feeling frustrated. That's completely understandable.",
        "fear": "I can sense that you're feeling anxious. It's okay to feel that way.",
        "surprise": "That sounds like quite a surprise! How are you processing that?",
        "neutral": "I appreciate you sharing that with me. How can I help you today?",
    }

    emotion_key = (
        getattr(user_emotion, "primary_emotion", "neutral")
        if user_emotion
        else "neutral"
    )
    response_text = templates.get(emotion_key, templates["neutral"])

    from dataclasses import dataclass

    @dataclass
    class TemplateEmpatheticResponse:
        response_text: str = response_text
        emotional_tone: str = emotion_key
        empathy_level: float = 0.7
        personalization_elements: list = None
        relationship_context: str = "template_based"

        def __post_init__(self):
            if self.personalization_elements is None:
                self.personalization_elements = []

    return TemplateEmpatheticResponse()


def _create_neutral_empathetic_response():
    """Create neutral empathetic response for fallback."""
    from dataclasses import dataclass

    @dataclass
    class NeutralEmpatheticResponse:
        response_text: str = "I understand. Thank you for sharing that with me."
        emotional_tone: str = "supportive"
        empathy_level: float = 0.5
        personalization_elements: list = None
        relationship_context: str = "neutral_fallback"

        def __post_init__(self):
            if self.personalization_elements is None:
                self.personalization_elements = []

    return NeutralEmpatheticResponse()


# Recovery procedures for companion features
def register_companion_recovery_procedures():
    """Register recovery procedures for companion features."""
    recovery_manager = get_recovery_manager()

    def recover_emotional_processing(
        error: EmotionalProcessingError, context: Dict[str, Any]
    ) -> bool:
        """Recover emotional processing by resetting to neutral state."""
        try:
            feature_manager = get_companion_feature_manager()

            # Reset feature health if error rate is acceptable
            metrics = feature_manager.feature_metrics["emotional_analysis"]
            total_ops = metrics["success_count"] + metrics["error_count"]

            if total_ops > 0:
                error_rate = metrics["error_count"] / total_ops
                if error_rate < 0.5:  # If error rate is below 50%, consider recoverable
                    feature_manager.feature_health[
                        CompanionFeatureType.EMOTIONAL_ANALYSIS
                    ] = True
                    logger.info("Emotional processing feature marked as recovered")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to recover emotional processing: {e}")
            return False

    def recover_companion_features(error: MorganError, context: Dict[str, Any]) -> bool:
        """General companion feature recovery."""
        try:
            # Apply minimal degradation instead of complete failure
            degradation_manager = get_degradation_manager()
            degradation_manager.force_degradation_level(
                DegradationLevel.MINIMAL, "companion_recovery_attempt"
            )

            logger.info("Applied minimal degradation for companion feature recovery")
            return True

        except Exception as e:
            logger.error(f"Failed to recover companion features: {e}")
            return False

    # Register recovery procedures
    from morgan.utils.error_handling import RecoveryProcedure, RecoveryStrategy

    recovery_manager.register_procedure(
        RecoveryProcedure(
            name="emotional_processing_recovery",
            strategy=RecoveryStrategy.FALLBACK,
            applicable_errors=[EmotionalProcessingError],
            recovery_function=recover_emotional_processing,
            description="Recover emotional processing with neutral fallback",
        )
    )

    recovery_manager.register_procedure(
        RecoveryProcedure(
            name="companion_feature_recovery",
            strategy=RecoveryStrategy.DEGRADE,
            applicable_errors=[RelationshipTrackingError, EmpathyGenerationError],
            recovery_function=recover_companion_features,
            description="Recover companion features with graceful degradation",
        )
    )

    logger.info("Registered companion feature recovery procedures")


def initialize_companion_error_handling():
    """Initialize companion error handling system."""
    logger.info("Initializing companion error handling")

    # Register recovery procedures
    register_companion_recovery_procedures()

    # Initialize feature manager
    get_companion_feature_manager()

    logger.info("Companion error handling initialized successfully")


if __name__ == "__main__":
    # Demo companion error handling
    print("💝 Companion Error Handling Demo")
    print("=" * 40)

    # Initialize system
    initialize_companion_error_handling()

    # Test emotional processing error handling
    @handle_emotional_processing_errors(
        enable_neutral_fallback=True, enable_sentiment_fallback=True
    )
    def test_emotion_detection(text):
        if "fail" in text.lower():
            raise Exception("Simulated emotion detection failure")

        # Simulate emotion detection result
        from dataclasses import dataclass

        @dataclass
        class EmotionResult:
            primary_emotion: str = "joy"
            intensity: float = 0.8
            confidence: float = 0.9

        return EmotionResult()

    # Test successful operation
    try:
        result = test_emotion_detection("I am happy today")
        print(
            f"Emotion detection successful: {result.primary_emotion} ({result.confidence})"
        )
    except Exception as e:
        print(f"Emotion detection failed: {e}")

    # Test failed operation with fallback
    try:
        result = test_emotion_detection("fail this detection")
        print(
            f"Emotion detection with fallback: {result.primary_emotion} ({result.confidence})"
        )
    except Exception as e:
        print(f"Emotion detection failed: {e}")

    # Test feature manager
    feature_manager = get_companion_feature_manager()
    print(
        f"Feature health: {feature_manager.get_feature_health(CompanionFeatureType.EMOTIONAL_ANALYSIS)}"
    )
    print(f"Feature metrics: {feature_manager.get_feature_metrics()}")

    print("\n" + "=" * 40)
    print("Companion error handling demo completed!")
