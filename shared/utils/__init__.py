"""Shared utilities for Morgan AI Assistant."""

from shared.utils.emotional import (
    EmotionType,
    EmotionalState,
    SimpleEmotionDetector,
    get_emotion_detector,
)
from shared.utils.error_handling import (
    ErrorHandler,
    ErrorContext,
    ErrorResponse,
    ErrorLogger,
    create_error_context,
    handle_async_errors,
)
from shared.utils.exceptions import (
    MorganException,
    ErrorCategory,
    ErrorSeverity,
)
from shared.utils.deduplication import (
    DeduplicationResult,
    deduplicate_by_content,
    deduplicate_by_id,
    deduplicate_by_similarity,
    deduplicate_search_results,
)
from shared.utils.text_extraction import (
    Entity,
    extract_entities,
    extract_keywords,
    extract_topics,
    extract_concepts,
    normalize_text,
    calculate_text_similarity,
)

__all__ = [
    # Emotional
    "EmotionType",
    "EmotionalState",
    "SimpleEmotionDetector",
    "get_emotion_detector",
    # Error handling
    "ErrorHandler",
    "ErrorContext",
    "ErrorResponse",
    "ErrorLogger",
    "create_error_context",
    "handle_async_errors",
    # Exceptions
    "MorganException",
    "ErrorCategory",
    "ErrorSeverity",
    # Deduplication
    "DeduplicationResult",
    "deduplicate_by_content",
    "deduplicate_by_id",
    "deduplicate_by_similarity",
    "deduplicate_search_results",
    # Text extraction
    "Entity",
    "extract_entities",
    "extract_keywords",
    "extract_topics",
    "extract_concepts",
    "normalize_text",
    "calculate_text_similarity",
]
