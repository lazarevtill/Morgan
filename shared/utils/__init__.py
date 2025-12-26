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
]

