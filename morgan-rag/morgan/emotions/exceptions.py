"""
Emotion detection system exceptions.

Provides a hierarchy of exceptions for different failure modes
in the emotion detection pipeline.
"""

from typing import Optional


class EmotionDetectionError(Exception):
    """Base exception for all emotion detection errors."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.recoverable = recoverable


class EmotionClassificationError(EmotionDetectionError):
    """Error during emotion classification."""

    pass


class EmotionAnalysisError(EmotionDetectionError):
    """Error during emotion analysis (intensity, patterns, triggers)."""

    pass


class EmotionValidationError(EmotionDetectionError):
    """Error validating emotion detection input or output."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(message, recoverable=False)
        self.field = field


class EmotionCacheError(EmotionDetectionError):
    """Error with emotion cache operations."""

    pass


class EmotionHistoryError(EmotionDetectionError):
    """Error with emotion history tracking."""

    pass


class EmotionContextError(EmotionDetectionError):
    """Error with emotion context management."""

    pass


class EmotionResourceError(EmotionDetectionError):
    """Error with emotion detection resources (models, connections, etc)."""

    def __init__(self, message: str, resource_type: str) -> None:
        super().__init__(message, recoverable=True)
        self.resource_type = resource_type
