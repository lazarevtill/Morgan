"""
Learning system exceptions.

Provides a hierarchy of exceptions for different failure modes
in the learning pipeline.
"""

from typing import Optional


class LearningError(Exception):
    """Base exception for all learning system errors."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        correlation_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.recoverable = recoverable
        self.correlation_id = correlation_id


class PatternDetectionError(LearningError):
    """Error during pattern detection."""

    def __init__(
        self,
        message: str,
        pattern_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.pattern_type = pattern_type


class FeedbackProcessingError(LearningError):
    """Error during feedback processing."""

    def __init__(
        self,
        message: str,
        feedback_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.feedback_id = feedback_id


class PreferenceLearningError(LearningError):
    """Error during preference learning."""

    def __init__(
        self,
        message: str,
        dimension: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.dimension = dimension


class AdaptationError(LearningError):
    """Error during adaptation."""

    def __init__(
        self,
        message: str,
        adaptation_id: Optional[str] = None,
        can_rollback: bool = True,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=can_rollback)
        self.adaptation_id = adaptation_id
        self.can_rollback = can_rollback


class ConsolidationError(LearningError):
    """Error during knowledge consolidation."""

    def __init__(
        self,
        message: str,
        phase: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.phase = phase


class LearningValidationError(LearningError):
    """Error validating learning system input or output."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[any] = None,
    ) -> None:
        super().__init__(message, recoverable=False)
        self.field = field
        self.value = value


class LearningStorageError(LearningError):
    """Error with learning data storage operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.operation = operation


class LearningCacheError(LearningError):
    """Error with learning cache operations."""

    def __init__(
        self,
        message: str,
        key: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.key = key


class LearningResourceError(LearningError):
    """Error with learning system resources (models, connections, etc)."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause, recoverable=True)
        self.resource_type = resource_type


class LearningConfigError(LearningError):
    """Error with learning system configuration."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
    ) -> None:
        super().__init__(message, recoverable=False)
        self.config_key = config_key


class CircuitBreakerOpenError(LearningError):
    """Error when circuit breaker is open."""

    def __init__(
        self,
        message: str,
        service: str,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, recoverable=True)
        self.service = service
        self.retry_after = retry_after
