"""
Error handling utilities for Morgan AI Assistant

DEPRECATED: This module is maintained for backward compatibility only.
New code should use:
- shared.utils.exceptions for exception classes
- shared.utils.error_handling for error handling utilities
- shared.utils.error_decorators for error decorators
"""
import logging
import warnings
from typing import Dict, Any, Optional
from enum import Enum

# Import new exception system
from shared.utils.exceptions import (
    MorganException,
    ErrorCategory,
    ServiceException,
    ModelException as NewModelException,
    AudioException as NewAudioException,
    ConfigurationException,
    ValidationException,
    ServiceUnavailableError,
    ServiceTimeoutError,
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
    AudioProcessingError,
    AudioFormatError,
    ConfigMissingError,
    ConfigInvalidError,
    InvalidInputError,
    ValidationError as NewValidationError
)

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """
    Standard error codes (DEPRECATED)

    Use ErrorCategory from shared.utils.exceptions instead
    """
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_ERROR = "SERVICE_ERROR"

    # Model errors
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INFERENCE_ERROR = "MODEL_INFERENCE_ERROR"

    # Audio errors
    AUDIO_PROCESSING_ERROR = "AUDIO_PROCESSING_ERROR"
    AUDIO_FORMAT_ERROR = "AUDIO_FORMAT_ERROR"

    # Configuration errors
    CONFIG_ERROR = "CONFIG_ERROR"
    CONFIG_MISSING = "CONFIG_MISSING"

    # Network errors
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"

    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"

    # Resource errors
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    GPU_MEMORY_ERROR = "GPU_MEMORY_ERROR"

    # Authentication errors
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"

    # Model not loaded
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"

    # Generic errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class MorganError(Exception):
    """
    Base exception for Morgan AI Assistant (DEPRECATED)

    Use MorganException from shared.utils.exceptions instead
    """

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None  # Will be set when logged

        # Issue deprecation warning
        warnings.warn(
            "MorganError is deprecated. Use MorganException from shared.utils.exceptions instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            "error_code": self.error_code.value,
            "message": str(self),
            "details": self.details,
            "timestamp": self.timestamp
        }


class ServiceError(MorganError):
    """Error related to external services (DEPRECATED)"""
    pass


class ModelError(MorganError):
    """Error related to AI models (DEPRECATED)"""
    pass


class AudioError(MorganError):
    """Error related to audio processing (DEPRECATED)"""
    pass


class ConfigurationError(MorganError):
    """Error related to configuration (DEPRECATED)"""
    pass


class ValidationError(MorganError):
    """Error related to input validation (DEPRECATED)"""
    pass


class ErrorHandler:
    """
    Central error handling and logging (DEPRECATED)

    Use ErrorHandler from shared.utils.error_handling instead
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Issue deprecation warning
        warnings.warn(
            "ErrorHandler from shared.utils.errors is deprecated. "
            "Use ErrorHandler from shared.utils.error_handling instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def handle_error(self, error: Exception, context: Optional[str] = None) -> MorganError:
        """Handle and convert errors to MorganError"""
        if isinstance(error, MorganError):
            morgan_error = error
        else:
            # Convert unknown errors to MorganError
            morgan_error = MorganError(
                str(error),
                ErrorCode.INTERNAL_ERROR,
                {"original_type": type(error).__name__, "context": context}
            )

        # Set timestamp and log
        morgan_error.timestamp = None  # Will be set by logging
        self.logger.error(f"MorganError: {morgan_error.error_code.value} - {morgan_error}",
                         extra={"error": morgan_error.to_dict()})

        return morgan_error

    def create_error_response(self, message: str, error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
                            details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": {
                "code": error_code.value,
                "message": message,
                "details": details or {}
            },
            "timestamp": None  # Will be set when sent
        }

    def create_success_response(self, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "success": True,
            "timestamp": None  # Will be set when sent
        }

        if data is not None:
            response["data"] = data

        if metadata:
            response["metadata"] = metadata

        return response


def handle_async_errors(logger: Optional[logging.Logger] = None):
    """
    Decorator to handle async function errors (DEPRECATED)

    Use decorators from shared.utils.error_decorators instead
    """
    warnings.warn(
        "handle_async_errors from shared.utils.errors is deprecated. "
        "Use decorators from shared.utils.error_decorators instead.",
        DeprecationWarning,
        stacklevel=2
    )

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler(logger)
                error_handler.handle_error(e, context=f"function: {func.__name__}")
                raise
        return wrapper
    return decorator
