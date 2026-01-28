"""
Production-grade error handling utilities for Morgan AI Assistant

Provides:
- Structured error responses
- Error context management
- Correlation ID tracking
- Error logging with context
- HTTP status code mapping
"""

import logging
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from http import HTTPStatus

from shared.utils.exceptions import (
    MorganException,
    ErrorCategory,
    ErrorSeverity,
    ServiceException,
    ModelException,
    AudioException,
    NetworkException,
    ConfigurationException,
    ValidationException,
    ResourceException,
    DatabaseException,
    ExternalIntegrationException,
)


logger = logging.getLogger(__name__)


class ErrorContext:
    """
    Error context manager for tracking correlation IDs and metadata
    across function calls
    """

    def __init__(self, correlation_id: Optional[str] = None, **metadata):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.metadata = metadata
        self.start_time = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "timestamp": self.start_time.isoformat(),
        }


class ErrorResponse:
    """
    Structured error response builder

    Provides consistent error response format across all services
    with appropriate HTTP status codes and user-friendly messages.
    """

    @staticmethod
    def from_exception(
        exception: Exception,
        include_stack_trace: bool = False,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create structured error response from exception

        Args:
            exception: The exception to convert
            include_stack_trace: Whether to include stack trace (dev mode)
            correlation_id: Override correlation ID

        Returns:
            Structured error response dictionary
        """
        if isinstance(exception, MorganException):
            return ErrorResponse._from_morgan_exception(
                exception, include_stack_trace, correlation_id
            )
        else:
            return ErrorResponse._from_generic_exception(
                exception, include_stack_trace, correlation_id
            )

    @staticmethod
    def _from_morgan_exception(
        exception: MorganException,
        include_stack_trace: bool = False,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create response from MorganException"""
        error_dict = exception.to_dict()

        # Override correlation ID if provided
        if correlation_id:
            error_dict["error"]["correlation_id"] = correlation_id

        # Add HTTP status code
        error_dict["error"]["http_status"] = ErrorResponse._get_http_status(exception)

        # Add stack trace if requested
        if include_stack_trace:
            error_dict["error"]["stack_trace"] = exception.stack_trace

        # Add timestamp if not present
        if "timestamp" not in error_dict["error"]:
            error_dict["error"]["timestamp"] = datetime.now(timezone.utc).isoformat()

        return error_dict

    @staticmethod
    def _from_generic_exception(
        exception: Exception,
        include_stack_trace: bool = False,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create response from generic exception"""
        import traceback

        error_dict = {
            "error": {
                "message": str(exception),
                "user_message": "An unexpected error occurred. Please try again.",
                "category": ErrorCategory.INTERNAL_ERROR.value,
                "severity": ErrorSeverity.ERROR.value,
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "is_transient": False,
                "is_retryable": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "recovery_suggestions": ["Contact support if the issue persists"],
            }
        }

        if include_stack_trace:
            error_dict["error"]["stack_trace"] = traceback.format_exc()

        return error_dict

    @staticmethod
    def _get_http_status(exception: MorganException) -> int:
        """Map exception to appropriate HTTP status code"""
        # Validation errors -> 400 Bad Request
        if isinstance(exception, ValidationException):
            return HTTPStatus.BAD_REQUEST.value

        # Configuration errors -> 500 Internal Server Error
        if isinstance(exception, ConfigurationException):
            return HTTPStatus.INTERNAL_SERVER_ERROR.value

        # Resource errors -> 429 Too Many Requests or 507 Insufficient Storage
        if isinstance(exception, ResourceException):
            if exception.category == ErrorCategory.QUOTA_EXCEEDED:
                return HTTPStatus.TOO_MANY_REQUESTS.value
            return HTTPStatus.INSUFFICIENT_STORAGE.value

        # Service errors -> 503 Service Unavailable or 504 Gateway Timeout
        if isinstance(exception, ServiceException):
            if exception.category == ErrorCategory.SERVICE_TIMEOUT:
                return HTTPStatus.GATEWAY_TIMEOUT.value
            return HTTPStatus.SERVICE_UNAVAILABLE.value

        # Network errors -> 503 Service Unavailable
        if isinstance(exception, NetworkException):
            return HTTPStatus.SERVICE_UNAVAILABLE.value

        # Model errors -> 503 Service Unavailable
        if isinstance(exception, ModelException):
            return HTTPStatus.SERVICE_UNAVAILABLE.value

        # Audio errors -> 422 Unprocessable Entity
        if isinstance(exception, AudioException):
            return HTTPStatus.UNPROCESSABLE_ENTITY.value

        # Database errors -> 503 Service Unavailable
        if isinstance(exception, DatabaseException):
            return HTTPStatus.SERVICE_UNAVAILABLE.value

        # External integration errors -> 502 Bad Gateway
        if isinstance(exception, ExternalIntegrationException):
            return HTTPStatus.BAD_GATEWAY.value

        # Default -> 500 Internal Server Error
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    @staticmethod
    def success(
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create structured success response

        Args:
            data: Response data
            metadata: Additional metadata
            correlation_id: Correlation ID for tracking

        Returns:
            Structured success response
        """
        response = {"success": True, "timestamp": datetime.now(timezone.utc).isoformat()}

        if data is not None:
            response["data"] = data

        if metadata:
            response["metadata"] = metadata

        if correlation_id:
            response["correlation_id"] = correlation_id

        return response


class ErrorLogger:
    """
    Enhanced error logger with structured logging and context

    Provides consistent error logging across all services with:
    - Correlation ID tracking
    - Structured log data
    - Severity-based logging
    - Error context capture
    """

    def __init__(
        self,
        logger_instance: Optional[logging.Logger] = None,
        service_name: str = "morgan",
    ):
        self.logger = logger_instance or logger
        self.service_name = service_name

    def log_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Log exception with full context

        Args:
            exception: Exception to log
            context: Error context with correlation ID
            additional_data: Additional data to include in log
        """
        # Determine log level
        if isinstance(exception, MorganException):
            log_level = self._get_log_level(exception.severity)
            correlation_id = exception.correlation_id
        else:
            log_level = logging.ERROR
            correlation_id = context.correlation_id if context else str(uuid.uuid4())

        # Build structured log data
        log_data = {
            "service": self.service_name,
            "correlation_id": correlation_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

        # Add context if available
        if context:
            log_data.update(context.to_dict())

        # Add exception-specific data
        if isinstance(exception, MorganException):
            log_data.update(
                {
                    "category": exception.category.value,
                    "severity": exception.severity.value,
                    "is_transient": exception.is_transient,
                    "is_retryable": exception.is_retryable,
                    "error_context": exception.context,
                }
            )

        # Add additional data
        if additional_data:
            log_data.update(additional_data)

        # Log with appropriate level
        self.logger.log(
            log_level, f"Error occurred: {exception}", extra=log_data, exc_info=True
        )

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Map error severity to log level"""
        severity_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.ERROR)


class ErrorHandler:
    """
    Unified error handler for Morgan services

    Provides:
    - Exception handling and conversion
    - Structured error responses
    - Error logging with context
    - Correlation ID tracking
    """

    def __init__(
        self,
        logger_instance: Optional[logging.Logger] = None,
        service_name: str = "morgan",
        include_stack_trace: bool = False,
    ):
        self.logger = ErrorLogger(logger_instance, service_name)
        self.service_name = service_name
        self.include_stack_trace = include_stack_trace

    def handle_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle exception and return structured error response

        Args:
            exception: Exception to handle
            context: Error context with correlation ID
            additional_data: Additional data for logging

        Returns:
            Structured error response
        """
        # Log the exception
        self.logger.log_exception(exception, context, additional_data)

        # Create structured response
        correlation_id = context.correlation_id if context else None
        return ErrorResponse.from_exception(
            exception,
            include_stack_trace=self.include_stack_trace,
            correlation_id=correlation_id,
        )

    def wrap_exception(
        self,
        exception: Exception,
        message: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        context: Optional[ErrorContext] = None,
    ) -> MorganException:
        """
        Wrap generic exception as MorganException

        Args:
            exception: Original exception
            message: Custom error message
            category: Error category
            context: Error context

        Returns:
            MorganException wrapping the original
        """
        msg = message or str(exception)
        correlation_id = context.correlation_id if context else None

        # Try to infer better category based on exception type
        if isinstance(exception, TimeoutError):
            category = ErrorCategory.SERVICE_TIMEOUT
        elif isinstance(exception, ConnectionError):
            category = ErrorCategory.NETWORK_CONNECTION_FAILED
        elif isinstance(exception, ValueError):
            category = ErrorCategory.VALIDATION_FAILED
        elif isinstance(exception, KeyError):
            category = ErrorCategory.MISSING_REQUIRED_FIELD

        return MorganException(
            message=msg,
            category=category,
            correlation_id=correlation_id,
            cause=exception,
            context=context.metadata if context else {},
        )


def create_error_context(**metadata) -> ErrorContext:
    """
    Create error context for tracking

    Args:
        **metadata: Metadata to include in context

    Returns:
        ErrorContext instance
    """
    return ErrorContext(**metadata)


def handle_async_errors(
    logger_instance: Optional[logging.Logger] = None,
    service_name: str = "morgan",
    suppress_errors: bool = False,
    default_return: Any = None,
):
    """
    Decorator for handling async function errors

    Args:
        logger_instance: Logger to use
        service_name: Service name for logging
        suppress_errors: If True, suppress errors and return default
        default_return: Default value to return on error (if suppress_errors=True)

    Example:
        @handle_async_errors(service_name="llm", suppress_errors=False)
        async def generate_text():
            return await llm.generate()
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = ErrorHandler(logger_instance, service_name)
            context = create_error_context(function=func.__name__, service=service_name)

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log and handle error
                error_response = handler.handle_error(e, context)

                if suppress_errors:
                    return default_return
                else:
                    # Re-raise as MorganException if not already
                    if isinstance(e, MorganException):
                        raise
                    else:
                        raise handler.wrap_exception(e, context=context)

        return wrapper

    return decorator
