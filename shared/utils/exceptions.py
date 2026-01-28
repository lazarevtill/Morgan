"""
Production-grade exception hierarchy for Morgan AI Assistant

This module provides a comprehensive exception system with:
- Error classification (transient vs permanent)
- Correlation ID support for distributed tracing
- Rich error context and metadata
- Structured error responses
"""

import uuid
import traceback
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""

    # Service errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_TIMEOUT = "service_timeout"
    SERVICE_DEGRADED = "service_degraded"

    # Model errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    MODEL_OUT_OF_MEMORY = "model_out_of_memory"

    # Audio errors
    AUDIO_INVALID_FORMAT = "audio_invalid_format"
    AUDIO_PROCESSING_FAILED = "audio_processing_failed"
    AUDIO_ENCODING_FAILED = "audio_encoding_failed"
    AUDIO_DECODING_FAILED = "audio_decoding_failed"

    # Network errors
    NETWORK_CONNECTION_FAILED = "network_connection_failed"
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_DNS_FAILED = "network_dns_failed"

    # Configuration errors
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    CONFIG_VALIDATION_FAILED = "config_validation_failed"

    # Validation errors
    VALIDATION_FAILED = "validation_failed"
    INVALID_INPUT = "invalid_input"
    MISSING_REQUIRED_FIELD = "missing_required_field"

    # Resource errors
    RESOURCE_EXHAUSTED = "resource_exhausted"
    QUOTA_EXCEEDED = "quota_exceeded"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    DISK_FULL = "disk_full"

    # Authentication/Authorization errors
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    TOKEN_EXPIRED = "token_expired"
    INVALID_CREDENTIALS = "invalid_credentials"

    # Database errors
    DATABASE_CONNECTION_FAILED = "database_connection_failed"
    DATABASE_QUERY_FAILED = "database_query_failed"
    DATABASE_INTEGRITY_ERROR = "database_integrity_error"

    # External integration errors
    EXTERNAL_API_ERROR = "external_api_error"
    EXTERNAL_SERVICE_UNAVAILABLE = "external_service_unavailable"

    # Internal errors
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"
    UNKNOWN_ERROR = "unknown_error"


class MorganException(Exception):
    """
    Base exception for all Morgan exceptions

    Provides:
    - Error classification (transient vs permanent)
    - Correlation ID for distributed tracing
    - Rich context and metadata
    - Structured error responses
    - Stack trace capture
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        is_transient: bool = False,
        is_retryable: bool = False,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)

        self.message = message
        self.category = category
        self.severity = severity
        self.is_transient = is_transient
        self.is_retryable = is_retryable
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.user_message = (
            user_message or "An error occurred while processing your request"
        )
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.utcnow()
        self.stack_trace = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary"""
        return {
            "error": {
                "message": self.message,
                "user_message": self.user_message,
                "category": self.category.value,
                "severity": self.severity.value,
                "correlation_id": self.correlation_id,
                "is_transient": self.is_transient,
                "is_retryable": self.is_retryable,
                "timestamp": self.timestamp.isoformat(),
                "context": self.context,
                "recovery_suggestions": self.recovery_suggestions,
                "caused_by": str(self.cause) if self.cause else None,
            }
        }

    def to_user_dict(self) -> Dict[str, Any]:
        """Convert to user-friendly dictionary (no stack traces)"""
        return {
            "error": {
                "message": self.user_message,
                "correlation_id": self.correlation_id,
                "recovery_suggestions": self.recovery_suggestions,
                "timestamp": self.timestamp.isoformat(),
            }
        }


# ============================================================================
# Service Exceptions
# ============================================================================


class ServiceException(MorganException):
    """Base exception for service-related errors"""

    def __init__(self, message: str, service_name: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context["service_name"] = service_name
        self.service_name = service_name


class ServiceUnavailableError(ServiceException):
    """Service is unavailable or unreachable"""

    def __init__(self, service_name: str, message: Optional[str] = None, **kwargs):
        message = message or f"Service '{service_name}' is unavailable"
        super().__init__(
            message=message,
            service_name=service_name,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message=f"The {service_name} service is temporarily unavailable. Please try again.",
            recovery_suggestions=[
                "Wait a few moments and try again",
                "Check service status",
            ],
            **kwargs,
        )


class ServiceTimeoutError(ServiceException):
    """Service request timed out"""

    def __init__(self, service_name: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message=f"Service '{service_name}' timed out after {timeout_seconds}s",
            service_name=service_name,
            category=ErrorCategory.SERVICE_TIMEOUT,
            severity=ErrorSeverity.WARNING,
            is_transient=True,
            is_retryable=True,
            user_message="The request took too long to process. Please try again.",
            recovery_suggestions=[
                "Try again with a simpler request",
                "Wait a moment and retry",
            ],
            **kwargs,
        )
        self.context["timeout_seconds"] = timeout_seconds


class ServiceDegradedError(ServiceException):
    """Service is degraded but partially functional"""

    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            message=message,
            service_name=service_name,
            category=ErrorCategory.SERVICE_DEGRADED,
            severity=ErrorSeverity.WARNING,
            is_transient=True,
            is_retryable=False,
            user_message=f"The {service_name} service is running with reduced capacity.",
            **kwargs,
        )


# ============================================================================
# Model Exceptions
# ============================================================================


class ModelException(MorganException):
    """Base exception for model-related errors"""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_name:
            self.context["model_name"] = model_name
        self.model_name = model_name


class ModelNotFoundError(ModelException):
    """Model not found or not loaded"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            message=f"Model '{model_name}' not found or not loaded",
            model_name=model_name,
            category=ErrorCategory.MODEL_NOT_FOUND,
            severity=ErrorSeverity.ERROR,
            is_transient=False,
            is_retryable=False,
            user_message="The requested AI model is not available.",
            recovery_suggestions=[
                "Check model name",
                "Load the model first",
                "Use a different model",
            ],
            **kwargs,
        )


class ModelLoadError(ModelException):
    """Failed to load model"""

    def __init__(self, model_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            model_name=model_name,
            category=ErrorCategory.MODEL_LOAD_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=False,
            is_retryable=False,
            user_message="Failed to load the AI model.",
            recovery_suggestions=[
                "Check model availability",
                "Verify system resources",
                "Contact support",
            ],
            **kwargs,
        )
        self.context["load_failure_reason"] = reason


class ModelInferenceError(ModelException):
    """Model inference failed"""

    def __init__(self, model_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Model inference failed for '{model_name}': {reason}",
            model_name=model_name,
            category=ErrorCategory.MODEL_INFERENCE_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to process your request with the AI model.",
            recovery_suggestions=[
                "Try again",
                "Simplify your request",
                "Try a different model",
            ],
            **kwargs,
        )
        self.context["inference_failure_reason"] = reason


class ModelOutOfMemoryError(ModelException):
    """Model ran out of memory during inference"""

    def __init__(
        self, model_name: str, memory_required: Optional[int] = None, **kwargs
    ):
        super().__init__(
            message=f"Out of memory during inference with model '{model_name}'",
            model_name=model_name,
            category=ErrorCategory.MODEL_OUT_OF_MEMORY,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Insufficient memory to process your request.",
            recovery_suggestions=[
                "Try a smaller input",
                "Use a smaller model",
                "Free up system memory",
            ],
            **kwargs,
        )
        if memory_required:
            self.context["memory_required_mb"] = memory_required


# ============================================================================
# Audio Exceptions
# ============================================================================


class AudioException(MorganException):
    """Base exception for audio-related errors"""

    pass


class AudioFormatError(AudioException):
    """Invalid audio format"""

    def __init__(self, message: str, expected_format: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUDIO_INVALID_FORMAT,
            severity=ErrorSeverity.ERROR,
            is_transient=False,
            is_retryable=False,
            user_message="The audio format is not supported.",
            recovery_suggestions=[
                "Use WAV or MP3 format",
                "Check audio encoding",
                "Re-encode the audio",
            ],
            **kwargs,
        )
        if expected_format:
            self.context["expected_format"] = expected_format


class AudioProcessingError(AudioException):
    """Audio processing failed"""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUDIO_PROCESSING_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to process the audio.",
            recovery_suggestions=[
                "Try again",
                "Check audio quality",
                "Use a different audio file",
            ],
            **kwargs,
        )
        if operation:
            self.context["failed_operation"] = operation


class AudioEncodingError(AudioException):
    """Audio encoding failed"""

    def __init__(self, message: str, target_format: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUDIO_ENCODING_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to encode the audio.",
            **kwargs,
        )
        if target_format:
            self.context["target_format"] = target_format


class AudioDecodingError(AudioException):
    """Audio decoding failed"""

    def __init__(self, message: str, source_format: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUDIO_DECODING_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to decode the audio.",
            **kwargs,
        )
        if source_format:
            self.context["source_format"] = source_format


# ============================================================================
# Network Exceptions
# ============================================================================


class NetworkException(MorganException):
    """Base exception for network-related errors"""

    pass


class NetworkConnectionError(NetworkException):
    """Network connection failed"""

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK_CONNECTION_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to establish network connection.",
            recovery_suggestions=[
                "Check network connectivity",
                "Verify service is running",
                "Retry in a moment",
            ],
            **kwargs,
        )
        if host:
            self.context["host"] = host
        if port:
            self.context["port"] = port


class NetworkTimeoutError(NetworkException):
    """Network request timed out"""

    def __init__(self, message: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK_TIMEOUT,
            severity=ErrorSeverity.WARNING,
            is_transient=True,
            is_retryable=True,
            user_message="Network request timed out.",
            recovery_suggestions=["Check network connectivity", "Retry the request"],
            **kwargs,
        )
        self.context["timeout_seconds"] = timeout_seconds


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationException(MorganException):
    """Base exception for configuration errors"""

    pass


class ConfigMissingError(ConfigurationException):
    """Required configuration is missing"""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            category=ErrorCategory.CONFIG_MISSING,
            severity=ErrorSeverity.CRITICAL,
            is_transient=False,
            is_retryable=False,
            user_message="System configuration is incomplete.",
            recovery_suggestions=[
                f"Set the '{config_key}' configuration",
                "Contact administrator",
            ],
            **kwargs,
        )
        self.context["missing_config_key"] = config_key


class ConfigInvalidError(ConfigurationException):
    """Configuration value is invalid"""

    def __init__(self, config_key: str, invalid_value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid configuration for '{config_key}': {reason}",
            category=ErrorCategory.CONFIG_INVALID,
            severity=ErrorSeverity.ERROR,
            is_transient=False,
            is_retryable=False,
            user_message="System configuration is invalid.",
            recovery_suggestions=[
                f"Fix the '{config_key}' configuration",
                "Contact administrator",
            ],
            **kwargs,
        )
        self.context.update(
            {
                "config_key": config_key,
                "invalid_value": str(invalid_value),
                "reason": reason,
            }
        )


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationException(MorganException):
    """Base exception for validation errors"""

    pass


class ValidationError(ValidationException):
    """Input validation failed"""

    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION_FAILED,
            severity=ErrorSeverity.WARNING,
            is_transient=False,
            is_retryable=False,
            user_message="Input validation failed.",
            recovery_suggestions=["Check input format", "Review validation rules"],
            **kwargs,
        )
        if field_name:
            self.context["field_name"] = field_name


class InvalidInputError(ValidationException):
    """Input is invalid"""

    def __init__(self, message: str, input_name: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.INVALID_INPUT,
            severity=ErrorSeverity.WARNING,
            is_transient=False,
            is_retryable=False,
            user_message=f"Invalid input: {input_name}",
            recovery_suggestions=["Correct the input", "Check input requirements"],
            **kwargs,
        )
        self.context["input_name"] = input_name


class MissingRequiredFieldError(ValidationException):
    """Required field is missing"""

    def __init__(self, field_name: str, **kwargs):
        super().__init__(
            message=f"Missing required field: {field_name}",
            category=ErrorCategory.MISSING_REQUIRED_FIELD,
            severity=ErrorSeverity.WARNING,
            is_transient=False,
            is_retryable=False,
            user_message=f"Required field '{field_name}' is missing.",
            recovery_suggestions=[f"Provide the '{field_name}' field"],
            **kwargs,
        )
        self.context["field_name"] = field_name


# ============================================================================
# Resource Exceptions
# ============================================================================


class ResourceException(MorganException):
    """Base exception for resource-related errors"""

    pass


class ResourceExhaustedError(ResourceException):
    """Resource exhausted"""

    def __init__(self, resource_name: str, **kwargs):
        super().__init__(
            message=f"Resource exhausted: {resource_name}",
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="System resources are exhausted.",
            recovery_suggestions=[
                "Wait and try again",
                "Reduce request size",
                "Contact administrator",
            ],
            **kwargs,
        )
        self.context["resource_name"] = resource_name


class QuotaExceededError(ResourceException):
    """Quota exceeded"""

    def __init__(self, quota_name: str, limit: int, current: int, **kwargs):
        super().__init__(
            message=f"Quota exceeded for '{quota_name}': {current}/{limit}",
            category=ErrorCategory.QUOTA_EXCEEDED,
            severity=ErrorSeverity.WARNING,
            is_transient=False,
            is_retryable=False,
            user_message=f"You have exceeded the {quota_name} quota.",
            recovery_suggestions=[
                "Wait for quota reset",
                "Upgrade quota",
                "Contact support",
            ],
            **kwargs,
        )
        self.context.update(
            {"quota_name": quota_name, "limit": limit, "current": current}
        )


class GPUOutOfMemoryError(ResourceException):
    """GPU out of memory"""

    def __init__(self, required_memory_mb: Optional[int] = None, **kwargs):
        super().__init__(
            message="GPU out of memory",
            category=ErrorCategory.GPU_OUT_OF_MEMORY,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Insufficient GPU memory to process your request.",
            recovery_suggestions=[
                "Try a smaller input",
                "Use a smaller model",
                "Free up GPU memory",
            ],
            **kwargs,
        )
        if required_memory_mb:
            self.context["required_memory_mb"] = required_memory_mb


# ============================================================================
# Database Exceptions
# ============================================================================


class DatabaseException(MorganException):
    """Base exception for database errors"""

    pass


class DatabaseConnectionError(DatabaseException):
    """Database connection failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE_CONNECTION_FAILED,
            severity=ErrorSeverity.CRITICAL,
            is_transient=True,
            is_retryable=True,
            user_message="Failed to connect to database.",
            recovery_suggestions=[
                "Check database connectivity",
                "Verify credentials",
                "Contact administrator",
            ],
            **kwargs,
        )


class DatabaseQueryError(DatabaseException):
    """Database query failed"""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE_QUERY_FAILED,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message="Database query failed.",
            recovery_suggestions=[
                "Retry the operation",
                "Contact support if issue persists",
            ],
            **kwargs,
        )
        if query:
            # Sanitize query before storing (remove sensitive data)
            self.context["query_preview"] = (
                query[:100] + "..." if len(query) > 100 else query
            )


# ============================================================================
# External Integration Exceptions
# ============================================================================


class ExternalIntegrationException(MorganException):
    """Base exception for external integration errors"""

    pass


class ExternalAPIError(ExternalIntegrationException):
    """External API returned an error"""

    def __init__(
        self,
        api_name: str,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        msg = f"External API error from '{api_name}'"
        if status_code:
            msg += f" (status {status_code})"
        if message:
            msg += f": {message}"

        super().__init__(
            message=msg,
            category=ErrorCategory.EXTERNAL_API_ERROR,
            severity=ErrorSeverity.ERROR,
            is_transient=True,
            is_retryable=True,
            user_message=f"External service '{api_name}' returned an error.",
            recovery_suggestions=[
                "Try again later",
                "Contact support if issue persists",
            ],
            **kwargs,
        )
        # Store as direct attributes for backward compatibility with RequestError
        self.status_code = status_code
        self.details = details or {}
        self.api_name = api_name
        self.context.update({"api_name": api_name, "status_code": status_code})
