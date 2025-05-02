"""
Enhanced error handling utilities for Morgan Core
"""
import logging
import traceback
import sys
import json
import time
from typing import Dict, Any, Callable, Coroutine, Optional, List, Type, Union
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class MorganError(Exception):
    """Base exception class for Morgan-specific errors"""

    def __init__(self, message: str, code: str = "general_error", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ConfigError(MorganError):
    """Error related to configuration issues"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "config_error", details)


class ServiceError(MorganError):
    """Error related to service connectivity or operation"""

    def __init__(self, service_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service"] = service_name
        super().__init__(message, "service_error", details)


class HomeAssistantError(MorganError):
    """Error related to Home Assistant integration"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "homeassistant_error", details)


class IntentError(MorganError):
    """Error related to intent parsing or resolution"""

    def __init__(self, message: str, intent: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if intent:
            details["intent"] = intent
        super().__init__(message, "intent_error", details)


class SystemError(MorganError):
    """Error related to system operations"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "system_error", details)


class AuthError(MorganError):
    """Error related to authentication or authorization"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "auth_error", details)


class ValidationError(MorganError):
    """Error related to input validation"""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, "validation_error", details)


class ErrorHandler:
    """Error handling utilities"""

    @staticmethod
    async def handle_with_fallback(
            func: Callable[..., Coroutine],
            fallback_response: Dict[str, Any],
            log_prefix: str,
            *args,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a function with error handling and fallback

        Args:
            func: Async function to execute
            fallback_response: Response to return on error
            log_prefix: Prefix for log messages
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function result or fallback response
        """
        try:
            return await func(*args, **kwargs)
        except MorganError as e:
            logger.error(f"{log_prefix}: {e.code} - {e.message}")
            logger.debug(f"Error details: {e.details}")
            fallback = fallback_response.copy()
            fallback["error"] = {
                "code": e.code,
                "message": e.message,
                "details": e.details
            }
            return fallback
        except Exception as e:
            logger.error(f"{log_prefix}: {e}")
            logger.debug(traceback.format_exc())
            return fallback_response

    @staticmethod
    def create_error_response(
            message: str,
            voice: bool = True,
            code: str = "general_error",
            details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standard error response

        Args:
            message: Error message
            voice: Whether to generate voice for this response
            code: Error code
            details: Optional error details

        Returns:
            Formatted error response
        """
        return {
            "text": message,
            "voice": voice,
            "actions": [],
            "error": True,
            "error_details": {
                "code": code,
                "message": message,
                "details": details or {}
            }
        }

    @staticmethod
    def format_exception(exc: Exception) -> Dict[str, Any]:
        """
        Format an exception into a structured dictionary

        Args:
            exc: The exception to format

        Returns:
            Structured error information
        """
        if isinstance(exc, MorganError):
            return {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "traceback": traceback.format_exc()
            }
        else:
            return {
                "code": "unhandled_error",
                "message": str(exc),
                "details": {
                    "type": exc.__class__.__name__
                },
                "traceback": traceback.format_exc()
            }

    @staticmethod
    def log_exception(exc: Exception, log_prefix: str = "Error"):
        """
        Log an exception with appropriate level and format

        Args:
            exc: The exception to log
            log_prefix: Prefix for the log message
        """
        if isinstance(exc, MorganError):
            logger.error(f"{log_prefix}: [{exc.code}] {exc.message}")
            if exc.details:
                logger.debug(f"Error details: {json.dumps(exc.details)}")
            logger.debug(traceback.format_exc())
        else:
            logger.error(f"{log_prefix}: {exc}")
            logger.debug(traceback.format_exc())

    @staticmethod
    def safe_execute(fallback_value: Any = None):
        """
        Decorator for safely executing functions with exception handling

        Args:
            fallback_value: Value to return if an exception occurs

        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                    return fallback_value

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                    return fallback_value

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
              exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception):
        """
        Decorator for retrying functions with exponential backoff

        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between attempts (seconds)
            backoff: Multiplier for delay after each attempt
            exceptions: Exception types to catch and retry

        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt < max_attempts:
                            logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                            logger.debug(f"Retrying in {current_delay:.2f} seconds...")

                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                            raise

                # This shouldn't be reached, but just in case
                raise last_exception

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt < max_attempts:
                            logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                            logger.debug(f"Retrying in {current_delay:.2f} seconds...")

                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                            raise

                # This shouldn't be reached, but just in case
                raise last_exception

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator