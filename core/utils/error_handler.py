"""
Error handling utilities for Morgan Core
"""
import logging
import traceback
from typing import Dict, Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


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
        except Exception as e:
            logger.error(f"{log_prefix}: {e}")
            logger.debug(traceback.format_exc())
            return fallback_response

    @staticmethod
    def create_error_response(message: str, voice: bool = True) -> Dict[str, Any]:
        """
        Create a standard error response

        Args:
            message: Error message
            voice: Whether to generate voice for this response

        Returns:
            Formatted error response
        """
        return {
            "text": message,
            "voice": voice,
            "actions": [],
            "error": True
        }