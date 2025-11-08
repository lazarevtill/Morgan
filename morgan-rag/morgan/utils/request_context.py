"""
Request context utilities for Morgan RAG.

Simple request ID tracking for debugging and tracing.
"""

import threading
import uuid
from typing import Optional

# Thread-local storage for request context
_context = threading.local()


def get_request_id() -> Optional[str]:
    """
    Get current request ID.

    Returns:
        Request ID or None if not set
    """
    return getattr(_context, "request_id", None)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for current thread.

    Args:
        request_id: Request ID (generates UUID if None)

    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]  # Short UUID

    _context.request_id = request_id
    return request_id


def clear_request_id():
    """Clear request ID for current thread."""
    if hasattr(_context, "request_id"):
        delattr(_context, "request_id")


class RequestContext:
    """Context manager for request ID."""

    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize request context.

        Args:
            request_id: Request ID (generates UUID if None)
        """
        self.request_id = request_id
        self.previous_id = None

    def __enter__(self) -> str:
        """Enter context and set request ID."""
        self.previous_id = get_request_id()
        return set_request_id(self.request_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous request ID."""
        if self.previous_id is not None:
            set_request_id(self.previous_id)
        else:
            clear_request_id()
