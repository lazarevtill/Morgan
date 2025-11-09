"""
Simple rate limiting for Morgan RAG.

Token bucket algorithm for API rate limiting.
"""

import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows burst requests up to bucket capacity,
    then limits to steady rate.
    """

    def __init__(self, rate_limit: int, time_window: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum requests per time window
            time_window: Time window in seconds (default: 60s)
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit  # Start with full bucket
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token (blocking).

        Args:
            timeout: Maximum time to wait for token (None = no timeout)

        Returns:
            True if token acquired, False if timeout

        Raises:
            TimeoutError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError("Rate limiter timeout")

            # Wait a bit before trying again
            time.sleep(0.1)

    def try_acquire(self) -> bool:
        """
        Try to acquire a token (non-blocking).

        Returns:
            True if token acquired, False if no tokens available
        """
        with self.lock:
            self._refill_tokens()

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        tokens_to_add = elapsed * (self.rate_limit / self.time_window)

        # Add tokens (up to bucket capacity)
        self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
        self.last_refill = now

    def get_available_tokens(self) -> float:
        """
        Get number of available tokens.

        Returns:
            Number of tokens available
        """
        with self.lock:
            self._refill_tokens()
            return self.tokens
