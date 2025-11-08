"""
Rate limiting implementations for API calls
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""

    pass


@dataclass
class RateLimitConfig:
    """Rate limiter configuration"""

    requests_per_second: float = 10.0
    burst_size: Optional[int] = None  # Max burst, defaults to requests_per_second


class RateLimiter(ABC):
    """Abstract base class for rate limiters"""

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens for rate limiting

        Args:
            tokens: Number of tokens to acquire

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get current rate limiter state"""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter implementation

    Allows burst traffic while maintaining average rate limit.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Set burst size to requests_per_second if not specified
        self.capacity = self.config.burst_size or int(self.config.requests_per_second)
        self.tokens = float(self.capacity)
        self.rate = self.config.requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

        logger.info(
            f"Token bucket rate limiter initialized: "
            f"rate={self.rate}/s, capacity={self.capacity}"
        )

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket

        Args:
            tokens: Number of tokens to acquire

        Raises:
            RateLimitExceeded: If not enough tokens available
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.rate))
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                logger.debug(
                    f"Rate limit: acquired {tokens} tokens, "
                    f"{self.tokens:.2f} remaining"
                )
            else:
                # Calculate wait time needed
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

                logger.warning(
                    f"Rate limit exceeded: need {tokens_needed:.2f} more tokens, "
                    f"wait {wait_time:.2f}s"
                )

                # Wait for tokens to become available
                await asyncio.sleep(wait_time)

                # Retry acquisition
                self.tokens = min(self.capacity, self.tokens + (wait_time * self.rate))
                self.last_update = time.time()
                self.tokens -= tokens

    def get_state(self) -> dict:
        """Get current rate limiter state"""
        # Update tokens before reporting state
        now = time.time()
        elapsed = now - self.last_update
        current_tokens = min(self.capacity, self.tokens + (elapsed * self.rate))

        return {
            "available_tokens": current_tokens,
            "capacity": self.capacity,
            "rate": self.rate,
            "utilization": 1.0 - (current_tokens / self.capacity),
        }


class SlidingWindowRateLimiter(RateLimiter):
    """
    Sliding window rate limiter implementation

    More accurate than fixed window but slightly more memory intensive.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.window_size = 1.0  # 1 second window
        self.max_requests = int(self.config.requests_per_second)
        self.requests: list[float] = []
        self.lock = asyncio.Lock()

        logger.info(
            f"Sliding window rate limiter initialized: "
            f"max_requests={self.max_requests}/s"
        )

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire permission for requests

        Args:
            tokens: Number of requests (typically 1)

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        async with self.lock:
            now = time.time()

            # Remove requests outside the window
            cutoff = now - self.window_size
            self.requests = [req for req in self.requests if req > cutoff]

            if len(self.requests) + tokens <= self.max_requests:
                # Add new requests
                for _ in range(tokens):
                    self.requests.append(now)

                logger.debug(
                    f"Rate limit: {len(self.requests)}/{self.max_requests} "
                    f"requests in window"
                )
            else:
                # Calculate wait time
                oldest_request = self.requests[0] if self.requests else now
                wait_time = (oldest_request + self.window_size) - now

                logger.warning(
                    f"Rate limit exceeded: {len(self.requests)}/{self.max_requests}, "
                    f"wait {wait_time:.2f}s"
                )

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Retry acquisition
                await self.acquire(tokens)

    def get_state(self) -> dict:
        """Get current rate limiter state"""
        now = time.time()
        cutoff = now - self.window_size
        active_requests = [req for req in self.requests if req > cutoff]

        return {
            "active_requests": len(active_requests),
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "utilization": len(active_requests) / self.max_requests,
        }
