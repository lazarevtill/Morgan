"""
Base classes and interfaces for learning modules.

Provides abstract base classes that all learning modules
must implement for consistency and testability.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Generic, Optional, TypeVar

from morgan.learning.exceptions import CircuitBreakerOpenError, LearningError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class HealthStatus:
    """Health status of a learning module."""

    healthy: bool
    message: str
    details: Dict[str, Any]
    last_check: float  # Timestamp


class BaseLearningModule(ABC):
    """
    Base class for all learning modules.

    All modules follow the same lifecycle:
    1. initialize() - Load resources, models, connect to services
    2. process() - Perform learning tasks
    3. health_check() - Monitor module health
    4. cleanup() - Release resources, close connections

    Modules are designed to be:
    - Async-first for performance
    - Independent and loosely coupled
    - Testable via dependency injection
    - Resilient with proper error handling
    - Observable with health checks
    """

    def __init__(self, name: str, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the module with a name.

        Args:
            name: Module name for logging and identification
            correlation_id: Optional correlation ID for request tracing
        """
        self.name = name
        self.correlation_id = correlation_id
        self._initialized = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"morgan.learning.{name}")

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the module.

        Load any required resources, models, or connections.
        This is called once before the module is used.

        Raises:
            LearningError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup module resources.

        Release any held resources, close connections, etc.
        This is called when the module is being shut down.
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Check module health.

        Returns:
            HealthStatus indicating module health

        Raises:
            LearningError: If health check fails critically
        """
        pass

    async def ensure_initialized(self) -> None:
        """Ensure the module is initialized before use."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    self._logger.info(
                        f"Initializing module {self.name}",
                        extra={"correlation_id": self.correlation_id},
                    )
                    await self.initialize()
                    self._initialized = True
                    self._logger.info(
                        f"Module {self.name} initialized successfully",
                        extra={"correlation_id": self.correlation_id},
                    )

    @property
    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[BaseLearningModule]:
        """
        Context manager for module lifecycle.

        Usage:
            async with module.lifecycle():
                result = await module.process(data)
        """
        try:
            await self.ensure_initialized()
            yield self
        finally:
            if self._initialized:
                await self.cleanup()
                self._initialized = False

    def _log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        **extra: Any,
    ) -> None:
        """Log error with correlation ID."""
        log_extra = {"correlation_id": self.correlation_id, **extra}
        if error:
            self._logger.error(
                f"{message}: {str(error)}",
                exc_info=error,
                extra=log_extra,
            )
        else:
            self._logger.error(message, extra=log_extra)

    def _log_info(self, message: str, **extra: Any) -> None:
        """Log info with correlation ID."""
        log_extra = {"correlation_id": self.correlation_id, **extra}
        self._logger.info(message, extra=log_extra)

    def _log_warning(self, message: str, **extra: Any) -> None:
        """Log warning with correlation ID."""
        log_extra = {"correlation_id": self.correlation_id, **extra}
        self._logger.warning(message, extra=log_extra)


class AsyncCache(Generic[T]):
    """
    Thread-safe async cache with TTL support.

    Used by modules that need to cache results for performance.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
    ) -> None:
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self._cache: Dict[str, tuple[T, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry = self._cache[key]
            current_time = asyncio.get_event_loop().time()

            if current_time > expiry:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return value

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            expiry = asyncio.get_event_loop().time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expiry)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry."""
        async with self._lock:
            self._cache.pop(key, None)

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Pattern to match (simple substring match)

        Returns:
            Number of keys invalidated
        """
        async with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by failing fast when a service
    is experiencing issues.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail immediately
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = LearningError,
        name: str = "circuit_breaker",
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            name: Circuit breaker name for logging
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_exception = expected_exception
        self._name = name
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"morgan.learning.circuit_breaker.{name}")

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception if call fails
        """
        async with self._lock:
            if self._state == "open":
                if self._should_attempt_reset():
                    self._state = "half-open"
                    self._logger.info(f"Circuit breaker {self._name} entering half-open state")
                else:
                    retry_after = (
                        self._recovery_timeout
                        - (asyncio.get_event_loop().time() - (self._last_failure_time or 0))
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self._name} is open",
                        service=self._name,
                        retry_after=retry_after,
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self._expected_exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = asyncio.get_event_loop().time() - self._last_failure_time
        return elapsed > self._recovery_timeout

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._success_count += 1
            self._failure_count = 0

            if self._state == "half-open":
                # Require a few successes before closing
                if self._success_count >= 3:
                    self._state = "closed"
                    self._logger.info(f"Circuit breaker {self._name} closed after recovery")
                    self._success_count = 0

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = asyncio.get_event_loop().time()

            if self._failure_count >= self._failure_threshold:
                if self._state != "open":
                    self._state = "open"
                    self._logger.error(
                        f"Circuit breaker {self._name} opened after {self._failure_count} failures"
                    )

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def is_healthy(self) -> bool:
        """Check if circuit breaker is healthy (closed)."""
        return self._state == "closed"

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._state = "closed"
            self._logger.info(f"Circuit breaker {self._name} manually reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self._failure_threshold,
            "last_failure_time": self._last_failure_time,
        }


class RateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    Useful for limiting API calls or expensive operations.
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        name: str = "rate_limiter",
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            capacity: Maximum token capacity
            name: Rate limiter name
        """
        self._rate = rate
        self._capacity = capacity
        self._name = name
        self._tokens = float(capacity)
        self._last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens acquired, False if timeout

        Raises:
            ValueError: If tokens > capacity
        """
        if tokens > self._capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens, capacity is {self._capacity}")

        start_time = asyncio.get_event_loop().time()

        while True:
            async with self._lock:
                self._refill_tokens()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return False

            # Wait a bit before retrying
            await asyncio.sleep(0.1)

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + (elapsed * self._rate))
        self._last_update = now

    async def get_available_tokens(self) -> float:
        """Get current available tokens."""
        async with self._lock:
            self._refill_tokens()
            return self._tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "name": self._name,
            "rate": self._rate,
            "capacity": self._capacity,
            "tokens": self._tokens,
        }
