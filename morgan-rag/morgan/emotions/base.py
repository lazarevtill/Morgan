"""
Base classes and interfaces for emotion detection modules.

Provides abstract base classes that all emotion detection modules
must implement for consistency and testability.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator, Generic, Optional, TypeVar

from morgan.emotions.exceptions import EmotionDetectionError


T = TypeVar("T")


class EmotionModule(ABC):
    """
    Base class for all emotion detection modules.

    All modules follow the same lifecycle:
    1. initialize() - Load resources, models, connect to services
    2. process() - Perform emotion detection tasks
    3. cleanup() - Release resources, close connections

    Modules are designed to be:
    - Async-first for performance
    - Independent and loosely coupled
    - Testable via dependency injection
    - Resilient with proper error handling
    """

    def __init__(self, name: str) -> None:
        """Initialize the module with a name."""
        self.name = name
        self._initialized = False
        self._lock = asyncio.Lock()

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the module.

        Load any required resources, models, or connections.
        This is called once before the module is used.
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

    async def ensure_initialized(self) -> None:
        """Ensure the module is initialized before use."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self.initialize()
                    self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[EmotionModule]:
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


class AsyncCache(Generic[T]):
    """
    Thread-safe async cache with TTL support.

    Used by modules that need to cache results for performance.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self._cache: dict[str, tuple[T, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if asyncio.get_event_loop().time() > expiry:
                del self._cache[key]
                return None

            return value

    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Simple eviction: remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            expiry = asyncio.get_event_loop().time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expiry)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry."""
        async with self._lock:
            self._cache.pop(key, None)


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by failing fast when a service
    is experiencing issues.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = EmotionDetectionError,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_exception = expected_exception
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

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
            EmotionDetectionError: If circuit is open
        """
        async with self._lock:
            if self._state == "open":
                if self._should_attempt_reset():
                    self._state = "half-open"
                else:
                    raise EmotionDetectionError(
                        f"Circuit breaker is open, last failure: {self._last_failure_time}",
                        recoverable=True,
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
        return (
            asyncio.get_event_loop().time() - self._last_failure_time
            > self._recovery_timeout
        )

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._failure_count = 0
            self._state = "closed"

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = asyncio.get_event_loop().time()

            if self._failure_count >= self._failure_threshold:
                self._state = "open"

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._failure_count = 0
            self._last_failure_time = None
            self._state = "closed"
