"""
Circuit Breaker pattern implementation for fault tolerance
"""
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional, TypeVar, Generic
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: float = 60.0  # Seconds to wait before half-open
    expected_exception: type = Exception  # Exception type to track


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker(Generic[T]):
    """
    Circuit Breaker pattern implementation

    Prevents cascading failures by stopping requests to failing services
    and allowing them time to recover.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = asyncio.Lock()

        logger.info(
            f"Circuit breaker initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout}s"
        )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. "
                        f"Retry after {self._time_until_reset():.1f}s"
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except self.config.expected_exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self.lock:
            self.failure_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                logger.debug(
                    f"Circuit breaker success in HALF_OPEN: "
                    f"{self.success_count}/{self.config.success_threshold}"
                )

                if self.success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker failure: "
                f"{self.failure_count}/{self.config.failure_threshold}"
            )

            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning back to OPEN from HALF_OPEN")
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0

            elif self.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"Circuit breaker transitioning to OPEN after "
                    f"{self.failure_count} failures"
                )
                self.state = CircuitBreakerState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.config.timeout

    def _time_until_reset(self) -> float:
        """Calculate time until circuit can be reset"""
        if self.last_failure_time is None:
            return 0.0

        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.timeout - elapsed)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")

    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_reset": self._time_until_reset() if self.state == CircuitBreakerState.OPEN else 0.0
        }
