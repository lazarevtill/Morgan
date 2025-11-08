"""
Enhanced HTTP client with connection pooling, circuit breaker, rate limiting, and retry logic
"""
import asyncio
import random
import time
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin
import logging

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from .rate_limiter import RateLimiter, TokenBucketRateLimiter, RateLimitConfig
from .health_monitor import HealthMonitor, HealthStatus
from ..utils.errors import ServiceError, ErrorCode

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """HTTP connection pool configuration"""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0  # seconds


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


@dataclass
class TimeoutConfig:
    """Timeout configuration"""
    connect: float = 5.0  # seconds
    read: float = 30.0  # seconds
    write: float = 10.0  # seconds
    pool: float = 5.0  # seconds


class EnhancedHTTPClient:
    """
    Production-quality HTTP client with:
    - Connection pooling
    - Circuit breaker pattern
    - Rate limiting
    - Retry with exponential backoff and jitter
    - Health monitoring
    - Comprehensive error handling
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        pool_config: Optional[ConnectionPoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        enable_health_monitoring: bool = True,
        health_check_interval: float = 30.0
    ):
        """
        Initialize enhanced HTTP client

        Args:
            service_name: Name of the service
            base_url: Base URL for the service
            pool_config: Connection pool configuration
            retry_config: Retry configuration
            timeout_config: Timeout configuration
            circuit_breaker_config: Circuit breaker configuration
            rate_limit_config: Rate limiter configuration
            enable_health_monitoring: Enable health monitoring
            health_check_interval: Seconds between health checks
        """
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')

        # Configurations
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or TimeoutConfig()

        # Components
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.rate_limiter: Optional[RateLimiter] = None
        if rate_limit_config:
            self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)

        self.health_monitor = HealthMonitor(
            name=service_name,
            check_interval=health_check_interval
        ) if enable_health_monitoring else None

        # HTTP client (initialized in start())
        self.client: Optional[httpx.AsyncClient] = None

        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0

        logger.info(
            f"Enhanced HTTP client initialized for '{service_name}': "
            f"base_url={base_url}, "
            f"pool_size={self.pool_config.max_connections}, "
            f"rate_limiting={'enabled' if self.rate_limiter else 'disabled'}"
        )

    async def start(self):
        """Initialize HTTP client and start health monitoring"""
        # Create HTTP client with connection pooling
        limits = httpx.Limits(
            max_connections=self.pool_config.max_connections,
            max_keepalive_connections=self.pool_config.max_keepalive_connections,
            keepalive_expiry=self.pool_config.keepalive_expiry
        )

        timeout = httpx.Timeout(
            connect=self.timeout_config.connect,
            read=self.timeout_config.read,
            write=self.timeout_config.write,
            pool=self.timeout_config.pool
        )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            limits=limits,
            timeout=timeout,
            follow_redirects=True
        )

        # Start health monitoring
        if self.health_monitor:
            await self.health_monitor.start_monitoring(self._health_check)

        logger.info(f"HTTP client started for '{self.service_name}'")

    async def stop(self):
        """Stop HTTP client and health monitoring"""
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()

        if self.client:
            await self.client.aclose()
            self.client = None

        logger.info(f"HTTP client stopped for '{self.service_name}'")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with all production features

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response

        Raises:
            ServiceError: On request failure
        """
        if not self.client:
            raise ServiceError(
                f"HTTP client not started for '{self.service_name}'",
                ErrorCode.SERVICE_ERROR
            )

        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Use circuit breaker
        try:
            return await self.circuit_breaker.call(
                self._request_with_retry,
                method,
                endpoint,
                **kwargs
            )
        except CircuitBreakerError as e:
            raise ServiceError(
                str(e),
                ErrorCode.SERVICE_UNAVAILABLE,
                {
                    "service": self.service_name,
                    "circuit_breaker_state": self.circuit_breaker.get_state()
                }
            )

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """Make request with retry logic"""
        url = urljoin(self.base_url, endpoint)
        last_exception = None

        for attempt in range(self.retry_config.max_retries):
            start_time = time.time()

            try:
                response = await self.client.request(method, url, **kwargs)

                # Track metrics
                response_time = time.time() - start_time
                self.request_count += 1
                self.total_response_time += response_time

                logger.debug(
                    f"HTTP {method} {url}: "
                    f"status={response.status_code}, "
                    f"time={response_time:.3f}s"
                )

                # Raise for HTTP errors
                response.raise_for_status()

                return response

            except httpx.HTTPStatusError as e:
                last_exception = e
                self.error_count += 1

                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise ServiceError(
                        f"HTTP {e.response.status_code}: {e.response.text}",
                        ErrorCode.VALIDATION_ERROR if e.response.status_code == 400
                        else ErrorCode.SERVICE_ERROR,
                        {
                            "status_code": e.response.status_code,
                            "url": url,
                            "service": self.service_name
                        }
                    )

                # Retry on server errors (5xx)
                if attempt < self.retry_config.max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"HTTP {method} {url} failed with {e.response.status_code}, "
                        f"retry {attempt + 1}/{self.retry_config.max_retries} "
                        f"in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise ServiceError(
                        f"HTTP {e.response.status_code} after {self.retry_config.max_retries} retries",
                        ErrorCode.SERVICE_ERROR,
                        {
                            "status_code": e.response.status_code,
                            "url": url,
                            "attempts": attempt + 1,
                            "service": self.service_name
                        }
                    )

            except httpx.TimeoutException as e:
                last_exception = e
                self.error_count += 1

                if attempt < self.retry_config.max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"HTTP {method} {url} timeout, "
                        f"retry {attempt + 1}/{self.retry_config.max_retries} "
                        f"in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise ServiceError(
                        f"Timeout after {self.retry_config.max_retries} retries",
                        ErrorCode.SERVICE_TIMEOUT,
                        {
                            "url": url,
                            "attempts": attempt + 1,
                            "service": self.service_name
                        }
                    )

            except httpx.ConnectError as e:
                last_exception = e
                self.error_count += 1

                if attempt < self.retry_config.max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Connection error to {url}, "
                        f"retry {attempt + 1}/{self.retry_config.max_retries} "
                        f"in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise ServiceError(
                        f"Connection failed after {self.retry_config.max_retries} retries",
                        ErrorCode.CONNECTION_ERROR,
                        {
                            "url": url,
                            "attempts": attempt + 1,
                            "service": self.service_name
                        }
                    )

            except Exception as e:
                last_exception = e
                self.error_count += 1
                logger.error(f"Unexpected error in HTTP request: {e}")

                raise ServiceError(
                    f"Unexpected error: {str(e)}",
                    ErrorCode.INTERNAL_ERROR,
                    {
                        "url": url,
                        "error_type": type(e).__name__,
                        "service": self.service_name
                    }
                )

        # Should not reach here, but handle just in case
        raise ServiceError(
            f"Max retries exceeded: {last_exception}",
            ErrorCode.SERVICE_ERROR,
            {"service": self.service_name}
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff delay with exponential backoff and jitter

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.retry_config.base_delay * (
            self.retry_config.exponential_base ** attempt
        )

        # Cap at max_delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            jitter = random.uniform(0, delay * 0.3)  # Up to 30% jitter
            delay += jitter

        return delay

    async def _health_check(self):
        """Internal health check"""
        try:
            response = await self.client.get("/health", timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            raise ServiceError(
                f"Health check failed: {e}",
                ErrorCode.SERVICE_ERROR
            )

    # Convenience methods
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request"""
        return await self.request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request"""
        return await self.request("POST", endpoint, **kwargs)

    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request"""
        return await self.request("PUT", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request"""
        return await self.request("DELETE", endpoint, **kwargs)

    async def patch(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PATCH request"""
        return await self.request("PATCH", endpoint, **kwargs)

    # Status and metrics
    def get_status(self) -> dict:
        """Get comprehensive client status"""
        status = {
            "service_name": self.service_name,
            "base_url": self.base_url,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.request_count
                if self.request_count > 0 else 0.0
            ),
            "avg_response_time": (
                self.total_response_time / self.request_count
                if self.request_count > 0 else 0.0
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
        }

        if self.rate_limiter:
            status["rate_limiter"] = self.rate_limiter.get_state()

        if self.health_monitor:
            status["health"] = self.health_monitor.get_summary()

        return status

    async def check_health(self) -> bool:
        """Perform synchronous health check"""
        try:
            await self._health_check()
            return True
        except Exception:
            return False

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.circuit_breaker.reset()
        logger.info(f"Circuit breaker reset for '{self.service_name}'")
