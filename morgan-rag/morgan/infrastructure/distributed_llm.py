"""
Distributed LLM Client for Multi-Host Setup

Manages LLM requests across multiple hosts with:
- Load balancing (round-robin, random, least-loaded)
- Automatic failover
- Health monitoring
- Performance tracking

This enables Morgan JARVIS to run across separate GPU hosts.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import httpx
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for distributed setup"""

    ROUND_ROBIN = "round_robin"  # Cycle through endpoints
    RANDOM = "random"  # Random selection
    LEAST_LOADED = "least_loaded"  # Lowest avg response time


@dataclass
class LLMEndpoint:
    """
    LLM endpoint configuration and statistics.

    Tracks health, performance, and error metrics for each endpoint.
    """

    url: str  # Endpoint URL
    model: str  # Model name
    healthy: bool = True  # Health status
    response_times: List[float] = field(default_factory=list)  # Response times (s)
    error_count: int = 0  # Consecutive errors
    total_requests: int = 0  # Total requests served
    total_errors: int = 0  # Total errors encountered
    last_health_check: Optional[float] = None  # Last health check timestamp
    last_request: Optional[float] = None  # Last request timestamp

    @property
    def average_response_time(self) -> float:
        """Calculate average response time from last 10 requests"""
        if not self.response_times:
            return 0.0
        recent = self.response_times[-10:]
        return sum(recent) / len(recent)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_errors) / self.total_requests

    def mark_success(self, response_time: float):
        """Mark successful request"""
        self.response_times.append(response_time)
        if len(self.response_times) > 100:  # Keep last 100
            self.response_times = self.response_times[-100:]
        self.total_requests += 1
        self.error_count = 0  # Reset consecutive errors
        self.last_request = time.time()
        self.healthy = True

    def mark_error(self):
        """Mark failed request"""
        self.total_requests += 1
        self.total_errors += 1
        self.error_count += 1
        self.last_request = time.time()

        # Mark unhealthy after 3 consecutive errors
        if self.error_count >= 3:
            self.healthy = False
            logger.warning(
                f"Endpoint {self.url} marked unhealthy after "
                f"{self.error_count} consecutive errors"
            )


class DistributedLLMClient:
    """
    Distributed LLM client for multi-host JARVIS setup.

    Manages LLM inference across multiple hosts with load balancing,
    failover, and health monitoring.

    Example:
        >>> client = DistributedLLMClient(
        ...     endpoints=[
        ...         "http://host1:11434/v1",
        ...         "http://host2:11434/v1"
        ...     ],
        ...     model="qwen2.5:32b-instruct-q4_K_M",
        ...     strategy="round_robin"
        ... )
        >>>
        >>> # Generate response
        >>> response = await client.generate(
        ...     prompt="What is Python?",
        ...     temperature=0.7
        ... )
        >>>
        >>> # Health check
        >>> await client.health_check()
        >>> stats = client.get_stats()
    """

    def __init__(
        self,
        endpoints: List[str],
        model: str,
        strategy: str = "round_robin",
        api_key: str = "ollama",
        timeout: float = 60.0,
        health_check_interval: int = 60,
    ):
        """
        Initialize distributed LLM client.

        Args:
            endpoints: List of LLM endpoint URLs (e.g., ["http://host1:11434/v1"])
            model: Model name (must be available on all endpoints)
            strategy: Load balancing strategy (round_robin, random, least_loaded)
            api_key: API key for endpoints (default: "ollama" for Ollama)
            timeout: Request timeout in seconds
            health_check_interval: Health check interval in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        if not endpoints:
            raise ValueError("At least one endpoint required")

        self.endpoints = [LLMEndpoint(url=url, model=model) for url in endpoints]
        self.model = model
        self.strategy = LoadBalancingStrategy(strategy)
        self.api_key = api_key
        self.timeout = timeout
        self.health_check_interval = health_check_interval

        self.current_index = 0  # For round-robin
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info(
            f"Initialized DistributedLLMClient with {len(self.endpoints)} endpoints"
        )
        logger.info(f"Strategy: {self.strategy.value}, Model: {model}")
        for i, endpoint in enumerate(self.endpoints):
            logger.info(f"  Endpoint {i+1}: {endpoint.url}")

    def start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("Started background health monitoring")

    def stop_health_monitoring(self):
        """Stop background health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped background health monitoring")

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")

    def _select_endpoint(self) -> LLMEndpoint:
        """
        Select endpoint based on load balancing strategy.

        Returns:
            Selected LLMEndpoint

        Raises:
            RuntimeError: If no healthy endpoints available
        """
        healthy_endpoints = [e for e in self.endpoints if e.healthy]

        if not healthy_endpoints:
            logger.error("No healthy endpoints available!")
            # Fallback: Try to use any endpoint
            if self.endpoints:
                logger.warning("Using unhealthy endpoint as fallback")
                return self.endpoints[0]
            else:
                raise RuntimeError("No endpoints available")

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round-robin through healthy endpoints
            endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
            self.current_index += 1
            return endpoint

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_endpoints)

        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select endpoint with lowest average response time
            return min(healthy_endpoints, key=lambda e: e.average_response_time)

        return healthy_endpoints[0]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        max_retries: int = 3,
    ) -> Any:
        """
        Generate response using distributed LLMs.

        Automatically retries with different endpoints on failure.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Enable streaming response
            max_retries: Maximum retry attempts

        Returns:
            Generated text (str) if stream=False, AsyncIterator if stream=True

        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            endpoint = self._select_endpoint()

            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries}: "
                    f"Using endpoint {endpoint.url}"
                )

                # Create client
                client = AsyncOpenAI(
                    base_url=endpoint.url,
                    api_key=self.api_key,
                    timeout=httpx.Timeout(self.timeout),
                )

                # Make request
                start_time = time.time()

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )

                elapsed = time.time() - start_time

                # Track success
                endpoint.mark_success(elapsed)

                logger.info(f"✓ Success in {elapsed:.2f}s using {endpoint.url}")

                # Return response
                if stream:
                    return response
                else:
                    return response.choices[0].message.content

            except Exception as e:
                last_error = e
                endpoint.mark_error()

                logger.error(
                    f"✗ Error with endpoint {endpoint.url} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt == max_retries - 1:
                    # Last attempt failed
                    raise RuntimeError(
                        f"All {max_retries} attempts failed. Last error: {e}"
                    ) from last_error

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2**attempt)
                continue

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all endpoints.

        Returns:
            Dict mapping endpoint URL to health status
        """
        logger.info("Running health check on all endpoints...")

        async def check_endpoint(endpoint: LLMEndpoint) -> bool:
            """Check single endpoint"""
            try:
                client = AsyncOpenAI(
                    base_url=endpoint.url,
                    api_key=self.api_key,
                    timeout=httpx.Timeout(5.0),
                )

                # Simple test query
                await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )

                endpoint.healthy = True
                endpoint.error_count = 0
                endpoint.last_health_check = time.time()

                logger.info(f"✓ {endpoint.url} is healthy")
                return True

            except Exception as e:
                endpoint.healthy = False
                endpoint.last_health_check = time.time()

                logger.error(f"✗ {endpoint.url} is unhealthy: {e}")
                return False

        # Check all endpoints in parallel
        results = await asyncio.gather(
            *[check_endpoint(endpoint) for endpoint in self.endpoints]
        )

        return {
            endpoint.url: healthy for endpoint, healthy in zip(self.endpoints, results)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all endpoints.

        Returns:
            Dict with comprehensive statistics
        """
        healthy_count = sum(1 for e in self.endpoints if e.healthy)

        return {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": healthy_count,
            "unhealthy_endpoints": len(self.endpoints) - healthy_count,
            "strategy": self.strategy.value,
            "model": self.model,
            "endpoints": [
                {
                    "url": e.url,
                    "healthy": e.healthy,
                    "avg_response_time": f"{e.average_response_time:.3f}s",
                    "success_rate": f"{e.success_rate * 100:.1f}%",
                    "total_requests": e.total_requests,
                    "total_errors": e.total_errors,
                    "error_count": e.error_count,
                    "last_request": (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(e.last_request)
                        )
                        if e.last_request
                        else "Never"
                    ),
                }
                for e in self.endpoints
            ],
        }

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DistributedLLMClient("
            f"endpoints={len(self.endpoints)}, "
            f"healthy={sum(1 for e in self.endpoints if e.healthy)}, "
            f"strategy={self.strategy.value}, "
            f"model={self.model})"
        )


# Global instance for singleton pattern
_client: Optional[DistributedLLMClient] = None


def get_distributed_llm_client(
    endpoints: Optional[List[str]] = None, model: Optional[str] = None, **kwargs
) -> DistributedLLMClient:
    """
    Get global distributed LLM client instance (singleton).

    Args:
        endpoints: List of endpoint URLs (required on first call)
        model: Model name (required on first call)
        **kwargs: Additional arguments for DistributedLLMClient

    Returns:
        DistributedLLMClient instance
    """
    global _client

    if _client is None:
        if endpoints is None or model is None:
            raise ValueError("endpoints and model required for first initialization")

        _client = DistributedLLMClient(endpoints=endpoints, model=model, **kwargs)

        # Start health monitoring
        _client.start_health_monitoring()

    return _client
