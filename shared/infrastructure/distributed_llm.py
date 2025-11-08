"""
Distributed LLM client for multi-host deployments

Provides:
- Load balancing across multiple LLM hosts
- Automatic failover
- Circuit breaker per host
- Health monitoring
- Request routing
"""
import asyncio
import random
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .http_client import EnhancedHTTPClient, ConnectionPoolConfig, RetryConfig, TimeoutConfig
from .circuit_breaker import CircuitBreakerConfig
from .rate_limiter import RateLimitConfig
from .health_monitor import HealthStatus
from ..utils.errors import ServiceError, ErrorCode

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_RANDOM = "weighted_random"


@dataclass
class LLMHost:
    """LLM host configuration"""
    name: str
    url: str
    weight: float = 1.0  # For weighted load balancing
    enabled: bool = True


class DistributedLLMClient:
    """
    Distributed LLM client with load balancing and failover

    Features:
    - Multiple host support with load balancing
    - Per-host circuit breakers
    - Automatic failover on host failure
    - Health monitoring for all hosts
    - Request routing based on strategy
    - Comprehensive metrics
    """

    def __init__(
        self,
        hosts: List[LLMHost],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        pool_config: Optional[ConnectionPoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        enable_health_monitoring: bool = True
    ):
        """
        Initialize distributed LLM client

        Args:
            hosts: List of LLM host configurations
            strategy: Load balancing strategy
            pool_config: Connection pool configuration
            retry_config: Retry configuration
            timeout_config: Timeout configuration
            circuit_breaker_config: Circuit breaker configuration
            rate_limit_config: Rate limiter configuration
            enable_health_monitoring: Enable health monitoring
        """
        self.hosts = hosts
        self.strategy = strategy
        self.enabled_hosts = [h for h in hosts if h.enabled]

        # Create HTTP clients for each host
        self.clients: Dict[str, EnhancedHTTPClient] = {}
        for host in hosts:
            client = EnhancedHTTPClient(
                service_name=f"llm_{host.name}",
                base_url=host.url,
                pool_config=pool_config,
                retry_config=retry_config,
                timeout_config=timeout_config,
                circuit_breaker_config=circuit_breaker_config,
                rate_limit_config=rate_limit_config,
                enable_health_monitoring=enable_health_monitoring
            )
            self.clients[host.name] = client

        # Load balancing state
        self.current_index = 0
        self.lock = asyncio.Lock()

        # Metrics
        self.request_counts: Dict[str, int] = {h.name: 0 for h in hosts}
        self.error_counts: Dict[str, int] = {h.name: 0 for h in hosts}

        logger.info(
            f"Distributed LLM client initialized: "
            f"hosts={len(hosts)}, strategy={strategy.value}"
        )

    async def start(self):
        """Start all host clients"""
        start_tasks = [
            client.start()
            for client in self.clients.values()
        ]
        await asyncio.gather(*start_tasks, return_exceptions=True)
        logger.info("Distributed LLM client started")

    async def stop(self):
        """Stop all host clients"""
        stop_tasks = [
            client.stop()
            for client in self.clients.values()
        ]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("Distributed LLM client stopped")

    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Any:
        """
        Make request with load balancing and failover

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request arguments

        Returns:
            Response from successful host

        Raises:
            ServiceError: If all hosts fail
        """
        # Get available hosts
        available_hosts = await self._get_available_hosts()

        if not available_hosts:
            raise ServiceError(
                "No available LLM hosts",
                ErrorCode.SERVICE_UNAVAILABLE,
                {"total_hosts": len(self.hosts)}
            )

        # Try each available host
        last_error = None

        for host in available_hosts:
            try:
                client = self.clients[host.name]

                # Make request
                response = await client.request(method, endpoint, **kwargs)

                # Track success
                self.request_counts[host.name] += 1

                logger.debug(
                    f"Request succeeded on host '{host.name}': "
                    f"{method} {endpoint}"
                )

                return response

            except Exception as e:
                last_error = e
                self.error_counts[host.name] += 1

                logger.warning(
                    f"Request failed on host '{host.name}': {e}, "
                    f"trying next host"
                )
                continue

        # All hosts failed
        raise ServiceError(
            f"All LLM hosts failed: {last_error}",
            ErrorCode.SERVICE_ERROR,
            {
                "attempted_hosts": [h.name for h in available_hosts],
                "last_error": str(last_error)
            }
        )

    async def _get_available_hosts(self) -> List[LLMHost]:
        """
        Get available hosts based on load balancing strategy

        Returns:
            Ordered list of hosts to try
        """
        # Filter to enabled hosts with healthy circuit breakers
        available = []

        for host in self.enabled_hosts:
            client = self.clients[host.name]
            cb_state = client.circuit_breaker.get_state()

            # Only include hosts with closed or half-open circuit breakers
            if cb_state["state"] in ["closed", "half_open"]:
                available.append(host)

        if not available:
            logger.warning("No hosts with healthy circuit breakers")
            # Fallback to all enabled hosts
            available = self.enabled_hosts

        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin(available)

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_order(available)

        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_order(available)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_order(available)

        else:
            return available

    async def _round_robin(self, hosts: List[LLMHost]) -> List[LLMHost]:
        """Round-robin load balancing"""
        if not hosts:
            return []

        async with self.lock:
            # Start with current index
            start_index = self.current_index % len(hosts)
            self.current_index += 1

        # Rotate list to start at current index
        return hosts[start_index:] + hosts[:start_index]

    def _random_order(self, hosts: List[LLMHost]) -> List[LLMHost]:
        """Random order"""
        shuffled = hosts.copy()
        random.shuffle(shuffled)
        return shuffled

    def _least_loaded_order(self, hosts: List[LLMHost]) -> List[LLMHost]:
        """Order by least loaded (fewest requests)"""
        return sorted(
            hosts,
            key=lambda h: self.request_counts.get(h.name, 0)
        )

    def _weighted_random_order(self, hosts: List[LLMHost]) -> List[LLMHost]:
        """Weighted random selection"""
        if not hosts:
            return []

        # Calculate weights
        weights = [h.weight for h in hosts]
        total_weight = sum(weights)

        if total_weight == 0:
            return hosts

        # Normalize weights
        probabilities = [w / total_weight for w in weights]

        # Select first host randomly with weights
        first_host = random.choices(hosts, weights=probabilities, k=1)[0]

        # Put selected host first, rest in random order
        remaining = [h for h in hosts if h != first_host]
        random.shuffle(remaining)

        return [first_host] + remaining

    # Convenience methods
    async def get(self, endpoint: str, **kwargs) -> Any:
        """Make GET request"""
        return await self.request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> Any:
        """Make POST request"""
        return await self.request("POST", endpoint, **kwargs)

    async def put(self, endpoint: str, **kwargs) -> Any:
        """Make PUT request"""
        return await self.request("PUT", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> Any:
        """Make DELETE request"""
        return await self.request("DELETE", endpoint, **kwargs)

    # Health and metrics
    async def health_check_all(self) -> Dict[str, Any]:
        """Check health of all hosts"""
        health_checks = {}

        for host in self.hosts:
            try:
                client = self.clients[host.name]
                is_healthy = await client.check_health()

                health_checks[host.name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "url": host.url,
                    "enabled": host.enabled,
                    "circuit_breaker": client.circuit_breaker.get_state()
                }
            except Exception as e:
                health_checks[host.name] = {
                    "status": "error",
                    "error": str(e),
                    "url": host.url,
                    "enabled": host.enabled
                }

        return health_checks

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            "total_hosts": len(self.hosts),
            "enabled_hosts": len(self.enabled_hosts),
            "strategy": self.strategy.value,
            "hosts": {}
        }

        for host in self.hosts:
            client = self.clients[host.name]
            client_status = client.get_status()

            metrics["hosts"][host.name] = {
                "url": host.url,
                "weight": host.weight,
                "enabled": host.enabled,
                "request_count": self.request_counts.get(host.name, 0),
                "error_count": self.error_counts.get(host.name, 0),
                "error_rate": (
                    self.error_counts.get(host.name, 0) /
                    max(self.request_counts.get(host.name, 1), 1)
                ),
                "client_status": client_status
            }

        return metrics

    def enable_host(self, host_name: str):
        """Enable a specific host"""
        for host in self.hosts:
            if host.name == host_name:
                host.enabled = True
                self.enabled_hosts = [h for h in self.hosts if h.enabled]
                logger.info(f"Host '{host_name}' enabled")
                return

        logger.warning(f"Host '{host_name}' not found")

    def disable_host(self, host_name: str):
        """Disable a specific host"""
        for host in self.hosts:
            if host.name == host_name:
                host.enabled = False
                self.enabled_hosts = [h for h in self.hosts if h.enabled]
                logger.info(f"Host '{host_name}' disabled")
                return

        logger.warning(f"Host '{host_name}' not found")

    def reset_circuit_breaker(self, host_name: str):
        """Reset circuit breaker for a specific host"""
        if host_name in self.clients:
            client = self.clients[host_name]
            client.reset_circuit_breaker()
            logger.info(f"Circuit breaker reset for host '{host_name}'")
        else:
            logger.warning(f"Host '{host_name}' not found")

    def reset_all_circuit_breakers(self):
        """Reset circuit breakers for all hosts"""
        for client in self.clients.values():
            client.reset_circuit_breaker()

        logger.info("All circuit breakers reset")
