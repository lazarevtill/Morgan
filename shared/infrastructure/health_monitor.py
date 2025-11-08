"""
Health monitoring for services and endpoints
"""
import asyncio
import time
from enum import Enum
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    response_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "response_time": self.response_time,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class HealthMetrics:
    """Health metrics over time"""
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    avg_response_time: float = 0.0
    last_check_time: Optional[float] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    uptime_percentage: float = 100.0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "failed_checks": self.failed_checks,
            "avg_response_time": self.avg_response_time,
            "last_check_time": self.last_check_time,
            "last_status": self.last_status.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "uptime_percentage": self.uptime_percentage
        }


class HealthMonitor:
    """
    Monitor health of services and endpoints

    Tracks health metrics, response times, and failure patterns.
    """

    def __init__(
        self,
        name: str,
        check_interval: float = 30.0,
        degraded_threshold: int = 2,
        unhealthy_threshold: int = 5
    ):
        """
        Initialize health monitor

        Args:
            name: Name of the monitored service
            check_interval: Seconds between health checks
            degraded_threshold: Consecutive failures before DEGRADED
            unhealthy_threshold: Consecutive failures before UNHEALTHY
        """
        self.name = name
        self.check_interval = check_interval
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold

        self.metrics = HealthMetrics()
        self.check_history: list[HealthCheckResult] = []
        self.max_history = 100  # Keep last 100 checks

        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

        logger.info(
            f"Health monitor initialized for '{name}': "
            f"interval={check_interval}s"
        )

    async def check_health(
        self,
        health_func: Callable[[], Any]
    ) -> HealthCheckResult:
        """
        Perform a single health check

        Args:
            health_func: Async function that performs health check

        Returns:
            HealthCheckResult
        """
        start_time = time.time()
        result = HealthCheckResult(status=HealthStatus.UNKNOWN)

        try:
            await asyncio.wait_for(health_func(), timeout=10.0)
            result.status = HealthStatus.HEALTHY
            result.response_time = time.time() - start_time

            async with self.lock:
                self.metrics.consecutive_failures = 0
                self.metrics.consecutive_successes += 1

        except asyncio.TimeoutError:
            result.status = HealthStatus.UNHEALTHY
            result.error = "Health check timeout"
            result.response_time = time.time() - start_time

            async with self.lock:
                self.metrics.consecutive_successes = 0
                self.metrics.consecutive_failures += 1

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error = str(e)
            result.response_time = time.time() - start_time

            async with self.lock:
                self.metrics.consecutive_successes = 0
                self.metrics.consecutive_failures += 1

            logger.error(f"Health check failed for '{self.name}': {e}")

        # Update metrics
        await self._update_metrics(result)

        return result

    async def _update_metrics(self, result: HealthCheckResult):
        """Update health metrics with new result"""
        async with self.lock:
            self.metrics.total_checks += 1
            self.metrics.last_check_time = result.timestamp

            if result.status == HealthStatus.HEALTHY:
                self.metrics.successful_checks += 1
            else:
                self.metrics.failed_checks += 1

            # Update average response time
            if result.response_time is not None:
                if self.metrics.avg_response_time == 0:
                    self.metrics.avg_response_time = result.response_time
                else:
                    # Exponential moving average
                    alpha = 0.3
                    self.metrics.avg_response_time = (
                        alpha * result.response_time +
                        (1 - alpha) * self.metrics.avg_response_time
                    )

            # Determine overall status
            if self.metrics.consecutive_failures >= self.unhealthy_threshold:
                self.metrics.last_status = HealthStatus.UNHEALTHY
            elif self.metrics.consecutive_failures >= self.degraded_threshold:
                self.metrics.last_status = HealthStatus.DEGRADED
            elif result.status == HealthStatus.HEALTHY:
                self.metrics.last_status = HealthStatus.HEALTHY
            else:
                self.metrics.last_status = result.status

            # Calculate uptime percentage
            if self.metrics.total_checks > 0:
                self.metrics.uptime_percentage = (
                    self.metrics.successful_checks / self.metrics.total_checks * 100
                )

            # Add to history
            self.check_history.append(result)
            if len(self.check_history) > self.max_history:
                self.check_history.pop(0)

    async def start_monitoring(
        self,
        health_func: Callable[[], Any]
    ):
        """
        Start continuous health monitoring

        Args:
            health_func: Async function that performs health check
        """
        if self.is_monitoring:
            logger.warning(f"Health monitor '{self.name}' already running")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(health_func)
        )
        logger.info(f"Started health monitoring for '{self.name}'")

    async def _monitor_loop(self, health_func: Callable[[], Any]):
        """Internal monitoring loop"""
        while self.is_monitoring:
            try:
                await self.check_health(health_func)
            except Exception as e:
                logger.error(f"Error in health monitor loop for '{self.name}': {e}")

            await asyncio.sleep(self.check_interval)

    async def stop_monitoring(self):
        """Stop continuous health monitoring"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped health monitoring for '{self.name}'")

    def get_status(self) -> HealthStatus:
        """Get current health status"""
        return self.metrics.last_status

    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.to_dict()

    def get_recent_history(self, count: int = 10) -> list[dict]:
        """Get recent health check history"""
        recent = self.check_history[-count:]
        return [result.to_dict() for result in recent]

    def get_summary(self) -> dict:
        """Get comprehensive health summary"""
        return {
            "name": self.name,
            "status": self.get_status().value,
            "metrics": self.get_metrics(),
            "recent_checks": self.get_recent_history(5)
        }
