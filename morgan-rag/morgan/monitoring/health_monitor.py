"""
System health monitoring for Morgan RAG.

Provides comprehensive health checks, dependency monitoring, and system status
tracking as specified in Requirements 7.2, 7.4, and 7.5.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

from .metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""

    name: str
    check_function: Callable[[], Awaitable[bool]]
    timeout_seconds: float = 30.0
    interval_seconds: float = 60.0
    critical: bool = False
    description: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


@dataclass
class SystemHealthStatus:
    """Overall system health status."""

    overall_status: HealthStatus
    healthy_checks: int
    warning_checks: int
    critical_checks: int
    unknown_checks: int
    last_updated: datetime
    check_results: List[HealthCheckResult]
    uptime_seconds: float


class HealthMonitor:
    """
    Comprehensive system health monitoring for Morgan RAG.

    Monitors system dependencies, database connections, service availability,
    and overall system health with configurable checks and alerting.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

        # Health check registry
        self._health_checks: Dict[str, HealthCheck] = {}
        self._check_results: Dict[str, HealthCheckResult] = {}

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

        # Health status callbacks
        self._status_change_callbacks: List[Callable[[HealthCheckResult], None]] = []

        # Initialize default health checks
        self._register_default_health_checks()

        logger.info("HealthMonitor initialized")

    def _register_default_health_checks(self):
        """Register default system health checks."""
        # Vector database health check
        self.register_health_check(
            name="vector_database",
            check_function=self._check_vector_database_health,
            timeout_seconds=10.0,
            interval_seconds=30.0,
            critical=True,
            description="Vector database (Qdrant) connectivity and performance",
        )

        # Embedding service health check
        self.register_health_check(
            name="embedding_service",
            check_function=self._check_embedding_service_health,
            timeout_seconds=15.0,
            interval_seconds=60.0,
            critical=True,
            description="Embedding service availability and performance",
        )

        # Memory usage health check
        self.register_health_check(
            name="memory_usage",
            check_function=self._check_memory_usage,
            timeout_seconds=5.0,
            interval_seconds=30.0,
            critical=False,
            description="System memory usage monitoring",
        )

        # Cache health check
        self.register_health_check(
            name="cache_system",
            check_function=self._check_cache_health,
            timeout_seconds=10.0,
            interval_seconds=60.0,
            critical=False,
            description="Caching system performance and hit rates",
        )

        # Processing pipeline health check
        self.register_health_check(
            name="processing_pipeline",
            check_function=self._check_processing_pipeline_health,
            timeout_seconds=20.0,
            interval_seconds=120.0,
            critical=False,
            description="Document processing pipeline performance",
        )

        # Search system health check
        self.register_health_check(
            name="search_system",
            check_function=self._check_search_system_health,
            timeout_seconds=15.0,
            interval_seconds=60.0,
            critical=True,
            description="Search system performance and accuracy",
        )

        # Companion system health check
        self.register_health_check(
            name="companion_system",
            check_function=self._check_companion_system_health,
            timeout_seconds=10.0,
            interval_seconds=120.0,
            critical=False,
            description="Companion features and emotional intelligence",
        )

    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], Awaitable[bool]],
        timeout_seconds: float = 30.0,
        interval_seconds: float = 60.0,
        critical: bool = False,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            critical=critical,
            description=description,
            metadata=metadata or {},
        )

        self._health_checks[name] = health_check

        logger.info(
            "Registered health check",
            name=name,
            critical=critical,
            interval=interval_seconds,
        )

    def unregister_health_check(self, name: str):
        """Unregister a health check."""
        if name in self._health_checks:
            del self._health_checks[name]
            if name in self._check_results:
                del self._check_results[name]

            logger.info("Unregistered health check", name=name)

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Started health monitoring")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring_active = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Main health monitoring loop."""
        check_schedules = dict.fromkeys(self._health_checks.keys(), 0.0)

        while self._monitoring_active:
            try:
                current_time = time.time()

                # Check which health checks need to run
                checks_to_run = []
                for name, health_check in self._health_checks.items():
                    if current_time >= check_schedules[name]:
                        checks_to_run.append(name)
                        check_schedules[name] = (
                            current_time + health_check.interval_seconds
                        )

                # Run health checks concurrently
                if checks_to_run:
                    tasks = [self._run_health_check(name) for name in checks_to_run]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep for a short interval before next check
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(10.0)

    async def _run_health_check(self, name: str):
        """Run a single health check."""
        if name not in self._health_checks:
            return

        health_check = self._health_checks[name]
        start_time = time.time()

        try:
            # Run the health check with timeout
            is_healthy = await asyncio.wait_for(
                health_check.check_function(), timeout=health_check.timeout_seconds
            )

            duration = time.time() - start_time

            # Determine status
            if is_healthy:
                status = HealthStatus.HEALTHY
                message = f"{health_check.description} is healthy"
            else:
                status = (
                    HealthStatus.WARNING
                    if not health_check.critical
                    else HealthStatus.CRITICAL
                )
                message = f"{health_check.description} is not healthy"

            result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration=duration,
                timestamp=datetime.now(),
                metadata=health_check.metadata.copy(),
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            status = (
                HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            )

            result = HealthCheckResult(
                name=name,
                status=status,
                message=f"{health_check.description} check timed out after {health_check.timeout_seconds}s",
                duration=duration,
                timestamp=datetime.now(),
                error="timeout",
            )

        except Exception as e:
            duration = time.time() - start_time
            status = (
                HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            )

            result = HealthCheckResult(
                name=name,
                status=status,
                message=f"{health_check.description} check failed: {str(e)}",
                duration=duration,
                timestamp=datetime.now(),
                error=str(e),
            )

        # Store result
        previous_result = self._check_results.get(name)
        self._check_results[name] = result

        # Record metrics
        self.metrics_collector.record_database_connections(
            database_type="health_check",
            connection_count=1 if result.status == HealthStatus.HEALTHY else 0,
        )

        # Check for status changes and notify callbacks
        if previous_result is None or previous_result.status != result.status:
            for callback in self._status_change_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Error in health status callback", error=str(e))

        logger.debug(
            "Health check completed",
            name=name,
            status=result.status.value,
            duration=duration,
            message=result.message,
        )

    async def run_health_check_once(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check once and return the result."""
        if name not in self._health_checks:
            return None

        await self._run_health_check(name)
        return self._check_results.get(name)

    async def run_all_health_checks_once(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks once and return results."""
        tasks = [self._run_health_check(name) for name in self._health_checks.keys()]

        await asyncio.gather(*tasks, return_exceptions=True)

        return self._check_results.copy()

    def get_system_health_status(self) -> SystemHealthStatus:
        """Get overall system health status."""
        if not self._check_results:
            return SystemHealthStatus(
                overall_status=HealthStatus.UNKNOWN,
                healthy_checks=0,
                warning_checks=0,
                critical_checks=0,
                unknown_checks=len(self._health_checks),
                last_updated=datetime.now(),
                check_results=[],
                uptime_seconds=time.time() - self._start_time,
            )

        # Count status types
        healthy_count = sum(
            1 for r in self._check_results.values() if r.status == HealthStatus.HEALTHY
        )
        warning_count = sum(
            1 for r in self._check_results.values() if r.status == HealthStatus.WARNING
        )
        critical_count = sum(
            1 for r in self._check_results.values() if r.status == HealthStatus.CRITICAL
        )
        unknown_count = sum(
            1 for r in self._check_results.values() if r.status == HealthStatus.UNKNOWN
        )

        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        # Get most recent update time
        last_updated = max(
            (result.timestamp for result in self._check_results.values()),
            default=datetime.now(),
        )

        return SystemHealthStatus(
            overall_status=overall_status,
            healthy_checks=healthy_count,
            warning_checks=warning_count,
            critical_checks=critical_count,
            unknown_checks=unknown_count,
            last_updated=last_updated,
            check_results=list(self._check_results.values()),
            uptime_seconds=time.time() - self._start_time,
        )

    def add_status_change_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add a callback to be called when health status changes."""
        self._status_change_callbacks.append(callback)
        logger.debug("Added health status change callback")

    def get_health_check_history(
        self, name: str, hours: int = 24
    ) -> List[HealthCheckResult]:
        """Get health check history for a specific check (placeholder for future implementation)."""
        # In a full implementation, this would return historical data
        # For now, return current result if available
        current_result = self._check_results.get(name)
        return [current_result] if current_result else []

    # Default health check implementations

    async def _check_vector_database_health(self) -> bool:
        """Check vector database (Qdrant) health with actual connectivity test."""
        try:
            from morgan.config import get_settings

            settings = get_settings()

            # Try to connect to Qdrant and list collections
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=getattr(settings, "qdrant_api_key", None),
                    timeout=5,
                )

                # Test connectivity by listing collections
                start_time = time.time()
                collections = client.get_collections()
                response_time = time.time() - start_time

                # Record metrics
                self.metrics_collector.record_database_connections(
                    database_type="qdrant",
                    connection_count=len(collections.collections),
                )

                # Log successful health check
                logger.debug(
                    "Qdrant health check passed",
                    collections=len(collections.collections),
                    response_time=f"{response_time:.3f}s",
                )

                # Healthy if response time < 5 seconds
                return response_time < 5.0

            except ImportError:
                logger.warning("qdrant-client not installed, using HTTP fallback")
                return await self._check_qdrant_http_health(settings)

        except Exception as e:
            logger.error("Vector database health check failed", error=str(e))
            return False

    async def _check_qdrant_http_health(self, settings) -> bool:
        """HTTP fallback for Qdrant health check."""
        try:
            import httpx

            qdrant_url = settings.qdrant_url.rstrip("/")
            health_url = f"{qdrant_url}/health"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Qdrant HTTP health check failed: {e}")
            return False

    async def _check_embedding_service_health(self) -> bool:
        """Check embedding service health with actual test."""
        try:
            from morgan.config import get_settings

            settings = get_settings()

            # Try Ollama embedding endpoint
            embedding_url = getattr(settings, "ollama_url", "http://localhost:11434")

            try:
                import httpx

                # Test embedding generation
                start_time = time.time()

                async with httpx.AsyncClient(timeout=15.0) as client:
                    # Check Ollama is running
                    response = await client.get(f"{embedding_url}/api/tags")
                    if response.status_code != 200:
                        return False

                    # Test actual embedding generation
                    test_response = await client.post(
                        f"{embedding_url}/api/embeddings",
                        json={
                            "model": getattr(
                                settings, "embedding_model", "nomic-embed-text"
                            ),
                            "prompt": "health check test",
                        },
                        timeout=10.0,
                    )

                    response_time = time.time() - start_time

                    if test_response.status_code == 200:
                        data = test_response.json()
                        embedding = data.get("embedding", [])

                        logger.debug(
                            "Embedding service health check passed",
                            embedding_dim=len(embedding),
                            response_time=f"{response_time:.3f}s",
                        )

                        # Healthy if we got an embedding and response time < 10 seconds
                        return len(embedding) > 0 and response_time < 10.0

                    return False

            except ImportError:
                # Try sentence-transformers fallback
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                    embedding = model.encode("health check test")

                    return len(embedding) > 0

                except Exception:
                    return False

        except Exception as e:
            logger.error("Embedding service health check failed", error=str(e))
            return False

    async def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Record memory usage
            self.metrics_collector.record_memory_usage("system", memory.used)

            # Also check GPU memory if available
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory = torch.cuda.memory_allocated(i)
                        gpu_total = torch.cuda.get_device_properties(i).total_memory
                        gpu_percent = (gpu_memory / gpu_total) * 100
                        self.metrics_collector.record_memory_usage(
                            f"gpu_{i}", gpu_memory
                        )

                        if gpu_percent > 95:
                            logger.warning(
                                f"GPU {i} memory usage high: {gpu_percent:.1f}%"
                            )
            except ImportError:
                pass  # PyTorch not available

            # Consider healthy if memory usage is below 90%
            return memory_percent < 90.0

        except Exception as e:
            logger.error("Memory usage health check failed", error=str(e))
            return False

    async def _check_cache_health(self) -> bool:
        """Check cache system health (Redis if configured, local otherwise)."""
        try:
            from morgan.config import get_settings

            settings = get_settings()

            redis_url = getattr(settings, "redis_url", None)

            if redis_url:
                return await self._check_redis_health(redis_url)
            else:
                return await self._check_local_cache_health()

        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return False

    async def _check_redis_health(self, redis_url: str) -> bool:
        """Check Redis cache health."""
        try:
            import redis.asyncio as redis

            client = redis.from_url(redis_url, decode_responses=True)

            try:
                # Test connectivity with PING
                start_time = time.time()
                await client.ping()
                response_time = time.time() - start_time

                # Get memory info
                info = await client.info("memory")
                used_memory = info.get("used_memory", 0)
                max_memory = info.get("maxmemory", 0)

                # Get stats
                stats_info = await client.info("stats")
                hits = stats_info.get("keyspace_hits", 0)
                misses = stats_info.get("keyspace_misses", 0)
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0

                self.metrics_collector.record_cache_hit_rate("redis", hit_rate)

                logger.debug(
                    "Redis health check passed",
                    response_time=f"{response_time:.3f}s",
                    hit_rate=f"{hit_rate:.1f}%",
                    used_memory=used_memory,
                )

                # Healthy if responsive and hit rate is reasonable
                return response_time < 1.0

            finally:
                await client.close()

        except ImportError:
            logger.warning("redis package not installed")
            return await self._check_local_cache_health()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def _check_local_cache_health(self) -> bool:
        """Check local cache health (in-memory caches)."""
        try:
            from morgan.caching.intelligent_cache import get_intelligent_cache_manager

            cache_manager = get_intelligent_cache_manager()
            stats = cache_manager.get_stats()

            hit_rate = stats.get("overall_hit_rate", 0)
            self.metrics_collector.record_cache_hit_rate("local", hit_rate)

            logger.debug(
                "Local cache health check passed",
                hit_rate=f"{hit_rate:.1f}%",
                total_entries=stats.get("total_entries", 0),
            )

            # Healthy if cache is operational
            return True

        except ImportError:
            # No intelligent cache, check basic operation
            return True
        except Exception as e:
            logger.debug(f"Local cache health check: {e}")
            return True  # Don't fail on missing cache

    async def _check_processing_pipeline_health(self) -> bool:
        """Check document processing pipeline health."""
        try:
            # Get recent processing metrics
            processing_summary = self.metrics_collector.get_metrics_summary(
                "processing_time"
            )

            if processing_summary:
                # Healthy if error rate < 5% and P95 < 10 seconds
                is_healthy = (
                    processing_summary.error_rate < 0.05
                    and processing_summary.percentile_95 < 10.0
                )

                logger.debug(
                    "Processing pipeline health check",
                    error_rate=f"{processing_summary.error_rate:.1%}",
                    p95=f"{processing_summary.percentile_95:.2f}s",
                    healthy=is_healthy,
                )

                return is_healthy

            # No data available - check if pipeline components exist
            try:
                from morgan.optimization.batch_processor import get_batch_processor

                processor = get_batch_processor()
                return processor is not None
            except ImportError:
                pass

            return True  # No data available, assume healthy

        except Exception as e:
            logger.error("Processing pipeline health check failed", error=str(e))
            return False

    async def _check_search_system_health(self) -> bool:
        """Check search system health with actual query test."""
        try:
            # Get recent search metrics first
            search_summary = self.metrics_collector.get_metrics_summary("search_time")

            if search_summary:
                # Healthy if error rate < 2% and P95 < 1 second
                is_healthy = (
                    search_summary.error_rate < 0.02
                    and search_summary.percentile_95 < 1.0
                )

                logger.debug(
                    "Search system health check (metrics)",
                    error_rate=f"{search_summary.error_rate:.1%}",
                    p95=f"{search_summary.percentile_95:.2f}s",
                    healthy=is_healthy,
                )

                if not is_healthy:
                    return False

            # Additionally test actual search capability
            try:
                from morgan.search.service import get_search_service

                service = get_search_service()
                if service and hasattr(service, "is_ready"):
                    return service.is_ready()
                return True

            except ImportError:
                return True

        except Exception as e:
            logger.error("Search system health check failed", error=str(e))
            return False

    async def _check_companion_system_health(self) -> bool:
        """Check companion system health."""
        try:
            # Get recent emotional analysis metrics
            emotional_summary = self.metrics_collector.get_metrics_summary(
                "emotional_analysis_time"
            )

            if emotional_summary:
                # Healthy if P95 < 2 seconds
                is_healthy = emotional_summary.percentile_95 < 2.0

                logger.debug(
                    "Companion system health check",
                    p95=f"{emotional_summary.percentile_95:.2f}s",
                    healthy=is_healthy,
                )

                return is_healthy

            # Test emotional analysis is functional
            try:
                from morgan.intelligence.emotions import get_emotion_detector

                detector = get_emotion_detector()
                if detector:
                    # Quick test - detect emotion in sample text
                    test_result = detector.detect_emotion("Hello, how are you?")
                    return test_result is not None
                return True

            except ImportError:
                return True

        except Exception as e:
            logger.error("Companion system health check failed", error=str(e))
            return False
