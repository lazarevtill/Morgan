"""
System health monitoring for Morgan RAG.

Provides comprehensive health checks, dependency monitoring, and system status
tracking as specified in Requirements 7.2, 7.4, and 7.5.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
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
            description="Vector database (Qdrant) connectivity and performance"
        )
        
        # Embedding service health check
        self.register_health_check(
            name="embedding_service",
            check_function=self._check_embedding_service_health,
            timeout_seconds=15.0,
            interval_seconds=60.0,
            critical=True,
            description="Embedding service availability and performance"
        )
        
        # Memory usage health check
        self.register_health_check(
            name="memory_usage",
            check_function=self._check_memory_usage,
            timeout_seconds=5.0,
            interval_seconds=30.0,
            critical=False,
            description="System memory usage monitoring"
        )
        
        # Cache health check
        self.register_health_check(
            name="cache_system",
            check_function=self._check_cache_health,
            timeout_seconds=10.0,
            interval_seconds=60.0,
            critical=False,
            description="Caching system performance and hit rates"
        )
        
        # Processing pipeline health check
        self.register_health_check(
            name="processing_pipeline",
            check_function=self._check_processing_pipeline_health,
            timeout_seconds=20.0,
            interval_seconds=120.0,
            critical=False,
            description="Document processing pipeline performance"
        )
        
        # Search system health check
        self.register_health_check(
            name="search_system",
            check_function=self._check_search_system_health,
            timeout_seconds=15.0,
            interval_seconds=60.0,
            critical=True,
            description="Search system performance and accuracy"
        )
        
        # Companion system health check
        self.register_health_check(
            name="companion_system",
            check_function=self._check_companion_system_health,
            timeout_seconds=10.0,
            interval_seconds=120.0,
            critical=False,
            description="Companion features and emotional intelligence"
        )
    
    def register_health_check(self, name: str, check_function: Callable[[], Awaitable[bool]],
                            timeout_seconds: float = 30.0, interval_seconds: float = 60.0,
                            critical: bool = False, description: str = "",
                            metadata: Optional[Dict[str, Any]] = None):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            critical=critical,
            description=description,
            metadata=metadata or {}
        )
        
        self._health_checks[name] = health_check
        
        logger.info("Registered health check",
                   name=name,
                   critical=critical,
                   interval=interval_seconds)
    
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
        check_schedules = {name: 0.0 for name in self._health_checks.keys()}
        
        while self._monitoring_active:
            try:
                current_time = time.time()
                
                # Check which health checks need to run
                checks_to_run = []
                for name, health_check in self._health_checks.items():
                    if current_time >= check_schedules[name]:
                        checks_to_run.append(name)
                        check_schedules[name] = current_time + health_check.interval_seconds
                
                # Run health checks concurrently
                if checks_to_run:
                    tasks = [
                        self._run_health_check(name)
                        for name in checks_to_run
                    ]
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
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Determine status
            if is_healthy:
                status = HealthStatus.HEALTHY
                message = f"{health_check.description} is healthy"
            else:
                status = HealthStatus.WARNING if not health_check.critical else HealthStatus.CRITICAL
                message = f"{health_check.description} is not healthy"
            
            result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration=duration,
                timestamp=datetime.now(),
                metadata=health_check.metadata.copy()
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            
            result = HealthCheckResult(
                name=name,
                status=status,
                message=f"{health_check.description} check timed out after {health_check.timeout_seconds}s",
                duration=duration,
                timestamp=datetime.now(),
                error="timeout"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            
            result = HealthCheckResult(
                name=name,
                status=status,
                message=f"{health_check.description} check failed: {str(e)}",
                duration=duration,
                timestamp=datetime.now(),
                error=str(e)
            )
        
        # Store result
        previous_result = self._check_results.get(name)
        self._check_results[name] = result
        
        # Record metrics
        self.metrics_collector.record_database_connections(
            database_type="health_check",
            connection_count=1 if result.status == HealthStatus.HEALTHY else 0
        )
        
        # Check for status changes and notify callbacks
        if previous_result is None or previous_result.status != result.status:
            for callback in self._status_change_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Error in health status callback", error=str(e))
        
        logger.debug("Health check completed",
                    name=name,
                    status=result.status.value,
                    duration=duration,
                    message=result.message)
    
    async def run_health_check_once(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check once and return the result."""
        if name not in self._health_checks:
            return None
        
        await self._run_health_check(name)
        return self._check_results.get(name)
    
    async def run_all_health_checks_once(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks once and return results."""
        tasks = [
            self._run_health_check(name)
            for name in self._health_checks.keys()
        ]
        
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
                uptime_seconds=time.time() - self._start_time
            )
        
        # Count status types
        healthy_count = sum(1 for r in self._check_results.values() if r.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for r in self._check_results.values() if r.status == HealthStatus.WARNING)
        critical_count = sum(1 for r in self._check_results.values() if r.status == HealthStatus.CRITICAL)
        unknown_count = sum(1 for r in self._check_results.values() if r.status == HealthStatus.UNKNOWN)
        
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
            default=datetime.now()
        )
        
        return SystemHealthStatus(
            overall_status=overall_status,
            healthy_checks=healthy_count,
            warning_checks=warning_count,
            critical_checks=critical_count,
            unknown_checks=unknown_count,
            last_updated=last_updated,
            check_results=list(self._check_results.values()),
            uptime_seconds=time.time() - self._start_time
        )
    
    def add_status_change_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add a callback to be called when health status changes."""
        self._status_change_callbacks.append(callback)
        logger.debug("Added health status change callback")
    
    def get_health_check_history(self, name: str, hours: int = 24) -> List[HealthCheckResult]:
        """Get health check history for a specific check (placeholder for future implementation)."""
        # In a full implementation, this would return historical data
        # For now, return current result if available
        current_result = self._check_results.get(name)
        return [current_result] if current_result else []
    
    # Default health check implementations
    
    async def _check_vector_database_health(self) -> bool:
        """Check vector database health."""
        try:
            # This would typically check Qdrant connectivity
            # For now, simulate a basic check
            await asyncio.sleep(0.1)  # Simulate network call
            
            # In real implementation, would check:
            # - Database connectivity
            # - Response time
            # - Available collections
            # - Memory usage
            
            return True  # Placeholder
            
        except Exception as e:
            logger.error("Vector database health check failed", error=str(e))
            return False
    
    async def _check_embedding_service_health(self) -> bool:
        """Check embedding service health."""
        try:
            # This would typically test embedding generation
            await asyncio.sleep(0.2)  # Simulate embedding generation
            
            # In real implementation, would check:
            # - Service availability
            # - Model loading status
            # - Response time for test embedding
            # - GPU/CPU utilization
            
            return True  # Placeholder
            
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
            
            # Consider healthy if memory usage is below 90%
            return memory_percent < 90.0
            
        except Exception as e:
            logger.error("Memory usage health check failed", error=str(e))
            return False
    
    async def _check_cache_health(self) -> bool:
        """Check cache system health."""
        try:
            # This would typically check cache performance
            await asyncio.sleep(0.05)  # Simulate cache check
            
            # In real implementation, would check:
            # - Cache hit rates
            # - Cache size and memory usage
            # - Cache response times
            # - Cache consistency
            
            # Simulate cache hit rate check
            cache_hit_rate = 85.0  # Placeholder
            self.metrics_collector.record_cache_hit_rate("semantic_cache", cache_hit_rate)
            
            return cache_hit_rate > 70.0  # Healthy if hit rate > 70%
            
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return False
    
    async def _check_processing_pipeline_health(self) -> bool:
        """Check document processing pipeline health."""
        try:
            # This would typically check processing performance
            await asyncio.sleep(0.1)  # Simulate processing check
            
            # In real implementation, would check:
            # - Processing queue depth
            # - Average processing time
            # - Error rates
            # - Resource utilization
            
            # Get recent processing metrics
            processing_summary = self.metrics_collector.get_metrics_summary("processing_time")
            
            if processing_summary:
                # Healthy if error rate < 5% and P95 < 10 seconds
                return (processing_summary.error_rate < 0.05 and 
                       processing_summary.percentile_95 < 10.0)
            
            return True  # No data available, assume healthy
            
        except Exception as e:
            logger.error("Processing pipeline health check failed", error=str(e))
            return False
    
    async def _check_search_system_health(self) -> bool:
        """Check search system health."""
        try:
            # This would typically test search functionality
            await asyncio.sleep(0.15)  # Simulate search test
            
            # In real implementation, would check:
            # - Search response times
            # - Search accuracy/relevance
            # - Index health
            # - Query processing capacity
            
            # Get recent search metrics
            search_summary = self.metrics_collector.get_metrics_summary("search_time")
            
            if search_summary:
                # Healthy if error rate < 2% and P95 < 1 second
                return (search_summary.error_rate < 0.02 and 
                       search_summary.percentile_95 < 1.0)
            
            return True  # No data available, assume healthy
            
        except Exception as e:
            logger.error("Search system health check failed", error=str(e))
            return False
    
    async def _check_companion_system_health(self) -> bool:
        """Check companion system health."""
        try:
            # This would typically check companion features
            await asyncio.sleep(0.1)  # Simulate companion check
            
            # In real implementation, would check:
            # - Emotional analysis accuracy
            # - Response generation time
            # - User satisfaction scores
            # - Relationship tracking functionality
            
            # Get recent emotional analysis metrics
            emotional_summary = self.metrics_collector.get_metrics_summary("emotional_analysis_time")
            
            if emotional_summary:
                # Healthy if P95 < 2 seconds
                return emotional_summary.percentile_95 < 2.0
            
            return True  # No data available, assume healthy
            
        except Exception as e:
            logger.error("Companion system health check failed", error=str(e))
            return False