"""
Health Check System for Morgan Server

This module provides health check endpoints and component monitoring.
It tracks component health, response times, and system metrics.
"""

import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from morgan_server.api.models import (
    HealthResponse,
    StatusResponse,
    ComponentStatus,
    SystemMetrics,
)


@dataclass
class HealthCheckResult:
    """Result of a component health check."""
    
    name: str
    status: str  # "up", "down", "degraded"
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthCheckSystem:
    """
    Health check system for monitoring server and component health.
    
    Tracks:
    - Component health (vector DB, LLM, etc.)
    - Response times
    - Request metrics
    - System uptime
    """
    
    def __init__(self, version: str = "0.1.0"):
        """
        Initialize health check system.
        
        Args:
            version: Server version string
        """
        self.version = version
        self.start_time = time.time()
        
        # Metrics tracking
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.response_times: list[float] = []
        self.active_sessions = 0
        
        # Component checkers (will be set by app initialization)
        self.component_checkers: Dict[str, Any] = {}
    
    def register_component(self, name: str, checker: Any) -> None:
        """
        Register a component health checker.
        
        Args:
            name: Component name (e.g., "vector_db", "llm")
            checker: Component instance with health check capability
        """
        self.component_checkers[name] = checker
    
    def record_request(self, success: bool, response_time_ms: float) -> None:
        """
        Record a request for metrics tracking.
        
        Args:
            success: Whether the request was successful
            response_time_ms: Response time in milliseconds
        """
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_error += 1
        
        # Keep last 1000 response times for metrics
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
    
    def increment_active_sessions(self) -> None:
        """Increment active session count."""
        self.active_sessions += 1
    
    def decrement_active_sessions(self) -> None:
        """Decrement active session count."""
        self.active_sessions = max(0, self.active_sessions - 1)
    
    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time
    
    async def check_component_health(
        self,
        name: str,
        checker: Any,
        timeout_seconds: float = 2.0
    ) -> HealthCheckResult:
        """
        Check health of a single component.
        
        Args:
            name: Component name
            checker: Component instance
            timeout_seconds: Timeout for health check
            
        Returns:
            HealthCheckResult with component status
        """
        start_time = time.time()
        
        try:
            # Try to check component health with timeout
            if hasattr(checker, 'health_check'):
                # Component has explicit health check method
                result = await asyncio.wait_for(
                    checker.health_check(),
                    timeout=timeout_seconds
                )
                latency_ms = (time.time() - start_time) * 1000
                
                if isinstance(result, dict):
                    return HealthCheckResult(
                        name=name,
                        status=result.get("status", "up"),
                        latency_ms=latency_ms,
                        details=result.get("details", {})
                    )
                else:
                    return HealthCheckResult(
                        name=name,
                        status="up" if result else "down",
                        latency_ms=latency_ms
                    )
            
            elif hasattr(checker, 'ping') or hasattr(checker, 'is_connected'):
                # Component has ping or is_connected method
                if hasattr(checker, 'ping'):
                    await asyncio.wait_for(
                        checker.ping(),
                        timeout=timeout_seconds
                    )
                else:
                    is_connected = await asyncio.wait_for(
                        checker.is_connected(),
                        timeout=timeout_seconds
                    )
                    if not is_connected:
                        return HealthCheckResult(
                            name=name,
                            status="down",
                            error="Component not connected"
                        )
                
                latency_ms = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    name=name,
                    status="up",
                    latency_ms=latency_ms
                )
            
            else:
                # No health check method, assume up if registered
                return HealthCheckResult(
                    name=name,
                    status="up",
                    latency_ms=0.0,
                    details={"note": "No health check method available"}
                )
        
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status="degraded",
                latency_ms=latency_ms,
                error=f"Health check timed out after {timeout_seconds}s"
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status="down",
                latency_ms=latency_ms,
                error=str(e)
            )
    
    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all registered components.
        
        Returns:
            Dictionary mapping component names to health check results
        """
        results = {}
        
        # Check all components concurrently
        tasks = []
        for name, checker in self.component_checkers.items():
            tasks.append(self.check_component_health(name, checker))
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in check_results:
                if isinstance(result, HealthCheckResult):
                    results[result.name] = result
                elif isinstance(result, Exception):
                    # Handle unexpected errors
                    results["unknown"] = HealthCheckResult(
                        name="unknown",
                        status="down",
                        error=str(result)
                    )
        
        return results
    
    def get_system_metrics(self) -> SystemMetrics:
        """
        Get current system metrics.
        
        Returns:
            SystemMetrics with current statistics
        """
        # Calculate average response time
        avg_response_time = 0.0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # Calculate error rate
        error_rate = 0.0
        if self.requests_total > 0:
            error_rate = self.requests_error / self.requests_total
        
        # Calculate requests per second (based on uptime)
        uptime = self.get_uptime_seconds()
        requests_per_second = 0.0
        if uptime > 0:
            requests_per_second = self.requests_total / uptime
        
        return SystemMetrics(
            requests_total=self.requests_total,
            requests_per_second=requests_per_second,
            average_response_time_ms=avg_response_time,
            error_rate=error_rate,
            active_sessions=self.active_sessions
        )
    
    async def get_health(self) -> HealthResponse:
        """
        Get simple health check response.
        
        This is a lightweight check that returns quickly for load balancers.
        
        Returns:
            HealthResponse with overall health status
        """
        # Quick check - just verify we're running
        uptime = self.get_uptime_seconds()
        
        # Determine status based on basic metrics
        status = "healthy"
        
        # Check if error rate is too high
        if self.requests_total > 10:  # Only check if we have enough data
            error_rate = self.requests_error / self.requests_total
            if error_rate > 0.5:  # More than 50% errors
                status = "unhealthy"
            elif error_rate > 0.1:  # More than 10% errors
                status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc),
            version=self.version,
            uptime_seconds=uptime
        )
    
    async def get_status(self) -> StatusResponse:
        """
        Get detailed status with component health checks.
        
        This performs actual health checks on all components and may take
        longer than get_health().
        
        Returns:
            StatusResponse with detailed component status and metrics
        """
        # Check all components
        component_results = await self.check_all_components()
        
        # Convert to ComponentStatus objects
        components = {}
        for name, result in component_results.items():
            components[name] = ComponentStatus(
                name=result.name,
                status=result.status,
                latency_ms=result.latency_ms,
                error=result.error,
                details=result.details
            )
        
        # Determine overall status
        if not components:
            overall_status = "healthy"
        elif all(c.status == "up" for c in components.values()):
            overall_status = "healthy"
        elif any(c.status == "down" for c in components.values()):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        # Get system metrics
        metrics = self.get_system_metrics()
        
        return StatusResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            components=components,
            metrics=metrics
        )


# Global health check system instance
_health_system: Optional[HealthCheckSystem] = None


def get_health_system() -> HealthCheckSystem:
    """
    Get the global health check system instance.
    
    Returns:
        HealthCheckSystem instance
    """
    global _health_system
    if _health_system is None:
        _health_system = HealthCheckSystem()
    return _health_system


def initialize_health_system(version: str = "0.1.0") -> HealthCheckSystem:
    """
    Initialize the global health check system.
    
    Args:
        version: Server version string
        
    Returns:
        HealthCheckSystem instance
    """
    global _health_system
    _health_system = HealthCheckSystem(version=version)
    return _health_system
