"""
Health Check API Routes

This module implements the health check endpoints for Morgan server:
- GET /health: Simple health check for load balancers
- GET /api/status: Detailed status with component health checks
- GET /metrics: Prometheus-compatible metrics endpoint
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status, Response

from morgan_server.api.models import (
    HealthResponse,
    StatusResponse,
)
from morgan_server.health import HealthCheckSystem


router = APIRouter(tags=["health"])


# Global health system instance (will be injected via dependency injection)
_health_system: Optional[HealthCheckSystem] = None


def set_health_system(health_system: HealthCheckSystem) -> None:
    """
    Set the global health system instance.
    
    This should be called during application startup.
    
    Args:
        health_system: HealthCheckSystem instance
    """
    global _health_system
    _health_system = health_system


def get_health_system() -> HealthCheckSystem:
    """
    Get the global health system instance.
    
    Returns:
        HealthCheckSystem instance
        
    Raises:
        HTTPException: If health system is not initialized
    """
    if _health_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health system not initialized",
        )
    return _health_system


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Simple health check",
    description="Lightweight health check endpoint for load balancers. "
                "Returns quickly with basic health status.",
    responses={
        200: {
            "description": "Service is healthy",
            "model": HealthResponse,
        },
        503: {
            "description": "Service is unhealthy or degraded",
            "model": HealthResponse,
        },
    },
)
async def health_check() -> HealthResponse:
    """
    Simple health check endpoint.
    
    This endpoint is designed to be lightweight and respond quickly
    for use by load balancers and monitoring systems.
    
    Returns:
        HealthResponse with basic health status
    """
    health_system = get_health_system()
    response = await health_system.get_health()
    
    # Return appropriate HTTP status code based on health
    if response.status == "unhealthy":
        # Still return 200 but with unhealthy status
        # Some load balancers prefer this approach
        pass
    
    return response


@router.get(
    "/api/status",
    response_model=StatusResponse,
    summary="Detailed status check",
    description="Comprehensive status check that includes component health checks, "
                "system metrics, and detailed diagnostics. May take longer than /health.",
    responses={
        200: {
            "description": "Status retrieved successfully",
            "model": StatusResponse,
        },
        503: {
            "description": "Service unavailable",
        },
    },
)
async def detailed_status() -> StatusResponse:
    """
    Detailed status endpoint with component health checks.
    
    This endpoint performs actual health checks on all registered components
    (vector DB, LLM, etc.) and returns detailed status information along with
    system metrics.
    
    This may take longer than the simple /health endpoint as it actively
    checks each component.
    
    Returns:
        StatusResponse with detailed component status and metrics
    """
    health_system = get_health_system()
    response = await health_system.get_status()
    
    return response


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint for monitoring. "
                "Returns metrics in Prometheus text format including request counts, "
                "response times, error rates, and active sessions.",
    responses={
        200: {
            "description": "Metrics in Prometheus text format",
            "content": {
                "text/plain": {
                    "example": "# HELP morgan_requests_total Total number of requests\n"
                               "# TYPE morgan_requests_total counter\n"
                               "morgan_requests_total 1234\n"
                }
            }
        },
        503: {
            "description": "Service unavailable",
        },
    },
)
async def prometheus_metrics() -> Response:
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping by Prometheus
    or compatible monitoring systems.
    
    Metrics included:
    - morgan_requests_total: Total number of requests processed
    - morgan_requests_success_total: Total number of successful requests
    - morgan_requests_error_total: Total number of failed requests
    - morgan_response_time_seconds: Average response time in seconds
    - morgan_error_rate: Current error rate (0.0 to 1.0)
    - morgan_active_sessions: Number of currently active sessions
    - morgan_uptime_seconds: Server uptime in seconds
    
    Returns:
        Response with Prometheus text format metrics
    """
    health_system = get_health_system()
    metrics = health_system.get_system_metrics()
    uptime = health_system.get_uptime_seconds()
    
    # Build Prometheus text format
    lines = []
    
    # Total requests counter
    lines.append("# HELP morgan_requests_total Total number of requests processed")
    lines.append("# TYPE morgan_requests_total counter")
    lines.append(f"morgan_requests_total {metrics.requests_total}")
    lines.append("")
    
    # Success requests counter
    lines.append("# HELP morgan_requests_success_total Total number of successful requests")
    lines.append("# TYPE morgan_requests_success_total counter")
    lines.append(f"morgan_requests_success_total {health_system.requests_success}")
    lines.append("")
    
    # Error requests counter
    lines.append("# HELP morgan_requests_error_total Total number of failed requests")
    lines.append("# TYPE morgan_requests_error_total counter")
    lines.append(f"morgan_requests_error_total {health_system.requests_error}")
    lines.append("")
    
    # Response time gauge (convert ms to seconds for Prometheus convention)
    lines.append("# HELP morgan_response_time_seconds Average response time in seconds")
    lines.append("# TYPE morgan_response_time_seconds gauge")
    response_time_seconds = metrics.average_response_time_ms / 1000.0
    lines.append(f"morgan_response_time_seconds {response_time_seconds:.6f}")
    lines.append("")
    
    # Error rate gauge
    lines.append("# HELP morgan_error_rate Current error rate (0.0 to 1.0)")
    lines.append("# TYPE morgan_error_rate gauge")
    lines.append(f"morgan_error_rate {metrics.error_rate:.6f}")
    lines.append("")
    
    # Active sessions gauge
    lines.append("# HELP morgan_active_sessions Number of currently active sessions")
    lines.append("# TYPE morgan_active_sessions gauge")
    lines.append(f"morgan_active_sessions {metrics.active_sessions}")
    lines.append("")
    
    # Uptime counter
    lines.append("# HELP morgan_uptime_seconds Server uptime in seconds")
    lines.append("# TYPE morgan_uptime_seconds counter")
    lines.append(f"morgan_uptime_seconds {uptime:.2f}")
    lines.append("")
    
    # Requests per second gauge
    lines.append("# HELP morgan_requests_per_second Current request rate")
    lines.append("# TYPE morgan_requests_per_second gauge")
    lines.append(f"morgan_requests_per_second {metrics.requests_per_second:.6f}")
    lines.append("")
    
    content = "\n".join(lines)
    
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )
