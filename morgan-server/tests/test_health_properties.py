"""
Property-based tests for health check system.

This module tests health check responsiveness and component monitoring
using property-based testing with Hypothesis.
"""

import asyncio
import time
from typing import Dict, Any, Optional

from hypothesis import given, strategies as st, settings
import pytest

from morgan_server.health import (
    HealthCheckSystem,
    HealthCheckResult,
)
from morgan_server.api.models import (
    HealthResponse,
    StatusResponse,
    ComponentStatus,
)


# ============================================================================
# Mock Components for Testing
# ============================================================================

class MockHealthyComponent:
    """Mock component that is always healthy."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Return healthy status."""
        await asyncio.sleep(0.001)  # Simulate minimal latency
        return {"status": "up", "details": {"mock": True}}


class MockSlowComponent:
    """Mock component with configurable latency."""
    
    def __init__(self, latency_ms: float = 100):
        self.latency_ms = latency_ms
    
    async def health_check(self) -> Dict[str, Any]:
        """Return healthy status after delay."""
        await asyncio.sleep(self.latency_ms / 1000)
        return {"status": "up", "details": {"latency_ms": self.latency_ms}}


class MockUnhealthyComponent:
    """Mock component that is always unhealthy."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Return unhealthy status."""
        await asyncio.sleep(0.001)
        return {"status": "down", "details": {"error": "Component unavailable"}}


class MockFailingComponent:
    """Mock component that raises exceptions."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Raise an exception."""
        await asyncio.sleep(0.001)
        raise Exception("Component check failed")


class MockPingComponent:
    """Mock component with ping method."""
    
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
    
    async def ping(self) -> None:
        """Ping the component."""
        await asyncio.sleep(0.001)
        if not self.should_succeed:
            raise Exception("Ping failed")


# ============================================================================
# Property 10: Health Check Responsiveness
# ============================================================================

class TestHealthCheckResponsiveness:
    """
    Property-based tests for health check responsiveness.

    **Feature: client-server-separation, Property 10: Health check responsiveness**

    For any health check request, the server should respond within 2 seconds with
    a status indicating whether all components are operational ("healthy") or if
    any component has failed ("unhealthy" with details).

    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @pytest.mark.asyncio
    @given(
        num_components=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=5000)
    async def test_property_health_check_responds_quickly(self, num_components):
        """
        Property: Health check responds within 2 seconds.

        For any number of registered components, the simple health check
        should respond within 2 seconds.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Register mock components
        for i in range(num_components):
            health_system.register_component(
                f"component_{i}",
                MockHealthyComponent()
            )
        
        # Measure response time
        start_time = time.time()
        response = await health_system.get_health()
        elapsed_time = time.time() - start_time
        
        # Verify response time is under 2 seconds
        assert elapsed_time < 2.0, f"Health check took {elapsed_time}s, expected < 2s"
        
        # Verify response structure
        assert isinstance(response, HealthResponse)
        assert response.status in ["healthy", "degraded", "unhealthy"]
        assert isinstance(response.uptime_seconds, float)
        assert response.uptime_seconds >= 0
        assert response.version == "test"

    @pytest.mark.asyncio
    @given(
        num_healthy=st.integers(min_value=0, max_value=5),
        num_unhealthy=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100, deadline=10000)
    async def test_property_status_reflects_component_health(self, num_healthy, num_unhealthy):
        """
        Property: Status endpoint reflects component health accurately.

        For any combination of healthy and unhealthy components, the status
        endpoint should accurately report the health of each component and
        determine the overall status correctly.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Register healthy components
        for i in range(num_healthy):
            health_system.register_component(
                f"healthy_{i}",
                MockHealthyComponent()
            )
        
        # Register unhealthy components
        for i in range(num_unhealthy):
            health_system.register_component(
                f"unhealthy_{i}",
                MockUnhealthyComponent()
            )
        
        # Get status
        status = await health_system.get_status()
        
        # Verify response structure
        assert isinstance(status, StatusResponse)
        assert status.status in ["healthy", "degraded", "unhealthy"]
        
        # Verify component count
        total_components = num_healthy + num_unhealthy
        assert len(status.components) == total_components
        
        # Verify overall status logic
        if total_components == 0:
            assert status.status == "healthy"
        elif num_unhealthy > 0:
            # If any component is down, overall status should be unhealthy
            assert status.status == "unhealthy"
        else:
            # All components are up
            assert status.status == "healthy"
        
        # Verify each component status
        for name, component in status.components.items():
            assert isinstance(component, ComponentStatus)
            assert component.status in ["up", "down", "degraded"]
            
            if name.startswith("healthy_"):
                assert component.status == "up"
            elif name.startswith("unhealthy_"):
                assert component.status == "down"

    @pytest.mark.asyncio
    @given(
        latency_ms=st.floats(min_value=1.0, max_value=500.0),
    )
    @settings(max_examples=100, deadline=10000)
    async def test_property_component_latency_tracked(self, latency_ms):
        """
        Property: Component latency is tracked accurately.

        For any component with a specific latency, the health check should
        accurately measure and report that latency.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Register slow component
        component = MockSlowComponent(latency_ms=latency_ms)
        health_system.register_component("slow_component", component)
        
        # Check component health
        result = await health_system.check_component_health(
            "slow_component",
            component,
            timeout_seconds=2.0
        )
        
        # Verify latency is tracked
        assert result.latency_ms is not None
        assert result.latency_ms > 0
        
        # Latency should be approximately the configured latency
        # Allow for some variance due to system overhead
        assert result.latency_ms >= latency_ms * 0.8
        assert result.latency_ms <= latency_ms * 2.0 + 50  # Allow overhead

    @pytest.mark.asyncio
    @given(
        num_requests=st.integers(min_value=1, max_value=100),
        success_rate=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=5000)
    async def test_property_metrics_accuracy(self, num_requests, success_rate):
        """
        Property: System metrics are accurate.

        For any sequence of requests with a given success rate, the metrics
        should accurately reflect the request count, success count, and error rate.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Record requests
        num_success = int(num_requests * success_rate)
        num_error = num_requests - num_success
        
        for i in range(num_success):
            health_system.record_request(success=True, response_time_ms=100.0)
        
        for i in range(num_error):
            health_system.record_request(success=False, response_time_ms=200.0)
        
        # Get metrics
        metrics = health_system.get_system_metrics()
        
        # Verify accuracy
        assert metrics.requests_total == num_requests
        
        # Calculate expected error rate
        expected_error_rate = num_error / num_requests if num_requests > 0 else 0.0
        
        # Allow for floating point precision
        assert abs(metrics.error_rate - expected_error_rate) < 0.01

    @pytest.mark.asyncio
    @given(
        timeout_seconds=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100, deadline=10000)
    async def test_property_timeout_handling(self, timeout_seconds):
        """
        Property: Health checks respect timeout settings.

        For any timeout value, if a component takes longer than the timeout,
        the health check should return a degraded status with timeout error.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Create component that takes longer than timeout
        slow_latency_ms = (timeout_seconds + 0.5) * 1000
        component = MockSlowComponent(latency_ms=slow_latency_ms)
        
        # Check component health with timeout
        start_time = time.time()
        result = await health_system.check_component_health(
            "slow_component",
            component,
            timeout_seconds=timeout_seconds
        )
        elapsed_time = time.time() - start_time
        
        # Verify timeout was respected
        assert elapsed_time <= timeout_seconds * 1.5  # Allow some overhead
        
        # Verify status is degraded due to timeout
        assert result.status == "degraded"
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    @given(
        num_failures=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=10000)
    async def test_property_error_handling(self, num_failures):
        """
        Property: Health checks handle component failures gracefully.

        For any number of failing components, the health check should handle
        exceptions gracefully and report them as down status with error details.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Register failing components
        for i in range(num_failures):
            health_system.register_component(
                f"failing_{i}",
                MockFailingComponent()
            )
        
        # Get status (should not raise exception)
        status = await health_system.get_status()
        
        # Verify all failing components are reported as down
        assert len(status.components) == num_failures
        
        for name, component in status.components.items():
            assert component.status == "down"
            assert component.error is not None
            assert len(component.error) > 0

    @pytest.mark.asyncio
    @given(
        has_ping=st.booleans(),
        ping_succeeds=st.booleans(),
    )
    @settings(max_examples=100, deadline=5000)
    async def test_property_ping_method_support(self, has_ping, ping_succeeds):
        """
        Property: Health checks support components with ping method.

        For any component with a ping method, the health check should use
        the ping method to determine health status.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        if has_ping:
            component = MockPingComponent(should_succeed=ping_succeeds)
        else:
            component = MockHealthyComponent()
        
        # Check component health
        result = await health_system.check_component_health(
            "test_component",
            component,
            timeout_seconds=2.0
        )
        
        # Verify result
        if has_ping:
            if ping_succeeds:
                assert result.status == "up"
            else:
                assert result.status == "down"
        else:
            # MockHealthyComponent has health_check method
            assert result.status == "up"

    @pytest.mark.asyncio
    @given(
        session_changes=st.lists(
            st.sampled_from(["increment", "decrement"]),
            min_size=0,
            max_size=50
        ),
    )
    @settings(max_examples=100, deadline=5000)
    async def test_property_session_tracking(self, session_changes):
        """
        Property: Active session count is tracked accurately.

        For any sequence of session increments and decrements, the active
        session count should accurately reflect the current number of sessions.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Apply session changes
        expected_count = 0
        for change in session_changes:
            if change == "increment":
                health_system.increment_active_sessions()
                expected_count += 1
            else:  # decrement
                health_system.decrement_active_sessions()
                expected_count = max(0, expected_count - 1)  # Can't go below 0
        
        # Get metrics
        metrics = health_system.get_system_metrics()
        
        # Verify session count
        assert metrics.active_sessions == expected_count
        assert metrics.active_sessions >= 0

    @pytest.mark.asyncio
    async def test_property_uptime_increases(self):
        """
        Property: Server uptime increases over time.

        The uptime should always increase as time passes.
        """
        # Create health system
        health_system = HealthCheckSystem(version="test")
        
        # Get initial uptime
        uptime1 = health_system.get_uptime_seconds()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Get uptime again
        uptime2 = health_system.get_uptime_seconds()
        
        # Verify uptime increased
        assert uptime2 > uptime1
        assert uptime2 >= uptime1 + 0.09  # Allow for timing variance
