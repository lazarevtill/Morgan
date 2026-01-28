"""
Unit tests for health check routes.

This module tests the health check API endpoints.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.health import router, set_health_system
from morgan_server.health import HealthCheckSystem


@pytest.fixture
def health_system():
    """Create a health check system for testing."""
    return HealthCheckSystem(version="test-1.0.0")


@pytest.fixture
def app(health_system):
    """Create a FastAPI app with health routes."""
    app = FastAPI()
    app.include_router(router)

    # Set the health system
    set_health_system(health_system)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestHealthRoutes:
    """Tests for health check routes."""

    def test_health_endpoint_returns_200(self, client):
        """Test that /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_structure(self, client):
        """Test that /health endpoint returns correct structure."""
        response = client.get("/health")
        data = response.json()

        # Verify required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["version"], str)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_endpoint_version(self, client):
        """Test that /health endpoint returns correct version."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == "test-1.0.0"

    def test_status_endpoint_returns_200(self, client):
        """Test that /api/status endpoint returns 200."""
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_status_endpoint_structure(self, client):
        """Test that /api/status endpoint returns correct structure."""
        response = client.get("/api/status")
        data = response.json()

        # Verify required fields
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "metrics" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["components"], dict)
        assert isinstance(data["metrics"], dict)

    def test_status_endpoint_metrics(self, client, health_system):
        """Test that /api/status endpoint returns metrics."""
        # Record some requests
        health_system.record_request(success=True, response_time_ms=100.0)
        health_system.record_request(success=True, response_time_ms=150.0)
        health_system.record_request(success=False, response_time_ms=200.0)

        response = client.get("/api/status")
        data = response.json()

        metrics = data["metrics"]

        # Verify metrics structure
        assert "requests_total" in metrics
        assert "requests_per_second" in metrics
        assert "average_response_time_ms" in metrics
        assert "error_rate" in metrics
        assert "active_sessions" in metrics

        # Verify values
        assert metrics["requests_total"] == 3
        assert metrics["error_rate"] > 0  # We had 1 error out of 3

    def test_status_with_components(self, client, health_system):
        """Test that /api/status includes registered components."""

        # Register a mock component
        class MockComponent:
            async def health_check(self):
                return {"status": "up", "details": {"test": True}}

        health_system.register_component("test_component", MockComponent())

        response = client.get("/api/status")
        data = response.json()

        # Verify component is included
        assert "test_component" in data["components"]
        component = data["components"]["test_component"]

        assert component["name"] == "test_component"
        assert component["status"] == "up"
        assert "latency_ms" in component

    def test_health_responds_quickly(self, client):
        """Test that /health endpoint responds quickly."""
        import time

        start_time = time.time()
        response = client.get("/health")
        elapsed_time = time.time() - start_time

        # Should respond in under 1 second (well under 2 second requirement)
        assert elapsed_time < 1.0
        assert response.status_code == 200

    def test_multiple_health_checks(self, client):
        """Test multiple health checks return consistent results."""
        responses = []

        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            responses.append(response.json())

        # All should have same version
        versions = [r["version"] for r in responses]
        assert all(v == "test-1.0.0" for v in versions)

        # Uptime should increase
        uptimes = [r["uptime_seconds"] for r in responses]
        for i in range(1, len(uptimes)):
            assert uptimes[i] >= uptimes[i - 1]


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint_returns_200(self, client):
        """Test that /metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self, client):
        """Test that /metrics endpoint returns Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_endpoint_format(self, client):
        """Test that /metrics endpoint returns valid Prometheus format."""
        response = client.get("/metrics")
        content = response.text

        # Verify Prometheus format elements
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "morgan_requests_total" in content
        assert "morgan_requests_success_total" in content
        assert "morgan_requests_error_total" in content
        assert "morgan_response_time_seconds" in content
        assert "morgan_error_rate" in content
        assert "morgan_active_sessions" in content
        assert "morgan_uptime_seconds" in content
        assert "morgan_requests_per_second" in content

    def test_metrics_with_no_requests(self, client):
        """Test metrics endpoint with no recorded requests."""
        response = client.get("/metrics")
        content = response.text

        # Should have zero values
        assert "morgan_requests_total 0" in content
        assert "morgan_requests_success_total 0" in content
        assert "morgan_requests_error_total 0" in content
        assert "morgan_error_rate 0.000000" in content
        assert "morgan_active_sessions 0" in content

    def test_metrics_with_requests(self, client, health_system):
        """Test metrics endpoint with recorded requests."""
        # Record some requests
        health_system.record_request(success=True, response_time_ms=100.0)
        health_system.record_request(success=True, response_time_ms=200.0)
        health_system.record_request(success=False, response_time_ms=300.0)

        response = client.get("/metrics")
        content = response.text

        # Verify counts
        assert "morgan_requests_total 3" in content
        assert "morgan_requests_success_total 2" in content
        assert "morgan_requests_error_total 1" in content

        # Verify error rate (1/3 ≈ 0.333333)
        assert "morgan_error_rate" in content
        lines = content.split("\n")
        error_rate_line = [l for l in lines if l.startswith("morgan_error_rate ")][0]
        error_rate = float(error_rate_line.split()[1])
        assert 0.33 < error_rate < 0.34

        # Verify average response time (150ms = 0.15s)
        response_time_line = [
            l for l in lines if l.startswith("morgan_response_time_seconds ")
        ][0]
        response_time = float(response_time_line.split()[1])
        assert 0.19 < response_time < 0.21  # (100+200+300)/3 = 200ms = 0.2s

    def test_metrics_with_active_sessions(self, client, health_system):
        """Test metrics endpoint with active sessions."""
        # Add some active sessions
        health_system.increment_active_sessions()
        health_system.increment_active_sessions()
        health_system.increment_active_sessions()

        response = client.get("/metrics")
        content = response.text

        # Verify active sessions count
        assert "morgan_active_sessions 3" in content

    def test_metrics_uptime_increases(self, client):
        """Test that uptime metric increases over time."""
        import time

        # Get initial metrics
        response1 = client.get("/metrics")
        content1 = response1.text
        lines1 = content1.split("\n")
        uptime_line1 = [l for l in lines1 if l.startswith("morgan_uptime_seconds ")][0]
        uptime1 = float(uptime_line1.split()[1])

        # Wait a bit
        time.sleep(0.1)

        # Get metrics again
        response2 = client.get("/metrics")
        content2 = response2.text
        lines2 = content2.split("\n")
        uptime_line2 = [l for l in lines2 if l.startswith("morgan_uptime_seconds ")][0]
        uptime2 = float(uptime_line2.split()[1])

        # Uptime should have increased
        assert uptime2 > uptime1

    def test_metrics_counter_types(self, client):
        """Test that counter metrics are properly typed."""
        response = client.get("/metrics")
        content = response.text
        lines = content.split("\n")

        # Find counter type declarations
        counter_types = [
            "# TYPE morgan_requests_total counter",
            "# TYPE morgan_requests_success_total counter",
            "# TYPE morgan_requests_error_total counter",
            "# TYPE morgan_uptime_seconds counter",
        ]

        for counter_type in counter_types:
            assert counter_type in content, f"Missing counter type: {counter_type}"

    def test_metrics_gauge_types(self, client):
        """Test that gauge metrics are properly typed."""
        response = client.get("/metrics")
        content = response.text

        # Find gauge type declarations
        gauge_types = [
            "# TYPE morgan_response_time_seconds gauge",
            "# TYPE morgan_error_rate gauge",
            "# TYPE morgan_active_sessions gauge",
            "# TYPE morgan_requests_per_second gauge",
        ]

        for gauge_type in gauge_types:
            assert gauge_type in content, f"Missing gauge type: {gauge_type}"

    def test_metrics_help_text(self, client):
        """Test that metrics have help text."""
        response = client.get("/metrics")
        content = response.text

        # Verify help text exists for each metric
        help_texts = [
            "# HELP morgan_requests_total",
            "# HELP morgan_requests_success_total",
            "# HELP morgan_requests_error_total",
            "# HELP morgan_response_time_seconds",
            "# HELP morgan_error_rate",
            "# HELP morgan_active_sessions",
            "# HELP morgan_uptime_seconds",
            "# HELP morgan_requests_per_second",
        ]

        for help_text in help_texts:
            assert help_text in content, f"Missing help text: {help_text}"
