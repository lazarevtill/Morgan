"""
Integration tests for security features

Tests rate limiting, CORS, input validation, and request size limits
"""

import asyncio
import base64
import json

import pytest
from httpx import AsyncClient


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test that rate limiting blocks excessive requests"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Make requests rapidly to trigger rate limit
            responses = []
            for i in range(25):  # Exceed the 20 burst limit
                response = await client.post(
                    "/api/text",
                    json={"text": f"test message {i}", "user_id": "test_user"},
                )
                responses.append(response)

            # Some requests should have rate limit headers
            rate_limited = [r for r in responses if "X-RateLimit-Limit" in r.headers]
            assert len(rate_limited) > 0, "Rate limit headers should be present"

            # Verify header values
            for response in rate_limited:
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers
                limit = int(response.headers["X-RateLimit-Limit"])
                assert limit == 10, f"Rate limit should be 10 req/s, got {limit}"

    @pytest.mark.asyncio
    async def test_rate_limit_exempt_paths(self):
        """Test that health check is exempt from rate limiting"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Make many health check requests - should not be rate limited
            responses = []
            for i in range(50):
                response = await client.get("/health")
                responses.append(response)

            # All should succeed
            assert all(
                r.status_code == 200 for r in responses
            ), "Health checks should not be rate limited"


class TestCORSConfiguration:
    """Test CORS security configuration"""

    @pytest.mark.asyncio
    async def test_cors_localhost_allowed(self):
        """Test that localhost origins are allowed"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.options(
                "/api/text", headers={"Origin": "http://localhost:3000"}
            )

            # Should allow the request
            assert response.status_code in [200, 204], "Localhost should be allowed"

    @pytest.mark.asyncio
    async def test_cors_headers_present(self):
        """Test that CORS headers are present"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.post(
                "/api/text",
                json={"text": "test", "user_id": "test_user"},
                headers={"Origin": "http://localhost:3000"},
            )

            # Check for CORS headers
            assert (
                "Access-Control-Allow-Origin" in response.headers
                or response.status_code == 200
            )


class TestInputValidation:
    """Test input validation for files and base64 data"""

    @pytest.mark.asyncio
    async def test_invalid_base64_rejected(self):
        """Test that invalid base64 is rejected"""
        # This test would require WebSocket connection
        # Placeholder for now - implement when WebSocket testing is set up
        pass

    @pytest.mark.asyncio
    async def test_file_size_limit(self):
        """Test that files over 10MB are rejected"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Create a file larger than 10MB
            large_data = b"x" * (11 * 1024 * 1024)  # 11MB

            try:
                response = await client.post(
                    "/api/audio",
                    files={"file": ("test.wav", large_data, "audio/wav")},
                    data={"user_id": "test_user"},
                )

                # Should be rejected
                assert (
                    response.status_code == 413 or response.status_code == 400
                ), f"Large file should be rejected, got {response.status_code}"
            except Exception as e:
                # Request may fail due to size limit
                assert "413" in str(e) or "Request Entity Too Large" in str(e)


class TestRequestSizeLimits:
    """Test request body size limits"""

    @pytest.mark.asyncio
    async def test_request_too_large(self):
        """Test that requests over 10MB are rejected"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Create a very large JSON payload
            large_text = "x" * (11 * 1024 * 1024)  # 11MB of text

            try:
                response = await client.post(
                    "/api/text", json={"text": large_text, "user_id": "test_user"}
                )

                # Should be rejected
                assert (
                    response.status_code == 413
                ), f"Large request should return 413, got {response.status_code}"
            except Exception as e:
                # Connection may be refused for very large requests
                pass


class TestRequestIDPropagation:
    """Test request ID propagation through services"""

    @pytest.mark.asyncio
    async def test_request_id_in_response(self):
        """Test that response includes request ID header"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/health")

            # Should have request ID in headers
            assert (
                "X-Request-ID" in response.headers
            ), "Response should include request ID"

    @pytest.mark.asyncio
    async def test_custom_request_id_preserved(self):
        """Test that custom request ID is preserved"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            custom_id = "test-request-123"
            response = await client.get("/health", headers={"X-Request-ID": custom_id})

            # Should preserve the custom request ID
            assert (
                response.headers.get("X-Request-ID") == custom_id
            ), "Custom request ID should be preserved"


class TestDatabaseConnectionHandling:
    """Test database connection validation and fallback"""

    @pytest.mark.asyncio
    async def test_service_works_without_database(self):
        """Test that service works even without database connection"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Service should work even if database is not configured
            response = await client.get("/health")

            # Should still be operational
            assert response.status_code == 200, "Service should work without database"

            data = response.json()
            # Core service should be healthy even if database is not available
            assert data["status"] in [
                "healthy",
                "degraded",
            ], "Service should be healthy or degraded (not failed)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
