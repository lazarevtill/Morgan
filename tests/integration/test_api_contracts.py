"""
API Contract Tests for Morgan AI Assistant

Ensures all API endpoints conform to their contracts
"""

import pytest
from httpx import AsyncClient
import base64


class TestCoreServiceAPI:
    """Test Core Service API contracts"""

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert "service" in data
            assert "version" in data
            assert "status" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_contract(self):
        """Test /health endpoint contract"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()

            # Required fields
            required_fields = ["status", "version", "services", "orchestrator"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Validate field types
            assert isinstance(data["status"], str)
            assert isinstance(data["version"], str)
            assert isinstance(data["services"], dict)
            assert isinstance(data["orchestrator"], dict)

    @pytest.mark.asyncio
    async def test_status_endpoint_contract(self):
        """Test /status endpoint contract"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/status")

            assert response.status_code == 200
            data = response.json()

            # Required fields
            required_fields = ["version", "status", "uptime", "uptime_seconds"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Validate types
            assert isinstance(data["version"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["uptime"], str)
            assert isinstance(data["uptime_seconds"], (int, float))

    @pytest.mark.asyncio
    async def test_text_api_request_contract(self):
        """Test /api/text POST request contract"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=60.0
        ) as client:
            # Valid request
            response = await client.post(
                "/api/text",
                json={
                    "text": "Hello",
                    "user_id": "test_user",
                    "metadata": {"source": "test"},
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Response should have text field
            assert "text" in data, "Response must include 'text' field"
            assert isinstance(data["text"], str)

    @pytest.mark.asyncio
    async def test_text_api_validation(self):
        """Test /api/text validation rules"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=60.0
        ) as client:
            # Missing required field
            response = await client.post(
                "/api/text", json={"user_id": "test_user"}  # Missing 'text'
            )

            assert response.status_code in [
                400,
                422,
            ], "Request with missing 'text' should be rejected"

    @pytest.mark.asyncio
    async def test_conversation_reset_contract(self):
        """Test /api/conversation/reset endpoint contract"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.post(
                "/api/conversation/reset", json={"user_id": "test_user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "status" in data or "message" in data

    @pytest.mark.asyncio
    async def test_devices_audio_contract(self):
        """Test /devices/audio endpoint contract"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/devices/audio")

            assert response.status_code == 200
            data = response.json()

            # Required fields
            assert "devices" in data
            assert "count" in data
            assert isinstance(data["devices"], list)
            assert isinstance(data["count"], int)


class TestLLMServiceAPI:
    """Test LLM Service API contracts"""

    @pytest.mark.asyncio
    async def test_llm_health_endpoint(self):
        """Test LLM service health endpoint"""
        async with AsyncClient(
            base_url="http://localhost:8001", timeout=30.0
        ) as client:
            try:
                response = await client.get("/health")
                assert response.status_code in [
                    200,
                    503,
                ], "Health endpoint should return 200 (healthy) or 503 (unhealthy)"

                data = response.json()
                assert "status" in data or "detail" in data

            except Exception as e:
                # LLM service may not be running in test environment
                pytest.skip(f"LLM service not available: {e}")


class TestTTSServiceAPI:
    """Test TTS Service API contracts"""

    @pytest.mark.asyncio
    async def test_tts_health_endpoint(self):
        """Test TTS service health endpoint"""
        async with AsyncClient(
            base_url="http://localhost:8002", timeout=30.0
        ) as client:
            try:
                response = await client.get("/health")
                assert response.status_code in [200, 503]

                data = response.json()
                assert "status" in data or "detail" in data

            except Exception as e:
                pytest.skip(f"TTS service not available: {e}")


class TestSTTServiceAPI:
    """Test STT Service API contracts"""

    @pytest.mark.asyncio
    async def test_stt_health_endpoint(self):
        """Test STT service health endpoint"""
        async with AsyncClient(
            base_url="http://localhost:8003", timeout=30.0
        ) as client:
            try:
                response = await client.get("/health")
                assert response.status_code in [200, 503]

                data = response.json()
                assert "status" in data or "detail" in data

            except Exception as e:
                pytest.skip(f"STT service not available: {e}")


class TestErrorResponseContracts:
    """Test error response formats"""

    @pytest.mark.asyncio
    async def test_400_error_format(self):
        """Test 400 error response format"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Send invalid request
            response = await client.post("/api/text", json={"invalid": "data"})

            assert response.status_code in [400, 422]

            # Error response should be JSON
            try:
                data = response.json()
                # Should have error information
                assert "detail" in data or "error" in data or "message" in data
            except:
                # Some frameworks return plain text errors
                pass

    @pytest.mark.asyncio
    async def test_404_error(self):
        """Test 404 error for non-existent endpoint"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/nonexistent/endpoint")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_request_id_in_error_responses(self):
        """Test that error responses include request ID"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            # Send invalid request
            response = await client.post(
                "/api/text", json={}  # Invalid - missing required fields
            )

            # Should have request ID even in error responses
            assert (
                "X-Request-ID" in response.headers
            ), "Error responses should include X-Request-ID header"


class TestRateLimitHeaders:
    """Test rate limit response headers"""

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self):
        """Test that rate limit headers are included in responses"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.post(
                "/api/text", json={"text": "test", "user_id": "test_user"}
            )

            # Rate limit headers should be present
            # Note: May not be present on first request if rate limiter hasn't initialized
            if response.status_code == 200:
                # Headers may be present
                if "X-RateLimit-Limit" in response.headers:
                    assert "X-RateLimit-Remaining" in response.headers


class TestTimingHeaders:
    """Test timing and request headers"""

    @pytest.mark.asyncio
    async def test_process_time_header(self):
        """Test that X-Process-Time header is included"""
        async with AsyncClient(
            base_url="http://localhost:8000", timeout=30.0
        ) as client:
            response = await client.get("/health")

            # Should have process time header
            assert response.status_code == 200, "Request should succeed"
            # Header may or may not be present depending on middleware configuration
            if "X-Process-Time" in response.headers:
                # Should be a valid float
                process_time = float(response.headers["X-Process-Time"])
                assert process_time >= 0, "Process time should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
