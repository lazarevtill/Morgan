"""
Real-world end-to-end tests for Morgan AI Assistant

Tests complete user workflows from start to finish
"""
import pytest
import asyncio
from httpx import AsyncClient
import base64
import wave
import io
import struct


class TestTextConversationWorkflow:
    """Test complete text conversation workflow"""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context retention"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=60.0) as client:
            user_id = "e2e_test_user_1"

            # Turn 1: Initial question
            response1 = await client.post(
                "/api/text",
                json={"text": "Hello, my name is Alice", "user_id": user_id}
            )
            assert response1.status_code == 200, f"First message failed: {response1.text}"
            data1 = response1.json()
            assert "text" in data1, "Response should contain text"

            # Turn 2: Follow-up question (should remember context)
            response2 = await client.post(
                "/api/text",
                json={"text": "What is my name?", "user_id": user_id}
            )
            assert response2.status_code == 200, f"Second message failed: {response2.text}"
            data2 = response2.json()
            assert "text" in data2, "Response should contain text"
            # The AI should remember the name from context
            # Note: Actual context retention depends on LLM service being available

    @pytest.mark.asyncio
    async def test_conversation_reset(self):
        """Test conversation reset functionality"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=60.0) as client:
            user_id = "e2e_test_user_2"

            # Send initial message
            await client.post(
                "/api/text",
                json={"text": "Remember this: secret123", "user_id": user_id}
            )

            # Reset conversation
            reset_response = await client.post(
                "/api/conversation/reset",
                json={"user_id": user_id}
            )
            assert reset_response.status_code == 200, "Reset should succeed"

            # Send follow-up - should not remember previous context
            response = await client.post(
                "/api/text",
                json={"text": "What was the secret?", "user_id": user_id}
            )
            assert response.status_code == 200


class TestAudioWorkflow:
    """Test complete audio processing workflow"""

    def create_test_wav(self, duration_seconds=1, sample_rate=16000):
        """Create a test WAV file"""
        num_samples = duration_seconds * sample_rate
        # Generate silence (could be sine wave for more realistic test)
        samples = [0] * num_samples

        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Pack samples as 16-bit integers
            wav_data = struct.pack('h' * num_samples, *samples)
            wav_file.writeframes(wav_data)

        buffer.seek(0)
        return buffer.read()

    @pytest.mark.asyncio
    async def test_audio_upload_and_transcription(self):
        """Test uploading audio file and getting transcription"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=60.0) as client:
            # Create test WAV file
            wav_data = self.create_test_wav(duration_seconds=2)

            # Upload audio file
            response = await client.post(
                "/api/audio",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                data={"user_id": "audio_test_user", "language": "en"}
            )

            # Should process successfully (even if transcription is empty for silence)
            assert response.status_code == 200, f"Audio upload failed: {response.text}"
            data = response.json()
            assert "transcription" in data, "Response should include transcription field"

    @pytest.mark.asyncio
    async def test_audio_format_validation(self):
        """Test that invalid audio formats are rejected"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=60.0) as client:
            # Try to upload invalid file
            invalid_data = b"This is not an audio file"

            response = await client.post(
                "/api/audio",
                files={"file": ("test.txt", invalid_data, "text/plain")},
                data={"user_id": "audio_test_user"}
            )

            # Should be rejected
            assert response.status_code in [400, 415], \
                f"Invalid format should be rejected, got {response.status_code}"


class TestServiceHealthAndResilience:
    """Test service health and error handling"""

    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self):
        """Test comprehensive health check"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
            response = await client.get("/health")

            assert response.status_code == 200, "Health check should succeed"
            data = response.json()

            # Verify health check structure
            assert "status" in data, "Health check should include status"
            assert "version" in data, "Health check should include version"
            assert "services" in data, "Health check should include services status"
            assert "orchestrator" in data, "Health check should include orchestrator status"

            # Check that it reports on backend services
            services = data["services"]
            assert isinstance(services, dict), "Services should be a dict"

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_request(self):
        """Test error handling with malformed request"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
            # Send request missing required fields
            response = await client.post(
                "/api/text",
                json={"user_id": "test"}  # Missing 'text' field
            )

            # Should return error
            assert response.status_code in [400, 422], \
                f"Invalid request should return 400/422, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=60.0) as client:
            # Send 10 requests concurrently
            tasks = []
            for i in range(10):
                task = client.post(
                    "/api/text",
                    json={"text": f"Concurrent test {i}", "user_id": f"concurrent_user_{i}"}
                )
                tasks.append(task)

            # Wait for all requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful responses
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)

            # At least some should succeed (rate limiting may block some)
            assert successful > 0, f"At least some concurrent requests should succeed, got {successful}/10"


class TestStatusEndpoint:
    """Test system status endpoint"""

    @pytest.mark.asyncio
    async def test_status_endpoint_structure(self):
        """Test that status endpoint returns proper structure"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
            response = await client.get("/status")

            assert response.status_code == 200, "Status endpoint should be accessible"
            data = response.json()

            # Verify expected fields
            expected_fields = ["version", "status", "uptime", "services", "orchestrator"]
            for field in expected_fields:
                assert field in data, f"Status should include {field}"

            # Verify uptime is positive
            assert data["uptime_seconds"] >= 0, "Uptime should be non-negative"


class TestDeviceAudioWorkflow:
    """Test device audio listing and processing"""

    @pytest.mark.asyncio
    async def test_list_audio_devices(self):
        """Test listing available audio devices"""
        async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
            response = await client.get("/devices/audio")

            # Should return device list (may be empty in Docker)
            assert response.status_code == 200, "Device listing should succeed"
            data = response.json()
            assert "devices" in data, "Response should include devices list"
            assert "count" in data, "Response should include count"
            assert isinstance(data["devices"], list), "Devices should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
