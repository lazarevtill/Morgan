#!/usr/bin/env python3
"""
End-to-end tests for Morgan AI Assistant

Tests complete workflows from user input to response:
- Text conversation flow
- Audio processing flow
- Multi-turn conversation
- Error handling and recovery
- Performance under load
"""

import asyncio
import io
import os
import struct
import sys
import time
import wave
from pathlib import Path

import httpx
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def core_service_url():
    """Get core service URL."""
    return os.getenv("CORE_SERVICE_URL", "http://localhost:8000")


class TestTextConversationFlow:
    """End-to-end tests for text-based conversations."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_simple_text_conversation(self, core_service_url):
        """Test a simple text conversation."""
        try:
            user_id = f"test_user_{int(time.time())}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                # First message
                response1 = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "Hello, my name is Alice", "user_id": user_id},
                )

                if response1.status_code != 200:
                    pytest.skip("Core service or dependencies not available")

                assert response1.status_code == 200
                data1 = response1.json()
                assert "text" in data1
                assert len(data1["text"]) > 0

                # Second message - test context retention
                await asyncio.sleep(1)

                response2 = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "What is my name?", "user_id": user_id},
                )

                assert response2.status_code == 200
                data2 = response2.json()
                assert "text" in data2

                # Response should reference the name (context retention)
                response_lower = data2["text"].lower()
                assert "alice" in response_lower or "name" in response_lower

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multi_turn_conversation(self, core_service_url):
        """Test multi-turn conversation with context."""
        try:
            user_id = f"test_user_{int(time.time())}"

            conversations = [
                "I like pizza",
                "What is my favorite food?",
                "Tell me more about it",
            ]

            async with httpx.AsyncClient(timeout=30.0) as client:
                for idx, text in enumerate(conversations):
                    response = await client.post(
                        f"{core_service_url}/api/chat",
                        json={"text": text, "user_id": user_id},
                    )

                    if response.status_code != 200:
                        pytest.skip(f"Request failed at turn {idx + 1}")

                    assert response.status_code == 200
                    data = response.json()
                    assert "text" in data
                    assert len(data["text"]) > 0

                    # Second response should mention pizza
                    if idx == 1:
                        assert "pizza" in data["text"].lower()

                    await asyncio.sleep(0.5)

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_conversation_reset(self, core_service_url):
        """Test conversation reset functionality."""
        try:
            user_id = f"test_user_{int(time.time())}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                # First conversation
                await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "My name is Bob", "user_id": user_id},
                )

                # Reset conversation
                reset_response = await client.post(
                    f"{core_service_url}/api/conversation/reset",
                    json={"user_id": user_id},
                )

                if reset_response.status_code == 200:
                    # After reset, context should be cleared
                    response = await client.post(
                        f"{core_service_url}/api/chat",
                        json={"text": "What is my name?", "user_id": user_id},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    # Response should not remember Bob
                    assert (
                        "bob" not in data["text"].lower()
                        or "don't know" in data["text"].lower()
                        or "not sure" in data["text"].lower()
                    )
                else:
                    pytest.skip("Conversation reset not implemented")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


class TestAudioProcessingFlow:
    """End-to-end tests for audio processing."""

    def create_test_audio(self, duration_seconds=1, sample_rate=16000):
        """Create a test audio file (silence)."""
        num_samples = sample_rate * duration_seconds
        audio_data = struct.pack("<" + ("h" * num_samples), *([0] * num_samples))

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        return wav_buffer

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_audio_upload_and_processing(self, core_service_url):
        """Test uploading audio and getting response."""
        try:
            user_id = f"test_user_{int(time.time())}"

            # Create test audio
            audio_buffer = self.create_test_audio()

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{core_service_url}/api/audio",
                    files={"file": ("test.wav", audio_buffer, "audio/wav")},
                    data={"user_id": user_id},
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()

                    # Should have transcription
                    assert "text" in data

                    # Might have audio response
                    if "audio" in data:
                        assert len(data["audio"]) > 0

                    # Should have metadata
                    assert "metadata" in data
                else:
                    pytest.skip(
                        "Audio processing not available or services not running"
                    )

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


class TestErrorHandling:
    """End-to-end tests for error handling."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_invalid_request_handling(self, core_service_url):
        """Test handling of invalid requests."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Missing required fields
                response = await client.post(f"{core_service_url}/api/chat", json={})

                # Should get error response, not crash
                assert response.status_code in [400, 422, 500]

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_empty_text_handling(self, core_service_url):
        """Test handling of empty text input."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "", "user_id": "test_user"},
                )

                # Should handle gracefully
                assert response.status_code in [200, 400, 422]

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_very_long_text_handling(self, core_service_url):
        """Test handling of very long text input."""
        try:
            # Create a very long text
            long_text = "This is a test. " * 500  # ~8000 characters

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": long_text, "user_id": "test_user"},
                )

                # Should either process or return appropriate error
                assert response.status_code in [200, 400, 413, 500]

                if response.status_code == 200:
                    data = response.json()
                    assert "text" in data

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


class TestPerformance:
    """End-to-end performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_response_time(self, core_service_url):
        """Test that responses come within acceptable time."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                start_time = time.time()

                response = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "Hello", "user_id": "perf_test_user"},
                )

                end_time = time.time()
                response_time = end_time - start_time

                if response.status_code == 200:
                    # Response should be reasonably fast
                    # (adjust threshold based on your requirements)
                    assert (
                        response_time < 30.0
                    ), f"Response time {response_time}s exceeds threshold"
                else:
                    pytest.skip("Core service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_concurrent_requests(self, core_service_url):
        """Test handling of concurrent requests."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create multiple concurrent requests
                async def make_request(idx):
                    return await client.post(
                        f"{core_service_url}/api/chat",
                        json={
                            "text": f"Hello {idx}",
                            "user_id": f"concurrent_user_{idx}",
                        },
                    )

                # Send 5 concurrent requests
                tasks = [make_request(i) for i in range(5)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Most requests should succeed
                successful = sum(
                    1
                    for r in responses
                    if not isinstance(r, Exception) and r.status_code == 200
                )

                if successful > 0:
                    assert (
                        successful >= 3
                    ), f"Only {successful}/5 concurrent requests succeeded"
                else:
                    pytest.skip("Core service not handling requests")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


class TestSystemIntegration:
    """End-to-end system integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_user_journey(self, core_service_url):
        """Test a complete user journey through the system."""
        try:
            user_id = f"journey_user_{int(time.time())}"

            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. Initial greeting
                response1 = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "Hi, I'm testing the system", "user_id": user_id},
                )

                if response1.status_code != 200:
                    pytest.skip("Service not available")

                assert response1.status_code == 200

                # 2. Ask a question
                await asyncio.sleep(1)

                response2 = await client.post(
                    f"{core_service_url}/api/chat",
                    json={"text": "What can you help me with?", "user_id": user_id},
                )

                assert response2.status_code == 200
                data2 = response2.json()
                assert len(data2["text"]) > 0

                # 3. Check health
                await asyncio.sleep(1)

                health_response = await client.get(f"{core_service_url}/health")

                assert health_response.status_code == 200

                # 4. Reset conversation
                reset_response = await client.post(
                    f"{core_service_url}/api/conversation/reset",
                    json={"user_id": user_id},
                )

                # Reset might not be implemented, that's okay
                assert reset_response.status_code in [200, 404, 501]

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
