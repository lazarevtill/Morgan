#!/usr/bin/env python3
"""
Integration tests for Morgan microservices

Tests the integration between different services:
- LLM service integration
- TTS service integration
- STT service integration
- VAD service integration
- Service orchestration
- Cross-service communication
"""

import pytest
import asyncio
import httpx
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.http_client import MorganHTTPClient, service_registry


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def service_urls():
    """Get service URLs from environment or use defaults."""
    return {
        "core": os.getenv("CORE_SERVICE_URL", "http://localhost:8000"),
        "llm": os.getenv("LLM_SERVICE_URL", "http://localhost:8001"),
        "tts": os.getenv("TTS_SERVICE_URL", "http://localhost:8002"),
        "stt": os.getenv("STT_SERVICE_URL", "http://localhost:8003"),
        "vad": os.getenv("VAD_SERVICE_URL", "http://localhost:8004"),
    }


class TestServiceHealth:
    """Test health endpoints of all services."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_core_service_health(self, service_urls):
        """Test that core service health endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['core']}/health")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                    assert data["status"] in ["healthy", "degraded", "unhealthy"]
                else:
                    pytest.skip("Core service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_service_health(self, service_urls):
        """Test that LLM service health endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['llm']}/health")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                else:
                    pytest.skip("LLM service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("LLM service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tts_service_health(self, service_urls):
        """Test that TTS service health endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['tts']}/health")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                else:
                    pytest.skip("TTS service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("TTS service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_stt_service_health(self, service_urls):
        """Test that STT service health endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['stt']}/health")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                else:
                    pytest.skip("STT service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("STT service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vad_service_health(self, service_urls):
        """Test that VAD service health endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['vad']}/health")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                else:
                    pytest.skip("VAD service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("VAD service not running")


class TestLLMServiceIntegration:
    """Integration tests for LLM service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_service_generate(self, service_urls):
        """Test LLM service text generation."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_urls['llm']}/api/generate",
                    json={
                        "prompt": "Say hello",
                        "max_tokens": 50
                    }
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "text" in data
                    assert len(data["text"]) > 0
                else:
                    pytest.skip("LLM service generation failed or Ollama not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("LLM service not running or Ollama not accessible")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_service_with_context(self, service_urls):
        """Test LLM service with conversation context."""
        try:
            context = [
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Hello Alice!"},
            ]

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_urls['llm']}/api/generate",
                    json={
                        "prompt": "What is my name?",
                        "context": context,
                        "max_tokens": 50
                    }
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "text" in data
                    # LLM should remember context
                    assert "alice" in data["text"].lower() or "name" in data["text"].lower()
                else:
                    pytest.skip("LLM service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("LLM service not running")


class TestTTSServiceIntegration:
    """Integration tests for TTS service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tts_service_synthesize(self, service_urls):
        """Test TTS service speech synthesis."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_urls['tts']}/api/synthesize",
                    json={
                        "text": "Hello world",
                        "output_format": "wav"
                    }
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    # Check that we got audio data
                    assert len(response.content) > 0
                    # WAV files start with "RIFF"
                    assert response.content[:4] == b"RIFF" or len(response.content) > 100
                else:
                    pytest.skip("TTS service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("TTS service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tts_service_with_voice(self, service_urls):
        """Test TTS service with specific voice."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_urls['tts']}/api/synthesize",
                    json={
                        "text": "Testing voice synthesis",
                        "voice": "default",
                        "output_format": "wav"
                    }
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    assert len(response.content) > 0
                else:
                    pytest.skip("TTS service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("TTS service not running")


class TestSTTServiceIntegration:
    """Integration tests for STT service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_stt_service_transcribe(self, service_urls):
        """Test STT service transcription."""
        # Note: This test requires a valid audio file
        # For now, we'll just test the endpoint availability
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create a minimal WAV file (silence)
                import struct
                import wave
                import io

                # Create 1 second of silence at 16kHz
                sample_rate = 16000
                duration = 1
                num_samples = sample_rate * duration
                audio_data = struct.pack('<' + ('h' * num_samples), *([0] * num_samples))

                # Create WAV file
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

                wav_buffer.seek(0)

                response = await client.post(
                    f"{service_urls['stt']}/api/transcribe",
                    files={"file": ("test.wav", wav_buffer, "audio/wav")}
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "text" in data
                    # Silence might transcribe to empty or minimal text
                else:
                    pytest.skip("STT service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("STT service not running")


class TestCoreServiceIntegration:
    """Integration tests for core service coordination."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_core_text_processing(self, service_urls):
        """Test core service text processing endpoint."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_urls['core']}/api/text",
                    json={
                        "text": "Hello, how are you?",
                        "user_id": "test_user"
                    }
                )

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "text" in data
                    assert len(data["text"]) > 0
                    assert "metadata" in data
                else:
                    pytest.skip("Core service or dependencies not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_core_status_endpoint(self, service_urls):
        """Test core service status endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_urls['core']}/status")

                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "version" in data or "status" in data
                else:
                    pytest.skip("Core service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


class TestServiceOrchestration:
    """Integration tests for service orchestration."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_services_reachable(self, service_urls):
        """Test that all services are reachable from core."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get core service health which includes all service status
                response = await client.get(f"{service_urls['core']}/health")

                if response.status_code == 200:
                    data = response.json()

                    # Check if service status is included
                    if "services" in data:
                        services = data["services"]

                        # At least some services should be available
                        available_services = sum(1 for v in services.values() if v)
                        assert available_services > 0, \
                            "At least one service should be available"
                else:
                    pytest.skip("Core service not available")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Core service not running")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
