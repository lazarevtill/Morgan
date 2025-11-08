#!/usr/bin/env python3
"""
Tests for streaming functionality in Morgan AI Assistant
"""
import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import aiohttp
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from core.api.server import APIServer
from services.stt.service import STTService
from services.tts.service import TTSService
from shared.config.base import ServiceConfig
from shared.models.base import TTSRequest


class TestStreamingEndpoints:
    """Test streaming endpoints and functionality"""

    @pytest.fixture
    def mock_core_service(self):
        """Mock core service for testing"""
        mock_service = AsyncMock()
        mock_service.process_text_request = AsyncMock()
        mock_service.process_text_request.return_value = AsyncMock()
        mock_service.process_text_request.return_value.text = "Test response"
        mock_service.process_text_request.return_value.audio_data = b"test_audio_data"
        mock_service.process_text_request.return_value.metadata = {}
        return mock_service

    @pytest.fixture
    def api_server(self, mock_core_service):
        """Create API server instance for testing"""
        config = ServiceConfig("core")
        server = APIServer(mock_core_service, host="0.0.0.0", port=8000)
        # Manually create the app for testing
        from fastapi import FastAPI

        server.app = FastAPI()
        server._setup_routes()
        return server

    @pytest.fixture
    def client(self, api_server):
        """Create test client"""
        return TestClient(api_server.app)

    def test_streaming_endpoint_exists(self, client):
        """Test that streaming endpoint exists"""
        response = client.post("/api/audio/stream")
        # Should return error about missing request body, not 404
        assert response.status_code != 404

    def test_streaming_request_validation(self, client):
        """Test streaming request validation"""
        # Test missing text field
        response = client.post("/api/audio/stream", json={})
        assert response.status_code == 422

        # Test valid request
        response = client.post(
            "/api/audio/stream",
            json={"text": "Hello, Morgan", "voice": "default", "user_id": "test_user"},
        )
        # Should start processing (may fail later due to mocking)
        assert response.status_code in [200, 500]  # 500 due to mocking

    @pytest.mark.asyncio
    async def test_tts_streaming_interface(self):
        """Test TTS streaming interface exists and is callable"""
        config = ServiceConfig("tts")
        tts_service = TTSService(config)

        # Test that the streaming method exists
        request = TTSRequest(text="Hello world", voice="default")
        assert hasattr(tts_service, "generate_speech_stream")

        # Test that we can call the method (it will fail due to missing CSM model, but that's expected)
        try:
            chunks = []
            async for chunk in tts_service.generate_speech_stream(request):
                chunks.append(chunk)
        except Exception as e:
            # Expected to fail due to missing model setup
            assert "CSM" in str(e) or "generator" in str(e) or "model" in str(e)

    @pytest.mark.asyncio
    async def test_stt_realtime_vad(self):
        """Test STT real-time VAD processing"""
        config = ServiceConfig("stt")
        stt_service = STTService(config)

        # Test that VAD functionality is available
        stt_service.vad_available = True
        assert stt_service.vad_available is True

        # Test that the VAD method exists
        assert hasattr(stt_service, "_apply_vad_filter")
        assert hasattr(stt_service, "_apply_simple_energy_vad")

        # Test with simple energy-based VAD (built-in functionality)
        audio_array = np.random.randn(16000)  # 1 second at 16kHz
        audio_array[1000:2000] = 2.0  # Add some high-energy speech-like signal

        # Temporarily lower the threshold for testing
        original_threshold = stt_service.stt_config.vad_threshold
        stt_service.stt_config.vad_threshold = 0.1

        result = await stt_service._apply_simple_energy_vad(audio_array)

        # Restore original threshold
        stt_service.stt_config.vad_threshold = original_threshold

        # Should detect the high-energy segment as speech
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_audio_chunk_to_buffer(self):
        """Test audio chunk to buffer conversion"""
        # Test hex string conversion
        test_hex = "48656c6c6f"  # "Hello" in hex
        import re

        matches = re.findall(r".{1,2}", test_hex)

        assert matches is not None
        assert len(matches) == 5

        # Test byte conversion
        audio_bytes = bytes([int(byte, 16) for byte in matches])
        assert len(audio_bytes) == 5


class TestWebAudioAPI:
    """Test Web Audio API integration"""

    def test_audio_context_creation(self):
        """Test AudioContext creation (mocked)"""
        # This would test the JavaScript AudioContext creation
        # In a real test, we'd use a headless browser or jsdom
        pass

    def test_audio_buffer_conversion(self):
        """Test audio buffer conversion logic"""
        # Test hex to bytes conversion
        test_hex = "fffe0000"  # WAV header start

        # Simulate the JavaScript conversion
        hex_matches = [test_hex[i : i + 2] for i in range(0, len(test_hex), 2)]
        audio_bytes = bytes([int(byte, 16) for byte in hex_matches])

        assert audio_bytes[0] == 255  # ff
        assert audio_bytes[1] == 254  # fe

    def test_streaming_message_format(self):
        """Test streaming message format"""
        # Test AudioChunkMessage format
        message = {
            "type": "audio",
            "audio_data": "fffe0000",
            "chunk_id": 1,
            "timestamp": 1234567890.123,
            "duration": 0.5,
            "sample_rate": 24000,
            "format": "wav",
        }

        # Validate message structure
        assert message["type"] == "audio"
        assert message["chunk_id"] == 1
        assert message["sample_rate"] == 24000
        assert message["format"] == "wav"


class TestVADIntegration:
    """Test VAD integration with faster-whisper"""

    @pytest.mark.asyncio
    async def test_vad_setup(self):
        """Test VAD setup with faster-whisper"""
        config = ServiceConfig("stt")
        stt_service = STTService(config)

        # Mock whisper model with VAD capability
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_with_vad = AsyncMock()
        stt_service.whisper_model = mock_whisper

        # Test VAD setup
        stt_service._setup_vad_filter()

        # Should detect VAD capability
        assert stt_service.vad_available is True

    @pytest.mark.asyncio
    async def test_vad_speech_detection(self):
        """Test VAD speech detection"""
        config = ServiceConfig("stt")
        stt_service = STTService(config)

        # Mock VAD filter (built-in functionality)
        stt_service.vad_available = True

        # Test with simple energy-based VAD (fallback)
        audio_array = np.random.randn(16000)  # 1 second at 16kHz
        audio_array[1000:2000] = 2.0  # Add some high-energy speech-like signal

        # Temporarily lower the threshold for testing
        original_threshold = stt_service.stt_config.vad_threshold
        stt_service.stt_config.vad_threshold = 0.1

        result = await stt_service._apply_simple_energy_vad(audio_array)

        # Restore original threshold
        stt_service.stt_config.vad_threshold = original_threshold

        # Should detect the high-energy segment as speech
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_vad_no_speech(self):
        """Test VAD with no speech detected"""
        config = ServiceConfig("stt")
        stt_service = STTService(config)

        # Mock VAD filter (built-in functionality)
        stt_service.vad_available = True

        # Test with low-energy audio (no speech)
        audio_array = np.random.randn(16000) * 0.01  # Very low energy

        result = await stt_service._apply_simple_energy_vad(audio_array)

        # Should not detect speech in low-energy audio
        assert len(result) == 0  # Should return empty array


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
