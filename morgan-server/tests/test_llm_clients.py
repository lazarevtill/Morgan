"""Unit tests for LLM clients."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientError, ClientTimeout
import asyncio

from morgan_server.llm import (
    LLMMessage,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
    OllamaClient,
    OpenAICompatibleClient,
)


# Test data
SAMPLE_MESSAGES = [
    LLMMessage(role="system", content="You are a helpful assistant."),
    LLMMessage(role="user", content="Hello, how are you?"),
]


class TestOllamaClient:
    """Tests for OllamaClient."""

    @pytest.fixture
    def ollama_client(self):
        """Create an Ollama client for testing."""
        return OllamaClient(
            endpoint="http://localhost:11434",
            model="gemma3",
            timeout=30,
            max_retries=2,
            retry_delay=0.1,
        )

    @pytest.mark.asyncio
    async def test_ollama_generate_success(self, ollama_client):
        """Test successful generation with Ollama."""
        mock_response = {
            "message": {"content": "Hello! I'm doing well, thank you."},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 15,
            "total_duration": 1000000,
        }

        with patch.object(
            ollama_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            response = await ollama_client.generate(SAMPLE_MESSAGES)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! I'm doing well, thank you."
            assert response.model == "gemma3"
            assert response.finish_reason == "stop"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 15
            assert response.usage["total_tokens"] == 25

    @pytest.mark.asyncio
    async def test_ollama_generate_with_parameters(self, ollama_client):
        """Test generation with custom parameters."""
        mock_response = {
            "message": {"content": "Response"},
            "done_reason": "stop",
        }

        with patch.object(
            ollama_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            await ollama_client.generate(
                SAMPLE_MESSAGES,
                temperature=0.5,
                max_tokens=100,
            )

            # Verify the request was made with correct parameters
            call_args = mock_request.call_args
            payload = call_args[0][1]
            assert payload["options"]["temperature"] == 0.5
            assert payload["options"]["num_predict"] == 100

    @pytest.mark.asyncio
    async def test_ollama_generate_stream(self, ollama_client):
        """Test streaming generation with Ollama - basic functionality."""
        # Test that streaming method exists and can be called
        # Full integration testing would require a real Ollama instance
        assert hasattr(ollama_client, "generate_stream")
        assert callable(ollama_client.generate_stream)

    @pytest.mark.asyncio
    async def test_ollama_connection_error(self, ollama_client):
        """Test handling of connection errors."""
        with patch.object(
            ollama_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMConnectionError("Connection failed")

            with pytest.raises(LLMConnectionError):
                await ollama_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_ollama_timeout_error(self, ollama_client):
        """Test handling of timeout errors."""
        with patch.object(
            ollama_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMTimeoutError("Request timed out")

            with pytest.raises(LLMTimeoutError):
                await ollama_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_ollama_rate_limit_error(self, ollama_client):
        """Test handling of rate limit errors."""
        with patch.object(
            ollama_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMRateLimitError("Rate limit exceeded")

            with pytest.raises(LLMRateLimitError):
                await ollama_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_ollama_retry_configuration(self, ollama_client):
        """Test that retry configuration is properly set."""
        assert ollama_client.max_retries == 2
        assert ollama_client.retry_delay == 0.1

    @pytest.mark.asyncio
    async def test_ollama_health_check_method_exists(self, ollama_client):
        """Test that health check method exists."""
        assert hasattr(ollama_client, "health_check")
        assert callable(ollama_client.health_check)

    @pytest.mark.asyncio
    async def test_ollama_health_check_failure(self, ollama_client):
        """Test failed health check."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = ClientError("Connection failed")

        with patch.object(ollama_client, "_get_session", return_value=mock_session):
            result = await ollama_client.health_check()
            assert result is False


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient."""

    @pytest.fixture
    def openai_client(self):
        """Create an OpenAI-compatible client for testing."""
        return OpenAICompatibleClient(
            endpoint="http://localhost:1234/v1",
            model="gpt-3.5-turbo",
            api_key="test-key",
            timeout=30,
            max_retries=2,
            retry_delay=0.1,
        )

    @pytest.mark.asyncio
    async def test_openai_generate_success(self, openai_client):
        """Test successful generation with OpenAI-compatible API."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello! How can I help?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
            },
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            response = await openai_client.generate(SAMPLE_MESSAGES)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help?"
            assert response.model == "gpt-3.5-turbo"
            assert response.finish_reason == "stop"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 15

    @pytest.mark.asyncio
    async def test_openai_generate_with_parameters(self, openai_client):
        """Test generation with custom parameters."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            await openai_client.generate(
                SAMPLE_MESSAGES,
                temperature=0.8,
                max_tokens=200,
            )

            # Verify the request was made with correct parameters
            call_args = mock_request.call_args
            payload = call_args[0][1]
            assert payload["temperature"] == 0.8
            assert payload["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_openai_generate_stream(self, openai_client):
        """Test streaming generation method exists."""
        # Test that streaming method exists and can be called
        assert hasattr(openai_client, "generate_stream")
        assert callable(openai_client.generate_stream)

    @pytest.mark.asyncio
    async def test_openai_connection_error(self, openai_client):
        """Test handling of connection errors."""
        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMConnectionError("Connection failed")

            with pytest.raises(LLMConnectionError):
                await openai_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_openai_timeout_error(self, openai_client):
        """Test handling of timeout errors."""
        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMTimeoutError("Request timed out")

            with pytest.raises(LLMTimeoutError):
                await openai_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self, openai_client):
        """Test handling of rate limit errors."""
        with patch.object(
            openai_client, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = LLMRateLimitError("Rate limit exceeded")

            with pytest.raises(LLMRateLimitError):
                await openai_client.generate(SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_openai_api_key_in_headers(self, openai_client):
        """Test that API key is included in request headers."""
        mock_session = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session) as mock_cls:
            await openai_client._get_session()

            # Verify session was created with Authorization header
            call_kwargs = mock_cls.call_args.kwargs
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_openai_health_check_method_exists(self, openai_client):
        """Test that health check method exists."""
        assert hasattr(openai_client, "health_check")
        assert callable(openai_client.health_check)

    @pytest.mark.asyncio
    async def test_openai_health_check_failure(self, openai_client):
        """Test failed health check."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = ClientError("Connection failed")

        with patch.object(openai_client, "_get_session", return_value=mock_session):
            result = await openai_client.health_check()
            assert result is False


class AsyncIterableMock:
    """Mock for async iterable (like aiohttp response.content)."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)
