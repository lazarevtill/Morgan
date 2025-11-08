#!/usr/bin/env python3
"""
Unit tests for Morgan Core Service

Tests the core service functionality including:
- Conversation management
- Service orchestration
- Request handling
- Configuration
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config.base import ServiceConfig, BaseConfig
from shared.models.base import Message, ConversationContext
from shared.utils.http_client import MorganHTTPClient, ServiceRegistry
from shared.utils.errors import ServiceError, ErrorCode


class TestServiceConfig:
    """Test configuration system."""

    def test_base_config_loads_yaml(self, tmp_path):
        """Test that BaseConfig can load YAML files."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
host: "0.0.0.0"
port: 8000
log_level: "INFO"
        """)

        config = BaseConfig(str(config_file))
        assert config.get("host") == "0.0.0.0"
        assert config.get("port") == 8000
        assert config.get("log_level") == "INFO"

    def test_config_get_with_default(self, tmp_path):
        """Test config get with default value."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("host: '0.0.0.0'")

        config = BaseConfig(str(config_file))
        assert config.get("nonexistent", "default") == "default"

    def test_config_env_override(self, tmp_path, monkeypatch):
        """Test that environment variables override config values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("port: 8000")

        monkeypatch.setenv("MORGAN_PORT", "9000")

        config = BaseConfig(str(config_file))
        assert config.get("port") == "9000"  # String from env var

    def test_service_config_auto_discovers_config_dir(self, tmp_path, monkeypatch):
        """Test that ServiceConfig can auto-discover config directory."""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create service config file
        service_config = config_dir / "core.yaml"
        service_config.write_text("test_value: 42")

        # Set config dir env var
        monkeypatch.setenv("MORGAN_CONFIG_DIR", str(config_dir))

        config = ServiceConfig("core")
        assert config.get("test_value") == 42


class TestDataModels:
    """Test data models."""

    def test_message_creation(self):
        """Test Message model creation."""
        msg = Message(
            role="user",
            content="Hello",
            timestamp="2025-01-01T00:00:00Z",
            metadata={"source": "test"}
        )

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == "2025-01-01T00:00:00Z"
        assert msg.metadata["source"] == "test"

    def test_message_to_dict(self):
        """Test Message serialization to dict."""
        msg = Message(
            role="assistant",
            content="Hi there",
            timestamp="2025-01-01T00:00:00Z",
            metadata={}
        )

        msg_dict = msg.to_dict()
        assert isinstance(msg_dict, dict)
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there"

    def test_message_to_json(self):
        """Test Message serialization to JSON."""
        msg = Message(
            role="user",
            content="Test",
            timestamp="2025-01-01T00:00:00Z",
            metadata={}
        )

        json_str = msg.to_json()
        assert isinstance(json_str, str)
        assert "user" in json_str
        assert "Test" in json_str

    def test_conversation_context_creation(self):
        """Test ConversationContext model creation."""
        msg1 = Message(role="user", content="Hello", timestamp="2025-01-01T00:00:00Z")
        msg2 = Message(role="assistant", content="Hi", timestamp="2025-01-01T00:00:01Z")

        ctx = ConversationContext(
            conversation_id="conv_123",
            user_id="user_456",
            messages=[msg1, msg2],
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:01Z",
            metadata={}
        )

        assert ctx.conversation_id == "conv_123"
        assert ctx.user_id == "user_456"
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"


class TestHTTPClient:
    """Test HTTP client functionality."""

    @pytest.mark.asyncio
    async def test_http_client_initialization(self):
        """Test MorganHTTPClient initialization."""
        client = MorganHTTPClient(
            base_url="http://test-service:8001",
            timeout=30.0,
            max_retries=3
        )

        assert client.base_url == "http://test-service:8001"
        assert client.timeout == 30.0
        assert client.max_retries == 3

    @pytest.mark.asyncio
    async def test_http_client_get_request(self):
        """Test HTTP GET request."""
        client = MorganHTTPClient("http://test-service:8001")

        with patch('shared.utils.http_client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.get("/health")

            assert result == {"status": "ok"}
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_client_post_request(self):
        """Test HTTP POST request."""
        client = MorganHTTPClient("http://test-service:8001")

        with patch('shared.utils.http_client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.post("/api/process", json={"text": "test"})

            assert result == {"result": "success"}
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_client_retry_on_failure(self):
        """Test that HTTP client retries on failure."""
        client = MorganHTTPClient("http://test-service:8001", max_retries=3)

        with patch('shared.utils.http_client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()

            # First two calls fail, third succeeds
            mock_response_fail = AsyncMock()
            mock_response_fail.status_code = 500
            mock_response_fail.raise_for_status.side_effect = Exception("Server error")

            mock_response_success = AsyncMock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {"status": "ok"}

            mock_client.get.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                mock_response_success
            ]

            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.get("/health")

            assert result == {"status": "ok"}
            assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_http_client_health_check(self):
        """Test HTTP client health check."""
        client = MorganHTTPClient("http://test-service:8001")

        with patch('shared.utils.http_client.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            is_healthy = await client.health_check()

            assert is_healthy is True


class TestServiceRegistry:
    """Test service registry."""

    def test_service_registry_register(self):
        """Test service registration."""
        registry = ServiceRegistry()

        client = registry.register_service(
            "llm",
            "http://llm-service:8001",
            timeout=30.0
        )

        assert isinstance(client, MorganHTTPClient)
        assert "llm" in registry._services

    def test_service_registry_get_service(self):
        """Test getting service from registry."""
        registry = ServiceRegistry()

        registry.register_service("llm", "http://llm-service:8001")
        client = registry.get_service("llm")

        assert isinstance(client, MorganHTTPClient)
        assert client.base_url == "http://llm-service:8001"

    def test_service_registry_get_nonexistent_service(self):
        """Test getting nonexistent service raises error."""
        registry = ServiceRegistry()

        with pytest.raises(KeyError):
            registry.get_service("nonexistent")

    @pytest.mark.asyncio
    async def test_service_registry_health_check_all(self):
        """Test checking health of all services."""
        registry = ServiceRegistry()

        with patch.object(MorganHTTPClient, 'health_check', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = True

            registry.register_service("llm", "http://llm-service:8001")
            registry.register_service("tts", "http://tts-service:8002")

            health_status = await registry.health_check_all()

            assert health_status == {"llm": True, "tts": True}


class TestCustomErrors:
    """Test custom error classes."""

    def test_service_error_creation(self):
        """Test ServiceError creation."""
        error = ServiceError(
            message="Service unavailable",
            code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=503
        )

        assert error.message == "Service unavailable"
        assert error.code == ErrorCode.SERVICE_UNAVAILABLE
        assert error.status_code == 503

    def test_service_error_string_representation(self):
        """Test ServiceError string representation."""
        error = ServiceError(
            message="Test error",
            code=ErrorCode.INVALID_REQUEST,
            status_code=400
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "INVALID_REQUEST" in error_str or "400" in error_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
