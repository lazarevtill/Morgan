"""
Unit tests for client configuration.

Tests configuration loading from environment variables,
command-line arguments, and default values.

Feature: client-server-separation
Requirements: 2.2
"""

import os
import pytest
from unittest.mock import patch

from morgan_cli.config import Config, get_config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_server_url(self):
        """Test default server URL is set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.server_url == "http://localhost:8080"

    def test_default_api_key_is_none(self):
        """Test default API key is None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.api_key is None

    def test_default_user_id_is_none(self):
        """Test default user ID is None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.user_id is None

    def test_default_timeout(self):
        """Test default timeout is 60 seconds."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.timeout_seconds == 60

    def test_default_retry_attempts(self):
        """Test default retry attempts is 3."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.retry_attempts == 3

    def test_default_retry_delay(self):
        """Test default retry delay is 2 seconds."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.retry_delay_seconds == 2


class TestConfigEnvironmentVariables:
    """Test configuration loading from environment variables."""

    def test_server_url_from_env(self):
        """Test loading server URL from environment variable."""
        test_url = "http://test-server:9090"
        with patch.dict(os.environ, {"MORGAN_SERVER_URL": test_url}):
            config = Config()
            assert config.server_url == test_url

    def test_api_key_from_env(self):
        """Test loading API key from environment variable."""
        test_key = "test-api-key-12345"
        with patch.dict(os.environ, {"MORGAN_API_KEY": test_key}):
            config = Config()
            assert config.api_key == test_key

    def test_user_id_from_env(self):
        """Test loading user ID from environment variable."""
        test_user = "test-user-123"
        with patch.dict(os.environ, {"MORGAN_USER_ID": test_user}):
            config = Config()
            assert config.user_id == test_user

    def test_timeout_from_env(self):
        """Test loading timeout from environment variable."""
        with patch.dict(os.environ, {"MORGAN_TIMEOUT": "120"}):
            config = Config()
            assert config.timeout_seconds == 120

    def test_retry_attempts_from_env(self):
        """Test loading retry attempts from environment variable."""
        with patch.dict(os.environ, {"MORGAN_RETRY_ATTEMPTS": "5"}):
            config = Config()
            assert config.retry_attempts == 5

    def test_retry_delay_from_env(self):
        """Test loading retry delay from environment variable."""
        with patch.dict(os.environ, {"MORGAN_RETRY_DELAY": "10"}):
            config = Config()
            assert config.retry_delay_seconds == 10

    def test_multiple_env_vars(self):
        """Test loading multiple configuration values from env vars."""
        env_vars = {
            "MORGAN_SERVER_URL": "http://prod-server:8080",
            "MORGAN_API_KEY": "prod-key",
            "MORGAN_USER_ID": "prod-user",
            "MORGAN_TIMEOUT": "90",
            "MORGAN_RETRY_ATTEMPTS": "4",
            "MORGAN_RETRY_DELAY": "5"
        }
        with patch.dict(os.environ, env_vars):
            config = Config()
            assert config.server_url == "http://prod-server:8080"
            assert config.api_key == "prod-key"
            assert config.user_id == "prod-user"
            assert config.timeout_seconds == 90
            assert config.retry_attempts == 4
            assert config.retry_delay_seconds == 5


class TestConfigFromEnv:
    """Test Config.from_env() class method."""

    def test_from_env_creates_config(self):
        """Test from_env creates a Config instance."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            assert isinstance(config, Config)

    def test_from_env_uses_environment(self):
        """Test from_env uses environment variables."""
        test_url = "http://env-server:7070"
        with patch.dict(os.environ, {"MORGAN_SERVER_URL": test_url}):
            config = Config.from_env()
            assert config.server_url == test_url


class TestConfigWithOverrides:
    """Test configuration overrides via with_overrides method."""

    def test_override_server_url(self):
        """Test overriding server URL."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(
                server_url="http://override:8888"
            )
            assert new_config.server_url == "http://override:8888"
            # Original should be unchanged
            assert config.server_url == "http://localhost:8080"

    def test_override_api_key(self):
        """Test overriding API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(api_key="override-key")
            assert new_config.api_key == "override-key"
            assert config.api_key is None

    def test_override_user_id(self):
        """Test overriding user ID."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(user_id="override-user")
            assert new_config.user_id == "override-user"
            assert config.user_id is None

    def test_override_timeout(self):
        """Test overriding timeout."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(timeout_seconds=180)
            assert new_config.timeout_seconds == 180
            assert config.timeout_seconds == 60

    def test_override_retry_attempts(self):
        """Test overriding retry attempts."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(retry_attempts=10)
            assert new_config.retry_attempts == 10
            assert config.retry_attempts == 3

    def test_override_retry_delay(self):
        """Test overriding retry delay."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(retry_delay_seconds=15)
            assert new_config.retry_delay_seconds == 15
            assert config.retry_delay_seconds == 2

    def test_override_multiple_values(self):
        """Test overriding multiple configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            new_config = config.with_overrides(
                server_url="http://multi:9999",
                api_key="multi-key",
                user_id="multi-user",
                timeout_seconds=200
            )
            assert new_config.server_url == "http://multi:9999"
            assert new_config.api_key == "multi-key"
            assert new_config.user_id == "multi-user"
            assert new_config.timeout_seconds == 200

    def test_override_none_values_ignored(self):
        """Test that None overrides don't change values."""
        with patch.dict(
            os.environ,
            {
                "MORGAN_SERVER_URL": "http://env:8080",
                "MORGAN_API_KEY": "env-key"
            }
        ):
            config = Config()
            new_config = config.with_overrides(
                server_url=None,
                api_key=None
            )
            # Should keep original values when None is passed
            assert new_config.server_url == "http://env:8080"
            assert new_config.api_key == "env-key"


class TestGetConfig:
    """Test get_config helper function."""

    def test_get_config_returns_config(self):
        """Test get_config returns a Config instance."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            assert isinstance(config, Config)

    def test_get_config_with_server_url_override(self):
        """Test get_config with server URL override."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config(server_url="http://override:7777")
            assert config.server_url == "http://override:7777"

    def test_get_config_with_api_key_override(self):
        """Test get_config with API key override."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config(api_key="override-key")
            assert config.api_key == "override-key"

    def test_get_config_with_user_id_override(self):
        """Test get_config with user ID override."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config(user_id="override-user")
            assert config.user_id == "override-user"

    def test_get_config_with_multiple_overrides(self):
        """Test get_config with multiple overrides."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config(
                server_url="http://multi:6666",
                api_key="multi-key",
                user_id="multi-user"
            )
            assert config.server_url == "http://multi:6666"
            assert config.api_key == "multi-key"
            assert config.user_id == "multi-user"

    def test_get_config_env_and_override(self):
        """Test get_config with both env vars and overrides."""
        env_vars = {
            "MORGAN_SERVER_URL": "http://env:5555",
            "MORGAN_API_KEY": "env-key"
        }
        with patch.dict(os.environ, env_vars):
            # Override should take precedence
            config = get_config(server_url="http://override:4444")
            assert config.server_url == "http://override:4444"
            # Non-overridden values should come from env
            assert config.api_key == "env-key"


class TestConfigValidation:
    """Test configuration validation."""

    def test_timeout_must_be_positive(self):
        """Test that timeout must be at least 1."""
        with pytest.raises(Exception):  # Pydantic validation error
            Config(timeout_seconds=0)

    def test_timeout_negative_rejected(self):
        """Test that negative timeout is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            Config(timeout_seconds=-1)

    def test_retry_attempts_can_be_zero(self):
        """Test that retry attempts can be 0 (no retries)."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(retry_attempts=0)
            assert config.retry_attempts == 0

    def test_retry_delay_can_be_zero(self):
        """Test that retry delay can be 0 (immediate retry)."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(retry_delay_seconds=0)
            assert config.retry_delay_seconds == 0


class TestConfigImmutability:
    """Test that configuration is immutable (frozen)."""

    def test_config_is_frozen(self):
        """Test that Config instances are immutable."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(Exception):  # Pydantic frozen model error
                config.server_url = "http://new:9999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
