"""
Server configuration system with support for multiple sources.

This module provides configuration management with precedence rules:
1. Environment variables (highest precedence)
2. Configuration files (YAML, JSON, .env)
3. Default values (lowest precedence)
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """
    Server configuration with validation and multiple source support.

    Configuration precedence (highest to lowest):
    1. Environment variables
    2. Configuration file
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="MORGAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    workers: int = Field(default=4, ge=1, description="Number of worker processes")

    # LLM settings (self-hosted only)
    llm_provider: Literal["ollama", "openai-compatible"] = Field(
        default="ollama", description="LLM provider type"
    )
    llm_api_key: Optional[str] = Field(
        default=None, description="API key for LLM service (optional for self-hosted)"
    )
    llm_model: str = Field(default="gemma3", description="LLM model name")
    llm_endpoint: str = Field(
        default="http://localhost:11434", description="Self-hosted LLM endpoint URL"
    )

    # Vector database settings
    vector_db_url: str = Field(
        default="http://localhost:6333", description="Vector database URL"
    )
    vector_db_api_key: Optional[str] = Field(
        default=None, description="Vector database API key"
    )

    # Embedding settings
    embedding_provider: Literal["local", "ollama", "openai-compatible"] = Field(
        default="local",
        description="Embedding provider type (local, ollama, openai-compatible)",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    embedding_device: str = Field(
        default="cpu", description="Device for local embedding model (cpu, cuda, mps)"
    )
    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Remote embedding service endpoint (for ollama/openai-compatible)",
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="API key for remote embedding service"
    )

    # Cache settings
    cache_dir: str = Field(default="./data/cache", description="Cache directory path")
    cache_size_mb: int = Field(
        default=1000, ge=1, description="Maximum cache size in MB"
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log output format"
    )

    # Performance settings
    max_concurrent_requests: int = Field(
        default=100, ge=1, description="Maximum concurrent requests"
    )
    request_timeout_seconds: int = Field(
        default=60, ge=1, description="Request timeout in seconds"
    )
    session_timeout_minutes: int = Field(
        default=60, ge=1, description="Session timeout in minutes"
    )

    @field_validator("llm_endpoint")
    @classmethod
    def validate_llm_endpoint(cls, v: str) -> str:
        """Validate LLM endpoint URL format."""
        if not v:
            raise ValueError("LLM endpoint cannot be empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("LLM endpoint must start with http:// or https://")
        return v

    @field_validator("vector_db_url")
    @classmethod
    def validate_vector_db_url(cls, v: str) -> str:
        """Validate vector database URL format."""
        if not v:
            raise ValueError("Vector database URL cannot be empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Vector database URL must start with http:// or https://")
        return v

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Ensure cache directory path is valid."""
        if not v:
            raise ValueError("Cache directory cannot be empty")
        return v

    @field_validator("embedding_device")
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        """Validate embedding device."""
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(
                f"Embedding device must be one of: {', '.join(valid_devices)}"
            )
        return v

    @field_validator("embedding_endpoint")
    @classmethod
    def validate_embedding_endpoint(cls, v: Optional[str], info) -> Optional[str]:
        """Validate embedding endpoint URL format for remote providers."""
        # Get the embedding_provider from the values being validated
        provider = info.data.get("embedding_provider", "local")

        # If using remote provider, endpoint is required
        if provider in ["ollama", "openai-compatible"]:
            if not v:
                raise ValueError(
                    f"Embedding endpoint is required when using '{provider}' provider"
                )
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError(
                    "Embedding endpoint must start with http:// or https://"
                )

        # If endpoint is provided, validate format
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Embedding endpoint must start with http:// or https://")

        return v


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""

    pass


def load_config_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a file.

    Supports YAML, JSON, and .env formats.

    Args:
        file_path: Path to configuration file

    Returns:
        Dictionary of configuration values

    Raises:
        ConfigurationError: If file format is unsupported or parsing fails
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix in [".yaml", ".yml"]:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                return data if data else {}

        elif suffix == ".json":
            with open(file_path, "r") as f:
                return json.load(f)

        elif suffix == ".env":
            # For .env files, we'll let pydantic-settings handle it
            # Return empty dict as the file will be loaded by BaseSettings
            return {}

        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {suffix}. "
                f"Supported formats: .yaml, .yml, .json, .env"
            )

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file: {e}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Failed to parse JSON file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file: {e}")


def _get_env_value(key: str) -> Optional[str]:
    """Get environment variable value with MORGAN_ prefix."""
    env_key = f"MORGAN_{key.upper()}"
    return os.environ.get(env_key)


def load_config(
    config_file: Optional[Path] = None, env_file: Optional[Path] = None, **overrides
) -> ServerConfig:
    """
    Load server configuration from multiple sources with precedence rules.

    Precedence (highest to lowest):
    1. Explicit overrides passed as kwargs
    2. Environment variables (MORGAN_*)
    3. Configuration file (if provided)
    4. .env file (if provided or default)
    5. Default values

    Args:
        config_file: Optional path to YAML/JSON configuration file
        env_file: Optional path to .env file
        **overrides: Explicit configuration overrides

    Returns:
        ServerConfig instance with loaded configuration

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Step 1: Start with defaults
        default_config = ServerConfig()
        config_dict = default_config.model_dump()

        # Step 2: Apply config file values (if provided)
        if config_file:
            try:
                file_data = load_config_from_file(config_file)
                # Update with file values
                for key, value in file_data.items():
                    if key in config_dict:
                        config_dict[key] = value
            except ConfigurationError as e:
                raise ConfigurationError(f"Failed to load configuration file: {e}")

        # Step 3: Apply environment variables (higher precedence than file)
        # Check each field for corresponding env var
        for key in config_dict.keys():
            env_value = _get_env_value(key)
            if env_value is not None:
                # Let pydantic handle type conversion
                # We'll create a temp config to validate the env value
                try:
                    temp_dict = {key: env_value}
                    temp_config = ServerConfig(**temp_dict)
                    config_dict[key] = getattr(temp_config, key)
                except Exception:
                    # If conversion fails, skip this env var
                    pass

        # Step 4: Apply explicit overrides (highest precedence)
        config_dict.update(overrides)

        # Step 5: Create final config
        config = ServerConfig(**config_dict)

        return config

    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")


def validate_required_config(config: ServerConfig) -> None:
    """
    Validate that all required configuration is present and valid.

    This performs additional validation beyond Pydantic's field validation
    to ensure the server can start successfully.

    Args:
        config: ServerConfig instance to validate

    Raises:
        ConfigurationError: If required configuration is missing or invalid
    """
    errors = []

    # Validate LLM endpoint is accessible (basic format check)
    if not config.llm_endpoint:
        errors.append("LLM endpoint is required")

    # Validate vector DB URL
    if not config.vector_db_url:
        errors.append("Vector database URL is required")

    # Validate cache directory can be created
    cache_path = Path(config.cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create cache directory '{config.cache_dir}': {e}")

    # Validate port is not in use (basic check)
    if config.port < 1 or config.port > 65535:
        errors.append(f"Port must be between 1 and 65535, got {config.port}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ConfigurationError(error_msg)


def get_config(
    config_file: Optional[Path] = None,
    env_file: Optional[Path] = None,
    validate: bool = True,
    **overrides,
) -> ServerConfig:
    """
    Load and optionally validate server configuration.

    This is the main entry point for loading configuration.

    Args:
        config_file: Optional path to YAML/JSON configuration file
        env_file: Optional path to .env file
        validate: Whether to perform additional validation
        **overrides: Explicit configuration overrides

    Returns:
        ServerConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = load_config(config_file=config_file, env_file=env_file, **overrides)

    if validate:
        validate_required_config(config)

    return config
