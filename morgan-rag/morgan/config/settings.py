"""
Settings for Morgan RAG - Human-First Configuration

Simple, secure configuration with sensible defaults.
Based on InspecTor's proven configuration patterns but simplified for human use.

KISS Principle: Easy to configure, secure by default, human-readable.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from morgan.utils.validators import ValidationError as ValidatorError
from morgan.utils.validators import (
    validate_int_range,
    validate_string_not_empty,
    validate_url,
)


class Settings(BaseSettings):
    """
    Morgan RAG Settings

    Human-friendly configuration with secure defaults.
    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ============================================================================
    # LLM Configuration (OpenAI Compatible)
    # ============================================================================

    llm_base_url: str = Field(
        default="https://gpt.lazarev.cloud/ollama/v1",
        description="OpenAI-compatible LLM endpoint",
    )

    llm_api_key: Optional[str] = Field(
        default=None, description="API key for LLM service"
    )

    llm_model: str = Field(default="llama3.1:8b", description="LLM model name")

    llm_max_tokens: int = Field(
        default=2048, description="Maximum tokens for responses", ge=100, le=32000
    )

    llm_temperature: float = Field(
        default=0.7,
        description="LLM temperature (0.0 = deterministic, 1.0 = creative)",
        ge=0.0,
        le=2.0,
    )

    # ============================================================================
    # Embedding Configuration (Same as InspecTor)
    # ============================================================================

    embedding_model: str = Field(
        default="qwen3-embedding:latest", description="Primary embedding model (remote)"
    )

    embedding_local_model: str = Field(
        default="all-MiniLM-L6-v2", description="Fallback local embedding model"
    )

    embedding_batch_size: int = Field(
        default=100, description="Batch size for embedding operations", ge=1, le=1000
    )

    embedding_device: str = Field(
        default="cpu", description="Device for local embeddings (cpu, cuda, mps)"
    )

    embedding_use_instructions: bool = Field(
        default=True, description="Use instruction prefixes for 22% better relevance"
    )

    embedding_force_remote: bool = Field(
        default=False,
        description="Force remote embeddings even in dev mode (no local fallback)",
    )

    # ============================================================================
    # Vector Database (Qdrant)
    # ============================================================================

    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant vector database URL"
    )

    qdrant_api_key: Optional[str] = Field(
        default=None, description="Qdrant API key (optional)"
    )

    # ============================================================================
    # Morgan System Settings
    # ============================================================================

    morgan_data_dir: Path = Field(
        default=Path("./data"), description="Directory for Morgan's data storage"
    )

    morgan_log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    morgan_debug: bool = Field(default=False, description="Enable debug mode")

    morgan_max_context: int = Field(
        default=8192, description="Maximum context length for LLM", ge=1000, le=32000
    )

    morgan_max_response_tokens: int = Field(
        default=2048, description="Maximum tokens for responses", ge=100, le=8000
    )

    # ============================================================================
    # Search & Retrieval Settings
    # ============================================================================

    morgan_max_search_results: int = Field(
        default=10, description="Maximum search results to return", ge=1, le=50
    )

    morgan_default_search_results: int = Field(
        default=5, description="Default number of search results", ge=1, le=20
    )

    morgan_min_search_score: float = Field(
        default=0.7,
        description="Minimum similarity score for search results",
        ge=0.0,
        le=1.0,
    )

    # ============================================================================
    # Memory & Learning Settings
    # ============================================================================

    morgan_memory_enabled: bool = Field(
        default=True, description="Enable conversation memory"
    )

    morgan_memory_max_conversations: int = Field(
        default=1000,
        description="Maximum conversations to keep in memory",
        ge=10,
        le=10000,
    )

    morgan_memory_max_turns_per_conversation: int = Field(
        default=100, description="Maximum turns per conversation", ge=5, le=500
    )

    morgan_learning_enabled: bool = Field(
        default=True, description="Enable learning from feedback"
    )

    # ============================================================================
    # Document Processing Settings
    # ============================================================================

    morgan_chunk_size: int = Field(
        default=1000,
        description="Default chunk size for document splitting",
        ge=100,
        le=5000,
    )

    morgan_chunk_overlap: int = Field(
        default=200, description="Chunk overlap for better context", ge=0, le=1000
    )

    morgan_max_file_size: int = Field(
        default=100, description="Maximum file size for upload (MB)", ge=1, le=1000
    )

    morgan_supported_types: str = Field(
        default="pdf,docx,txt,md,html,py,js,ts,go,java,cpp,c,h,json,yaml,yml",
        description="Supported file types (comma-separated)",
    )

    # ============================================================================
    # Web Interface Settings
    # ============================================================================

    morgan_host: str = Field(default="0.0.0.0", description="Web server host")

    morgan_port: int = Field(
        default=8080, description="Web server port", ge=1024, le=65535
    )

    morgan_api_port: int = Field(
        default=8000, description="API server port", ge=1024, le=65535
    )

    morgan_web_enabled: bool = Field(default=True, description="Enable web interface")

    morgan_api_key: Optional[str] = Field(
        default=None, description="API key for Morgan API (optional)"
    )

    morgan_cors_origins: str = Field(
        default="*", description="Allowed CORS origins (comma-separated)"
    )

    # ============================================================================
    # Performance Settings
    # ============================================================================

    morgan_workers: int = Field(
        default=4, description="Number of worker processes", ge=1, le=16
    )

    morgan_cache_size: int = Field(
        default=1000, description="Cache size (number of items)", ge=100, le=10000
    )

    morgan_cache_ttl: int = Field(
        default=3600, description="Cache TTL in seconds", ge=60, le=86400
    )

    # ============================================================================
    # Security Settings
    # ============================================================================

    morgan_allow_file_upload: bool = Field(
        default=True, description="Allow file uploads"
    )

    morgan_allow_url_ingestion: bool = Field(
        default=True, description="Allow URL ingestion"
    )

    morgan_allow_code_execution: bool = Field(
        default=False, description="Allow code execution (dangerous!)"
    )

    morgan_session_secret: str = Field(
        default="change-me-in-production",
        description="Session secret for web interface",
    )

    # ============================================================================
    # Validators (Security & Correctness)
    # ============================================================================

    @field_validator("morgan_log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(
                f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
            )
        return v

    @field_validator("morgan_data_dir")
    @classmethod
    def ensure_data_dir(cls, v):
        """Ensure data directory exists."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("llm_base_url", "qdrant_url")
    @classmethod
    def validate_service_urls(cls, v, info):
        """Validate service URLs for security."""
        if v is None:
            return v

        try:
            validate_url(v, info.field_name)
            return v
        except ValidatorError as e:
            raise ValueError(f"Invalid URL for {info.field_name}: {e}")

    @field_validator("morgan_port", "morgan_api_port")
    @classmethod
    def validate_port_range(cls, v, info):
        """Validate port is in valid range."""
        try:
            validate_int_range(
                v, min_value=1024, max_value=65535, field_name=info.field_name
            )
            return v
        except ValidatorError as e:
            raise ValueError(f"Invalid port: {e}")

    @field_validator("llm_api_key", "qdrant_api_key", "morgan_api_key")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Validate API keys."""
        if v is None or v == "":
            return v  # Optional keys can be None

        try:
            validate_string_not_empty(v, info.field_name)

            # Check minimum length
            if len(v) < 8:
                raise ValueError(
                    f"{info.field_name} is too short (must be at least 8 characters)"
                )

            # Warn if key looks like a placeholder
            placeholder_patterns = [
                "xxx",
                "test",
                "sample",
                "example",
                "placeholder",
                "changeme",
                "change-me",
            ]
            if any(pattern in v.lower() for pattern in placeholder_patterns):
                import logging

                logger = logging.getLogger("morgan.settings")
                logger.warning(
                    f"SECURITY WARNING: {info.field_name} appears to be a placeholder value. "
                    f"Please use a real API key in production."
                )

            return v

        except ValidatorError as e:
            raise ValueError(f"Invalid API key for {info.field_name}: {e}")

    @field_validator("morgan_session_secret")
    @classmethod
    def validate_session_secret(cls, v):
        """Validate session secret."""
        if v == "change-me-in-production":
            import logging

            logger = logging.getLogger("morgan.settings")
            logger.warning(
                "SECURITY WARNING: Using default session secret. "
                "Please change MORGAN_SESSION_SECRET in production."
            )

        if len(v) < 16:
            raise ValueError("Session secret must be at least 16 characters long")

        return v

    @field_validator("morgan_data_dir", mode="after")
    @classmethod
    def validate_data_directory_security(cls, v):
        """Validate data directory for security."""
        if isinstance(v, str):
            v = Path(v)

        # Resolve to absolute path
        v = v.resolve()

        # Check for path traversal attempts
        try:
            # Ensure data dir is not in system directories
            if str(v).startswith(
                ("/etc", "/root", "/sys", "/proc", "C:\\Windows", "C:\\Program Files")
            ):
                raise ValueError(
                    f"SECURITY: Data directory cannot be in system directories: {v}"
                )
        except Exception:
            pass

        # Create directory with restricted permissions
        try:
            v.mkdir(parents=True, exist_ok=True)

            # Set permissions to 700 (owner read/write/execute only) on Unix
            import os

            if os.name != "nt":  # Not Windows
                os.chmod(v, 0o700)

        except PermissionError:
            raise ValueError(f"Cannot create data directory (permission denied): {v}")
        except Exception as e:
            raise ValueError(f"Failed to create data directory: {e}")

        return v


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get cached settings instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Settings: Validated settings object

    Raises:
        ValidationError: If configuration has invalid values
        FileNotFoundError: If .env file is missing
    """
    # Find .env file
    if config_path:
        env_path = Path(config_path)
    else:
        env_path = Path(".env")
        if not env_path.exists():
            # Try parent directory
            env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        # Load environment variables (override shell environment)
        load_dotenv(env_path, override=True)
    else:
        # No .env file found - use environment variables and defaults
        import logging

        logger = logging.getLogger("morgan.settings")
        logger.info("No .env file found, using environment variables and defaults")

    # Create and validate settings
    try:
        settings = Settings()
    except Exception as e:
        raise ValueError(f"Failed to load settings: {e}")

    return settings


def validate_settings(settings: Settings) -> bool:
    """
    Validate that all required services are accessible.

    Args:
        settings: Settings object to validate

    Returns:
        bool: True if all services are accessible

    Raises:
        ConnectionError: If any required service is not accessible
    """
    import requests
    from openai import OpenAI

    errors = []

    # Check LLM API (OpenAI compatible)
    try:
        client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key or "dummy-key"
        )

        # Test with a minimal request
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )

    except Exception as e:
        errors.append(f"LLM API error: {e}")

    # Check Qdrant
    try:
        headers = {}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key

        response = requests.get(
            f"{settings.qdrant_url}/collections", headers=headers, timeout=5
        )
        response.raise_for_status()

    except Exception as e:
        errors.append(f"Qdrant error: {e}")

    if errors:
        raise ConnectionError(
            "Service validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


if __name__ == "__main__":
    # Test settings loading
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully")
        print(f"  - LLM Model: {settings.llm_model}")
        print(f"  - Embedding Model: {settings.embedding_model}")
        print(f"  - Qdrant URL: {settings.qdrant_url}")
        print(f"  - Data Directory: {settings.morgan_data_dir}")
        print(f"  - Debug Mode: {settings.morgan_debug}")

        # Validate services
        print("\n🔍 Validating services...")
        validate_settings(settings)
        print("✅ All services accessible")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
