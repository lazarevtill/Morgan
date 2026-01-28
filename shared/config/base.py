"""
Base configuration for all Morgan services.

All service-specific settings should inherit from MorganBaseSettings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
from pathlib import Path


class MorganBaseSettings(BaseSettings):
    """
    Base settings inherited by all Morgan services.

    Environment variables use MORGAN_ prefix and override defaults.
    """

    # LLM Configuration
    llm_endpoint: str = Field(
        default="http://localhost:11434/v1",
        description="LLM service endpoint"
    )
    llm_model: str = Field(
        default="qwen2.5:32b-instruct-q4_K_M",
        description="Main LLM model name"
    )
    llm_fast_model: str = Field(
        default="qwen2.5:7b-instruct-q5_K_M",
        description="Fast LLM model for simple queries"
    )

    # Embedding Configuration
    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Embedding service endpoint (defaults to llm_endpoint)"
    )
    embedding_model: str = Field(
        default="qwen3-embedding:4b",
        description="Embedding model name"
    )
    embedding_dimensions: int = Field(
        default=2048,
        description="Embedding vector dimensions"
    )

    # Reranking Configuration
    reranking_endpoint: Optional[str] = Field(
        default=None,
        description="Reranking service endpoint"
    )
    reranking_model: str = Field(
        default="ms-marco-MiniLM-L-6-v2",
        description="Reranking model name"
    )

    # Vector Database
    vector_db_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis cache URL"
    )

    # Paths
    cache_dir: Path = Field(
        default=Path.home() / ".morgan" / "cache",
        description="Cache directory path"
    )
    data_dir: Path = Field(
        default=Path.home() / ".morgan" / "data",
        description="Data directory path"
    )
    log_dir: Path = Field(
        default=Path.home() / ".morgan" / "logs",
        description="Log directory path"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    # Timeouts
    default_timeout: float = Field(
        default=60.0,
        description="Default request timeout in seconds"
    )
    llm_timeout: float = Field(
        default=120.0,
        description="LLM request timeout in seconds"
    )
    embedding_timeout: float = Field(
        default=30.0,
        description="Embedding request timeout in seconds"
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )

    model_config = {
        "env_prefix": "MORGAN_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    @field_validator("llm_endpoint", "vector_db_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v} (must start with http:// or https://)")
        return v.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v} (must be one of {valid_levels})")
        return v_upper

    @field_validator("cache_dir", "data_dir", "log_dir", mode="before")
    @classmethod
    def expand_path(cls, v) -> Path:
        """Expand user path."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser()

    def get_effective_embedding_endpoint(self) -> str:
        """Get embedding endpoint, defaulting to LLM endpoint."""
        return self.embedding_endpoint or self.llm_endpoint

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for directory in [self.cache_dir, self.data_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
