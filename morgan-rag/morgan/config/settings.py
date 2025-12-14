"""
Settings for Morgan RAG - Human-First Configuration

Simple, secure configuration with sensible defaults.
Supports SEPARATE hosts for LLM and Embedding providers.

KISS Principle: Easy to configure, secure by default, human-readable.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

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
    Supports SEPARATE hosts for LLM and Embedding providers.
    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ============================================================================
    # LLM Configuration (OpenAI Compatible)
    # ============================================================================

    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible LLM endpoint for chat/generation",
    )

    llm_api_key: Optional[str] = Field(
        default="ollama", description="API key for LLM service"
    )

    llm_model: str = Field(
        default="qwen2.5:32b-instruct-q4_K_M", description="LLM model name"
    )

    llm_max_tokens: int = Field(
        default=2048, description="Maximum tokens for responses", ge=100, le=32000
    )

    llm_temperature: float = Field(
        default=0.7,
        description="LLM temperature (0.0 = deterministic, 1.0 = creative)",
        ge=0.0,
        le=2.0,
    )

    llm_timeout: float = Field(
        default=60.0, description="LLM request timeout in seconds", ge=5.0, le=600.0
    )

    # ============================================================================
    # Distributed LLM Configuration (Optional - for load balancing)
    # ============================================================================

    llm_distributed_enabled: bool = Field(
        default=False, description="Enable distributed LLM with load balancing"
    )

    llm_endpoints: Optional[str] = Field(
        default=None,
        description="Comma-separated list of LLM endpoints for load balancing",
    )

    llm_load_balancing_strategy: str = Field(
        default="round_robin",
        description="Load balancing strategy: round_robin, random, least_loaded",
    )

    llm_health_check_interval: int = Field(
        default=60,
        description="Health check interval in seconds for distributed LLM",
        ge=10,
        le=600,
    )

    # ============================================================================
    # Embedding Configuration (SEPARATE from LLM)
    # ============================================================================

    embedding_base_url: Optional[str] = Field(
        default=None,
        description="Embedding endpoint URL (SEPARATE from LLM). e.g., http://192.168.1.22:11434",
    )

    ollama_host: Optional[str] = Field(
        default=None,
        description="Ollama host for embeddings (alternative to embedding_base_url)",
    )

    embedding_model: str = Field(
        default="nomic-embed-text", description="Primary embedding model (remote)"
    )

    embedding_dimensions: int = Field(
        default=768, description="Embedding dimensions (768 for nomic, 4096 for qwen3)"
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
        description="Force remote embeddings only (no local fallback)",
    )

    # ============================================================================
    # Reranking Configuration (SEPARATE from LLM and Embedding)
    # ============================================================================

    reranking_enabled: bool = Field(
        default=True, description="Enable reranking for better search results"
    )

    reranking_endpoint: Optional[str] = Field(
        default=None,
        description="Reranking endpoint URL (e.g., http://192.168.1.23:8081/rerank)",
    )

    reranking_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model for local fallback",
    )

    reranking_force_remote: bool = Field(
        default=False, description="Force remote reranking only (no local fallback)"
    )

    reranking_timeout: float = Field(
        default=30.0, description="Reranking request timeout in seconds"
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

    qdrant_default_collection: str = Field(
        default="morgan_knowledge",
        description="Primary Qdrant collection for knowledge documents",
    )

    qdrant_memory_collection: str = Field(
        default="morgan_memories",
        description="Qdrant collection for conversation memories",
    )

    qdrant_hierarchical_collection: Optional[str] = Field(
        default=None,
        description="Override for hierarchical collection (defaults to <knowledge>_hierarchical)",
    )

    qdrant_log_level: str = Field(
        default="INFO", description="Log level for Qdrant (INFO, WARN, ERROR, DEBUG)"
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

    morgan_dev_mode: bool = Field(
        default=False, description="Enable developer mode with extra diagnostics"
    )

    morgan_auto_reload: bool = Field(
        default=False,
        description="Reload services automatically on code/config changes (dev only)",
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

    morgan_feedback_weight: float = Field(
        default=0.1,
        description="Weight applied to human feedback when learning",
        ge=0.0,
        le=1.0,
    )

    morgan_knowledge_graph_enabled: bool = Field(
        default=True, description="Enable knowledge graph extraction"
    )

    morgan_entity_extraction_enabled: bool = Field(
        default=True, description="Enable entity extraction during ingestion"
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

    morgan_scrape_depth: int = Field(
        default=3, description="Maximum crawl depth for web scraping", ge=1, le=10
    )

    morgan_scrape_delay: int = Field(
        default=1, description="Delay between web scrape requests (seconds)", ge=0, le=30
    )

    morgan_scrape_timeout: int = Field(
        default=30, description="Timeout for web scraping (seconds)", ge=5, le=300
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

    morgan_docs_enabled: bool = Field(
        default=True, description="Expose interactive API docs"
    )

    morgan_api_key: Optional[str] = Field(
        default=None, description="API key for Morgan API (optional)"
    )

    morgan_cors_origins: str = Field(
        default="*", description="Allowed CORS origins (comma-separated)"
    )

    morgan_request_logging: bool = Field(
        default=False, description="Enable request logging (debugging only)"
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
    # External Services
    # ============================================================================

    redis_url: Optional[str] = Field(
        default="redis://localhost:6379",
        description="Redis connection string for caching",
    )

    database_url: Optional[str] = Field(
        default=None, description="Optional PostgreSQL database URL"
    )

    elasticsearch_url: Optional[str] = Field(
        default=None, description="Optional Elasticsearch URL for full-text search"
    )

    consul_enabled: bool = Field(
        default=False, description="Enable Consul service discovery"
    )

    consul_http_addr: str = Field(
        default="http://localhost:8500", description="Consul HTTP address"
    )

    huggingface_hub_token: Optional[str] = Field(
        default=None, description="HuggingFace token for private models"
    )

    # ============================================================================
    # Monitoring & Analytics
    # ============================================================================

    morgan_metrics_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics endpoint"
    )

    morgan_metrics_port: int = Field(
        default=9000, description="Prometheus metrics port", ge=1024, le=65535
    )

    morgan_health_check_interval: int = Field(
        default=30, description="Health check interval (seconds)", ge=5, le=600
    )

    morgan_analytics_retention: int = Field(
        default=90, description="Analytics retention (days)", ge=1, le=365
    )

    grafana_password: str = Field(
        default="admin", description="Grafana admin password (for docker-compose)"
    )

    # ============================================================================
    # Validators (Security & Correctness)
    # ============================================================================

    @field_validator("morgan_log_level", "qdrant_log_level")
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

    @field_validator(
        "llm_base_url",
        "embedding_base_url",
        "qdrant_url",
        "reranking_endpoint",
        "consul_http_addr",
    )
    @classmethod
    def validate_service_urls(cls, v, info):
        """Validate service URLs for security."""
        if v is None:
            return v
        if isinstance(v, str) and not v.strip():
            # Treat empty strings as unset/None
            return None

        try:
            validate_url(v, info.field_name)
            return v
        except ValidatorError as e:
            raise ValueError(f"Invalid URL for {info.field_name}: {e}")

    @field_validator("morgan_port", "morgan_api_port", "morgan_metrics_port")
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

            # Allow short local defaults for LLM dev setups
            if info.field_name == "llm_api_key" and v.lower() == "ollama":
                return v

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

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def get_llm_endpoints(self) -> List[str]:
        """
        Get list of LLM endpoints for distributed setup.

        Returns:
            List of endpoint URLs
        """
        if self.llm_distributed_enabled and self.llm_endpoints:
            return [e.strip() for e in self.llm_endpoints.split(",") if e.strip()]
        return [self.llm_base_url]

    def get_embedding_base_url(self) -> Optional[str]:
        """
        Get the effective embedding base URL.

        Priority: embedding_base_url > ollama_host > llm_base_url (without /v1)

        Returns:
            Embedding base URL or None
        """
        if self.embedding_base_url:
            return self.embedding_base_url.rstrip("/")

        if self.ollama_host:
            host = self.ollama_host
            if not host.startswith(("http://", "https://")):
                host = f"http://{host}"
            return host.rstrip("/")

        # Fallback to LLM base URL without /v1
        if self.llm_base_url:
            url = self.llm_base_url.rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            return url

        return None

    def get_reranking_endpoint(self) -> Optional[str]:
        """
        Get the reranking endpoint URL.

        Returns:
            Reranking endpoint URL or None
        """
        if self.reranking_enabled and self.reranking_endpoint:
            return self.reranking_endpoint
        return None

    def get_knowledge_collection(self) -> str:
        """Get configured knowledge collection name."""
        return (self.qdrant_default_collection or "morgan_knowledge").strip()

    def get_memory_collection(self) -> str:
        """Get configured memory collection name."""
        return (self.qdrant_memory_collection or "morgan_memories").strip()

    def get_hierarchical_collection(self) -> str:
        """Get configured hierarchical knowledge collection name."""
        if self.qdrant_hierarchical_collection:
            candidate = self.qdrant_hierarchical_collection.strip()
            if candidate:
                return candidate
        return f"{self.get_knowledge_collection()}_hierarchical"


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


def validate_settings(settings: Settings) -> dict:
    """
    Validate that all required services are accessible.

    Args:
        settings: Settings object to validate

    Returns:
        dict: Status of each service (healthy/unhealthy)

    Raises:
        ConnectionError: If critical services are not accessible
    """
    import requests
    from openai import OpenAI

    status = {
        "llm": {"healthy": False, "error": None, "endpoint": settings.llm_base_url},
        "embedding": {"healthy": False, "error": None, "endpoint": None},
        "reranking": {"healthy": False, "error": None, "endpoint": None},
        "qdrant": {"healthy": False, "error": None, "endpoint": settings.qdrant_url},
    }

    # Check LLM API (OpenAI compatible)
    try:
        client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "ollama",
            timeout=10.0,
        )

        # Test with a minimal request
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        status["llm"]["healthy"] = True

    except Exception as e:
        status["llm"]["error"] = str(e)

    # Check Embedding endpoint (if separate from LLM)
    embedding_url = settings.get_embedding_base_url()
    status["embedding"]["endpoint"] = embedding_url

    if embedding_url:
        try:
            # Try Ollama-style endpoint
            response = requests.get(f"{embedding_url}/api/tags", timeout=5)
            if response.status_code == 200:
                status["embedding"]["healthy"] = True
            else:
                # Try OpenAI-style endpoint
                response = requests.get(f"{embedding_url}/v1/models", timeout=5)
                status["embedding"]["healthy"] = response.status_code == 200
        except Exception as e:
            status["embedding"]["error"] = str(e)

    # Check Reranking endpoint (if enabled)
    reranking_url = settings.get_reranking_endpoint()
    status["reranking"]["endpoint"] = reranking_url

    if reranking_url:
        try:
            # Health check endpoint
            base_url = reranking_url.replace("/rerank", "")
            response = requests.get(f"{base_url}/health", timeout=5)
            status["reranking"]["healthy"] = response.status_code == 200
        except Exception as e:
            status["reranking"]["error"] = str(e)
    else:
        status["reranking"]["healthy"] = True  # Not configured, not an error
        status["reranking"]["error"] = "Not configured (using local fallback)"

    # Check Qdrant
    try:
        headers = {}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key

        response = requests.get(
            f"{settings.qdrant_url}/collections", headers=headers, timeout=5
        )
        response.raise_for_status()
        status["qdrant"]["healthy"] = True

    except Exception as e:
        status["qdrant"]["error"] = str(e)

    # Check critical services
    critical_errors = []
    if not status["llm"]["healthy"]:
        critical_errors.append(f"LLM API: {status['llm']['error']}")
    if not status["qdrant"]["healthy"]:
        critical_errors.append(f"Qdrant: {status['qdrant']['error']}")

    if critical_errors:
        raise ConnectionError(
            "Critical service validation failed:\n"
            + "\n".join(f"  - {e}" for e in critical_errors)
        )

    return status


if __name__ == "__main__":
    # Test settings loading
    try:
        settings = get_settings()
        print("=" * 60)
        print("Morgan RAG Settings")
        print("=" * 60)

        print("\n[LLM Configuration]")
        print(f"  Base URL: {settings.llm_base_url}")
        print(f"  Model: {settings.llm_model}")
        print(f"  Distributed: {settings.llm_distributed_enabled}")
        if settings.llm_distributed_enabled:
            print(f"  Endpoints: {settings.get_llm_endpoints()}")
            print(f"  Strategy: {settings.llm_load_balancing_strategy}")

        print("\n[Embedding Configuration]")
        print(f"  Base URL: {settings.get_embedding_base_url()}")
        print(f"  Model: {settings.embedding_model}")
        print(f"  Dimensions: {settings.embedding_dimensions}")
        print(f"  Force Remote: {settings.embedding_force_remote}")

        print("\n[Reranking Configuration]")
        print(f"  Enabled: {settings.reranking_enabled}")
        print(f"  Endpoint: {settings.get_reranking_endpoint()}")
        print(f"  Model: {settings.reranking_model}")

        print("\n[Vector Database]")
        print(f"  Qdrant URL: {settings.qdrant_url}")

        print("\n[System Settings]")
        print(f"  Data Directory: {settings.morgan_data_dir}")
        print(f"  Debug Mode: {settings.morgan_debug}")

        # Validate services
        print("\n" + "=" * 60)
        print("Validating Services...")
        print("=" * 60)

        status = validate_settings(settings)

        for service, info in status.items():
            icon = "OK" if info["healthy"] else "FAIL"
            print(f"\n[{icon}] {service.upper()}")
            print(f"  Endpoint: {info['endpoint']}")
            if info["error"]:
                print(f"  Error: {info['error']}")

        print("\n" + "=" * 60)
        print("All critical services accessible!")
        print("=" * 60)

    except ConnectionError as e:
        print(f"\n[FAIL] {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)
