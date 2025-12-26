"""
Configuration Defaults for Morgan AI Assistant.

Single source of truth for all default configuration values.
Import from here instead of hardcoding values in code.

Usage:
    from morgan.config.defaults import Defaults

    # Use defaults
    port = Defaults.MORGAN_PORT
    model = Defaults.LLM_MODEL

    # Or get all defaults as dict
    all_defaults = Defaults.to_dict()
"""

from typing import Any, Dict


class Defaults:
    """
    Central configuration defaults.

    All default values are defined here to avoid scattered hardcoded values.
    """

    # =========================================================================
    # Server Configuration
    # =========================================================================

    MORGAN_HOST = "0.0.0.0"
    MORGAN_PORT = 8080
    MORGAN_API_PORT = 8000
    MORGAN_WORKERS = 4

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_API_KEY = "ollama"
    LLM_MODEL = "qwen2.5:32b-instruct-q4_K_M"
    LLM_FAST_MODEL = "qwen2.5:7b-instruct-q5_K_M"
    LLM_MAX_TOKENS = 2048
    LLM_TEMPERATURE = 0.7
    LLM_TIMEOUT = 60.0
    LLM_MODE = "single"  # "single" or "distributed"
    LLM_STRATEGY = "round_robin"  # Load balancing strategy

    # =========================================================================
    # Embedding Configuration
    # =========================================================================

    EMBEDDING_MODEL = "qwen3-embedding:4b"
    EMBEDDING_DIMENSIONS = 2048
    EMBEDDING_LOCAL_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 100
    EMBEDDING_DEVICE = "cpu"
    EMBEDDING_TIMEOUT = 30.0
    EMBEDDING_FORCE_REMOTE = False

    # =========================================================================
    # Reranking Configuration
    # =========================================================================

    RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKING_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKING_BATCH_SIZE = 100
    RERANKING_TIMEOUT = 30.0
    RERANKING_TOP_K = 20
    RERANKING_WEIGHT = 0.6
    RERANKING_ORIGINAL_WEIGHT = 0.4

    # =========================================================================
    # Vector Database (Qdrant)
    # =========================================================================

    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_GRPC_PORT = 6334
    QDRANT_DEFAULT_COLLECTION = "morgan_knowledge"
    QDRANT_MEMORY_COLLECTION = "morgan_memory"

    # =========================================================================
    # Cache (Redis)
    # =========================================================================

    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = ""
    REDIS_PREFIX = "morgan:"
    REDIS_TTL = 3600  # 1 hour

    # =========================================================================
    # Model Cache
    # =========================================================================

    MODEL_CACHE_DIR = "~/.morgan/models"

    # =========================================================================
    # Search Configuration
    # =========================================================================

    SEARCH_MAX_RESULTS = 50
    SEARCH_DEFAULT_RESULTS = 10
    SEARCH_SIMILARITY_THRESHOLD = 0.7

    # =========================================================================
    # Memory Configuration
    # =========================================================================

    MEMORY_ENABLED = True
    MEMORY_MAX_CONVERSATIONS = 1000
    MEMORY_MAX_TURNS = 100

    # =========================================================================
    # Logging
    # =========================================================================

    LOG_LEVEL = "INFO"
    LOG_FORMAT = "json"

    # =========================================================================
    # Monitoring
    # =========================================================================

    PROMETHEUS_PORT = 9090
    GRAFANA_PORT = 3000
    HEALTH_CHECK_INTERVAL = 60

    # =========================================================================
    # Timeouts and Limits
    # =========================================================================

    REQUEST_TIMEOUT = 60
    MAX_CONCURRENT_REQUESTS = 100
    MAX_FILE_SIZE_MB = 100

    # =========================================================================
    # Feature Flags
    # =========================================================================

    REASONING_ENABLED = True
    REASONING_MAX_STEPS = 10
    PROACTIVE_ENABLED = True
    PROACTIVE_SUGGESTIONS_MAX = 5
    LEARNING_ENABLED = True
    KNOWLEDGE_GRAPH_ENABLED = True

    # =========================================================================
    # Security
    # =========================================================================

    CORS_ORIGINS = "*"
    ALLOW_FILE_UPLOAD = True
    ALLOW_URL_INGESTION = True
    ALLOW_CODE_EXECUTION = False

    # =========================================================================
    # Class Methods
    # =========================================================================

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Get all defaults as dictionary."""
        return {
            key: value
            for key, value in vars(cls).items()
            if not key.startswith("_") and not callable(value)
        }

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a default value by key.

        Args:
            key: Configuration key (case-insensitive)
            default: Fallback if key not found

        Returns:
            Default value or fallback
        """
        return getattr(cls, key.upper(), default)


# Environment variable name mappings
# Maps old/alternative names to canonical names
ENV_VAR_ALIASES = {
    # LLM
    "LLM_BASE_URL": "MORGAN_LLM_ENDPOINT",
    "LLM_API_KEY": "MORGAN_LLM_API_KEY",
    "LLM_MODEL": "MORGAN_LLM_MODEL",
    "OLLAMA_BASE_URL": "MORGAN_LLM_ENDPOINT",
    # Embedding
    "EMBEDDING_MODEL": "MORGAN_EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL": "MORGAN_EMBEDDING_ENDPOINT",
    # Qdrant
    "QDRANT_URL": "MORGAN_VECTOR_DB_URL",
    "QDRANT_HOST": "MORGAN_QDRANT_HOST",
    # Redis
    "REDIS_URL": "MORGAN_REDIS_URL",
    "REDIS_HOST": "MORGAN_REDIS_HOST",
}


def get_env_with_fallback(primary: str, *fallbacks: str, default: Any = None) -> Any:
    """
    Get environment variable with fallback names.

    Args:
        primary: Primary environment variable name
        *fallbacks: Alternative names to try
        default: Default value if none found

    Returns:
        Environment variable value or default
    """
    import os

    # Try primary
    value = os.environ.get(primary)
    if value is not None:
        return value

    # Try fallbacks
    for fallback in fallbacks:
        value = os.environ.get(fallback)
        if value is not None:
            return value

    return default
