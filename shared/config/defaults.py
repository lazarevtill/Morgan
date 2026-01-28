"""
Centralized default values for Morgan configuration.

Single source of truth for all default values used across services.
"""
from typing import Dict, Any


# Default model configurations
DEFAULTS: Dict[str, Any] = {
    # LLM Models
    "llm": {
        "main_model": "qwen2.5:32b-instruct-q4_K_M",
        "fast_model": "qwen2.5:7b-instruct-q5_K_M",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },

    # Embedding
    "embedding": {
        "model": "qwen3-embedding:4b",
        "dimensions": 2048,
        "batch_size": 32,
        "normalize": True,
    },

    # Reranking
    "reranking": {
        "model": "ms-marco-MiniLM-L-6-v2",
        "top_k": 10,
        "batch_size": 32,
    },

    # Search
    "search": {
        "top_k": 10,
        "min_score": 0.5,
        "rerank_top_k": 50,
        "hybrid_alpha": 0.7,  # Weight for vector vs keyword search
    },

    # Memory
    "memory": {
        "collection_name": "morgan_memories",
        "max_history": 100,
        "ttl_days": 30,
    },

    # Chunking
    "chunking": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "min_chunk_size": 100,
    },

    # Timeouts (seconds)
    "timeouts": {
        "default": 60.0,
        "llm": 120.0,
        "embedding": 30.0,
        "reranking": 30.0,
        "search": 10.0,
    },

    # Retry
    "retry": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "exponential_base": 2.0,
    },

    # Endpoints (local development)
    "endpoints": {
        "llm": "http://localhost:11434/v1",
        "embedding": None,  # Uses LLM endpoint by default
        "reranking": None,  # Uses local model by default
        "vector_db": "http://localhost:6333",
        "redis": "redis://localhost:6379",
    },

    # Logging
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    },

    # Distributed deployment hosts
    "hosts": {
        "orchestrator": "192.168.1.10",
        "main_llm_1": "192.168.1.20",
        "main_llm_2": "192.168.1.21",
        "embeddings": "192.168.1.22",
        "reranking": "192.168.1.23",
    },
}


def get_default(path: str, fallback: Any = None) -> Any:
    """
    Get a default value by dot-separated path.

    Args:
        path: Dot-separated path like "llm.temperature"
        fallback: Value to return if path not found

    Returns:
        The default value or fallback
    """
    parts = path.split(".")
    current = DEFAULTS

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return fallback

    return current
