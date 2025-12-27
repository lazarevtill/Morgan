"""
Shared enumerations for Morgan infrastructure.

Consolidates duplicate enum definitions from:
- distributed_gpu_manager.py
- distributed_manager.py
- client.py
"""
from enum import Enum


class HostRole(str, Enum):
    """Roles for distributed hosts."""

    # Central management
    ORCHESTRATOR = "orchestrator"
    MANAGER = "manager"
    BACKGROUND = "background"

    # LLM hosts
    MAIN_LLM = "main_llm"
    MAIN_LLM_1 = "main_llm_1"  # First 3090
    MAIN_LLM_2 = "main_llm_2"  # Second 3090
    FAST_LLM = "fast_llm"

    # Specialized services
    EMBEDDINGS = "embeddings"
    RERANKING = "reranking"


class HostStatus(str, Enum):
    """Status of a distributed host."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Service types per host."""

    MORGAN_CORE = "morgan_core"
    OLLAMA = "ollama"
    QDRANT = "qdrant"
    REDIS = "redis"
    RERANKING_API = "reranking_api"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"


class GPURole(str, Enum):
    """Roles for GPU allocation."""

    MAIN_LLM = "main_llm"
    FAST_LLM = "fast_llm"
    EMBEDDINGS = "embeddings"
    RERANKING = "reranking"
    UTILITY = "utility"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for distributed LLM."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LATENCY_BASED = "latency_based"
    RANDOM = "random"


class ConnectionStatus(str, Enum):
    """Connection status states."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ServiceStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """Types of ML models."""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    CLASSIFIER = "classifier"
