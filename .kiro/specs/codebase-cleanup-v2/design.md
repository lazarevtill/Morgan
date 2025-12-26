# Design Document: Codebase Cleanup v2

## Overview

This design document outlines the architectural changes needed to address 150+ issues identified across the Morgan codebase. The design focuses on consolidation, standardization, and elimination of technical debt while maintaining backward compatibility.

**Date**: 2025-12-26
**Status**: Design Complete
**Related Requirements**: requirements.md

---

## 1. Architecture Overview

### 1.1 Current State (Problems)

```
Morgan/
├── morgan-rag/
│   └── morgan/
│       ├── exceptions.py           # Defines 6 unused exceptions
│       ├── config/
│       │   └── settings.py         # Settings class
│       ├── services/
│       │   ├── embeddings/
│       │   │   └── service.py      # Duplicate setup_model_cache()
│       │   └── reranking/
│       │       └── service.py      # Duplicate setup_model_cache()
│       ├── infrastructure/
│       │   ├── distributed_gpu_manager.py  # HostRole enum v1
│       │   └── distributed_manager.py      # HostRole enum v2 (conflict!)
│       ├── utils/
│       │   ├── error_handling.py   # Duplicate exceptions
│       │   ├── companion_error_handling.py  # More duplicates
│       │   └── validators.py       # ValidationError duplicate
│       └── memory/
│           └── memory_processor.py # Uses "morgan_memories"
├── morgan-server/
│   └── morgan_server/
│       └── config.py              # ServerConfig (different from Settings)
├── morgan-cli/
│   └── morgan_cli/
│       └── config.py              # Config (third config class!)
└── shared/                        # Underutilized
```

### 1.2 Target State (Solution)

```
Morgan/
├── shared/                        # Centralized shared code
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseSettings (all services inherit)
│   │   ├── defaults.py           # Single source of defaults
│   │   └── validators.py         # Shared validation logic
│   ├── exceptions/
│   │   ├── __init__.py
│   │   └── base.py               # MorganError + all exceptions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── singleton.py          # SingletonFactory
│   │   ├── deduplication.py      # Unified deduplication
│   │   ├── text_extraction.py    # Entity/keyword extraction
│   │   └── health_monitor.py     # Health monitoring mixin
│   └── models/
│       ├── __init__.py
│       └── enums.py              # HostRole, GPURole, etc.
├── morgan-rag/
│   └── morgan/
│       ├── config/
│       │   └── settings.py       # Inherits from shared.config.base
│       ├── services/             # Uses shared utilities
│       ├── infrastructure/       # Uses shared.models.enums
│       └── intelligence/
│           └── constants.py      # EMOTION_VALENCE, etc.
├── morgan-server/
│   └── morgan_server/
│       └── config.py             # Inherits from shared.config.base
└── morgan-cli/
    └── morgan_cli/
        └── config.py             # Inherits from shared.config.base
```

---

## 2. Component Designs

### 2.1 Shared Exception Hierarchy

**File**: `shared/exceptions/base.py`

```python
"""
Unified exception hierarchy for Morgan.
All custom exceptions inherit from MorganError.
"""

class MorganError(Exception):
    """Base exception for all Morgan errors."""

    def __init__(
        self,
        message: str,
        service: str = "unknown",
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.service = service
        self.operation = operation
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "service": self.service,
            "operation": self.operation,
            "details": self.details
        }


class ConfigurationError(MorganError):
    """Configuration-related errors."""
    pass


class ValidationError(MorganError):
    """Input validation errors."""
    pass


class ServiceError(MorganError):
    """Base for all service errors."""
    pass


class LLMError(ServiceError):
    """LLM service errors."""
    pass


class EmbeddingError(ServiceError):
    """Embedding service errors."""
    pass


class RerankingError(ServiceError):
    """Reranking service errors."""
    pass


class VectorDBError(ServiceError):
    """Vector database errors."""
    pass


class MemoryError(ServiceError):
    """Memory service errors."""
    pass


class SearchError(ServiceError):
    """Search service errors."""
    pass


class InfrastructureError(MorganError):
    """Infrastructure-related errors."""
    pass


class ConnectionError(InfrastructureError):
    """Connection errors."""
    pass


class TimeoutError(InfrastructureError):
    """Timeout errors."""
    pass
```

**Migration Strategy**:
1. Create new `shared/exceptions/base.py`
2. Update `morgan/exceptions.py` to re-export from shared
3. Update all imports to use `from morgan.exceptions import X`
4. Delete duplicate definitions in `utils/error_handling.py`, etc.

---

### 2.2 Shared Configuration Base

**File**: `shared/config/base.py`

```python
"""
Base configuration for all Morgan services.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from pathlib import Path


class MorganBaseSettings(BaseSettings):
    """Base settings inherited by all Morgan services."""

    # LLM Configuration
    llm_endpoint: str = Field(
        default="http://localhost:11434/v1",
        alias="MORGAN_LLM_ENDPOINT"
    )
    llm_model: str = Field(
        default="qwen2.5:32b-instruct-q4_K_M",
        alias="MORGAN_LLM_MODEL"
    )

    # Embedding Configuration
    embedding_endpoint: Optional[str] = Field(
        default=None,
        alias="MORGAN_EMBEDDING_ENDPOINT"
    )
    embedding_model: str = Field(
        default="qwen3-embedding:4b",
        alias="MORGAN_EMBEDDING_MODEL"
    )

    # Vector Database
    vector_db_url: str = Field(
        default="http://localhost:6333",
        alias="MORGAN_VECTOR_DB_URL"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        alias="MORGAN_REDIS_URL"
    )

    # Cache
    cache_dir: Path = Field(
        default=Path("~/.morgan/cache").expanduser(),
        alias="MORGAN_CACHE_DIR"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        alias="MORGAN_LOG_LEVEL"
    )

    class Config:
        env_prefix = "MORGAN_"
        env_file = ".env"
        extra = "ignore"

    @field_validator("llm_endpoint", "vector_db_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v}")
        return v.rstrip("/")
```

**Service-Specific Extensions**:

```python
# morgan-rag/morgan/config/settings.py
from shared.config.base import MorganBaseSettings

class Settings(MorganBaseSettings):
    """Morgan RAG specific settings."""

    # RAG-specific settings
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=10)

    # Distributed settings
    distributed_mode: bool = Field(default=False)
    endpoints: List[str] = Field(default_factory=list)
```

---

### 2.3 Singleton Factory

**File**: `shared/utils/singleton.py`

```python
"""
Thread-safe singleton factory for services.
"""
import threading
from typing import TypeVar, Type, Optional, Dict, Any, Callable

T = TypeVar('T')


class SingletonFactory:
    """Thread-safe singleton factory with cleanup support."""

    _instances: Dict[Type, Any] = {}
    _locks: Dict[Type, threading.Lock] = {}
    _global_lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        service_class: Type[T],
        factory: Optional[Callable[[], T]] = None,
        force_new: bool = False,
        **kwargs
    ) -> T:
        """Get existing instance or create new one."""
        # Ensure lock exists for this class
        with cls._global_lock:
            if service_class not in cls._locks:
                cls._locks[service_class] = threading.Lock()

        # Double-checked locking
        if service_class not in cls._instances or force_new:
            with cls._locks[service_class]:
                if service_class not in cls._instances or force_new:
                    if factory:
                        cls._instances[service_class] = factory()
                    else:
                        cls._instances[service_class] = service_class(**kwargs)

        return cls._instances[service_class]

    @classmethod
    def reset(cls, service_class: Type[T]) -> None:
        """Reset singleton instance with cleanup."""
        with cls._global_lock:
            if service_class in cls._instances:
                instance = cls._instances[service_class]
                # Call cleanup method if exists
                if hasattr(instance, 'shutdown'):
                    instance.shutdown()
                elif hasattr(instance, 'close'):
                    instance.close()
                elif hasattr(instance, 'clear_cache'):
                    instance.clear_cache()
                del cls._instances[service_class]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all singleton instances."""
        with cls._global_lock:
            for service_class in list(cls._instances.keys()):
                cls.reset(service_class)


def singleton(cls: Type[T]) -> Type[T]:
    """Decorator to make a class a singleton."""
    original_new = cls.__new__

    def new_new(cls, *args, **kwargs):
        return SingletonFactory.get_or_create(cls, lambda: original_new(cls))

    cls.__new__ = new_new
    return cls
```

---

### 2.4 Collection Name Fix

**Problem**: Memory stored in `morgan_memories`, searched in `morgan_turns`

**File Changes**:

```python
# morgan-rag/morgan/search/multi_stage_search.py
# Line 176: Change from:
self.memory_collection = "morgan_turns"
# To:
self.memory_collection = "morgan_memories"  # Must match memory_processor.py
```

**Validation**: Add integration test to verify store-then-search works.

---

### 2.5 Shared Enums

**File**: `shared/models/enums.py`

```python
"""
Shared enumerations for Morgan infrastructure.
"""
from enum import Enum


class HostRole(str, Enum):
    """Roles for distributed hosts."""
    ORCHESTRATOR = "orchestrator"
    MAIN_LLM = "main_llm"
    FAST_LLM = "fast_llm"
    EMBEDDINGS = "embeddings"
    RERANKING = "reranking"
    BACKGROUND = "background"
    MANAGER = "manager"


class GPURole(str, Enum):
    """Roles for GPU allocation."""
    MAIN_LLM = "main_llm"
    FAST_LLM = "fast_llm"
    EMBEDDINGS = "embeddings"
    RERANKING = "reranking"
    UTILITY = "utility"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LATENCY_BASED = "latency_based"
    RANDOM = "random"


class ConnectionStatus(str, Enum):
    """Connection status states."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
```

---

### 2.6 Text Extraction Utility

**File**: `shared/utils/text_extraction.py`

```python
"""
Unified text extraction utilities.
"""
import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class Entity:
    """Extracted entity."""
    text: str
    entity_type: str
    confidence: float = 1.0


# Common patterns
ENTITY_PATTERNS = {
    "person": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
    "technology": r"\b(?:Python|JavaScript|Docker|Kubernetes|AWS|React|Node\.js|SQL|API|REST|GraphQL)\b",
    "organization": r"\b(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Group))?)\b",
}

STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "although", "though",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
}


def extract_entities(text: str) -> List[Entity]:
    """Extract named entities from text."""
    entities = []

    for entity_type, pattern in ENTITY_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(),
                entity_type=entity_type
            ))

    return entities


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    """Extract meaningful keywords from text."""
    # Tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Remove stop words
    keywords = [w for w in words if w not in STOP_WORDS]

    # Count frequency
    freq: Dict[str, int] = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency
    sorted_keywords = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)

    return sorted_keywords[:max_keywords]


def extract_topics(text: str) -> List[str]:
    """Extract topic categories from text."""
    topic_keywords = {
        "technology": ["tech", "computer", "software", "code", "programming", "api"],
        "health": ["health", "fitness", "exercise", "medical", "wellness"],
        "finance": ["money", "investment", "budget", "financial", "stock"],
        "education": ["learn", "study", "course", "training", "education"],
        "travel": ["travel", "trip", "vacation", "destination", "flight"],
    }

    text_lower = text.lower()
    detected = []

    for topic, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(topic)

    return detected
```

---

### 2.7 Deduplication Utility

**File**: `shared/utils/deduplication.py`

```python
"""
Unified deduplication utilities.
"""
import hashlib
from typing import List, TypeVar, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class DeduplicationResult:
    """Result of deduplication."""
    unique_items: List
    duplicates_removed: int
    original_count: int


def deduplicate_by_content(
    items: List[T],
    content_getter: Callable[[T], str]
) -> DeduplicationResult:
    """Deduplicate items by content hash."""
    seen_hashes = set()
    unique = []

    for item in items:
        content = content_getter(item)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique.append(item)

    return DeduplicationResult(
        unique_items=unique,
        duplicates_removed=len(items) - len(unique),
        original_count=len(items)
    )


def deduplicate_by_similarity(
    items: List[T],
    embedding_getter: Callable[[T], List[float]],
    threshold: float = 0.95
) -> DeduplicationResult:
    """Deduplicate items by embedding similarity."""
    import numpy as np

    if not items:
        return DeduplicationResult([], 0, 0)

    unique = [items[0]]
    unique_embeddings = [np.array(embedding_getter(items[0]))]

    for item in items[1:]:
        embedding = np.array(embedding_getter(item))

        # Check similarity with all unique items
        is_duplicate = False
        for unique_emb in unique_embeddings:
            similarity = np.dot(embedding, unique_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(unique_emb)
            )
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(item)
            unique_embeddings.append(embedding)

    return DeduplicationResult(
        unique_items=unique,
        duplicates_removed=len(items) - len(unique),
        original_count=len(items)
    )
```

---

## 3. Migration Strategy

### 3.1 Phase Order

1. **Phase 1**: Critical fixes (collection names, test imports)
2. **Phase 2**: Create shared module structure
3. **Phase 3**: Migrate exceptions to shared
4. **Phase 4**: Migrate configuration to shared
5. **Phase 5**: Migrate utilities to shared
6. **Phase 6**: Update all imports
7. **Phase 7**: Remove dead code
8. **Phase 8**: Validation and testing

### 3.2 Backward Compatibility

- Keep `morgan/exceptions.py` as re-export layer
- Keep `morgan/config/settings.py` as service-specific extension
- Add deprecation warnings for old import paths
- Provide migration guide for environment variables

### 3.3 Testing Strategy

1. Run existing tests after each phase
2. Add integration tests for critical paths
3. Validate collection name fix with store-then-search test
4. Validate exception hierarchy with unit tests

---

## 4. File Changes Summary

### Files to Create

| File | Purpose |
|------|---------|
| `shared/config/__init__.py` | Config module init |
| `shared/config/base.py` | Base settings class |
| `shared/config/defaults.py` | Centralized defaults |
| `shared/config/validators.py` | Shared validators |
| `shared/exceptions/__init__.py` | Exceptions module init |
| `shared/exceptions/base.py` | Exception hierarchy |
| `shared/utils/singleton.py` | Singleton factory |
| `shared/utils/deduplication.py` | Deduplication utilities |
| `shared/utils/text_extraction.py` | Text extraction utilities |
| `shared/utils/health_monitor.py` | Health monitoring mixin |
| `shared/models/__init__.py` | Models module init |
| `shared/models/enums.py` | Shared enums |
| `intelligence/constants.py` | Emotion constants |

### Files to Modify

| File | Changes |
|------|---------|
| `multi_stage_search.py:176` | Fix collection name |
| `test_client_properties.py:372` | Fix exception import |
| `test_client_properties.py:15-22` | Add WebSocketClient import |
| `morgan/exceptions.py` | Re-export from shared |
| `embeddings/service.py` | Remove duplicate setup_model_cache |
| `reranking/service.py` | Remove duplicate setup_model_cache |
| `distributed_gpu_manager.py` | Import HostRole from shared |
| `distributed_manager.py` | Import HostRole from shared |
| `error_handling.py` | Import exceptions from shared |
| `client.py` | Fix cleanup_memory params bug |
| `profile.py` | Fix silent error suppression |

### Files to Delete

| File | Reason |
|------|--------|
| `cli.py` | Deprecated stub |
| `consul_client.py` | Unused |
| `docker/env.example` | Merge into `.env.example` |

---

## 5. Validation Checklist

- [ ] All tests pass after each phase
- [ ] No duplicate exception definitions
- [ ] No duplicate setup_model_cache implementations
- [ ] Single HostRole enum definition
- [ ] Collection names consistent
- [ ] All imports resolve correctly
- [ ] No silent error suppression
- [ ] Client bugs fixed
- [ ] Dead code removed
