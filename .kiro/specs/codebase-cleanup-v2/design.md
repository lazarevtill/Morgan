# Design Document: Codebase Cleanup v2

## Overview

This design document outlines the architectural changes needed to address 87 issues identified across the Morgan codebase. Since the app is in development, we take aggressive cleanup approach with NO backward compatibility concerns.

**Date**: 2025-12-27
**Status**: Design Complete
**Related Requirements**: requirements.md
**Note**: NO backward compatibility - delete duplicates directly

---

## 1. Architecture Overview

### 1.1 Current State (Problems)

```
Morgan/
├── morgan-rag/
│   └── morgan/
│       ├── exceptions.py           # MorganError v1
│       ├── config/
│       │   ├── settings.py         # llm_model = "gemma3:latest" !!!
│       │   ├── defaults.py         # llm_model = "qwen2.5:32b"
│       │   └── distributed_config.py  # Dataclasses + manual _cached_config
│       ├── services/
│       │   ├── llm/service.py      # Own singleton pattern
│       │   ├── embeddings/service.py  # Own singleton + race condition
│       │   └── reranking/service.py   # os.environ.get() + async-first
│       ├── infrastructure/
│       │   ├── distributed_gpu_manager.py  # HostRole enum v1
│       │   └── distributed_manager.py      # HostRole enum v2 (conflict!)
│       ├── utils/
│       │   ├── singleton.py        # SingletonFactory (good)
│       │   ├── error_handling.py   # MorganError v2 + manual singletons
│       │   ├── deduplication.py    # ResultDeduplicator (UNUSED!)
│       │   └── validators.py       # ValidationError duplicate
│       ├── intelligence/
│       │   ├── emotions/detector.py        # Emotion logic v1
│       │   └── core/intelligence_engine.py # Emotion logic v2 (duplicate!)
│       ├── search/
│       │   ├── multi_stage_search.py       # 3 dedup methods + memory search
│       │   └── companion_memory_search.py  # 1 dedup + memory search duplicate
│       ├── memory/
│       │   └── memory_processor.py  # 1 dedup method
│       └── communication/
│           └── cultural.py          # EMPTY FILE!
├── shared/
│   └── utils/
│       ├── singleton.py            # Different SingletonFactory!
│       ├── deduplication.py        # Different dedup utility!
│       └── exceptions.py           # Yet another exception hierarchy!
└── cli.py                          # DEPRECATED but exists
```

### 1.2 Target State (Clean)

```
Morgan/
├── morgan-rag/
│   └── morgan/
│       ├── exceptions.py           # SINGLE exception hierarchy
│       ├── config/
│       │   ├── settings.py         # Uses defaults.py values
│       │   ├── defaults.py         # SINGLE source of truth
│       │   └── distributed_config.py  # Uses SingletonFactory
│       ├── services/
│       │   ├── llm/service.py      # Uses SingletonFactory
│       │   ├── embeddings/service.py  # Uses SingletonFactory + locks
│       │   └── reranking/service.py   # Sync-first + uses settings
│       ├── infrastructure/
│       │   └── distributed_*.py    # Import HostRole from shared
│       ├── utils/
│       │   ├── singleton.py        # SINGLE implementation
│       │   ├── deduplication.py    # USED by all modules
│       │   └── validators.py       # Import from exceptions.py
│       ├── intelligence/
│       │   ├── constants.py        # SINGLE emotion patterns source
│       │   ├── emotions/detector.py  # SINGLE emotion detection
│       │   └── core/intelligence_engine.py  # Delegates to detector
│       ├── search/
│       │   └── multi_stage_search.py  # Uses ResultDeduplicator
│       └── communication/
│           └── cultural.py          # Removed from exports OR implemented
├── shared/
│   └── models/
│       └── enums.py                # HostRole, GPURole
└── (cli.py DELETED)
```

---

## 2. Component Designs

### 2.1 Unified Exception Hierarchy

**File**: `morgan-rag/morgan/exceptions.py` (KEEP AND ENHANCE)

```python
"""
SINGLE exception hierarchy for Morgan.
All custom exceptions inherit from MorganError.
DELETE duplicates from error_handling.py and validators.py.
"""
from typing import Optional, Dict, Any


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


# Configuration errors
class ConfigurationError(MorganError):
    """Configuration-related errors."""
    pass


class ValidationError(MorganError):
    """Input validation errors."""
    pass


# Service errors
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


# Companion errors
class CompanionError(MorganError):
    """Companion-related errors."""
    pass


class EmotionalProcessingError(CompanionError):
    """Emotional processing errors."""
    pass


class MemoryProcessingError(CompanionError):
    """Memory processing errors."""
    pass


# Infrastructure errors
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

**DELETE from these files**:
- `utils/error_handling.py:94-262` - All exception class definitions
- `utils/validators.py:10` - ValidationError class
- `utils/companion_error_handling.py` - EmotionalProcessingError, MemoryProcessingError
- `shared/utils/exceptions.py` - Entire file (if not needed elsewhere)

---

### 2.2 Unified Singleton Factory

**File**: `morgan-rag/morgan/utils/singleton.py` (KEEP - more complete)

```python
"""
SINGLE Singleton Factory implementation.
DELETE manual singleton patterns from:
- error_handling.py:1109-1136
- distributed_config.py:511-534
- All service _*_instance patterns (migrate to use this)
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
        with cls._global_lock:
            if service_class not in cls._locks:
                cls._locks[service_class] = threading.Lock()

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

    def new_new(klass, *args, **kwargs):
        return SingletonFactory.get_or_create(klass, lambda: object.__new__(klass))

    cls.__new__ = new_new
    return cls
```

**Migrate services to use factory**:

```python
# services/llm/service.py - REPLACE get_llm_service()
from morgan.utils.singleton import SingletonFactory

def get_llm_service(**kwargs) -> LLMService:
    return SingletonFactory.get_or_create(LLMService, **kwargs)

def reset_llm_service() -> None:
    SingletonFactory.reset(LLMService)
```

---

### 2.3 Unified Configuration Defaults

**File**: `morgan-rag/morgan/config/defaults.py` (SINGLE source of truth)

```python
"""
SINGLE source for all default values.
settings.py and distributed_config.py MUST use these values.
"""

class Defaults:
    """Centralized default values for Morgan configuration."""

    # LLM Configuration
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_MODEL = "qwen2.5:32b-instruct-q4_K_M"  # NOT gemma3!
    LLM_API_KEY = "ollama"
    LLM_TIMEOUT = 120.0
    LLM_MAX_TOKENS = 4096
    LLM_TEMPERATURE = 0.7

    # Fast LLM
    FAST_LLM_MODEL = "qwen2.5:7b-instruct-q5_K_M"

    # Embedding Configuration
    EMBEDDING_BASE_URL = "http://localhost:11434/v1"
    EMBEDDING_MODEL = "qwen3-embedding:4b"
    EMBEDDING_DIMENSIONS = 2048

    # Reranking Configuration
    RERANKING_MODEL = "ms-marco-MiniLM-L-6-v2"
    RERANKING_TOP_K = 10

    # Vector Database
    QDRANT_URL = "http://localhost:6333"
    QDRANT_COLLECTION = "morgan_memories"  # Consistent name!

    # Redis
    REDIS_URL = "redis://localhost:6379"

    # Search
    SEARCH_TOP_K = 10
    SEARCH_MIN_SCORE = 0.3
    SIMILARITY_THRESHOLD = 0.95

    # Emotional Intelligence
    EMOTION_CONFIDENCE_THRESHOLD = 0.7
```

**Update settings.py**:

```python
# morgan-rag/morgan/config/settings.py
from .defaults import Defaults

class Settings(BaseSettings):
    llm_base_url: str = Field(default=Defaults.LLM_BASE_URL)
    llm_model: str = Field(default=Defaults.LLM_MODEL)  # Uses Qwen, not Gemma!
    embedding_model: str = Field(default=Defaults.EMBEDDING_MODEL)
    # ... etc
```

---

### 2.4 Intelligence Constants (Centralized)

**File**: `morgan-rag/morgan/intelligence/constants.py` (NEW)

```python
"""
SINGLE source for emotion-related constants.
DELETE duplicates from detector.py, intelligence_engine.py, analyzer.py, etc.
"""
from enum import Enum
from typing import Dict, List


class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


# SINGLE valence mapping - DELETE from analyzer.py:44-52, context.py:135-143, recovery.py:110-119
EMOTION_VALENCE: Dict[EmotionType, float] = {
    EmotionType.JOY: 1.0,
    EmotionType.TRUST: 0.8,
    EmotionType.ANTICIPATION: 0.6,
    EmotionType.SURPRISE: 0.3,
    EmotionType.NEUTRAL: 0.0,
    EmotionType.FEAR: -0.6,
    EmotionType.SADNESS: -0.8,
    EmotionType.ANGER: -0.9,
    EmotionType.DISGUST: -0.7,
}


# SINGLE patterns - DELETE from detector.py:40-82, intelligence_engine.py:51-88
EMOTION_PATTERNS: Dict[EmotionType, List[str]] = {
    EmotionType.JOY: [
        r"\b(?:happy|glad|delighted|excited|thrilled|joyful|pleased)\b",
        r"\b(?:love|adore|wonderful|amazing|fantastic)\b",
        r"[!]{2,}",
        r"[:;]-?[)D]",
    ],
    EmotionType.SADNESS: [
        r"\b(?:sad|unhappy|depressed|down|blue|miserable|heartbroken)\b",
        r"\b(?:sorry|regret|miss|lonely|hopeless)\b",
        r"[:;]-?[(]",
    ],
    EmotionType.ANGER: [
        r"\b(?:angry|furious|mad|annoyed|irritated|frustrated)\b",
        r"\b(?:hate|rage|outraged)\b",
        r"[!]{3,}",
    ],
    EmotionType.FEAR: [
        r"\b(?:afraid|scared|terrified|anxious|worried|nervous)\b",
        r"\b(?:panic|dread|horror)\b",
    ],
    EmotionType.SURPRISE: [
        r"\b(?:surprised|shocked|amazed|astonished|stunned)\b",
        r"\b(?:wow|omg|whoa|unexpected)\b",
    ],
    EmotionType.DISGUST: [
        r"\b(?:disgusted|gross|revolting|sick|nasty)\b",
    ],
    EmotionType.NEUTRAL: [],
}


# SINGLE intensity modifiers - DELETE from detector.py:85-98, intelligence_engine.py:91-102, intensity.py:49-77
INTENSITY_MODIFIERS: Dict[str, float] = {
    # Amplifiers
    "very": 1.3,
    "extremely": 1.5,
    "incredibly": 1.4,
    "really": 1.2,
    "so": 1.2,
    "absolutely": 1.5,
    "completely": 1.4,
    "totally": 1.3,
    # Diminishers
    "slightly": 0.6,
    "somewhat": 0.7,
    "a bit": 0.7,
    "a little": 0.7,
    "kind of": 0.8,
    "sort of": 0.8,
    # Negators
    "not": -0.5,
    "never": -0.6,
    "no": -0.4,
}


# Formality indicators - DELETE from wherever duplicated
FORMALITY_INDICATORS: Dict[str, List[str]] = {
    "formal": [
        r"\b(?:therefore|furthermore|consequently|nevertheless)\b",
        r"\b(?:please|kindly|respectfully)\b",
        r"\b(?:dear|sincerely|regards)\b",
    ],
    "informal": [
        r"\b(?:gonna|wanna|gotta|kinda|sorta)\b",
        r"\b(?:yeah|yup|nope|hey|yo)\b",
        r"[!?]{2,}",
    ],
}
```

---

### 2.5 Service Pattern Updates

**File**: `services/reranking/service.py` - Key changes:

```python
# BEFORE (async-first, wrong pattern)
async def rerank(self, query, documents, top_k=10):
    ...

def rerank_sync(self, query, documents, top_k=10):
    loop = asyncio.new_event_loop()  # DEPRECATED!
    try:
        return loop.run_until_complete(self.rerank(...))
    finally:
        loop.close()

# AFTER (sync-first like other services)
def rerank(self, query, documents, top_k=10):
    """Synchronous reranking."""
    ...

async def arerank(self, query, documents, top_k=10):
    """Async wrapper for reranking."""
    return await asyncio.to_thread(self.rerank, query, documents, top_k)


# BEFORE (os.environ.get)
self.endpoint = endpoint or os.environ.get("MORGAN_RERANKING_ENDPOINT")

# AFTER (use settings)
self.endpoint = endpoint or getattr(self.settings, "reranking_endpoint", None)
```

**Add locks for thread safety**:

```python
# services/embeddings/service.py
class EmbeddingService:
    def __init__(self):
        self._availability_lock = threading.Lock()
        self._remote_available = None

    def _check_remote_available(self):
        with self._availability_lock:  # ADD LOCK
            if self._remote_available is None:
                self._remote_available = self._probe_remote()
            return self._remote_available
```

---

### 2.6 Deduplication Consolidation

**DELETE these methods and use `ResultDeduplicator`**:

```python
# DELETE from multi_stage_search.py:
# - _deduplicate_results() (lines 1729-1775)
# - _apply_rrf_deduplication() (lines 1660-1727)
# - _deduplicate_memory_results() (lines 2300-2334)

# DELETE from companion_memory_search.py:
# - _deduplicate_search_results() (lines 1057-1075)

# DELETE from memory_processor.py:
# - _deduplicate_memories() (lines 608-623)

# USE INSTEAD:
from morgan.utils.deduplication import ResultDeduplicator

class MultiStageSearchEngine:
    def __init__(self):
        self.deduplicator = ResultDeduplicator()

    def _deduplicate_results(self, results):
        return self.deduplicator.deduplicate_by_similarity(
            results,
            embedding_getter=lambda r: r.embedding,
            threshold=0.95
        )
```

---

### 2.7 Communication Cultural Module

**Option A - Remove from exports** (simpler):

```python
# communication/__init__.py - REMOVE this line:
# from .cultural import CulturalEmotionalAwareness

# DELETE: communication/cultural.py
```

**Option B - Implement stub** (if needed later):

```python
# communication/cultural.py
"""Cultural emotional awareness module."""

class CulturalEmotionalAwareness:
    """Placeholder for cultural awareness features."""

    def __init__(self):
        raise NotImplementedError(
            "CulturalEmotionalAwareness is not yet implemented. "
            "Remove from imports or implement this class."
        )
```

---

## 3. File Changes Summary

### Files to DELETE

| File | Reason |
|------|--------|
| `cli.py` (root) | DEPRECATED stub |
| `shared/utils/singleton.py` | Duplicate (use morgan/utils/singleton.py) |
| `shared/utils/deduplication.py` | Duplicate (use morgan/utils/deduplication.py) |
| `communication/cultural.py` | Empty (or implement) |

### Files to CREATE

| File | Purpose |
|------|---------|
| `intelligence/constants.py` | Centralized emotion patterns, valence, modifiers |

### Files to MODIFY (Major)

| File | Changes |
|------|---------|
| `utils/error_handling.py` | DELETE lines 94-262 (exception classes) |
| `utils/validators.py` | DELETE ValidationError class, import from exceptions.py |
| `utils/companion_error_handling.py` | DELETE exception classes, import from exceptions.py |
| `services/reranking/service.py` | Sync-first, use settings, fix event loop |
| `services/embeddings/service.py` | Add availability lock |
| `config/settings.py` | Use Defaults class values |
| `search/multi_stage_search.py` | Use ResultDeduplicator, delete 3 methods |
| `search/companion_memory_search.py` | Use ResultDeduplicator, delete dedup method |
| `memory/memory_processor.py` | Use ResultDeduplicator, delete dedup method |
| `intelligence/emotions/detector.py` | Import patterns from constants.py |
| `intelligence/core/intelligence_engine.py` | Delegate emotion detection to detector.py |

### Files to MODIFY (Minor)

| File | Changes |
|------|---------|
| `distributed_config.py` | Use SingletonFactory, delete _cached_config |
| `llm/service.py` | Use SingletonFactory |
| `embeddings/service.py` | Use SingletonFactory, delete unused Path import |
| `reranking/service.py` | Delete unused Path import |
| `infrastructure/distributed_gpu_manager.py` | Import HostRole from shared |
| `infrastructure/distributed_manager.py` | Import HostRole from shared |

---

## 4. Migration Steps (No Backward Compatibility)

### Phase 1: Critical Fixes (2-3 hours)
1. Fix `communication/cultural.py` - remove from exports or implement
2. Add locks to embeddings/reranking availability flags
3. Fix collection name mismatch (if exists)

### Phase 2: Exception Consolidation (2-3 hours)
1. Enhance `exceptions.py` with all needed exceptions
2. DELETE exception definitions from `error_handling.py:94-262`
3. DELETE ValidationError from `validators.py`
4. UPDATE all imports to use `from morgan.exceptions import X`

### Phase 3: Singleton Consolidation (2-3 hours)
1. DELETE `shared/utils/singleton.py`
2. DELETE manual singletons from `error_handling.py:1109-1136`
3. DELETE `_cached_config` from `distributed_config.py`
4. MIGRATE all services to use `SingletonFactory`

### Phase 4: Configuration Unification (1-2 hours)
1. UPDATE `settings.py` to use `Defaults` class values
2. UPDATE `distributed_config.py` to use `Defaults` class values
3. FIX `reranking/service.py` to use settings instead of os.environ

### Phase 5: Intelligence Consolidation (2-3 hours)
1. CREATE `intelligence/constants.py`
2. MOVE patterns from detector.py and intelligence_engine.py to constants.py
3. UPDATE detector.py to import from constants.py
4. UPDATE intelligence_engine.py to delegate to detector.py

### Phase 6: Deduplication Consolidation (2-3 hours)
1. DELETE `shared/utils/deduplication.py`
2. UPDATE search modules to use `ResultDeduplicator`
3. DELETE 5 duplicate dedup methods

### Phase 7: API Standardization (1-2 hours)
1. FIX reranking to sync-first pattern
2. FIX LLM streaming method naming
3. FIX deprecated event loop patterns

### Phase 8: Dead Code Removal (1-2 hours)
1. DELETE `cli.py`
2. DELETE unused imports (Path)
3. DELETE or document unused methods

---

## 5. Validation Checklist

- [ ] No duplicate exception definitions (grep for "class.*Error.*Exception")
- [ ] No duplicate singleton patterns (grep for "_instance = None")
- [ ] No duplicate deduplication methods (grep for "def.*dedup")
- [ ] Single HostRole enum definition
- [ ] Collection names consistent across memory/search
- [ ] All imports resolve correctly (run python -c "import morgan")
- [ ] All tests pass
- [ ] No silent error suppression (grep for "except.*pass")
- [ ] No deprecated asyncio patterns (grep for "new_event_loop")

---

*Last Updated: 2025-12-27*
