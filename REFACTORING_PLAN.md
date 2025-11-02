# Morgan RAG - Complete Refactoring Plan
## Transformation to Production-Grade Clean Architecture

**Date:** November 2, 2025
**Current Version:** 1.0.0
**Target Version:** 2.0.0
**Estimated Timeline:** 2-3 weeks

---

## Executive Summary

Transform Morgan from a functional but scattered architecture into a **production-grade, enterprise-ready system** using:
- **Clean Architecture** (Uncle Bob Martin)
- **Domain-Driven Design** (DDD)
- **SOLID Principles**
- **Dependency Injection**
- **Hexagonal Architecture** (Ports & Adapters)

### Current Problems
1. ❌ **Circular Dependencies**: Core modules depend on each other tightly
2. ❌ **Mixed Concerns**: Business logic mixed with infrastructure
3. ❌ **Hard to Test**: Dependencies hardcoded, no interfaces
4. ❌ **Difficult to Extend**: Adding features requires touching many files
5. ❌ **No Clear Boundaries**: 60+ modules without clear hierarchy

### Target Benefits
1. ✅ **Testable**: Every component can be tested in isolation
2. ✅ **Maintainable**: Clear separation of concerns
3. ✅ **Extensible**: Easy to add new features
4. ✅ **Replaceable**: Swap implementations without changing business logic
5. ✅ **Clear**: Obvious where code belongs

---

## Architecture Transformation

### Current Structure (Anti-Pattern)
```
morgan/
├── background/         # Infrastructure
├── caching/           # Infrastructure
├── cli/               # Interface
├── communication/     # Domain?
├── companion/         # Domain
├── config/            # Infrastructure
├── core/              # Everything mixed!
│   ├── assistant.py      # God object (400+ lines)
│   ├── knowledge.py
│   ├── memory.py
│   └── search.py
├── emotional/         # Domain
├── emotions/          # Domain (duplicate?)
├── empathy/           # Domain
├── ingestion/         # Application?
├── interfaces/        # Interface
├── jina/              # Infrastructure
├── learning/          # Domain
├── memory/            # Domain (duplicate?)
├── migration/         # Infrastructure
├── models/            # Infrastructure
├── monitoring/        # Infrastructure
├── optimization/      # Infrastructure
├── personality/       # Domain
├── relationships/     # Domain
├── search/            # Application?
├── services/          # Infrastructure
├── storage/           # Infrastructure
├── utils/             # Cross-cutting
├── vector_db/         # Infrastructure
└── vectorization/     # Infrastructure
```

**Problems:**
- No clear layers
- Duplicate responsibilities (emotions/, emotional/, empathy/)
- Mixed business logic and infrastructure
- Circular dependencies
- Hard to understand dependencies

---

### Target Structure (Clean Architecture)

```
morgan/
├── domain/                      # 🔵 CORE - Business Logic (No Dependencies)
│   ├── __init__.py
│   ├── entities/                # Core business objects
│   │   ├── __init__.py
│   │   ├── conversation.py      # Conversation, Turn, Message
│   │   ├── user.py              # User, UserProfile, Preferences
│   │   ├── knowledge.py         # Document, Chunk, Source
│   │   ├── emotion.py           # EmotionalState, EmotionalContext
│   │   ├── relationship.py      # CompanionProfile, Milestone
│   │   └── memory.py            # Memory, MemoryContext
│   │
│   ├── value_objects/           # Immutable domain values
│   │   ├── __init__.py
│   │   ├── emotion_types.py     # EmotionType, IntensityLevel
│   │   ├── communication.py     # CommunicationStyle, Tone
│   │   ├── search_params.py     # SearchQuery, SearchResult
│   │   └── embeddings.py        # Embedding, EmbeddingScale
│   │
│   ├── repositories/            # Abstract interfaces (ports)
│   │   ├── __init__.py
│   │   ├── conversation.py      # IConversationRepository
│   │   ├── user.py              # IUserRepository
│   │   ├── knowledge.py         # IKnowledgeRepository
│   │   ├── memory.py            # IMemoryRepository
│   │   └── vector_store.py      # IVectorStoreRepository
│   │
│   ├── services/                # Domain services (business logic)
│   │   ├── __init__.py
│   │   ├── emotion_analyzer.py  # Pure emotion analysis logic
│   │   ├── relationship_builder.py
│   │   ├── memory_scorer.py     # Importance scoring
│   │   ├── conversation_flow.py
│   │   └── learning_engine.py   # Preference extraction logic
│   │
│   └── events/                  # Domain events
│       ├── __init__.py
│       ├── conversation.py      # ConversationStarted, TurnCompleted
│       ├── emotion.py           # EmotionDetected, MoodChanged
│       └── relationship.py      # MilestoneReached, BondStrengthened
│
├── application/                 # 🟢 USE CASES - Application Logic
│   ├── __init__.py
│   ├── use_cases/               # Business operations
│   │   ├── __init__.py
│   │   ├── conversation/
│   │   │   ├── __init__.py
│   │   │   ├── start_conversation.py
│   │   │   ├── process_query.py      # Main query processing
│   │   │   ├── provide_feedback.py
│   │   │   └── end_conversation.py
│   │   │
│   │   ├── knowledge/
│   │   │   ├── __init__.py
│   │   │   ├── ingest_documents.py
│   │   │   ├── search_knowledge.py
│   │   │   └── update_knowledge.py
│   │   │
│   │   ├── emotion/
│   │   │   ├── __init__.py
│   │   │   ├── detect_emotion.py
│   │   │   ├── track_mood.py
│   │   │   └── generate_empathy.py
│   │   │
│   │   ├── relationship/
│   │   │   ├── __init__.py
│   │   │   ├── build_profile.py
│   │   │   ├── detect_milestones.py
│   │   │   └── adapt_communication.py
│   │   │
│   │   └── learning/
│   │       ├── __init__.py
│   │       ├── extract_preferences.py
│   │       ├── analyze_patterns.py
│   │       └── adapt_behavior.py
│   │
│   ├── dto/                     # Data Transfer Objects
│   │   ├── __init__.py
│   │   ├── query_request.py
│   │   ├── query_response.py
│   │   ├── ingestion_request.py
│   │   └── profile_dto.py
│   │
│   └── ports/                   # Interfaces for infrastructure
│       ├── __init__.py
│       ├── llm_service.py       # ILLMService
│       ├── embedding_service.py # IEmbeddingService
│       ├── cache_service.py     # ICacheService
│       └── event_bus.py         # IEventBus
│
├── infrastructure/              # 🟡 INFRASTRUCTURE - External Concerns
│   ├── __init__.py
│   ├── persistence/             # Database adapters
│   │   ├── __init__.py
│   │   ├── qdrant/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── conversation_repository.py
│   │   │   ├── knowledge_repository.py
│   │   │   └── memory_repository.py
│   │   │
│   │   └── sqlite/              # For metadata
│   │       ├── __init__.py
│   │       ├── client.py
│   │       └── user_repository.py
│   │
│   ├── ai_services/             # External AI APIs
│   │   ├── __init__.py
│   │   ├── openai_compatible/
│   │   │   ├── __init__.py
│   │   │   ├── llm_adapter.py
│   │   │   └── embedding_adapter.py
│   │   │
│   │   ├── jina/
│   │   │   ├── __init__.py
│   │   │   ├── embedding_adapter.py
│   │   │   ├── reranker_adapter.py
│   │   │   └── scraper_adapter.py
│   │   │
│   │   └── local/
│   │       ├── __init__.py
│   │       ├── sentence_transformers.py
│   │       └── ollama_adapter.py
│   │
│   ├── caching/                 # Caching implementations
│   │   ├── __init__.py
│   │   ├── redis_cache.py
│   │   ├── memory_cache.py
│   │   └── git_hash_cache.py
│   │
│   ├── search/                  # Search engine implementations
│   │   ├── __init__.py
│   │   ├── hierarchical_search.py
│   │   ├── multi_stage_search.py
│   │   └── reranker.py
│   │
│   ├── vectorization/           # Embedding processing
│   │   ├── __init__.py
│   │   ├── hierarchical_embedder.py
│   │   ├── contrastive_clustering.py
│   │   └── batch_processor.py
│   │
│   ├── background/              # Background jobs
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── executor.py
│   │   └── tasks/
│   │
│   ├── monitoring/              # Observability
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics_collector.py
│   │   └── health_checker.py
│   │
│   └── config/                  # Configuration
│       ├── __init__.py
│       ├── settings.py
│       └── validators.py
│
├── interfaces/                  # 🟠 INTERFACES - External Access
│   ├── __init__.py
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── commands/
│   │   └── formatters/
│   │
│   ├── api/                     # REST/GraphQL API
│   │   ├── __init__.py
│   │   ├── rest/
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   ├── routes/
│   │   │   └── middleware/
│   │   │
│   │   └── websocket/
│   │       ├── __init__.py
│   │       └── handlers.py
│   │
│   └── web/                     # Web interface
│       ├── __init__.py
│       ├── static/
│       └── templates/
│
├── shared/                      # 🔴 SHARED - Cross-cutting Concerns
│   ├── __init__.py
│   ├── errors/                  # Custom exceptions
│   │   ├── __init__.py
│   │   ├── domain_errors.py
│   │   ├── application_errors.py
│   │   └── infrastructure_errors.py
│   │
│   ├── utils/                   # Pure utilities
│   │   ├── __init__.py
│   │   ├── text_processing.py
│   │   ├── validators.py
│   │   └── datetime_utils.py
│   │
│   └── types/                   # Shared types
│       ├── __init__.py
│       └── common.py
│
├── di/                          # 🟣 DEPENDENCY INJECTION
│   ├── __init__.py
│   ├── container.py             # DI container setup
│   ├── providers.py             # Service providers
│   └── scopes.py                # Lifecycle scopes
│
└── __init__.py                  # Package root
```

---

## Layer Responsibilities

### 🔵 Domain Layer (Core)
**Purpose:** Pure business logic, no external dependencies

**Rules:**
- ✅ No imports from other layers
- ✅ Only Python standard library + dataclasses/Pydantic
- ✅ Contains all business rules
- ✅ Framework-agnostic

**Examples:**
```python
# domain/entities/emotion.py
from dataclasses import dataclass
from enum import Enum

class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

@dataclass(frozen=True)
class EmotionalState:
    """Pure business object - no dependencies"""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float

    def __post_init__(self):
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0 and 1")
```

---

### 🟢 Application Layer
**Purpose:** Orchestrate business operations (use cases)

**Rules:**
- ✅ Can import from domain layer
- ✅ Depends on ports (interfaces), not implementations
- ✅ Contains no business logic (delegates to domain)
- ✅ Coordinates domain entities and services

**Examples:**
```python
# application/use_cases/conversation/process_query.py
from typing import Protocol
from domain.entities.conversation import Conversation
from domain.services.emotion_analyzer import EmotionAnalyzer
from application.ports.llm_service import ILLMService

class ProcessQueryUseCase:
    """Pure use case - depends only on interfaces"""

    def __init__(
        self,
        emotion_analyzer: EmotionAnalyzer,  # Domain service
        llm_service: ILLMService,           # Port (interface)
        conversation_repo: IConversationRepository
    ):
        self._emotion_analyzer = emotion_analyzer
        self._llm_service = llm_service
        self._conversation_repo = conversation_repo

    async def execute(self, request: QueryRequest) -> QueryResponse:
        # Orchestrate domain logic
        emotion = self._emotion_analyzer.analyze(request.query)
        conversation = await self._conversation_repo.get(request.conversation_id)

        # Use ports (interfaces)
        response = await self._llm_service.generate(
            query=request.query,
            emotion=emotion,
            context=conversation.context
        )

        return QueryResponse(answer=response, emotion=emotion)
```

---

### 🟡 Infrastructure Layer
**Purpose:** Implement technical details and external services

**Rules:**
- ✅ Implements application ports (interfaces)
- ✅ Can import from domain and application
- ✅ Contains all framework/library code
- ✅ Replaceable without changing business logic

**Examples:**
```python
# infrastructure/ai_services/openai_compatible/llm_adapter.py
from application.ports.llm_service import ILLMService
from infrastructure.config.settings import get_settings

class OpenAICompatibleLLMService(ILLMService):
    """Adapter implementing the port interface"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key
        )

    async def generate(
        self,
        query: str,
        emotion: EmotionalState,
        context: str
    ) -> str:
        # Implementation details hidden from application layer
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
```

---

### 🟠 Interfaces Layer
**Purpose:** External access points (CLI, API, Web)

**Rules:**
- ✅ Depends on application layer (use cases)
- ✅ Converts external requests to use case calls
- ✅ Formats responses for external consumers
- ✅ Handles HTTP/CLI-specific concerns

**Examples:**
```python
# interfaces/api/rest/routes/conversation.py
from fastapi import APIRouter, Depends
from application.use_cases.conversation.process_query import ProcessQueryUseCase
from application.dto.query_request import QueryRequest
from di.container import get_container

router = APIRouter()

@router.post("/query")
async def process_query(
    request: QueryRequest,
    use_case: ProcessQueryUseCase = Depends(get_container().process_query_use_case)
):
    """REST endpoint - just converts HTTP to use case"""
    response = await use_case.execute(request)
    return response.dict()
```

---

## Dependency Injection

### Container Setup
```python
# di/container.py
from dependency_injector import containers, providers
from application.use_cases.conversation.process_query import ProcessQueryUseCase
from domain.services.emotion_analyzer import EmotionAnalyzer
from infrastructure.ai_services.openai_compatible.llm_adapter import OpenAICompatibleLLMService

class Container(containers.DeclarativeContainer):
    """Central DI container"""

    # Configuration
    config = providers.Configuration()

    # Infrastructure (Adapters)
    llm_service = providers.Singleton(
        OpenAICompatibleLLMService
    )

    embedding_service = providers.Singleton(
        JinaEmbeddingService
    )

    conversation_repository = providers.Singleton(
        QdrantConversationRepository,
        client=providers.Singleton(QdrantClient)
    )

    # Domain Services
    emotion_analyzer = providers.Factory(
        EmotionAnalyzer
    )

    # Application Use Cases
    process_query_use_case = providers.Factory(
        ProcessQueryUseCase,
        emotion_analyzer=emotion_analyzer,
        llm_service=llm_service,
        conversation_repo=conversation_repository
    )
```

---

## Migration Strategy

### Phase 1: Create New Structure (Week 1)
1. Create new directory structure
2. Define domain entities and value objects
3. Define repository interfaces (ports)
4. Create application DTOs

### Phase 2: Extract Business Logic (Week 1-2)
1. Extract pure business logic to domain services
2. Move emotion detection to domain/services
3. Move relationship logic to domain/services
4. Create use cases in application layer

### Phase 3: Implement Infrastructure (Week 2)
1. Create adapters for external services
2. Implement repository interfaces
3. Set up dependency injection
4. Migrate configuration

### Phase 4: Update Interfaces (Week 2-3)
1. Refactor CLI to use use cases
2. Refactor API to use use cases
3. Update web interface

### Phase 5: Testing & Documentation (Week 3)
1. Update all tests
2. Create integration tests
3. Write migration guide
4. Update documentation

---

## Benefits Summary

### Before (Current)
- ❌ 400-line god object (assistant.py)
- ❌ Circular dependencies
- ❌ Hard to test
- ❌ Difficult to understand
- ❌ Business logic mixed with infrastructure

### After (Target)
- ✅ Clear separation of concerns
- ✅ Testable in isolation
- ✅ Easy to extend
- ✅ Replaceable components
- ✅ Self-documenting architecture
- ✅ Enterprise-ready

---

## Key Principles Applied

1. **Dependency Inversion**: High-level modules don't depend on low-level modules
2. **Single Responsibility**: Each class has one reason to change
3. **Open/Closed**: Open for extension, closed for modification
4. **Interface Segregation**: Many specific interfaces > one general interface
5. **Liskov Substitution**: Implementations interchangeable
6. **Don't Repeat Yourself (DRY)**: Shared logic in one place
7. **Keep It Simple, Stupid (KISS)**: Simplicity over cleverness
8. **Separation of Concerns**: Each layer has distinct responsibility

---

## Next Steps

1. Review and approve this plan
2. Create feature branch: `refactor/clean-architecture`
3. Start Phase 1: Directory structure
4. Implement iteratively with tests
5. Merge to main when stable

