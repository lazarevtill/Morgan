# Morgan RAG - Complete Refactoring Guide
## Step-by-Step Implementation

This guide provides a complete, actionable plan to refactor Morgan to Clean Architecture.

---

## Overview

**Current Status**: 70% complete codebase with scattered architecture
**Target**: Production-grade Clean Architecture
**Timeline**: You can choose your speed
**Compatibility**: Complete replacement - no backward compatibility

---

## Approach Options

### Option A: Automated Refactoring (Recommended - Fastest)
Run the automation script I created:
```bash
cd /c/Users/lazarev/Documents/GitHub/Morgan
python scripts/refactor_to_v2.py --dry-run  # Preview changes
python scripts/refactor_to_v2.py            # Execute refactoring
```

**Pros**: Fastest (2-4 hours total), consistent structure
**Cons**: Need to review generated code

### Option B: Manual Refactoring (Learning Approach)
Follow this guide step-by-step to understand every change.

**Pros**: Deep understanding, full control
**Cons**: Slower (1-2 weeks)

### Option C: Hybrid (Balanced)
1. Run script to generate structure
2. Manually review and adjust business logic
3. Test thoroughly

**Pros**: Fast + controlled
**Cons**: Requires both technical and domain knowledge

---

## I Recommend: Option C (Hybrid Approach)

Here's why and how:

### Step 1: Run Structure Generation
```bash
# This creates the directory structure and templates
python scripts/refactor_to_v2.py --dry-run --keep-old
```

Review the output to see what will be created.

### Step 2: Execute Actual Refactoring
```bash
# This will:
# 1. Create new morgan_v2/ directory
# 2. Generate all files
# 3. Move old morgan/ to morgan_old/
python scripts/refactor_to_v2.py --keep-old
```

### Step 3: Manual Business Logic Migration
Now you manually migrate the important business logic:

1. **Emotion Detection Logic** (morgan_old/emotional/* → morgan_v2/domain/services/emotion_analyzer.py)
2. **Learning Engine** (morgan_old/learning/* → morgan_v2/domain/services/learning_engine.py)
3. **Relationship Logic** (morgan_old/companion/* → morgan_v2/domain/services/relationship_builder.py)
4. **Search Logic** (morgan_old/search/* → morgan_v2/infrastructure/search/)
5. **LLM Service** (morgan_old/services/llm_service.py → morgan_v2/infrastructure/ai_services/openai_compatible/llm_adapter.py)

### Step 4: Wire Everything with DI
Update `morgan_v2/di/container.py` to inject all dependencies.

### Step 5: Test
```bash
pytest morgan-rag/tests/
```

### Step 6: Remove Old Code
```bash
rm -rf morgan-rag/morgan_old
mv morgan-rag/morgan_v2 morgan-rag/morgan
```

---

## Detailed Manual Steps (If You Choose Option B)

### Week 1: Domain Layer

#### Day 1: Setup Structure
```bash
cd morgan-rag
mkdir -p morgan_v2/{domain,application,infrastructure,interfaces,shared,di}
mkdir -p morgan_v2/domain/{entities,value_objects,repositories,services,events}
```

#### Day 2: Create Domain Entities
Create these files (I've already started some):

1. `domain/entities/emotion.py` ✅ (Already created - 300 lines)
2. `domain/entities/conversation.py` (Use template from script)
3. `domain/entities/user.py` (Use template from script)
4. `domain/entities/knowledge.py` (Use template from script)
5. `domain/entities/relationship.py` (Use template from script)
6. `domain/entities/memory.py` (Use template from script)

#### Day 3: Create Value Objects
Value objects are immutable. Create:

1. `domain/value_objects/communication.py` - CommunicationStyle, Tone
2. `domain/value_objects/search_params.py` - SearchQuery, SearchResult
3. `domain/value_objects/embeddings.py` - Embedding, EmbeddingScale

#### Day 4-5: Extract Business Logic to Domain Services

**Important**: Domain services contain pure business logic with NO infrastructure dependencies.

Example - Extract emotion detection:
```python
# domain/services/emotion_analyzer.py
from domain.entities.emotion import EmotionalState, EmotionType

class EmotionAnalyzer:
    """Pure business logic - no external dependencies"""

    def analyze_text(self, text: str) -> EmotionalState:
        """
        Analyze text and detect emotion.
        Pure logic - no AI calls, no database.
        """
        # Extract from old code: morgan/emotional/intelligence_engine.py
        # Keep ONLY the pattern matching logic
        # Remove all infrastructure (LLM calls, DB access)

        intensity = self._calculate_intensity(text)
        emotion_type = self._detect_emotion_type(text)

        return EmotionalState(
            primary_emotion=emotion_type,
            intensity=intensity,
            confidence=0.85
        )
```

Migrate these domain services:
- [ ] `emotion_analyzer.py` - from morgan/emotional/*
- [ ] `relationship_builder.py` - from morgan/companion/*
- [ ] `memory_scorer.py` - from morgan/memory/memory_processor.py
- [ ] `learning_engine.py` - from morgan/learning/*

#### Day 6-7: Create Repository Interfaces

These are abstract interfaces (ports):

```python
# domain/repositories/conversation.py
from abc import ABC, abstractmethod
from domain.entities.conversation import Conversation

class IConversationRepository(ABC):
    """Port - interface for conversation storage"""

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        pass

    @abstractmethod
    async def get(self, conversation_id: str) -> Conversation:
        pass
```

Create interfaces for:
- [ ] IConversationRepository
- [ ] IUserRepository
- [ ] IKnowledgeRepository
- [ ] IMemoryRepository
- [ ] IVectorStoreRepository

---

### Week 2: Application & Infrastructure Layers

#### Day 8-9: Create Use Cases

Use cases orchestrate domain logic:

```python
# application/use_cases/conversation/process_query.py
from domain.entities.conversation import Conversation
from domain.services.emotion_analyzer import EmotionAnalyzer
from application.ports.llm_service import ILLMService
from application.dto.query_request import QueryRequest
from application.dto.query_response import QueryResponse

class ProcessQueryUseCase:
    """Orchestrate query processing"""

    def __init__(
        self,
        emotion_analyzer: EmotionAnalyzer,
        llm_service: ILLMService,
        conversation_repo: IConversationRepository
    ):
        self._emotion_analyzer = emotion_analyzer
        self._llm_service = llm_service
        self._conversation_repo = conversation_repo

    async def execute(self, request: QueryRequest) -> QueryResponse:
        # 1. Load conversation
        conversation = await self._conversation_repo.get(request.conversation_id)

        # 2. Analyze emotion (domain service)
        emotion = self._emotion_analyzer.analyze_text(request.query)

        # 3. Generate response (infrastructure service via port)
        answer = await self._llm_service.generate(
            query=request.query,
            emotion=emotion,
            context=conversation.get_context()
        )

        # 4. Save conversation
        # ...

        # 5. Return response
        return QueryResponse(answer=answer, emotion=emotion)
```

Create use cases for:
- [ ] ProcessQueryUseCase
- [ ] IngestDocumentsUseCase
- [ ] DetectEmotionUseCase
- [ ] BuildProfileUseCase
- [ ] ExtractPreferencesUseCase

#### Day 10-11: Implement Infrastructure Adapters

Now implement the concrete classes:

```python
# infrastructure/ai_services/openai_compatible/llm_adapter.py
from application.ports.llm_service import ILLMService
from openai import AsyncOpenAI

class OpenAICompatibleLLMService(ILLMService):
    """Adapter implementing the port"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate(self, query: str, emotion, context: str) -> str:
        # Migrate from: morgan/services/llm_service.py
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
```

Implement adapters for:
- [ ] OpenAICompatibleLLMService
- [ ] JinaEmbeddingService
- [ ] QdrantVectorStoreRepository
- [ ] RedisCache
- [ ] HierarchicalSearchEngine

#### Day 12-13: Setup Dependency Injection

```python
# di/container.py
from dependency_injector import containers, providers
from domain.services.emotion_analyzer import EmotionAnalyzer
from infrastructure.ai_services.openai_compatible.llm_adapter import OpenAICompatibleLLMService
from application.use_cases.conversation.process_query import ProcessQueryUseCase

class Container(containers.DeclarativeContainer):
    """Central DI container"""

    # Config
    config = providers.Configuration()

    # Infrastructure
    llm_service = providers.Singleton(
        OpenAICompatibleLLMService,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        model=config.llm.model
    )

    # Domain Services
    emotion_analyzer = providers.Factory(EmotionAnalyzer)

    # Use Cases
    process_query = providers.Factory(
        ProcessQueryUseCase,
        emotion_analyzer=emotion_analyzer,
        llm_service=llm_service
    )
```

#### Day 14: Create Interfaces (CLI/API)

```python
# interfaces/cli/main.py
from morgan_v2 import create_assistant
import asyncio

async def main():
    container = await create_assistant()
    process_query = container.process_query()

    while True:
        query = input("You: ")
        if query == "exit":
            break

        response = await process_query.execute(QueryRequest(query=query))
        print(f"Morgan: {response.answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Week 3: Testing & Finalization

#### Day 15-17: Update Tests
Move tests to new structure, update imports.

#### Day 18-19: Integration Testing
Test end-to-end flows.

#### Day 20: Documentation
Update README, add architecture diagrams.

#### Day 21: Cleanup & Deploy
- Remove old code
- Rename morgan_v2 to morgan
- Deploy

---

## Quick Decision Matrix

**If you want:**
- ✅ **Fastest (2-4 hours)**: Run automation script, review, test
- ✅ **Most learning (1-2 weeks)**: Manual step-by-step
- ✅ **Balanced (3-5 days)**: Script generates structure, you migrate business logic

**My recommendation for you**: **Hybrid approach** (Option C)

Why? Because:
1. You have working code - don't throw it away
2. Script can generate boilerplate quickly
3. You manually migrate important business logic (emotion detection, learning)
4. You maintain full control over critical parts
5. Fastest path while keeping quality high

---

## Next Action

**To proceed, tell me which option you prefer:**

1. **"Run the automation script"** - I'll help you execute and review
2. **"Manual step-by-step"** - I'll guide you through each file
3. **"Hybrid approach"** - I'll run script then help you migrate business logic

Then we can start immediately!

What's your choice?
