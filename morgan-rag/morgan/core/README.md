# Morgan Core Assistant System

Production-quality core assistant and memory systems with full async/await architecture, emotion integration, learning capabilities, and RAG-enhanced responses.

## Overview

The Morgan core system is the flagship component that demonstrates how all refactored systems work together seamlessly:

- **MorganAssistant**: Main orchestrator coordinating all subsystems
- **MemorySystem**: Multi-layer memory management (short-term, working, long-term, consolidated)
- **ContextManager**: Intelligent context handling with pruning strategies
- **ResponseGenerator**: LLM-powered response generation with streaming
- **MultiStageSearch**: Production RAG with hierarchical search and reranking

## Architecture

```
User Message
    ↓
[MorganAssistant.process_message]
    ↓
Phase 1: Gather Context (Parallel)
    ├─ Emotion Detection → EmotionDetector
    ├─ Memory Retrieval → MemorySystem
    └─ User Profile → MemorySystem
    ↓
Phase 2: RAG Search
    └─ Hierarchical Search → MultiStageSearch
    ↓
Phase 3: Build Context
    └─ Context Assembly → ContextManager
    ↓
Phase 4: Generate Response
    └─ LLM Generation → ResponseGenerator
    ↓
Phase 5: Update Systems (Background)
    ├─ Store Messages → MemorySystem
    ├─ Update Emotion State → MemorySystem
    └─ Learning Update → LearningEngine
    ↓
Assistant Response
```

## Components

### 1. MorganAssistant

Main orchestrator that coordinates all subsystems.

**Key Features:**
- Full async/await architecture
- Parallel processing where possible
- Circuit breakers for resilience
- Graceful degradation
- Performance tracking
- < 2s response latency target (P95)

**Usage:**
```python
from morgan.core import MorganAssistant

assistant = MorganAssistant(
    storage_path=Path.home() / ".morgan",
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",
    enable_emotion_detection=True,
    enable_learning=True,
    enable_rag=True,
)

await assistant.initialize()

response = await assistant.process_message(
    user_id="user_001",
    message="Hello! How are you?",
    session_id="session_001",
)

print(response.content)
await assistant.cleanup()
```

### 2. MemorySystem

Multi-layer memory management with automatic cleanup and consolidation.

**Memory Layers:**
1. **Short-term**: Current conversation (fast, in-memory)
2. **Working**: Processing buffer (Redis/in-memory)
3. **Long-term**: Historical conversations (persistent)
4. **Consolidated**: Important patterns (background processed)

**Key Features:**
- Fast retrieval (< 100ms target)
- Automatic cleanup of expired memories
- Background consolidation
- Importance-based storage
- Session management

**Usage:**
```python
from morgan.core import MemorySystem, Message, MessageRole

memory = MemorySystem(storage_path=Path("./memory"))
await memory.initialize()

# Store message
message = Message(
    role=MessageRole.USER,
    content="Hello!",
    timestamp=datetime.now(),
    message_id="msg_001",
)

await memory.store_message("session_001", message)

# Retrieve context
messages = await memory.retrieve_context("session_001", n_messages=10)

# Search memories
results = await memory.search_memories("user_001", "machine learning")

await memory.cleanup()
```

### 3. ContextManager

Intelligent context handling with multiple pruning strategies.

**Pruning Strategies:**
- `SLIDING_WINDOW`: Keep most recent messages
- `IMPORTANCE_BASED`: Keep highest importance scores
- `RECENCY_WEIGHTED`: Balance recency and importance
- `HYBRID`: Combination (default)

**Key Features:**
- Token counting and limits
- Importance-based scoring
- Context compression
- Fast operation (< 50ms target)

**Usage:**
```python
from morgan.core import ContextManager, ContextPruningStrategy

context_mgr = ContextManager(
    max_context_tokens=8000,
    target_context_tokens=6000,
    default_pruning_strategy=ContextPruningStrategy.HYBRID,
)

# Build context
context = await context_mgr.build_context(
    messages=messages,
    user_id="user_001",
    session_id="session_001",
)

# Manual pruning
pruned = await context_mgr.prune_context(
    messages=messages,
    target_tokens=4000,
    strategy=ContextPruningStrategy.IMPORTANCE_BASED,
)
```

### 4. ResponseGenerator

LLM-powered response generation with retry logic and streaming.

**Key Features:**
- Emotion-aware prompting
- RAG-enhanced responses with citations
- Retry logic with exponential backoff
- Streaming support
- Response validation
- Performance tracking

**Usage:**
```python
from morgan.core import ResponseGenerator

generator = ResponseGenerator(
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",
    temperature=0.7,
)

# Standard generation
response = await generator.generate(
    context=conversation_context,
    user_message="What is machine learning?",
    rag_results=search_results,
    detected_emotion=emotion_result,
)

# Streaming
async for chunk in generator.generate_stream(
    context=conversation_context,
    user_message="Tell me a story.",
):
    print(chunk, end="", flush=True)

await generator.cleanup()
```

### 5. MultiStageSearch

Production-quality RAG with hierarchical search and reranking.

**Key Features:**
- Hierarchical search (coarse → medium → fine)
- Reciprocal Rank Fusion for result merging
- Optional cross-encoder reranking
- Async concurrent search
- Circuit breaker for resilience
- Performance metrics

**Usage:**
```python
from morgan.core import MultiStageSearch, SearchConfig

search = MultiStageSearch(
    vector_db=qdrant_client,
    embedding_service=embedding_service,
    reranking_service=reranking_service,
    config=SearchConfig(
        coarse_top_k=50,
        medium_top_k=30,
        fine_top_k=20,
        final_top_k=10,
        enable_reranking=True,
    ),
)

results, metrics = await search.search(
    query="What is quantum computing?",
    top_k=5,
)

for result in results:
    print(f"[{result.rank}] {result.content} (score: {result.score:.3f})")
```

## Types

### Core Types

```python
from morgan.core.types import (
    # Messages
    Message,
    MessageRole,

    # Context
    ConversationContext,
    EmotionalState,
    UserProfile,

    # Response
    AssistantResponse,
    SearchSource,

    # Memory
    MemoryEntry,
    MemoryType,

    # Metrics
    AssistantMetrics,
    ProcessingContext,
)
```

### Message
```python
@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    timestamp: datetime
    message_id: str
    emotion: Optional[EmotionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None
    importance_score: float = 1.0
```

### ConversationContext
```python
@dataclass(frozen=True)
class ConversationContext:
    messages: List[Message]
    user_id: str
    session_id: str
    user_profile: Optional[UserProfile] = None
    emotional_state: Optional[EmotionalState] = None
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### AssistantResponse
```python
@dataclass(frozen=True)
class AssistantResponse:
    content: str
    response_id: str
    timestamp: datetime
    sources: List[SearchSource] = field(default_factory=list)
    emotion: Optional[EmotionResult] = None
    confidence: float = 1.0
    generation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Error Handling

The system uses a comprehensive exception hierarchy:

```python
from morgan.core import (
    AssistantError,
    MemoryError,
    MemoryRetrievalError,
    MemoryStorageError,
    ContextError,
    ContextOverflowError,
    GenerationError,
    ValidationError,
)
```

All errors include:
- Correlation IDs for tracing
- Recoverable flag
- Detailed error messages

## Performance Targets

| Metric | Target | Component |
|--------|--------|-----------|
| Total latency | < 2s (P95) | MorganAssistant |
| Memory retrieval | < 100ms | MemorySystem |
| Context building | < 50ms | ContextManager |
| Emotion detection | < 200ms | EmotionDetector |
| RAG search | < 500ms | MultiStageSearch |

## Integration with Refactored Systems

### Emotion System Integration

```python
# Automatic emotion detection
response = await assistant.process_message(
    user_id="user_001",
    message="I'm feeling great today!",
    session_id="session_001",
)

if response.emotion and response.emotion.primary_emotion:
    print(f"Detected: {response.emotion.primary_emotion.emotion_type}")
```

### Learning System Integration

```python
# Automatic learning from interactions
# Learning updates happen in background after each message
response = await assistant.process_message(
    user_id="user_001",
    message="I prefer detailed explanations.",
    session_id="session_001",
)

# Learning system will:
# 1. Detect pattern (preference for detail)
# 2. Update user profile
# 3. Adapt future responses
```

### RAG Integration

```python
# Automatic RAG when vector DB is available
assistant = MorganAssistant(
    vector_db=qdrant_client,
    embedding_service=embedding_service,
    reranking_service=reranking_service,
    enable_rag=True,
)

response = await assistant.process_message(
    user_id="user_001",
    message="What is the capital of France?",
    session_id="session_001",
)

# Response includes sources
for source in response.sources:
    print(f"Source: {source.source} (score: {source.score})")
```

## Circuit Breakers

The system includes circuit breakers for resilience:

```python
# Circuit breaker per component
stats = assistant.get_stats()

for component, breaker in stats["circuit_breakers"].items():
    if breaker["is_open"]:
        print(f"{component} circuit breaker is OPEN")
        print(f"Failures: {breaker['failure_count']}")
```

When a circuit breaker opens:
1. Component enters degraded mode
2. Requests continue without that component
3. Circuit resets after timeout
4. System continues to function

## Monitoring & Metrics

```python
# Get comprehensive stats
stats = assistant.get_stats()

print(f"Total requests: {stats['metrics']['total_requests']}")
print(f"Success rate: {stats['metrics']['successful_requests'] / stats['metrics']['total_requests']:.2%}")

# Per-component metrics
print(f"\nMemory:")
print(f"  Sessions: {stats['memory']['short_term_sessions']}")
print(f"  Messages: {stats['memory']['short_term_messages']}")

print(f"\nContext:")
print(f"  Built: {stats['context']['metrics']['contexts_built']}")
print(f"  Pruned: {stats['context']['metrics']['contexts_pruned']}")

print(f"\nResponse Generation:")
print(f"  Total: {stats['response_generator']['metrics']['total_generations']}")
print(f"  Success: {stats['response_generator']['metrics']['successful_generations']}")
```

## Examples

See `/examples/core_assistant_example.py` for comprehensive examples including:

1. Simple message processing
2. Context-aware conversations
3. Emotional message handling
4. Streaming responses
5. Performance testing
6. Direct component usage

## Best Practices

### 1. Always Initialize and Cleanup

```python
assistant = MorganAssistant(...)
await assistant.initialize()

try:
    # Use assistant
    pass
finally:
    await assistant.cleanup()
```

### 2. Use Context Managers for Components

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_assistant():
    assistant = MorganAssistant(...)
    await assistant.initialize()
    try:
        yield assistant
    finally:
        await assistant.cleanup()

async with get_assistant() as assistant:
    response = await assistant.process_message(...)
```

### 3. Handle Errors Gracefully

```python
try:
    response = await assistant.process_message(...)
except AssistantError as e:
    if e.recoverable:
        # Retry or use fallback
        response = await assistant.process_message(...)
    else:
        # Log and handle
        logger.error(f"Unrecoverable error: {e}")
```

### 4. Monitor Performance

```python
response = await assistant.process_message(...)

if response.metadata.get("metrics"):
    metrics = response.metadata["metrics"]
    if metrics["total_duration_ms"] > 2000:
        logger.warning("Response time exceeded target")
```

### 5. Use Streaming for Long Responses

```python
# For better UX on long responses
async for chunk in assistant.stream_response(...):
    print(chunk, end="", flush=True)
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
MORGAN_LLM_URL=http://localhost:11434
MORGAN_LLM_MODEL=llama3.2:latest

# Vector DB
MORGAN_QDRANT_HOST=localhost
MORGAN_QDRANT_PORT=6333

# Storage
MORGAN_STORAGE_PATH=~/.morgan

# Feature Flags
MORGAN_ENABLE_EMOTION=true
MORGAN_ENABLE_LEARNING=true
MORGAN_ENABLE_RAG=true
```

### Programmatic Configuration

```python
assistant = MorganAssistant(
    storage_path=Path.home() / ".morgan",

    # LLM
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",

    # Services
    vector_db=qdrant_client,
    embedding_service=embedding_service,
    reranking_service=reranking_service,

    # Features
    enable_emotion_detection=True,
    enable_learning=True,
    enable_rag=True,

    # Performance
    max_concurrent_operations=10,
)
```

## Testing

Run tests:
```bash
pytest morgan/core/tests/ -v
```

Run performance tests:
```bash
pytest morgan/core/tests/test_performance.py -v
```

## Contributing

When extending the core system:

1. Maintain async/await architecture
2. Add proper error handling with correlation IDs
3. Include performance metrics
4. Write comprehensive tests
5. Update documentation
6. Follow existing patterns

## License

See main project LICENSE file.
