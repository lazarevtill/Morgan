# Emotion Detection System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EmotionDetector                             │
│                   (Main Orchestrator)                            │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Detection Pipeline (<200ms)                  │  │
│  │                                                            │  │
│  │  Stage 1: Classification                                  │  │
│  │  ────────────────────────                                 │  │
│  │  EmotionClassifier ──> [Emotion, Emotion, ...]            │  │
│  │                                                            │  │
│  │  Stage 2: Intensity Analysis                              │  │
│  │  ────────────────────────────                             │  │
│  │  IntensityAnalyzer ──> [Adjusted Emotions]                │  │
│  │                                                            │  │
│  │  Stage 3: Context Analysis                                │  │
│  │  ──────────────────────                                   │  │
│  │  ContextAnalyzer ──> [Context-Adjusted Emotions]          │  │
│  │                                                            │  │
│  │  Stage 4: Parallel Analysis                               │  │
│  │  ───────────────────────────                              │  │
│  │  ┌─ TriggerDetector ──> [Triggers]                        │  │
│  │  ├─ PatternDetector ──> [Patterns]                        │  │
│  │  └─ MultiEmotionDetector ──> (dominant, valence, arousal) │  │
│  │                                                            │  │
│  │  Stage 5: Aggregation                                     │  │
│  │  ─────────────────────                                    │  │
│  │  EmotionAggregator ──> EmotionResult                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Supporting Modules:                                             │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ EmotionCache     │  │ CircuitBreaker   │                     │
│  │ (Performance)    │  │ (Resilience)     │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │HistoryTracker    │  │TemporalAnalyzer  │                     │
│  │ (Persistence)    │  │ (Trends)         │                     │
│  └──────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Architecture

```
morgan/emotions/
│
├── Core Components
│   ├── types.py              ← Domain models (Emotion, EmotionResult, etc.)
│   ├── exceptions.py         ← Exception hierarchy
│   ├── base.py              ← Base classes (EmotionModule, AsyncCache, CircuitBreaker)
│   ├── detector.py          ← Main orchestrator
│   └── utils.py             ← Helper functions
│
├── 11 Specialized Modules
│   ├── classifier.py         ← Module 1: Emotion classification
│   ├── intensity.py          ← Module 2: Intensity analysis
│   ├── pattern_detector.py  ← Module 3: Pattern detection
│   ├── trigger_detector.py  ← Module 4: Trigger detection
│   ├── history_tracker.py   ← Module 5: History tracking
│   ├── context_analyzer.py  ← Module 6: Context analysis
│   ├── multi_emotion.py     ← Module 7: Multi-emotion handling
│   ├── temporal_analyzer.py ← Module 8: Temporal analysis
│   ├── cache.py             ← Module 9: Result caching
│   └── aggregator.py        ← Module 10: Result aggregation
│                               (Module 11 is detector.py)
│
└── Documentation
    ├── README.md            ← Comprehensive documentation
    └── ARCHITECTURE.md      ← This file
```

## Data Flow

```
Input Text
    │
    ├──> [Cache Check] ──> Hit? ──> Return cached result
    │                          │
    │                          No
    ▼                          ▼
[Validate Input]          Continue
    │
    ▼
[Stage 1: Classification]
EmotionClassifier
    │
    ├─ Keyword matching (95+ keywords/emotion)
    ├─ Pattern matching (32+ regex patterns)
    ├─ Negation handling (20+ negations)
    └─ Intensity modifiers (24+ modifiers)
    │
    ▼ List[Emotion]
[Stage 2: Intensity Analysis]
IntensityAnalyzer
    │
    ├─ Text markers (CAPS, punctuation, repetition)
    ├─ Context factors (timing, frequency)
    ├─ Emotion interactions (amplify/suppress)
    └─ Confidence calculation
    │
    ▼ List[Emotion] (adjusted)
[Stage 3: Context Analysis]
ContextAnalyzer (if context provided)
    │
    ├─ Emotional shifts
    ├─ Continuity effects
    └─ Implicit emotions
    │
    ▼ List[Emotion] (context-adjusted)
[Stage 4: Parallel Analysis]
    │
    ├─────────────┬─────────────┬──────────────┐
    ▼             ▼             ▼              ▼
TriggerDetector PatternDetector MultiEmotionDetector
    │             │             │
    │             │             ├─ Find dominant
    │             │             ├─ Calculate valence
    │             │             └─ Calculate arousal
    │             │
    ▼             ▼             ▼
List[Trigger] List[Pattern] (dominant, valence, arousal)
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
[Stage 5: Aggregation]
EmotionAggregator
    │
    ├─ Combine all results
    ├─ Detect conflicts
    ├─ Add warnings
    └─ Create EmotionResult
    │
    ▼
EmotionResult
    │
    ├──> [Update History] (if enabled)
    ├──> [Update Temporal] (if enabled)
    └──> [Cache Result] (if enabled)
    │
    ▼
Return to caller
```

## Async Execution Flow

```python
async def detect(text: str) -> EmotionResult:
    # Stage 1: Sequential (depends on text)
    emotions = await classifier.classify(text)

    # Stage 2: Sequential (depends on emotions)
    emotions = await intensity_analyzer.analyze(emotions, text)

    # Stage 3: Sequential (depends on adjusted emotions)
    emotions = await context_analyzer.analyze(emotions, context)

    # Stage 4: PARALLEL (independent operations)
    triggers, patterns, (dominant, valence, arousal) = await asyncio.gather(
        trigger_detector.detect_triggers(text, emotions),
        pattern_detector.detect_patterns(emotions, context),
        multi_emotion_detector.analyze_multi_emotions(emotions),
    )

    # Stage 5: Sequential (depends on all results)
    result = await aggregator.aggregate(
        emotions, dominant, valence, arousal,
        triggers, patterns, context
    )

    return result
```

## Module Dependencies

```
EmotionDetector
    │
    ├─ depends on ──> EmotionClassifier
    ├─ depends on ──> IntensityAnalyzer
    │                     └─ uses ──> EmotionContext
    ├─ depends on ──> ContextAnalyzer
    │                     └─ uses ──> EmotionContext
    ├─ depends on ──> TriggerDetector
    ├─ depends on ──> PatternDetector
    │                     └─ uses ──> EmotionContext
    ├─ depends on ──> MultiEmotionDetector
    ├─ depends on ──> EmotionAggregator
    ├─ optionally ──> EmotionCache
    ├─ optionally ──> EmotionHistoryTracker
    └─ optionally ──> TemporalAnalyzer

All modules inherit from EmotionModule (base.py)
All modules use types from types.py
All modules raise exceptions from exceptions.py
```

## State Management

```
┌─────────────────────────────────────────────────────────┐
│                  EmotionDetector                         │
│                                                          │
│  Instance State:                                         │
│  ├─ _initialized: bool                                  │
│  ├─ _circuit_breaker: CircuitBreaker                    │
│  ├─ _semaphore: asyncio.Semaphore                       │
│  ├─ _total_detections: int                              │
│  └─ _cache_hits: int                                    │
└─────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
┌─────────────────┐                    ┌──────────────────┐
│  EmotionCache   │                    │HistoryTracker    │
│                 │                    │                  │
│  State:         │                    │  State:          │
│  └─ cache: dict │                    │  └─ history: dict│
└─────────────────┘                    └──────────────────┘
        │                                         │
        └─────────────────┬───────────────────────┘
                          ▼
              ┌────────────────────┐
              │TemporalAnalyzer    │
              │                    │
              │  State:            │
              │  └─ timelines: dict│
              └────────────────────┘
```

## Error Handling Flow

```
User calls detect(text)
    │
    ▼
[Validate Input] ──> Invalid? ──> Raise EmotionValidationError
    │
    ▼ Valid
[Circuit Breaker Check]
    │
    ├─ Open? ──> Raise EmotionDetectionError (recoverable)
    │
    ▼ Closed/Half-Open
[Acquire Semaphore] ──> Timeout? ──> Raise EmotionResourceError
    │
    ▼ Acquired
[Run Pipeline]
    │
    ├─ Classification fails ──> Raise EmotionClassificationError
    │                               │
    │                               └──> Caught by circuit breaker
    │                                        │
    │                                        └──> Increment failure count
    │
    ├─ Analysis fails ──> Raise EmotionAnalysisError
    │                          │
    │                          └──> Caught by circuit breaker
    │
    └─ Cache fails ──> Log warning, continue (graceful degradation)
    │
    ▼
[Return Result]
    │
    └──> Circuit breaker records success
```

## Performance Optimization Strategies

### 1. Caching (EmotionCache)
```
First Request:  Text → [Full Pipeline 50-150ms] → Result → Cache
Second Request: Text → [Cache Lookup <5ms] → Result
```

### 2. Parallel Processing
```
Sequential:  T1 + T2 + T3 = 150ms
Parallel:    max(T1, T2, T3) = 60ms
```

### 3. Early Returns
```python
if not text:
    return empty_result  # <1ms

if len(emotions) == 0:
    return neutral_result  # <5ms

if cached:
    return cached  # <5ms
```

### 4. Concurrency Control
```
Semaphore(5) limits parallel operations:
- Prevents resource exhaustion
- Maintains predictable latency
- Protects downstream services
```

### 5. Circuit Breaker
```
Normal: All requests processed
Failures: 1, 2, 3, 4, 5 consecutive failures
Circuit Opens: Fast-fail for 60 seconds
Half-Open: Attempt recovery
Success: Circuit closes
```

## Integration Patterns

### With Empathy Engine
```python
# Empathy engine consumes emotion results
emotion_result = await emotion_detector.detect(user_message)

empathy_response = await empathy_engine.generate_response(
    emotion=emotion_result.dominant_emotion,
    valence=emotion_result.valence,
    triggers=emotion_result.triggers,
    is_crisis=emotion_result.is_crisis,
)
```

### With Learning System
```python
# Learning system tracks patterns
emotion_result = await emotion_detector.detect(user_message)

await learning_system.update_profile(
    user_id=user_id,
    emotional_state=emotion_result,
    patterns=emotion_result.patterns,
)

# Get trajectory for personalization
trajectory = await emotion_detector.get_user_trajectory(user_id)
```

### With RAG System
```python
# RAG filters content by emotional context
emotion_result = await emotion_detector.detect(query)

docs = await rag_system.search(
    query=query,
    emotion_filter=emotion_result.dominant_emotion.emotion_type,
    valence_range=(emotion_result.valence - 0.2, emotion_result.valence + 0.2),
)
```

## Scalability Considerations

### Horizontal Scaling
```
Load Balancer
    │
    ├─ Instance 1 (EmotionDetector)
    ├─ Instance 2 (EmotionDetector)
    └─ Instance 3 (EmotionDetector)
         │
         └─> Shared Storage
              ├─ Redis (EmotionCache)
              └─ PostgreSQL (EmotionHistory)
```

### Resource Limits
- **Memory**: ~100MB per instance
- **CPU**: 1-2 cores per instance
- **Connections**: Semaphore(5-10)
- **Cache Size**: 10,000 entries (~50MB)
- **History**: 1,000 users (~50MB)

## Production Deployment

```yaml
# Docker/Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emotion-detector
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: emotion-detector
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        env:
        - name: EMOTION_CACHE_ENABLED
          value: "true"
        - name: EMOTION_HISTORY_PATH
          value: "/data/emotions"
```

## Monitoring & Observability

```python
# Metrics to track
stats = detector.stats

# Response time
avg_time = stats["average_processing_time_ms"]  # Target: <200ms
assert avg_time < 200, "Performance degradation"

# Cache efficiency
cache_hit_rate = stats["cache_hit_rate"]  # Target: >50%
assert cache_hit_rate > 0.5, "Cache underutilized"

# Circuit breaker state
circuit_state = stats["circuit_breaker_state"]  # Should be: "closed"
assert circuit_state == "closed", "Circuit open - investigate failures"

# Throughput
total_detections = stats["total_detections"]
```

## Testing Strategy

```
Unit Tests (per module)
    ├─ EmotionClassifier
    ├─ IntensityAnalyzer
    ├─ PatternDetector
    └─ ... (all 11 modules)

Integration Tests
    ├─ Full pipeline
    ├─ Context awareness
    ├─ Pattern detection
    └─ Crisis detection

Performance Tests
    ├─ Response time <200ms
    ├─ Cache performance
    └─ Concurrent requests

End-to-End Tests
    └─ Real-world scenarios
```

## Summary

The emotion detection system is architected for:

✅ **Performance**: <200ms response time via caching and parallel processing
✅ **Reliability**: Circuit breaker, graceful degradation, proper error handling
✅ **Scalability**: Stateless design, horizontal scaling, resource limits
✅ **Maintainability**: Clean architecture, DI, comprehensive tests
✅ **Observability**: Stats tracking, logging, monitoring hooks
✅ **Production-Ready**: No placeholders, real implementations, proper async patterns
