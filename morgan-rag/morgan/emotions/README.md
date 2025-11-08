# Morgan Emotion Detection System

A production-quality emotion detection system with 11 specialized modules for comprehensive emotional analysis.

## Overview

The emotion detection system analyzes text to identify emotional content, intensity, patterns, and triggers. It achieves <200ms response time while providing deep emotional intelligence.

## Architecture

### 11 Specialized Modules

1. **EmotionClassifier** - Classifies text into 8 basic emotion types using keyword matching, pattern recognition, and negation handling
2. **IntensityAnalyzer** - Analyzes and adjusts emotion intensities based on linguistic markers, context, and emotion interactions
3. **PatternDetector** - Detects emotional patterns over time (recurring, escalating, alternating, suppressed)
4. **TriggerDetector** - Identifies specific words, phrases, or patterns that trigger emotional responses
5. **EmotionHistoryTracker** - Maintains persistent emotional history for users with configurable retention
6. **ContextAnalyzer** - Analyzes conversational context to refine emotion detection accuracy
7. **MultiEmotionDetector** - Handles multiple simultaneous emotions and calculates valence/arousal
8. **TemporalAnalyzer** - Analyzes emotional changes over time to detect trends and trajectories
9. **EmotionCache** - Provides caching for performance optimization with content-based hashing
10. **EmotionAggregator** - Aggregates results from all modules into coherent final output
11. **EmotionDetector** - Main orchestration with async pipeline, circuit breaker, and error handling

### Core Design Principles

- **Async-first**: All operations are async for maximum performance
- **Dependency Injection**: Modules are loosely coupled and independently testable
- **Clean Architecture**: Clear separation of concerns with well-defined interfaces
- **Error Resilience**: Circuit breaker pattern prevents cascading failures
- **Resource Management**: Proper lifecycle management with initialize/cleanup
- **Performance Optimized**: Caching, parallel processing, <200ms target

## Eight Basic Emotions (Plutchik Model)

1. **Joy** - Happiness, pleasure, delight
2. **Sadness** - Grief, sorrow, melancholy
3. **Anger** - Frustration, rage, irritation
4. **Fear** - Anxiety, worry, terror
5. **Surprise** - Shock, amazement, astonishment
6. **Disgust** - Revulsion, distaste, aversion
7. **Trust** - Confidence, faith, belief
8. **Anticipation** - Expectation, hope, eagerness

## Usage

### Basic Usage

```python
from morgan.emotions import EmotionDetector, EmotionContext

# Initialize detector
detector = EmotionDetector(
    enable_cache=True,
    enable_history=True,
)

await detector.initialize()

try:
    # Detect emotions in text
    result = await detector.detect(
        "I'm so happy today! This is wonderful!",
        context=EmotionContext(user_id="user123"),
    )

    # Access results
    print(f"Dominant emotion: {result.dominant_emotion.emotion_type}")
    print(f"Intensity: {result.dominant_emotion.intensity}")
    print(f"Valence: {result.valence}")  # -1 (negative) to +1 (positive)
    print(f"Arousal: {result.arousal}")  # 0 (calm) to 1 (excited)
    print(f"Processing time: {result.processing_time_ms}ms")

    # Check for crisis
    if result.is_crisis:
        print("⚠️  Crisis state detected!")

finally:
    await detector.cleanup()
```

### With Context and History

```python
from morgan.emotions import EmotionDetector, EmotionContext

detector = EmotionDetector(enable_history=True)
await detector.initialize()

try:
    # Create context
    context = EmotionContext(
        user_id="user123",
        conversation_id="conv456",
        message_index=5,
    )

    # Detect with context
    result = await detector.detect(
        "Still feeling down about everything.",
        context=context,
    )

    # Get user's emotional history
    history = await detector.get_user_history("user123", limit=10)

    # Get emotional trajectory
    trajectory = await detector.get_user_trajectory("user123")
    print(f"Direction: {trajectory['direction']}")  # improving/worsening/stable
    print(f"Volatility: {trajectory['volatility']}")

finally:
    await detector.cleanup()
```

### Using Utilities

```python
from morgan.emotions import (
    format_emotion_summary,
    format_colored_output,
    get_emotion_emoji,
    is_crisis_state,
)

result = await detector.detect("I'm terrified!")

# Format for display
summary = format_emotion_summary(result)
colored = format_colored_output(result)  # ANSI colors for terminal

# Get emoji
emoji = get_emotion_emoji(result.dominant_emotion.emotion_type)

# Check crisis
if is_crisis_state(result):
    print("🚨 Crisis detected!")
```

### Advanced Features

```python
# Parallel detection
results = await asyncio.gather(
    detector.detect("I'm happy!"),
    detector.detect("This is terrible."),
    detector.detect("I'm worried about this."),
)

# Get statistics
stats = detector.stats
print(f"Total detections: {stats['total_detections']}")
print(f"Average time: {stats['average_processing_time_ms']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

# Clear user data (GDPR compliance)
await detector.clear_user_data("user123")
```

## Data Structures

### EmotionResult

```python
@dataclass
class EmotionResult:
    primary_emotions: List[Emotion]  # Sorted by significance
    dominant_emotion: Optional[Emotion]  # Highest intensity * confidence
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    triggers: List[EmotionTrigger]  # Detected triggers
    patterns: List[EmotionPattern]  # Detected patterns
    context: Optional[EmotionContext]  # Context used
    timestamp: datetime  # Detection time
    processing_time_ms: float  # Processing duration
    warnings: List[str]  # Any warnings
```

### Emotion

```python
@dataclass(frozen=True)
class Emotion:
    emotion_type: EmotionType  # One of 8 basic emotions
    intensity: EmotionIntensity  # 0-1 scale
    confidence: float  # 0-1, detection confidence
```

### EmotionContext

```python
@dataclass
class EmotionContext:
    user_id: Optional[str]  # User identifier
    conversation_id: Optional[str]  # Conversation identifier
    message_index: int  # Position in conversation
    previous_emotions: List[Emotion]  # Recent emotional history
    time_since_last_message: Optional[float]  # Seconds
    conversation_topic: Optional[str]  # Topic being discussed
```

## Performance

### Targets

- **Response Time**: <200ms per message (P95)
- **Throughput**: 50-100 detections/second (with caching)
- **Memory**: <100MB for typical usage
- **Accuracy**: 70-85% emotion classification accuracy

### Optimizations

1. **Caching**: Content-based caching with SHA256 keys
2. **Parallel Processing**: Concurrent analysis of triggers, patterns, emotions
3. **Early Returns**: Skip expensive operations for simple cases
4. **Concurrency Control**: Semaphore limits parallel operations
5. **Circuit Breaker**: Fast-fail during outages

## Error Handling

### Exception Hierarchy

```
EmotionDetectionError (base)
├── EmotionClassificationError - Classification failures
├── EmotionAnalysisError - Analysis failures
├── EmotionValidationError - Input validation failures
├── EmotionCacheError - Cache operation failures
├── EmotionHistoryError - History tracking failures
├── EmotionContextError - Context management failures
└── EmotionResourceError - Resource loading failures
```

### Circuit Breaker

The detector includes a circuit breaker that:
- Opens after 5 consecutive failures
- Stays open for 60 seconds
- Attempts recovery in half-open state
- Prevents cascading failures

## Testing

Run comprehensive tests:

```bash
# Run all tests
pytest tests/test_emotion_detection.py -v

# Run specific test
pytest tests/test_emotion_detection.py::test_basic_emotion_detection -v

# Run with coverage
pytest tests/test_emotion_detection.py --cov=morgan.emotions --cov-report=html
```

## Module Details

### 1. EmotionClassifier

**Purpose**: Classify text into emotion types

**Strategies**:
- Keyword-based lexicon matching
- Regex pattern recognition
- Negation handling ("not happy" → sadness)
- Intensity modifiers (very, slightly, etc.)

**Example**:
```python
classifier = EmotionClassifier()
await classifier.initialize()
emotions = await classifier.classify("I'm so happy!")
# Returns: [Emotion(JOY, 0.8, 0.8)]
```

### 2. IntensityAnalyzer

**Purpose**: Analyze and adjust emotion intensities

**Factors**:
- Text markers (CAPS, punctuation, repetition)
- Contextual factors (timing, history)
- Emotion interactions (fear + sadness amplify)
- User baseline

**Example**:
```python
analyzer = IntensityAnalyzer()
await analyzer.initialize()
adjusted = await analyzer.analyze_intensity(emotions, text, context)
```

### 3. PatternDetector

**Purpose**: Detect emotional patterns over time

**Pattern Types**:
- **Recurring**: Same emotion repeatedly
- **Escalating**: Intensifying emotions
- **Alternating**: Flip-flopping between opposites
- **Suppressed**: Hidden emotions

**Example**:
```python
detector = PatternDetector()
patterns = await detector.detect_patterns(emotions, context)
# Returns: [EmotionPattern(recurring, [SADNESS], ...)]
```

### 4. TriggerDetector

**Purpose**: Identify emotional triggers

**Detection**:
- Keyword triggers (death, birthday, fired)
- Pattern triggers (regex-based)
- Contextual triggers

**Example**:
```python
detector = TriggerDetector()
triggers = await detector.detect_triggers(text, emotions)
# Returns: [EmotionTrigger("died", keyword, [SADNESS], 0.8, 15)]
```

### 5. EmotionHistoryTracker

**Purpose**: Maintain emotional history

**Features**:
- Persistent storage (optional)
- Configurable retention (default: 90 days)
- Dominant emotion tracking
- Baseline state calculation

**Example**:
```python
tracker = EmotionHistoryTracker(storage_path=Path("data/emotions.json"))
await tracker.add_result("user123", result)
history = await tracker.get_recent("user123", limit=10)
baseline = await tracker.get_baseline_state("user123")
```

### 6. ContextAnalyzer

**Purpose**: Refine detection using context

**Analysis**:
- Emotional shifts detection
- Continuity over rapid messages
- Implicit emotion inference

**Example**:
```python
analyzer = ContextAnalyzer()
adjusted = await analyzer.analyze_context(emotions, context)
```

### 7. MultiEmotionDetector

**Purpose**: Handle multiple simultaneous emotions

**Calculates**:
- Dominant emotion (highest intensity * confidence)
- Valence (positive/negative dimension)
- Arousal (activation/energy dimension)

**Example**:
```python
detector = MultiEmotionDetector()
dominant, valence, arousal = await detector.analyze_multi_emotions(emotions)
```

### 8. TemporalAnalyzer

**Purpose**: Analyze emotional changes over time

**Metrics**:
- Trajectory (improving/worsening/stable)
- Velocity (rate of change)
- Volatility (fluctuation)
- Cycle detection

**Example**:
```python
analyzer = TemporalAnalyzer()
trajectory = await analyzer.analyze_trajectory("user123")
# {direction: "improving", velocity: 0.15, volatility: 0.3}
```

### 9. EmotionCache

**Purpose**: Cache results for performance

**Features**:
- Content-based hashing (SHA256)
- Configurable TTL (default: 1 hour)
- LRU eviction
- Context-aware caching

**Example**:
```python
cache = EmotionCache(max_size=10000, default_ttl=3600)
cached = await cache.get(text, context_key)
if not cached:
    result = await detect(text)
    await cache.set(text, result, context_key)
```

### 10. EmotionAggregator

**Purpose**: Aggregate all module results

**Responsibilities**:
- Combine results into EmotionResult
- Detect conflicting emotions
- Add warnings
- Validate outputs

**Example**:
```python
aggregator = EmotionAggregator()
result = await aggregator.aggregate(
    emotions, dominant, valence, arousal,
    triggers, patterns, context
)
```

## Production Considerations

### Deployment

```python
# Production configuration
detector = EmotionDetector(
    enable_cache=True,
    enable_history=True,
    history_storage_path=Path("/var/lib/morgan/emotions"),
    max_concurrent_operations=10,
)
```

### Monitoring

```python
# Log statistics periodically
stats = detector.stats
logger.info(
    "Emotion detector stats",
    extra={
        "total_detections": stats["total_detections"],
        "avg_time_ms": stats["average_processing_time_ms"],
        "cache_hit_rate": stats["cache_hit_rate"],
        "circuit_state": stats["circuit_breaker_state"],
    }
)
```

### Resource Limits

- Set appropriate `max_concurrent_operations` based on CPU cores
- Configure cache size based on available memory
- Set history retention based on storage capacity
- Monitor processing times and adjust timeouts

### Security

- Sanitize user input before processing
- Don't log sensitive emotional data
- Implement GDPR-compliant data deletion
- Encrypt history storage at rest

## Future Enhancements

1. **ML-based Classification**: Replace keyword matching with transformer models
2. **Multi-language Support**: Extend beyond English
3. **Emotion Intensity Models**: Fine-tune intensity calculations
4. **Real-time Streaming**: Support for streaming detection
5. **Integration with LLM**: Feed emotional context to language model
6. **Advanced Analytics**: Emotion clustering, anomaly detection
7. **Personalization**: User-specific emotion models
8. **Audio/Voice**: Emotion detection from speech

## License

Part of the Morgan AI Assistant system. See main project for license.
