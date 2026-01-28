# Empathic Engine - Emotional Intelligence

**Last Updated:** January 29, 2026

## Overview

Morgan's Emotional Intelligence system uses a **semantic-first approach** for understanding emotions. This means the LLM analyzes the meaning and context of messages to detect emotions, rather than relying primarily on keyword patterns.

## Architecture

The emotional intelligence system is located in `morgan-rag/morgan/intelligence/`:

```
intelligence/
├── emotions/           # Emotion detection and analysis
│   ├── analyzer.py     # Main emotion analyzer
│   ├── context.py      # Emotional context management
│   ├── intensity.py    # Intensity detection
│   ├── memory.py       # Emotional memory
│   ├── patterns.py     # Pattern validation (secondary)
│   ├── recovery.py     # Emotional recovery tracking
│   ├── regulator.py    # Emotion regulation
│   ├── tracker.py      # Emotion tracking over time
│   └── triggers.py     # Trigger detection
├── empathy/            # Empathic response generation
│   ├── generator.py    # Empathy response generator
│   ├── mirror.py       # Emotional mirroring
│   ├── support.py      # Supportive responses
│   ├── tone.py         # Tone analysis
│   └── validator.py    # Response validation
└── core/
    ├── intelligence_engine.py  # Main orchestrator
    └── models.py               # Data models
```

## Semantic-First Approach

### How It Works

1. **PRIMARY: LLM Semantic Analysis**
   - The LLM analyzes the full meaning of the message
   - Understands context, sarcasm, emotional masking
   - Detects both surface emotions and hidden emotions

2. **SECONDARY: Pattern Validation**
   - Pattern matching validates the LLM's detection
   - Boosts confidence when patterns agree
   - Does NOT override semantic results

3. **FALLBACK: Pattern-Only Detection**
   - Used only when LLM analysis fails
   - Provides basic emotion detection as backup

### Why Semantic-First?

Pattern-based detection fails for:
- **Sarcasm**: "Great, another meeting" (frustration, not joy)
- **Emotional masking**: "I'm fine" (often hides sadness)
- **Context-dependent meanings**: "Oh wonderful, the system crashed"
- **Complex emotions**: Bittersweet feelings, mixed emotions

The semantic approach correctly handles these cases by understanding meaning, not just words.

## Features

### 1. Emotion Detection

Detects emotions across multiple dimensions:
- **Surface emotion**: What the user appears to feel
- **Hidden emotion**: What they might actually feel
- **Sarcasm detection**: Identifies when meaning differs from words
- **Masking detection**: Identifies when emotions are hidden

### 2. Emotional Categories

- **Positive**: Joy, Excitement, Contentment, Gratitude, Hope
- **Negative**: Sadness, Anxiety, Frustration, Anger
- **Neutral**: Neutral, Concern, Confusion

### 3. Empathic Response Generation

The empathy system generates appropriate responses by:
- Mirroring appropriate emotions
- Providing emotional support
- Adjusting tone based on user's emotional state
- Validating response appropriateness

### 4. Emotional Memory

Tracks emotional patterns over time:
- Emotional history per user
- Trend analysis (improving, declining, stable)
- Shift detection (sudden emotional changes)
- Recovery tracking

## Usage

### Through the Intelligence Engine

```python
from morgan.intelligence.core.intelligence_engine import IntelligenceEngine

engine = IntelligenceEngine()

# Analyze a message
result = await engine.analyze("I'm so frustrated with this project!")

print(f"Emotion: {result.emotion}")
print(f"Confidence: {result.confidence}")
print(f"Is sarcasm: {result.sarcasm_detected}")
```

### Through the API

The emotional intelligence is automatically applied during chat:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am feeling really down today"}'
```

The response will be emotionally appropriate, providing comfort and support.

## Integration

The emotional intelligence integrates with:

- **Memory System**: Stores emotional context with memories
- **Search Pipeline**: Weights results by emotional relevance
- **Response Generation**: Adjusts tone and content

## Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Emotion detection | <100ms | With LLM cache |
| Pattern validation | <10ms | Local processing |
| Full empathy pipeline | <500ms | Including response adjustment |

## Further Reading

- [Architecture Documentation](../../morgan-rag/docs/ARCHITECTURE.md) - Full system architecture
- [Configuration Guide](./CONFIGURATION.md) - Server configuration
