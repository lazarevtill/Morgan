

# Emotional Intelligence Integration Guide

## Overview

This guide shows how to integrate emotional awareness into Morgan's core service.

## Components Added

1. **CLI** (`cli.py`) - Interactive command-line interface with markdown rendering
2. **Emotional Detection** (`shared/utils/emotional.py`) - Simple emotion detector
3. **Emotional Handler** (`core/emotional_handler.py`) - Integration layer for emotional awareness

## Quick Integration

### Step 1: Add Emotional Handler to Core Service

In `core/app.py`, add to imports:

```python
from core.emotional_handler import get_emotional_handler
```

In `MorganCore.__init__()`, add:

```python
# Initialize emotional handler
self.emotional_handler = get_emotional_handler()
```

### Step 2: Enhance Text Processing with Emotional Awareness

In `process_text_request()` method, add emotional detection:

```python
async def process_text_request(
    self, text: str, user_id: str = "default", metadata: Optional[Dict[str, Any]] = None
) -> Response:
    """Process text input request with emotional awareness"""
    with Timer(self.logger, f"Text processing for user {user_id}"):
        try:
            self.request_count += 1

            # Detect emotion from user input
            emotion = self.emotional_handler.detect_emotion(text)
            self.emotional_handler.track_user_emotion(user_id, emotion)

            # Log emotional state
            if emotion.intensity > 0.6:
                self.logger.info(
                    f"Detected {emotion.primary_emotion.value} "
                    f"(intensity: {emotion.intensity:.2f}) from user {user_id}"
                )

            # Get conversation context
            context = await self.conversation_manager.get_context(user_id)

            # Add user message
            user_message = Message(
                role="user",
                content=text,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "emotion": emotion.primary_emotion.value,
                    "intensity": emotion.intensity,
                    "confidence": emotion.confidence,
                }
            )
            await self.conversation_manager.add_message(user_id, user_message)

            # Get emotional context for LLM
            emotional_context = self.emotional_handler.get_emotional_context_for_llm(
                user_id, emotion
            )

            # Enhance system prompt with emotional awareness
            enhanced_messages = context.messages.copy()
            if emotional_context:
                # Add emotional context as system message
                enhanced_messages.insert(0, Message(
                    role="system",
                    content=emotional_context,
                    timestamp=datetime.utcnow().isoformat()
                ))

            # Process with LLM (existing code)
            llm_response = await self.streaming_orchestrator.process_message(
                text, enhanced_messages
            )

            # Enhance response with empathy
            enhanced_text = self.emotional_handler.enhance_response_with_empathy(
                llm_response.text, emotion
            )

            # Store response
            assistant_message = Message(
                role="assistant",
                content=enhanced_text,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "original_response": llm_response.text,
                    "empathy_enhanced": enhanced_text != llm_response.text
                }
            )
            await self.conversation_manager.add_message(user_id, assistant_message)

            return Response(
                text=enhanced_text,
                metadata={
                    "conversation_id": context.conversation_id,
                    "emotion_detected": emotion.primary_emotion.value,
                    "emotion_intensity": emotion.intensity,
                    "processing_time": 0.0  # Add actual timing
                }
            )

        except Exception as e:
            self.logger.error(f"Text processing error: {e}", exc_info=True)
            raise
```

### Step 3: Add Emotional Stats to Health/Status Endpoints

In `core/api/server.py`, enhance the status endpoint:

```python
@app.get("/status")
async def get_status():
    """Get detailed system status"""
    # ... existing code ...

    # Add emotional stats
    emotional_stats = core_service.emotional_handler.get_stats()

    return {
        # ... existing fields ...
        "emotional_intelligence": {
            "tracked_users": emotional_stats["tracked_users"],
            "total_states_tracked": emotional_stats["total_emotional_states"],
            "active_users": emotional_stats["active_users"],
        }
    }
```

## Using the CLI

### Install Requirements

```bash
pip install rich aiohttp
```

### Basic Usage

```bash
# Interactive chat
python cli.py chat

# Ask a single question
python cli.py ask "What is Docker?"

# Ask with sources
python cli.py ask "Explain containers" --sources

# View memory stats
python cli.py memory --stats

# Check health
python cli.py health
```

### CLI Features

- **Markdown rendering** - Beautiful formatted responses
- **Conversation memory** - Remembers context within session
- **Emotional awareness** - Detects and responds to your emotional state
- **Rich formatting** - Panels, syntax highlighting, progress bars
- **Commands**: `help`, `reset`, `exit`/`quit`

## Emotional Detection Features

### Supported Emotions

- **Joy** - happiness, excitement, gratitude
- **Sadness** - disappointment, loneliness, hurt
- **Anger** - frustration, annoyance, hate
- **Fear** - worry, anxiety, panic
- **Surprise** - amazement, shock, wonder
- **Neutral** - default state

### Mood Pattern Tracking

Tracks:
- Dominant emotion over time
- Average emotional intensity
- Emotional stability (variance)
- Trends (improving, declining, stable)
- Emotion distribution

### Empathetic Responses

For strong emotions (intensity > 0.7), responses include:
- **Sadness**: "I understand this might be difficult."
- **Anger**: "I hear your frustration."
- **Joy**: "I'm glad to help with this!"
- **Fear**: "I understand your concern."

## Advanced Features (Future)

1. **Personalization** - Learn user preferences over time
2. **Relationship Milestones** - Track significant moments
3. **Emotional Memory** - Remember emotional context across sessions
4. **Adaptive Responses** - Adjust communication style based on user mood
5. **Reranking** - Prioritize results based on emotional context

## Testing

```bash
# Test emotion detection
python -c "from shared.utils.emotional import get_emotion_detector; \
d = get_emotion_detector(); \
print(d.detect('I am so happy!').primary_emotion)"

# Test CLI health check
python cli.py health

# Test interactive chat
python cli.py chat
```

## Configuration

No additional configuration required. Emotional features work out-of-the-box with the existing Morgan setup.

## Troubleshooting

**CLI can't connect**:
- Ensure Morgan service is running on http://localhost:8000
- Check `python cli.py health`
- Use `--url` flag to specify different URL

**Emotions not detected**:
- Check logs for emotional detection (intensity > 0.6 logged)
- Verify emotional handler is initialized in core service
- Test detector directly with test script above

## Performance Impact

- **Minimal overhead** - Simple keyword matching, no ML models
- **Async processing** - No blocking operations
- **Memory efficient** - Only keeps last 20 emotional states per user

## Next Steps

1. Integrate emotional handler into core service (Step 1-2 above)
2. Test CLI: `python cli.py chat`
3. Monitor emotional stats: `python cli.py memory --stats`
4. Customize empathetic responses in `core/emotional_handler.py`
5. Add personalization based on emotional patterns

## Example Conversation

```
$ python cli.py chat

╭──────────────── Welcome to Morgan ────────────────╮
│   Morgan AI Assistant                             │
│                                                    │
│ Type your message and press Enter.                │
│ Type 'exit' or 'quit' to end.                     │
│ Type 'help' for available commands.               │
╰────────────────────────────────────────────────────╯

You: I'm really frustrated with this Docker setup!

Morgan is thinking...

Morgan: I hear your frustration. Let me help you with Docker setup.
What specific issues are you encountering? I can guide you through
the installation and configuration step by step.

[Detected: anger (intensity: 0.8)]
```

