# Morgan CLI & Interfaces - Quick Reference

## Installation

```bash
cd morgan-rag
pip install -e .
```

## CLI Commands

### User CLI (morgan)

```bash
# Setup
morgan init                                    # Initialize configuration

# Chat
morgan chat                                    # Interactive chat
morgan chat --session-id abc123                # Resume session
morgan ask "Your question"                     # Single question

# Knowledge
morgan learn ./docs --recursive                # Ingest documents
morgan knowledge                               # Show KB stats

# System
morgan health                                  # Health check
morgan stats                                   # Learning statistics
morgan history                                 # Conversation history

# Feedback
morgan rate 0.9 --comment "Excellent!"        # Rate response

# Config
morgan config                                  # View all config
morgan config llm_model                        # Get value
morgan config llm_model llama3.2               # Set value
```

### Admin CLI (morgan-admin)

```bash
# Deployment
morgan-admin deploy --environment production --replicas 3
morgan-admin status --watch

# Service Management
morgan-admin restart morgan-api
morgan-admin logs --service morgan-api --follow

# Monitoring
morgan-admin metrics --service morgan-api
morgan-admin alerts --severity critical
```

## Web API

### Start Server

```bash
# Development
python -m morgan.interfaces.web_interface

# Production
uvicorn morgan.interfaces.web_interface:app --host 0.0.0.0 --port 8000 --workers 4
```

### Endpoints

```http
GET  /                           # Root
GET  /health                     # Health check
POST /chat                       # Chat (sync)
POST /chat/stream                # Chat (streaming)
POST /feedback                   # Submit feedback
GET  /learning/stats?user_id=X   # Learning stats
GET  /sessions/{id}/history      # Session history
DELETE /sessions/{id}            # Delete session
```

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is AI?",
    "user_id": "user123",
    "include_sources": true,
    "include_emotion": true
  }'
```

## WebSocket

### Connect

```javascript
const ws = new WebSocket('ws://localhost:8000/ws?user_id=user123');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message);
};
```

### Send Message

```javascript
// Chat
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello Morgan!',
    session_id: 'session123'
  }
}));

// Feedback
ws.send(JSON.stringify({
  type: 'feedback',
  data: {
    response_id: 'resp789',
    rating: 0.9
  }
}));

// Ping
ws.send(JSON.stringify({type: 'ping'}));
```

## Configuration

### File Location

`~/.morgan/config.json`

### Environment Variables

```bash
export MORGAN_LLM_BASE_URL="http://localhost:11434"
export MORGAN_LLM_MODEL="llama3.2:latest"
export MORGAN_VECTOR_DB_URL="http://localhost:6333"
export MORGAN_ENABLE_EMOTIONS="true"
export MORGAN_ENABLE_LEARNING="true"
export MORGAN_ENABLE_RAG="true"
export MORGAN_LOG_LEVEL="INFO"
export MORGAN_VERBOSE="true"
```

### Python Configuration

```python
from morgan.cli import CLIConfig
from morgan.interfaces import create_app

# CLI Config
config = CLIConfig(
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",
    enable_emotion_detection=True,
    enable_learning=True,
    enable_rag=True,
)

# Web App
app = create_app(
    storage_path="/path/to/storage",
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",
    vector_db_url="http://localhost:6333",
)
```

## Common Patterns

### CLI Chat Session

```bash
$ morgan chat
Morgan AI Assistant

You: What is machine learning?

🤖 Morgan: Machine learning is a branch of artificial intelligence...

💭 Emotion: NEUTRAL (intensity: 0.75)

📚 Sources (3):
  1. ml_intro.pdf (score: 0.892)
  2. ai_basics.txt (score: 0.845)

You: exit
Goodbye! 👋
```

### API Integration

```python
import httpx
import asyncio

async def chat(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "message": message,
                "user_id": "user123",
                "include_sources": True,
                "include_emotion": True,
            }
        )
        return response.json()

# Usage
response = asyncio.run(chat("What is AI?"))
print(response["content"])
```

### WebSocket Integration

```javascript
class MorganClient {
  constructor(userId) {
    this.ws = new WebSocket(`ws://localhost:8000/ws?user_id=${userId}`);
    this.setupHandlers();
  }

  setupHandlers() {
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
  }

  handleMessage(message) {
    switch(message.type) {
      case 'connected':
        console.log('Connected:', message.data);
        break;
      case 'chat_chunk':
        console.log('Chunk:', message.data.content);
        break;
      case 'chat_complete':
        console.log('Complete:', message.data);
        break;
      case 'error':
        console.error('Error:', message.data);
        break;
    }
  }

  sendChat(message, sessionId) {
    this.ws.send(JSON.stringify({
      type: 'chat',
      data: {message, session_id: sessionId}
    }));
  }

  sendFeedback(responseId, rating) {
    this.ws.send(JSON.stringify({
      type: 'feedback',
      data: {response_id: responseId, rating}
    }));
  }
}

// Usage
const client = new MorganClient('user123');
client.sendChat('Hello Morgan!', 'session456');
```

## Troubleshooting

### CLI Issues

```bash
# Check health
morgan health

# Verbose mode
morgan --verbose chat

# Check logs
tail -f ~/.morgan/logs/morgan.log
```

### API Issues

```bash
# Check health
curl http://localhost:8000/health

# Check logs
# (stdout when running directly)

# Test endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test","user_id":"test"}'
```

### WebSocket Issues

```javascript
// Check connection status
fetch('http://localhost:8000/ws/status')
  .then(r => r.json())
  .then(console.log);

// Test ping
ws.send(JSON.stringify({type: 'ping'}));
```

## Performance Tips

1. **Enable caching**: Set `cache_enabled=true`
2. **Use streaming**: Lower latency for long responses
3. **Adjust context**: Reduce `max_context_tokens` for faster processing
4. **Batch operations**: Process multiple documents together
5. **Connection pooling**: Reuse HTTP clients
6. **Scale workers**: Use multiple uvicorn workers for API

## Error Handling

### CLI Errors

```bash
# Recoverable error
❌ Error: Service temporarily unavailable
💡 This error is recoverable. Please try again.

# With correlation ID (verbose mode)
❌ Error: Request failed
Correlation ID: abc123-def456
```

### API Errors

```json
{
  "error": "Assistant error",
  "detail": "Service unavailable",
  "correlation_id": "abc123",
  "timestamp": "2025-11-08T10:30:00"
}
```

### WebSocket Errors

```json
{
  "type": "error",
  "data": {
    "error": "Processing error",
    "message": "Failed to process request"
  }
}
```

## Next Steps

1. Read full documentation: `CLI_INTERFACES_README.md`
2. Explore examples: `examples/`
3. Run tests: `pytest tests/`
4. Report issues: GitHub Issues

## Support

- Documentation: https://morgan.readthedocs.io/
- GitHub: https://github.com/yourusername/morgan
- Email: support@morgan.ai
