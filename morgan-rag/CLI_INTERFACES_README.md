# Morgan CLI and Interfaces

Complete implementation of user-facing interfaces with full async/await patterns and integration with all refactored systems.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [User CLI](#user-cli)
4. [Admin CLI](#admin-cli)
5. [Web API](#web-api)
6. [WebSocket Interface](#websocket-interface)
7. [Configuration](#configuration)
8. [Architecture](#architecture)
9. [Performance](#performance)

## Overview

The Morgan interfaces provide multiple ways to interact with the AI assistant:

- **User CLI (`morgan`)**: Interactive command-line interface for end users
- **Admin CLI (`morgan-admin`)**: Administrative tools for system management
- **Web API**: RESTful API with FastAPI for web/mobile applications
- **WebSocket**: Real-time bidirectional communication for interactive experiences

All interfaces are built with:
- ✅ Full async/await architecture
- ✅ Integration with refactored systems (emotions, learning, RAG)
- ✅ Rich formatting and excellent UX
- ✅ Comprehensive error handling
- ✅ Performance monitoring
- ✅ NO placeholders, NO TODOs

## Installation

### Install with pip

```bash
cd morgan-rag
pip install -e .
```

This installs:
- `morgan` command (user CLI)
- `morgan-admin` command (admin CLI)
- Python packages for web/websocket interfaces

### Dependencies

Required dependencies are automatically installed:
- `click>=8.1.0` - CLI framework with async support
- `rich>=13.7.0` - Terminal formatting
- `fastapi>=0.108.0` - Web framework
- `uvicorn>=0.25.0` - ASGI server
- `websockets>=12.0` - WebSocket support

## User CLI

### Quick Start

```bash
# Initialize configuration
morgan init

# Start interactive chat
morgan chat

# Ask a single question
morgan ask "What is quantum computing?"

# Check system health
morgan health

# Ingest documents
morgan learn ./documents --recursive

# View learning statistics
morgan stats
```

### Available Commands

#### Interactive Chat

```bash
morgan chat [OPTIONS]

Options:
  --session-id TEXT      Resume specific session
  --streaming/--no-streaming  Enable/disable streaming (default: enabled)
```

Features:
- Real-time streaming responses
- Emotion detection and display
- RAG source citations
- Graceful error handling
- Session persistence

Example:
```bash
$ morgan chat
Starting new session: abc123...

Morgan AI Assistant
Type 'exit', 'quit', or 'bye' to end the session.

You: Tell me about neural networks

🤖 Morgan: Neural networks are computational models inspired by biological
neural networks in the brain. They consist of interconnected nodes (neurons)
organized in layers...

💭 Emotion: NEUTRAL (intensity: 0.75)

📚 Sources (3):
  1. deep_learning_basics.pdf (score: 0.892)
  2. neural_networks_intro.md (score: 0.845)
  3. ai_fundamentals.txt (score: 0.798)
```

#### Single Question

```bash
morgan ask "Your question here" [OPTIONS]

Options:
  --session-id TEXT  Use existing session for context
```

#### Health Check

```bash
morgan health
```

Checks all system components:
- Memory system
- Emotion detector
- Learning engine
- RAG search
- LLM connection

#### Document Ingestion

```bash
morgan learn PATH [OPTIONS]

Options:
  --recursive          Process directories recursively
  --file-types TEXT    Filter by file type (e.g., .pdf, .txt)
```

Supported formats: PDF, TXT, DOCX, MD, HTML

#### Configuration

```bash
# Initialize configuration interactively
morgan init

# View all configuration
morgan config

# Get specific value
morgan config llm_model

# Set value
morgan config llm_model llama3.2
```

#### Session Management

```bash
# View conversation history
morgan history [--limit 10]

# Resume session
morgan resume SESSION_ID
```

#### Feedback and Learning

```bash
# Rate last response (0.0 - 1.0)
morgan rate 0.9 [--comment "Great answer!"]

# View learning statistics
morgan stats
```

### CLI Options

Global options available for all commands:

```bash
--config PATH     Path to configuration file
--verbose         Enable verbose output
--no-rich         Disable rich formatting
```

## Admin CLI

Administrative tools for managing distributed Morgan deployments.

### Available Commands

#### Deployment

```bash
morgan-admin deploy [OPTIONS]

Options:
  --environment TEXT   Target environment (production, staging, development)
  --replicas INTEGER   Number of replicas (default: 3)
  --force              Force deployment without confirmation
```

#### Cluster Status

```bash
morgan-admin status [OPTIONS]

Options:
  --environment TEXT  Target environment
  --watch            Watch for changes (refresh every 5s)
```

#### Service Management

```bash
# Restart service
morgan-admin restart SERVICE_NAME [--environment TEXT]

# View logs
morgan-admin logs [OPTIONS]

Options:
  --service TEXT      Filter by service name
  --level TEXT        Minimum log level (DEBUG, INFO, WARNING, ERROR)
  --follow            Follow logs (tail -f)
  --lines INTEGER     Number of lines to show (default: 100)
```

#### Monitoring

```bash
# View metrics
morgan-admin metrics [OPTIONS]

Options:
  --service TEXT      Filter by service name
  --format TEXT       Output format (table, json)

# View alerts
morgan-admin alerts [OPTIONS]

Options:
  --severity TEXT     Minimum severity (info, warning, critical)
```

## Web API

FastAPI-based REST API for web and mobile applications.

### Starting the Server

#### Development

```python
# morgan-rag/morgan/interfaces/web_interface.py
from morgan.interfaces import create_app

app = create_app(
    storage_path="/path/to/storage",
    llm_base_url="http://localhost:11434",
    llm_model="llama3.2:latest",
    vector_db_url="http://localhost:6333",
    enable_emotion_detection=True,
    enable_learning=True,
    enable_rag=True,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or run directly:
```bash
cd morgan-rag
python -m morgan.interfaces.web_interface
```

#### Production

```bash
uvicorn morgan.interfaces.web_interface:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop
```

### API Endpoints

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00",
  "components": {
    "assistant": {"status": "operational", "healthy": true},
    "emotion": {"status": "operational", "healthy": true},
    "learning": {"status": "operational", "healthy": true},
    "rag": {"status": "operational", "healthy": true}
  }
}
```

#### Chat (Synchronous)

```http
POST /chat
Content-Type: application/json

{
  "message": "What is machine learning?",
  "user_id": "user123",
  "session_id": "session456",
  "include_sources": true,
  "include_emotion": true,
  "include_metrics": false
}
```

Response:
```json
{
  "response_id": "resp789",
  "content": "Machine learning is a branch of AI...",
  "timestamp": "2025-11-08T10:30:00",
  "emotion": {
    "primary_emotion": "neutral",
    "intensity": 0.75,
    "confidence": 0.92,
    "emotions": {"neutral": 0.85, "curiosity": 0.15}
  },
  "sources": [
    {
      "content": "Machine learning definition...",
      "source": "ml_intro.pdf",
      "score": 0.89,
      "document_id": "doc123"
    }
  ],
  "confidence": 0.95
}
```

#### Chat (Streaming)

```http
POST /chat/stream
Content-Type: application/json

{
  "message": "Explain neural networks",
  "user_id": "user123",
  "session_id": "session456"
}
```

Response (Server-Sent Events):
```
data: {"type": "start", "session_id": "session456"}

data: {"type": "chunk", "content": "Neural"}

data: {"type": "chunk", "content": " networks"}

data: {"type": "chunk", "content": " are..."}

data: {"type": "complete"}
```

#### Submit Feedback

```http
POST /feedback
Content-Type: application/json

{
  "response_id": "resp789",
  "user_id": "user123",
  "session_id": "session456",
  "rating": 0.9,
  "comment": "Very helpful!"
}
```

#### Learning Statistics

```http
GET /learning/stats?user_id=user123
```

Response:
```json
{
  "patterns_detected": 15,
  "feedback_processed": 42,
  "preferences_learned": 8,
  "adaptations_made": 23,
  "consolidations_performed": 3,
  "avg_confidence": 0.87
}
```

#### Session Management

```http
# Get session history
GET /sessions/{session_id}/history?limit=50

# Delete session
DELETE /sessions/{session_id}
```

### CORS Configuration

CORS is enabled by default. Configure origins:

```python
app = create_app(
    cors_origins=["https://yourdomain.com", "http://localhost:3000"]
)
```

## WebSocket Interface

Real-time bidirectional communication for interactive experiences.

### Connecting

```javascript
const ws = new WebSocket('ws://localhost:8000/ws?user_id=user123');

ws.onopen = () => {
  console.log('Connected to Morgan');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleMessage(message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from Morgan');
};
```

### Message Format

All messages follow this format:

```json
{
  "type": "message_type",
  "timestamp": "2025-11-08T10:30:00",
  "data": {
    // Type-specific data
  }
}
```

### Client -> Server Messages

#### Chat Message

```javascript
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello Morgan!',
    session_id: 'session123',
    include_emotion: true,
    include_sources: true
  }
}));
```

#### Feedback

```javascript
ws.send(JSON.stringify({
  type: 'feedback',
  data: {
    response_id: 'resp789',
    rating: 0.9,
    comment: 'Great response!',
    session_id: 'session123'
  }
}));
```

#### Typing Indicator

```javascript
ws.send(JSON.stringify({
  type: 'typing',
  data: {
    is_typing: true
  }
}));
```

#### Ping

```javascript
ws.send(JSON.stringify({
  type: 'ping'
}));
```

### Server -> Client Messages

#### Connected

```json
{
  "type": "connected",
  "data": {
    "connection_id": "conn123",
    "user_id": "user123",
    "message": "Connected to Morgan AI"
  }
}
```

#### Chat Start

```json
{
  "type": "chat_start",
  "data": {
    "response_id": "resp789",
    "session_id": "session123"
  }
}
```

#### Chat Chunk (Streaming)

```json
{
  "type": "chat_chunk",
  "data": {
    "response_id": "resp789",
    "content": "Neural networks"
  }
}
```

#### Chat Complete

```json
{
  "type": "chat_complete",
  "data": {
    "response_id": "resp789",
    "session_id": "session123",
    "content": "Full response text..."
  }
}
```

#### Error

```json
{
  "type": "error",
  "data": {
    "error": "Processing error",
    "message": "Failed to process request"
  }
}
```

#### Pong

```json
{
  "type": "pong",
  "data": {
    "timestamp": "2025-11-08T10:30:00"
  }
}
```

### Connection Status

Check active connections:

```http
GET /ws/status
```

Response:
```json
{
  "active_connections": 15,
  "connected_users": 12,
  "timestamp": "2025-11-08T10:30:00"
}
```

## Configuration

### Configuration File

Default location: `~/.morgan/config.json`

```json
{
  "storage_path": "/home/user/.morgan",
  "llm_base_url": "http://localhost:11434",
  "llm_model": "llama3.2:latest",
  "llm_timeout": 30,
  "vector_db_url": "http://localhost:6333",
  "vector_db_collection": "morgan_knowledge",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_batch_size": 32,
  "reranking_enabled": true,
  "reranking_model": "jina-reranker-v2-base-multilingual",
  "reranking_top_n": 5,
  "enable_emotion_detection": true,
  "enable_learning": true,
  "enable_rag": true,
  "max_context_tokens": 8000,
  "target_context_tokens": 6000,
  "context_pruning_strategy": "hybrid",
  "use_rich_formatting": true,
  "show_emotions": true,
  "show_sources": true,
  "show_metrics": false,
  "use_streaming": true,
  "max_concurrent_operations": 10,
  "cache_enabled": true,
  "log_level": "INFO",
  "log_file": null,
  "verbose": false
}
```

### Environment Variables

Override configuration with environment variables:

```bash
export MORGAN_STORAGE_PATH="/custom/path"
export MORGAN_LLM_BASE_URL="http://remote-llm:11434"
export MORGAN_LLM_MODEL="llama3.2:latest"
export MORGAN_VECTOR_DB_URL="http://remote-qdrant:6333"
export MORGAN_ENABLE_EMOTIONS="true"
export MORGAN_ENABLE_LEARNING="true"
export MORGAN_ENABLE_RAG="true"
export MORGAN_LOG_LEVEL="DEBUG"
export MORGAN_VERBOSE="true"
```

### Interactive Initialization

```bash
morgan init
```

Prompts for key configuration values and creates config file.

## Architecture

### CLI Architecture

```
morgan (CLI command)
    ↓
[app.py:cli()]
    ↓
1. Load configuration (config.py)
2. Initialize formatter (formatters.py)
3. Set up logging (utils.py)
    ↓
[Command execution]
    ↓
[utils.py:create_assistant()]
    ├─ Initialize MorganAssistant
    ├─ Set up emotion detection
    ├─ Set up learning engine
    └─ Set up RAG search
    ↓
[Assistant operations]
    ├─ process_message()
    ├─ stream_response()
    └─ process_feedback()
    ↓
[formatters.py:display_response()]
    ├─ Rich console output
    ├─ Emotion display
    ├─ Source citations
    └─ Metrics (if enabled)
```

### Web API Architecture

```
FastAPI App
    ↓
[Startup]
    ├─ Initialize MorganAssistant
    ├─ Initialize all systems
    └─ Register routes
    ↓
[Request handling]
    ├─ /chat → process_message()
    ├─ /chat/stream → stream_response()
    ├─ /feedback → process_feedback()
    └─ /health → health_check()
    ↓
[Response formatting]
    └─ Pydantic models
```

### WebSocket Architecture

```
WebSocket Connection
    ↓
[ConnectionManager]
    ├─ Track connections
    ├─ User session mapping
    └─ Message routing
    ↓
[WebSocketHandler]
    ├─ Message processing
    ├─ Chat streaming
    ├─ Feedback handling
    └─ Heartbeat
    ↓
[Assistant integration]
    └─ Full async operations
```

## Performance

### CLI Performance Targets

- **Startup time**: < 500ms
- **Command execution**: < 2s total (including assistant processing)
- **Streaming latency**: < 100ms first chunk
- **Memory usage**: < 200MB per session

### Web API Performance

- **Request latency**: < 2s (P95)
- **Streaming first byte**: < 100ms
- **Throughput**: > 100 requests/second (with proper scaling)
- **Concurrent connections**: Scales with workers

### WebSocket Performance

- **Connection latency**: < 50ms
- **Message latency**: < 10ms
- **Concurrent connections**: > 1000 per instance
- **Heartbeat interval**: 30s

### Optimization Features

1. **Async/await throughout**: All I/O operations are non-blocking
2. **Connection pooling**: Reuse HTTP connections
3. **Caching**: Emotion results, embeddings, responses
4. **Circuit breakers**: Graceful degradation on failures
5. **Rate limiting**: Protect against overload
6. **Batch processing**: Efficient document ingestion

## Error Handling

All interfaces implement comprehensive error handling:

1. **User-friendly messages**: Clear, actionable error messages
2. **Graceful degradation**: Continue operation when non-critical systems fail
3. **Correlation IDs**: Track errors across distributed systems
4. **Verbose mode**: Detailed debugging information when needed
5. **Exit codes**: Proper status codes for scripting

### Error Examples

#### CLI Error

```bash
$ morgan chat
❌ Error: Failed to connect to LLM service

💡 This error is recoverable. Please try again.

# With verbose mode
$ morgan --verbose chat
❌ Error: Failed to connect to LLM service
Correlation ID: abc123-def456
Connection timeout after 30s to http://localhost:11434

Traceback:
  ...detailed stack trace...
```

#### API Error

```json
{
  "error": "Assistant error",
  "detail": "Emotion detection service unavailable",
  "correlation_id": "abc123-def456",
  "timestamp": "2025-11-08T10:30:00"
}
```

## Testing

All components include comprehensive tests. See individual test files:

- `tests/cli/test_app.py` - CLI tests
- `tests/interfaces/test_web_interface.py` - API tests
- `tests/interfaces/test_websocket_interface.py` - WebSocket tests

Run tests:

```bash
pytest tests/cli/
pytest tests/interfaces/
```

## Contributing

When contributing to CLI or interfaces:

1. Follow async/await patterns throughout
2. Add proper type hints
3. Include docstrings
4. Write tests for new features
5. Update this README
6. Maintain rich formatting support
7. Ensure backward compatibility

## License

MIT License - see LICENSE file for details.

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/yourusername/morgan/issues
- Documentation: https://morgan.readthedocs.io/
- Email: support@morgan.ai
