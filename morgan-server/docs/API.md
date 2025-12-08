# API Documentation

Morgan Server exposes a comprehensive REST and WebSocket API for client communication. This document provides detailed information about all available endpoints.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Chat Endpoints](#chat-endpoints)
- [Memory Endpoints](#memory-endpoints)
- [Knowledge Endpoints](#knowledge-endpoints)
- [Profile Endpoints](#profile-endpoints)
- [System Endpoints](#system-endpoints)
- [WebSocket API](#websocket-api)
- [Client Libraries](#client-libraries)

## Base URL

```
http://localhost:8080
```

For production deployments, replace with your server's URL.

## Authentication

Morgan Server is designed for single-user self-hosted deployment and does not require authentication by default. All endpoints are accessible without API keys.

For multi-user deployments (future), authentication can be enabled via configuration.

## Response Format

All API responses follow a consistent JSON format:

**Success Response:**
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2024-12-08T10:30:00Z"
}
```

**Error Response:**
```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": { ... },
  "timestamp": "2024-12-08T10:30:00Z",
  "request_id": "req_123abc"
}
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request format or parameters
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request format or parameters invalid |
| `VALIDATION_ERROR` | Request validation failed |
| `NOT_FOUND` | Requested resource not found |
| `SERVICE_UNAVAILABLE` | External service (LLM, vector DB) unavailable |
| `TIMEOUT` | Request timeout |
| `INTERNAL_ERROR` | Unexpected server error |

## Rate Limiting

Rate limiting is configured per server instance:

- Default: 100 concurrent requests
- Configurable via `MORGAN_MAX_CONCURRENT_REQUESTS`

When rate limit is exceeded:
- Status: `429 Too Many Requests`
- Header: `Retry-After: <seconds>`

## Chat Endpoints

### POST /api/chat

Send a message and receive a response from Morgan.

**Request:**
```json
{
  "message": "Hello, Morgan!",
  "user_id": "user123",
  "conversation_id": "conv456",
  "metadata": {
    "source": "cli",
    "version": "1.0.0"
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User message |
| `user_id` | string | No | User identifier |
| `conversation_id` | string | No | Conversation identifier (auto-generated if not provided) |
| `metadata` | object | No | Additional metadata |

**Response:**
```json
{
  "answer": "Hello! How can I help you today?",
  "conversation_id": "conv456",
  "emotional_tone": "friendly",
  "empathy_level": 0.8,
  "personalization_elements": [
    "greeting_style",
    "tone_matching"
  ],
  "milestone_celebration": null,
  "confidence": 0.95,
  "sources": [
    {
      "content": "Relevant context...",
      "score": 0.87,
      "metadata": {
        "source": "document.pdf",
        "page": 5
      }
    }
  ],
  "metadata": {
    "processing_time_ms": 1234,
    "model": "gemma3"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather today?",
    "user_id": "user123"
  }'
```

## Memory Endpoints

### GET /api/memory/stats

Get memory statistics for the current user.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | No | User identifier (query parameter) |

**Response:**
```json
{
  "total_conversations": 42,
  "active_conversations": 5,
  "total_messages": 1234,
  "oldest_conversation": "2024-01-01T00:00:00Z",
  "newest_conversation": "2024-12-08T10:30:00Z",
  "storage_size_mb": 15.7
}
```

**Example:**
```bash
curl http://localhost:8080/api/memory/stats?user_id=user123
```

### GET /api/memory/search

Search conversation history.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query (query parameter) |
| `user_id` | string | No | User identifier (query parameter) |
| `limit` | integer | No | Maximum results (default: 10) |

**Response:**
```json
{
  "results": [
    {
      "conversation_id": "conv456",
      "timestamp": "2024-12-08T10:00:00Z",
      "message": "User message...",
      "response": "Morgan's response...",
      "relevance_score": 0.92
    }
  ],
  "total": 15,
  "query": "search term"
}
```

**Example:**
```bash
curl "http://localhost:8080/api/memory/search?query=weather&user_id=user123&limit=5"
```

### DELETE /api/memory/cleanup

Clean up old conversations.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | No | User identifier (query parameter) |
| `older_than_days` | integer | No | Delete conversations older than N days (default: 90) |

**Response:**
```json
{
  "deleted_conversations": 10,
  "deleted_messages": 234,
  "freed_space_mb": 5.2
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8080/api/memory/cleanup?user_id=user123&older_than_days=30"
```

## Knowledge Endpoints

### POST /api/knowledge/learn

Add documents to the knowledge base.

**Request:**
```json
{
  "source": "/path/to/document.pdf",
  "url": "https://example.com/article",
  "content": "Direct text content...",
  "doc_type": "pdf",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "tags": ["tag1", "tag2"]
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | No* | File path to document |
| `url` | string | No* | URL to fetch document from |
| `content` | string | No* | Direct text content |
| `doc_type` | string | No | Document type (auto-detected if not provided) |
| `metadata` | object | No | Additional metadata |

*One of `source`, `url`, or `content` is required.

**Response:**
```json
{
  "status": "success",
  "documents_processed": 1,
  "chunks_created": 42,
  "processing_time_seconds": 3.5,
  "document_id": "doc_789"
}
```

**Example:**
```bash
# From file
curl -X POST http://localhost:8080/api/knowledge/learn \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/path/to/document.pdf",
    "doc_type": "pdf"
  }'

# From URL
curl -X POST http://localhost:8080/api/knowledge/learn \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "doc_type": "html"
  }'

# Direct content
curl -X POST http://localhost:8080/api/knowledge/learn \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is the document content...",
    "doc_type": "text",
    "metadata": {"title": "My Document"}
  }'
```

### GET /api/knowledge/search

Search the knowledge base.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query (query parameter) |
| `limit` | integer | No | Maximum results (default: 10) |
| `min_score` | float | No | Minimum relevance score (0-1, default: 0.5) |

**Response:**
```json
{
  "results": [
    {
      "content": "Relevant text chunk...",
      "score": 0.92,
      "document_id": "doc_789",
      "metadata": {
        "source": "document.pdf",
        "page": 5,
        "title": "Document Title"
      }
    }
  ],
  "total": 25,
  "query": "search term"
}
```

**Example:**
```bash
curl "http://localhost:8080/api/knowledge/search?query=machine+learning&limit=5&min_score=0.7"
```

### GET /api/knowledge/stats

Get knowledge base statistics.

**Response:**
```json
{
  "total_documents": 150,
  "total_chunks": 3420,
  "total_size_bytes": 52428800,
  "collections": ["main", "archive"],
  "last_updated": "2024-12-08T10:30:00Z"
}
```

**Example:**
```bash
curl http://localhost:8080/api/knowledge/stats
```

## Profile Endpoints

### GET /api/profile/{user_id}

Get user profile and preferences.

**Response:**
```json
{
  "user_id": "user123",
  "preferred_name": "Alex",
  "relationship_age_days": 45,
  "interaction_count": 234,
  "trust_level": 0.85,
  "engagement_score": 0.92,
  "communication_style": "friendly",
  "response_length": "detailed",
  "topics_of_interest": [
    "AI",
    "Python",
    "Machine Learning"
  ],
  "created_at": "2024-10-24T00:00:00Z",
  "last_interaction": "2024-12-08T10:30:00Z"
}
```

**Example:**
```bash
curl http://localhost:8080/api/profile/user123
```

### PUT /api/profile/{user_id}

Update user preferences.

**Request:**
```json
{
  "communication_style": "professional",
  "response_length": "concise",
  "topics_of_interest": ["AI", "Python"],
  "preferred_name": "Alex"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `communication_style` | string | No | Communication style preference |
| `response_length` | string | No | Response length preference |
| `topics_of_interest` | array | No | Topics of interest |
| `preferred_name` | string | No | Preferred name |

**Response:**
```json
{
  "user_id": "user123",
  "updated_fields": [
    "communication_style",
    "response_length"
  ],
  "profile": { ... }
}
```

**Example:**
```bash
curl -X PUT http://localhost:8080/api/profile/user123 \
  -H "Content-Type: application/json" \
  -d '{
    "communication_style": "professional",
    "response_length": "concise"
  }'
```

### GET /api/timeline/{user_id}

Get relationship timeline and milestones.

**Response:**
```json
{
  "user_id": "user123",
  "milestones": [
    {
      "type": "first_conversation",
      "timestamp": "2024-10-24T00:00:00Z",
      "description": "First conversation with Morgan"
    },
    {
      "type": "trust_milestone",
      "timestamp": "2024-11-15T00:00:00Z",
      "description": "Reached trust level 0.5"
    },
    {
      "type": "interaction_milestone",
      "timestamp": "2024-12-01T00:00:00Z",
      "description": "100 interactions completed"
    }
  ],
  "relationship_summary": {
    "age_days": 45,
    "total_interactions": 234,
    "current_trust_level": 0.85
  }
}
```

**Example:**
```bash
curl http://localhost:8080/api/timeline/user123
```

## System Endpoints

### GET /health

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-08T10:30:00Z",
  "version": "0.1.0",
  "uptime_seconds": 3600.0
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Some non-critical issues
- `unhealthy` - Critical issues

**Example:**
```bash
curl http://localhost:8080/health
```

### GET /api/status

Detailed system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-08T10:30:00Z",
  "components": {
    "vector_db": {
      "name": "Qdrant",
      "status": "up",
      "latency_ms": 15.3,
      "error": null,
      "details": {
        "collections": 1,
        "total_vectors": 3420
      }
    },
    "llm": {
      "name": "Ollama",
      "status": "up",
      "latency_ms": 234.5,
      "error": null,
      "details": {
        "model": "gemma3",
        "provider": "ollama"
      }
    },
    "memory": {
      "name": "Memory System",
      "status": "up",
      "latency_ms": 5.2,
      "error": null,
      "details": {
        "total_conversations": 42,
        "cache_hit_rate": 0.87
      }
    }
  },
  "metrics": {
    "requests_total": 1234,
    "requests_per_second": 2.5,
    "average_response_time_ms": 456.7,
    "error_rate": 0.02,
    "active_sessions": 5
  }
}
```

**Example:**
```bash
curl http://localhost:8080/api/status
```

### GET /metrics

Prometheus-compatible metrics endpoint.

**Response:**
```
# HELP morgan_requests_total Total number of requests
# TYPE morgan_requests_total counter
morgan_requests_total{endpoint="/api/chat",method="POST"} 1234

# HELP morgan_response_time_seconds Response time in seconds
# TYPE morgan_response_time_seconds histogram
morgan_response_time_seconds_bucket{le="0.1"} 100
morgan_response_time_seconds_bucket{le="0.5"} 450
morgan_response_time_seconds_bucket{le="1.0"} 890
morgan_response_time_seconds_bucket{le="+Inf"} 1234

# HELP morgan_errors_total Total number of errors
# TYPE morgan_errors_total counter
morgan_errors_total{type="validation"} 5
morgan_errors_total{type="service"} 2
```

**Example:**
```bash
curl http://localhost:8080/metrics
```

### GET /docs

Interactive API documentation (Swagger UI).

Access at: http://localhost:8080/docs

### GET /redoc

Alternative API documentation (ReDoc).

Access at: http://localhost:8080/redoc

## WebSocket API

### WS /ws/{user_id}

Real-time chat via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/user123');
```

**Send Message:**
```json
{
  "type": "message",
  "message": "Hello, Morgan!",
  "conversation_id": "conv456",
  "metadata": {}
}
```

**Receive Response:**
```json
{
  "type": "response",
  "answer": "Hello! How can I help you?",
  "conversation_id": "conv456",
  "emotional_tone": "friendly",
  "confidence": 0.95,
  "sources": [],
  "metadata": {}
}
```

**Receive Streaming Response:**
```json
{
  "type": "stream",
  "chunk": "Hello! ",
  "conversation_id": "conv456",
  "done": false
}
```

**Connection Status:**
```json
{
  "type": "status",
  "status": "connected",
  "user_id": "user123",
  "session_id": "sess_789"
}
```

**Error:**
```json
{
  "type": "error",
  "error": "VALIDATION_ERROR",
  "message": "Message cannot be empty",
  "details": {}
}
```

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/user123');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'message',
    message: 'Hello, Morgan!'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

**Example (Python):**
```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8080/ws/user123"
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "message",
            "message": "Hello, Morgan!"
        }))
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(data["answer"])

asyncio.run(chat())
```

## Client Libraries

### Python Client

```bash
pip install morgan-cli
```

```python
import asyncio
from morgan_cli.client import MorganClient, ClientConfig

async def main():
    config = ClientConfig(
        server_url="http://localhost:8080",
        user_id="user123"
    )
    
    async with MorganClient(config) as client:
        # Chat
        response = await client.http.chat("Hello!")
        print(response["answer"])
        
        # Search knowledge
        results = await client.http.search_knowledge("AI")
        for result in results["results"]:
            print(result["content"])

asyncio.run(main())
```

### JavaScript Client (Coming Soon)

```javascript
import { MorganClient } from 'morgan-client';

const client = new MorganClient({
  serverUrl: 'http://localhost:8080',
  userId: 'user123'
});

// Chat
const response = await client.chat('Hello!');
console.log(response.answer);

// WebSocket
await client.connectWebSocket();
client.on('message', (data) => {
  console.log(data.answer);
});
await client.sendMessage('Hello!');
```

## Best Practices

### Error Handling

Always handle errors gracefully:

```python
try:
    response = await client.http.chat("Hello!")
except ConnectionError:
    print("Server unavailable")
except TimeoutError:
    print("Request timed out")
except RequestError as e:
    print(f"Request failed: {e.message}")
```

### Rate Limiting

Implement exponential backoff for rate limit errors:

```python
import time

max_retries = 3
retry_delay = 1

for attempt in range(max_retries):
    try:
        response = await client.http.chat("Hello!")
        break
    except RateLimitError as e:
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))
        else:
            raise
```

### Connection Management

Reuse connections when possible:

```python
# Good: Reuse client
async with MorganClient(config) as client:
    for message in messages:
        response = await client.http.chat(message)

# Bad: Create new client each time
for message in messages:
    async with MorganClient(config) as client:
        response = await client.http.chat(message)
```

### WebSocket Reconnection

Implement automatic reconnection:

```python
async def connect_with_retry(client, max_retries=5):
    for attempt in range(max_retries):
        try:
            await client.ws.connect()
            return
        except ConnectionError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
```

## Further Reading

- [Configuration Guide](./CONFIGURATION.md) - Server configuration
- [Deployment Guide](./DEPLOYMENT.md) - Deployment options
- [Client Documentation](../../morgan-cli/README.md) - Python client library
- [Migration Guide](../../MIGRATION.md) - Migrating from old system

## Support

For API issues or questions:
- Check server logs for error details
- Review API documentation at http://localhost:8080/docs
- Test endpoints with curl or Postman
- Check GitHub issues for known problems
