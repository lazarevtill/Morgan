# Morgan AI Assistant - Real-Time Streaming Architecture

> **Version**: 0.2.0  
> **Date**: 2025-10-27  
> **Status**: Production Ready

## Overview

Morgan now supports full real-time streaming capabilities with PostgreSQL persistence, Redis caching, and WebSocket communication. This document describes the complete streaming architecture.

## Architecture Components

### 1. Database Layer

#### PostgreSQL (Port 5432)
- **Purpose**: Persistent storage for conversations, messages, and system state
- **Tables**:
  - `conversations` - Conversation metadata and history
  - `messages` - Individual messages with embeddings
  - `streaming_sessions` - Active streaming session tracking
  - `audio_transcriptions` - STT results cache
  - `tts_generations` - TTS audio cache
  - `user_preferences` - User settings and preferences
  - `system_metrics` - Performance and usage metrics

#### Redis (Port 6379)
- **Purpose**: High-speed caching and session state
- **Usage**:
  - Conversation context caching (TTL: 1 hour)
  - Streaming session state (TTL: 30 minutes)
  - User preferences cache (TTL: 24 hours)
  - Audio buffer management
  - Active user tracking

#### Qdrant (Ports 6333/6334)
- **Purpose**: Vector database for semantic memory search
- **Features**:
  - Message embeddings storage
  - Semantic search capabilities
  - Context-aware memory retrieval

### 2. Core Service (Port 8000)

#### WebSocket Endpoint
**Path**: `/ws/audio`

**Message Types**:
- `start` - Initialize streaming session
- `audio` - Stream audio chunk (base64 encoded)
- `stop` - End streaming session
- `config` - Update session configuration

**Response Types**:
- `transcription` - Real-time transcription result
- `response` - LLM response
- `audio` - TTS audio response
- `status` - Session status update
- `error` - Error message

#### REST Endpoints
- `POST /api/text` - Text conversation endpoint
- `POST /api/audio` - Audio file upload
- `POST /api/conversation/reset` - Reset conversation
- `GET /health` - Health check with full system status

### 3. LLM Service (Port 8001)

#### Streaming Support
- **Endpoint**: `POST /stream`
- **Format**: Server-Sent Events (SSE)
- **Features**:
  - Token-by-token streaming
  - Low latency response
  - Conversation context management

#### Standard Endpoints
- `POST /generate` - Non-streaming generation
- `POST /embed` - Text embeddings
- `GET /models` - Available models list

### 4. TTS Service (Port 8002)

#### Streaming Support
- **Endpoint**: `POST /generate` with `stream: true`
- **Format**: Chunked audio streaming
- **Features**:
  - 16KB audio chunks
  - Multi-voice support (Kokoro: af_heart, am_michael, bf_emma, bm_george)
  - GPU-accelerated synthesis

#### Standard Endpoints
- `POST /generate` - Full audio generation
- `POST /generate/audio` - Raw audio response
- `GET /voices` - Available voices list

### 5. STT Service (Port 8003)

#### Real-Time Streaming
- **WebSocket**: `/ws/stream/{session_id}`
- **HTTP Streaming**: 
  - `POST /stream/start` - Initialize session
  - `POST /stream/{session_id}/chunk` - Add audio chunk
  - `POST /stream/{session_id}/end` - Finalize transcription

#### Features
- **Faster Whisper Integration**: High-speed transcription
- **Silero VAD**: Voice activity detection
- **Supported Models**: distil-large-v3, large-v3, medium, small
- **Languages**: 16+ languages with auto-detection

#### Standard Endpoints
- `POST /transcribe` - JSON transcription
- `POST /transcribe/file` - File upload
- `POST /transcribe/realtime` - Real-time with VAD
- `GET /sessions` - Active sessions list

## Data Flow

### Real-Time Voice Conversation

```
1. Client → WebSocket Connection → Core Service (/ws/audio)
2. Client → Audio Chunk (base64) → Core Service
3. Core Service → STT Service (WebSocket) → Transcription
4. Core Service → ConversationManager (Redis + PostgreSQL)
5. Core Service → LLM Service (SSE Stream) → Response Tokens
6. Core Service → TTS Service (Streaming) → Audio Chunks
7. Core Service → Client (WebSocket) → Audio Response
```

### Text Conversation with Streaming

```
1. Client → HTTP POST → Core Service (/api/text)
2. Core Service → ConversationManager (Redis Cache Check)
3. Core Service → LLM Service (SSE Stream)
4. LLM Service → Stream Tokens → Core Service
5. Core Service → PostgreSQL (Save Message)
6. Core Service → Redis (Update Cache)
7. Core Service → Client (HTTP Response)
```

### Audio File Processing

```
1. Client → Upload Audio → Core Service (/api/audio)
2. Core Service → STT Service (File Upload)
3. STT Service → Whisper + VAD → Transcription
4. Core Service → LLM Service → Response
5. Core Service → TTS Service → Audio Generation
6. TTS Service → Cache Check (PostgreSQL)
7. Core Service → Client (JSON + Base64 Audio)
```

## Session Management

### Streaming Session Lifecycle

1. **Initialization**
   - Create session in PostgreSQL
   - Cache session state in Redis
   - Initialize audio buffer

2. **Active Streaming**
   - Buffer audio chunks in Redis
   - Process chunks when threshold reached
   - Update session metadata
   - Emit real-time transcriptions

3. **Termination**
   - Finalize transcription
   - Save complete conversation to PostgreSQL
   - Clear Redis cache
   - Update session status

### Session State Tracking

**Redis Keys**:
- `stream_session:{session_id}` - Session metadata
- `audio_buffer:{session_id}` - Audio chunks buffer
- `active_user:{user_id}` - Active user flag

**PostgreSQL Tables**:
- `streaming_sessions` - Full session history
- `audio_transcriptions` - Transcription results
- `messages` - Conversation messages

## Performance Optimization

### Caching Strategy

1. **Conversation Context** (Redis)
   - TTL: 1 hour
   - Key: `conversation:{user_id}`
   - Fallback: PostgreSQL

2. **User Preferences** (Redis)
   - TTL: 24 hours
   - Key: `user_prefs:{user_id}`
   - Write-through to PostgreSQL

3. **TTS Audio** (PostgreSQL)
   - Hash-based cache
   - Key: SHA256(text + voice + speed)
   - Binary storage

4. **Streaming Sessions** (Redis)
   - TTL: 30 minutes
   - Key: `stream_session:{session_id}`
   - Periodic sync to PostgreSQL

### Database Connection Pooling

**PostgreSQL**:
- Min connections: 5
- Max connections: 20
- Connection timeout: 60s

**Redis**:
- Max connections: 50
- Decode responses: true

## Error Handling

### Graceful Degradation

1. **Redis Unavailable**
   - Fall back to in-memory cache
   - Continue with PostgreSQL
   - Log warning

2. **PostgreSQL Unavailable**
   - Use in-memory storage
   - Queue writes for retry
   - Degraded mode indicator

3. **Service Unavailable**
   - Retry with exponential backoff
   - Circuit breaker pattern
   - Fallback responses

### Error Recovery

- Automatic reconnection to databases
- Session state recovery from PostgreSQL
- Conversation history restoration
- Audio buffer persistence

## Monitoring & Metrics

### Health Checks

**Endpoints**:
- `GET /health` - Overall system health
- Individual service health checks
- Database connectivity checks
- Redis availability checks

**Response Format**:
```json
{
  "status": "healthy|degraded|unhealthy",
  "services": {
    "llm": true,
    "tts": true,
    "stt": true,
    "vad": true
  },
  "databases": {
    "postgres": true,
    "redis": true,
    "qdrant": true
  },
  "conversations": {
    "total": 10,
    "active": 5
  }
}
```

### System Metrics

Stored in `system_metrics` table:
- Response times by service
- Request counts
- Error rates
- Database query performance
- Cache hit rates

## Security

### Data Protection

1. **PostgreSQL**
   - Password authentication
   - Connection encryption (optional)
   - Row-level security (future)

2. **Redis**
   - Password protection (optional)
   - TTL-based expiration
   - Automatic key eviction

3. **API Security**
   - CORS configuration
   - Rate limiting (planned)
   - Request validation
   - Error message sanitization

## Configuration

### Environment Variables

```bash
# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=morgan
POSTGRES_USER=morgan
POSTGRES_PASSWORD=morgan_secure_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Services
MORGAN_LLM_SERVICE_URL=http://llm-service:8001
MORGAN_TTS_SERVICE_URL=http://tts-service:8002
MORGAN_STT_SERVICE_URL=http://stt-service:8003
```

### Configuration Files

- `config/core.yaml` - Core service settings
- `config/llm.yaml` - LLM configuration
- `config/tts.yaml` - TTS settings
- `config/stt.yaml` - STT configuration

## Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f core

# Restart service
docker-compose restart core

# Scale services (future)
docker-compose up -d --scale stt-service=3
```

### Dependencies

**Start Order**:
1. Redis
2. PostgreSQL
3. Qdrant
4. LLM Service
5. TTS Service
6. STT Service
7. Core Service

## API Examples

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/audio');

// Start session
ws.send(JSON.stringify({
  type: 'start',
  user_id: 'user123',
  language: 'en'
}));

// Send audio chunk
ws.send(JSON.stringify({
  type: 'audio',
  audio_data: base64AudioChunk
}));

// Receive responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'transcription') {
    console.log('Transcription:', data.text);
  } else if (data.type === 'audio') {
    playAudio(data.text); // base64 audio
  }
};
```

### HTTP Streaming (LLM)

```python
import requests

response = requests.post(
    'http://localhost:8001/stream',
    json={
        'prompt': 'Tell me a story',
        'stream': True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(data['text'], end='', flush=True)
```

## Future Enhancements

1. **Horizontal Scaling**
   - Load balancing across service instances
   - Redis Cluster support
   - PostgreSQL read replicas

2. **Advanced Caching**
   - Conversation prefetching
   - Predictive TTS caching
   - Embedding cache

3. **Real-Time Features**
   - Voice interruption handling
   - Simultaneous multi-user sessions
   - Live transcription streaming

4. **Analytics**
   - Real-time dashboard
   - Performance analytics
   - User behavior tracking

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check firewall rules
   - Verify CORS settings
   - Check service health

2. **High Latency**
   - Check database connection pool
   - Monitor Redis cache hit rate
   - Review service logs

3. **Memory Issues**
   - Adjust Redis maxmemory
   - Increase PostgreSQL shared_buffers
   - Monitor conversation cleanup

### Debug Commands

```bash
# Check database connection
docker-compose exec core python -c "from shared.utils.database import get_db_manager; import asyncio; asyncio.run(get_db_manager())"

# Check Redis connection
docker-compose exec redis redis-cli ping

# View active sessions
curl http://localhost:8003/sessions

# Check system health
curl http://localhost:8000/health | jq
```

## Contributing

When contributing streaming features:

1. Follow async/await patterns
2. Implement proper error handling
3. Add database migrations
4. Update this documentation
5. Add integration tests
6. Monitor performance impact

## License

MIT License - See LICENSE file for details

---

**Morgan AI Assistant** - Real-time streaming with PostgreSQL, Redis, and WebSocket support.

