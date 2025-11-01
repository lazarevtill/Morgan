# Morgan AI Assistant - API Reference

## 📋 Overview

Morgan AI Assistant provides a comprehensive REST API for AI-powered conversational interactions. The system is built with a microservices architecture, providing specialized endpoints for different AI capabilities.

> **Note**: Voice Activity Detection (VAD) using Silero VAD is **integrated into the STT service** via faster-whisper, not a separate service.

## 🔗 Base URLs

- **Core Service**: `http://localhost:8000` (Main orchestration)
- **LLM Service**: `http://localhost:8001` (OpenAI-compatible)
- **TTS Service**: `http://localhost:8002` (Text-to-speech)
- **STT Service**: `http://localhost:8003` (Speech-to-text with integrated VAD)

## 📚 API Documentation

All services provide interactive API documentation at:
- Core: `http://localhost:8000/docs`
- LLM: `http://localhost:8001/docs`
- TTS: `http://localhost:8002/docs`
- STT: `http://localhost:8003/docs`

## 🏠 Core Service API

The Core Service acts as the main API gateway and orchestrates interactions between all AI services.

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "core",
  "version": "0.2.0",
  "uptime": 3600,
  "dependencies": {
    "llm_service": "healthy",
    "tts_service": "healthy",
    "stt_service": "healthy"
  },
  "note": "VAD (Silero) is integrated into STT service via faster-whisper"
}
```

### Process Text Input
```http
POST /api/text
Content-Type: application/json

{
  "text": "Turn on the living room lights",
  "user_id": "user123",
  "metadata": {
    "generate_audio": true,
    "voice": "default",
    "language": "en"
  }
}
```

**Response:**
```json
{
  "success": true,
  "text_response": "I'll turn on the living room lights for you.",
  "audio_data": "base64_encoded_wav_data",
  "metadata": {
    "processing_time": 1.23,
    "services_used": ["llm", "tts"],
    "model": "llama3.2:latest"
  }
}
```

### Process Audio Input
```http
POST /api/audio
Content-Type: multipart/form-data

file: audio.wav
user_id: user123
language: en
```

**Response:**
```json
{
  "success": true,
  "transcription": "Turn on the living room lights",
  "confidence": 0.95,
  "text_response": "I'll turn on the living room lights for you.",
  "audio_data": "base64_encoded_wav_data"
}
```

### Get Conversation History
```http
GET /api/conversations/{user_id}
```

**Response:**
```json
{
  "user_id": "user123",
  "messages": [
    {
      "role": "user",
      "content": "Hello Morgan",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ],
  "total_messages": 50
}
```

### Clear Conversation History
```http
DELETE /api/conversations/{user_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Conversation history cleared"
}
```

## 🤖 LLM Service API (OpenAI Compatible)

The LLM service provides OpenAI-compatible endpoints for text generation and embeddings.

### List Available Models
```http
GET /v1/models
Authorization: Bearer ollama
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2:latest",
      "object": "model",
      "created": 1705312800,
      "owned_by": "ollama"
    },
    {
      "id": "mistral:7b",
      "object": "model",
      "created": 1705312800,
      "owned_by": "ollama"
    }
  ]
}
```

### Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer ollama

{
  "model": "llama3.2:latest",
  "messages": [
    {
      "role": "system",
      "content": "You are Morgan, a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "What's the weather like?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-12345",
  "object": "chat.completion",
  "created": 1705312800,
  "model": "llama3.2:latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I don't have access to current weather data, but I can help you with other tasks like controlling your smart home devices or answering general questions."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 35,
    "total_tokens": 60
  }
}
```

### Streaming Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama3.2:latest",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

**Streaming Response:**
```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"llama3.2:latest","choices":[{"index":0,"delta":{"role":"assistant","content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"llama3.2:latest","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1705312800,"model":"llama3.2:latest","choices":[{"index":0,"delta":{"content":" a time..."},"finish_reason":"stop"}]}
```

### Text Embeddings
```http
POST /v1/embeddings
Content-Type: application/json

{
  "input": "Hello world",
  "model": "llama3.2:latest"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "llama3.2:latest",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

## 🎵 TTS Service API

The TTS service provides text-to-speech synthesis with multiple voice options.

### Generate Speech
```http
POST /generate
Content-Type: application/json

{
  "text": "Hello, this is a test message.",
  "voice": "default",
  "speed": 1.0,
  "language": "en",
  "format": "wav"
}
```

**Response:**
```json
{
  "audio_data": "base64_encoded_wav_data",
  "format": "wav",
  "sample_rate": 22050,
  "duration": 2.5,
  "metadata": {
    "model": "csm-streaming",
    "voice": "default",
    "processing_time": 0.8
  }
}
```

### List Available Voices
```http
GET /voices
```

**Response:**
```json
{
  "voices": [
    {
      "name": "default",
      "language": "en",
      "gender": "female",
      "description": "American female voice"
    },
    {
      "name": "default",
      "language": "en",
      "gender": "male",
      "description": "American male voice"
    }
  ]
}
```

### Get Voice Information
```http
GET /voices/{voice_name}
```

**Response:**
```json
{
  "name": "default",
  "language": "en",
  "gender": "female",
  "sample_rate": 22050,
  "description": "American female voice - warm and friendly"
}
```

## 🎤 STT Service API

The STT service provides speech-to-text transcription with voice activity detection.

### Transcribe Audio
```http
POST /transcribe
Content-Type: application/json

{
  "audio_data": "base64_encoded_audio_data",
  "language": "en",
  "model": "whisper-base",
  "temperature": 0.0,
  "prompt": "This is a conversation with an AI assistant."
}
```

**Response:**
```json
{
  "text": "Turn on the living room lights",
  "language": "en",
  "confidence": 0.95,
  "duration": 2.1,
  "segments": [
    {
      "start": 0.0,
      "end": 0.8,
      "text": "Turn on",
      "confidence": 0.98
    },
    {
      "start": 0.8,
      "end": 1.5,
      "text": "the living room",
      "confidence": 0.94
    },
    {
      "start": 1.5,
      "end": 2.1,
      "text": "lights",
      "confidence": 0.96
    }
  ]
}
```

### Detect Language
```http
POST /detect-language
Content-Type: application/json

{
  "audio_data": "base64_encoded_audio_data"
}
```

**Response:**
```json
{
  "language": "en",
  "confidence": 0.99,
  "alternatives": [
    {"language": "en", "confidence": 0.99},
    {"language": "es", "confidence": 0.01}
  ]
}
```

### List Available Models
```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "name": "whisper-tiny",
      "size": "39MB",
      "languages": ["en", "es", "fr", "de"],
      "performance": "fast"
    },
    {
      "name": "whisper-base",
      "size": "74MB",
      "languages": ["en", "es", "fr", "de", "it", "pt"],
      "performance": "balanced"
    },
    {
      "name": "whisper-distil-large-v3.5 ",
      "size": "1.5GB",
      "languages": ["auto"],
      "performance": "accurate"
    }
  ]
}
```

## 🎯 Voice Activity Detection (Integrated)

Voice Activity Detection (VAD) is **integrated directly into the STT service** using Silero VAD through faster-whisper. There is no separate VAD service.

### VAD Configuration in STT

VAD parameters are configured in the STT transcription request:

```http
POST /transcribe
Content-Type: application/json

{
  "audio_data": "base64_encoded_audio_data",
  "language": "en",
  "use_vad": true,
  "vad_threshold": 0.5,
  "min_speech_duration": 0.25,
  "max_speech_duration": 30.0
}
```

**Response includes VAD information:**
```json
{
  "text": "Turn on the living room lights",
  "language": "en",
  "confidence": 0.95,
  "duration": 2.1,
  "vad_segments": [
    {
      "start": 0.5,
      "end": 2.1,
      "duration": 1.6,
      "speech_detected": true
    }
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 0.8,
      "text": "Turn on",
      "confidence": 0.98
    }
  ]
}
```

### VAD Benefits

- **Automatic noise filtering**: Silero VAD filters out non-speech audio
- **Improved accuracy**: Focuses transcription on actual speech segments
- **Reduced latency**: Skips silent portions
- **Energy efficient**: Processes only relevant audio

## 🔧 Error Handling

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Normal response |
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Endpoint or resource not found |
| 429 | Rate Limited | Too many requests |
| 500 | Internal Error | Server error |

### Error Response Format
```json
{
  "error": "MODEL_INFERENCE_ERROR",
  "message": "Failed to generate response from LLM service",
  "details": {
    "service": "llm_service",
    "timeout": 30,
    "model": "llama3.2:latest"
  },
  "request_id": "req_12345"
}
```

### Common Error Codes

#### Core Service Errors
- `INVALID_INPUT`: Malformed request data
- `SERVICE_UNAVAILABLE`: Required service not available
- `CONVERSATION_EXPIRED`: Conversation context expired
- `RATE_LIMITED`: Too many requests from user

#### LLM Service Errors
- `MODEL_LOAD_ERROR`: Failed to load language model
- `MODEL_INFERENCE_ERROR`: Error during text generation
- `OLLAMA_UNAVAILABLE`: External Ollama service not reachable
- `INVALID_MODEL`: Requested model not available

#### TTS Service Errors
- `AUDIO_SYNTHESIS_ERROR`: Failed to generate audio
- `VOICE_NOT_FOUND`: Requested voice not available
- `INVALID_AUDIO_FORMAT`: Unsupported audio format

#### STT Service Errors
- `TRANSCRIPTION_ERROR`: Failed to transcribe audio
- `INVALID_AUDIO`: Malformed or corrupted audio data
- `LANGUAGE_NOT_SUPPORTED`: Requested language not supported

## 📊 Rate Limiting

### Default Limits
- **Core Service**: 100 requests/minute per user
- **LLM Service**: 50 requests/minute (due to resource intensity)
- **TTS Service**: 30 requests/minute
- **STT Service**: 20 requests/minute
- **VAD Service**: 1000 requests/minute (lightweight)

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312800
```

### Rate Limit Exceeded Response
```json
{
  "error": "RATE_LIMITED",
  "message": "Too many requests",
  "retry_after": 60
}
```

## 🔐 Authentication

### API Key Authentication
```http
Authorization: Bearer your-api-key-here
```

### Service-to-Service Authentication
```python
# Internal service communication
headers = {
    "Authorization": f"Bearer {INTERNAL_SERVICE_TOKEN}",
    "X-Service-Name": "core",
    "X-Request-ID": request_id
}
```

## 📈 Metrics and Monitoring

### Service Metrics
```http
GET /metrics
```

**Response:**
```text
# HELP morgan_requests_total Total requests
# TYPE morgan_requests_total counter
morgan_requests_total{service="core",endpoint="/api/text"} 150

# HELP morgan_request_duration_seconds Request duration
# TYPE morgan_request_duration_seconds histogram
morgan_request_duration_seconds_bucket{service="llm",le="0.1"} 45
morgan_request_duration_seconds_bucket{service="llm",le="0.5"} 120
```

### Health Metrics
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600,
  "memory_usage": {
    "rss": 156780000,
    "vms": 234560000,
    "gpu_memory": 1200000000
  },
  "active_requests": 5,
  "error_rate": 0.001
}
```

## 🔄 Webhooks

### Conversation Events
```http
POST /webhooks/conversation
Content-Type: application/json

{
  "event": "conversation_start",
  "user_id": "user123",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "source": "voice",
    "language": "en",
    "duration": 2.1
  }
}
```

### Error Notifications
```http
POST /webhooks/error
Content-Type: application/json

{
  "event": "service_error",
  "service": "tts_service",
  "error_code": "AUDIO_SYNTHESIS_ERROR",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345"
}
```

## 📝 Request/Response Examples

### Complete Conversation Flow

#### 1. Text Input
```bash
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the weather like today?",
    "user_id": "user123",
    "metadata": {
      "generate_audio": true,
      "voice": "default"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "text_response": "I don't have access to current weather data, but I can help you control your smart home devices or answer general questions. Would you like me to turn on some lights or adjust the temperature?",
  "audio_data": "UklGRnoGAABXQVZFZm10IBAAAA...",
  "metadata": {
    "processing_time": 1.45,
    "services_used": ["llm", "tts"],
    "model": "llama3.2:latest",
    "voice": "default"
  }
}
```

#### 2. Voice Input
```bash
curl -X POST http://localhost:8000/api/audio \
  -F "file=@weather_question.wav" \
  -F "user_id=user123"
```

**Response:**
```json
{
  "success": true,
  "transcription": "What is the weather like today?",
  "confidence": 0.94,
  "text_response": "I don't have access to current weather data, but I can help you control your smart home devices or answer general questions. Would you like me to turn on some lights or adjust the temperature?",
  "audio_data": "UklGRnoGAABXQVZFZm10IBAAAA..."
}
```

### LLM Service Usage

#### Simple Chat
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Advanced Configuration
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.3,
    "max_tokens": 500,
    "stream": true
  }'
```

### TTS Service Usage

#### Basic Synthesis
```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is Morgan AI Assistant.",
    "voice": "default"
  }'
```

#### Advanced Configuration
```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to your smart home!",
    "voice": "default",
    "speed": 1.2,
    "language": "en",
    "format": "mp3",
    "sample_rate": 44100
  }'
```

### STT Service Usage

#### Basic Transcription
```bash
curl -X POST http://localhost:8003/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "UklGRnoGAABXQVZFZm10IBAAAA...",
    "language": "en"
  }'
```

#### Advanced Transcription
```bash
curl -X POST http://localhost:8003/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "UklGRnoGAABXQVZFZm10IBAAAA...",
    "model": "whisper-distil-large-v3.5 ",
    "temperature": 0.0,
    "prompt": "This is a conversation with Morgan AI Assistant."
  }'
```

## 🚀 SDK Examples

### Python SDK
```python
import asyncio
from morgan_sdk import MorganClient

async def main():
    client = MorganClient()

    # Text conversation
    response = await client.text("Turn on the lights")
    print(response.text)
    # Play audio if available
    if response.audio_data:
        play_audio(response.audio_data)

    # Voice conversation
    with open("audio.wav", "rb") as f:
        response = await client.audio(f.read())
        print(f"Transcription: {response.transcription}")

asyncio.run(main())
```

### JavaScript SDK
```javascript
import { MorganClient } from 'morgan-sdk';

const client = new MorganClient();

async function chat() {
    // Text chat
    const response = await client.text('Hello Morgan!');
    console.log(response.text);

    // Voice chat
    const audioFile = await fetch('audio.wav');
    const audioBlob = await audioFile.blob();
    const response = await client.audio(audioBlob);
    console.log(`Transcription: ${response.transcription}`);
}

chat();
```

---

**Morgan AI Assistant API Reference** - Comprehensive documentation for integrating with the distributed AI assistant system.
