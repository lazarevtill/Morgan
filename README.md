# Morgan AI Assistant

Morgan is a fully self-hosted, distributed personal AI assistant with emotional intelligence, deep knowledge management, and natural conversation capabilities.

## Key Features

- **Emotional Intelligence** - Understands and responds empathetically
- **Self-Hosted** - Complete privacy, all processing on your local hardware
- **Distributed Architecture** - Supports multi-host GPU deployment with load balancing
- **Separate Providers** - Configure different hosts for LLM, Embeddings, and Reranking
- **No External APIs** - Zero dependency on cloud services

## Quick Start

### 1. Configure Environment

Copy the example configuration:

```bash
cd morgan-rag
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# LLM Provider (for chat/generation)
LLM_BASE_URL=http://192.168.1.20:11434/v1
LLM_MODEL=qwen2.5:32b-instruct-q4_K_M

# Embedding Provider (can be different host)
EMBEDDING_BASE_URL=http://192.168.1.22:11434
EMBEDDING_MODEL=nomic-embed-text

# Vector Database
QDRANT_URL=http://localhost:6333
```

### 2. Start Dependencies

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Start Ollama (on your LLM host)
ollama serve
ollama pull qwen2.5:32b-instruct-q4_K_M

# Start Ollama (on your Embedding host if separate)
ollama pull nomic-embed-text
```

### 3. Install and Run

```bash
cd morgan-rag
pip install -r requirements.txt
python -m morgan chat
```

### Docker Deployment

```bash
cd morgan-rag
docker-compose up -d
```

## Configuration

Morgan supports **separate hosts** for each provider:

| Service | Environment Variable | Description |
|---------|---------------------|-------------|
| LLM | `LLM_BASE_URL` | OpenAI-compatible endpoint for chat |
| Embedding | `EMBEDDING_BASE_URL` | Endpoint for vector embeddings |
| Reranking | `RERANKING_ENDPOINT` | Endpoint for result reranking |
| Vector DB | `QDRANT_URL` | Qdrant vector database |

### Distributed LLM (Load Balancing)

Enable load balancing across multiple LLM hosts:

```bash
LLM_DISTRIBUTED_ENABLED=true
LLM_ENDPOINTS=http://192.168.1.20:11434/v1,http://192.168.1.21:11434/v1
LLM_LOAD_BALANCING_STRATEGY=round_robin
```

### 6-Host Architecture Example

```text
Host 1 (192.168.1.10): Morgan Core + Qdrant + Redis
Host 2 (192.168.1.11): Background Services + Monitoring
Host 3 (192.168.1.20): RTX 3090 - Main LLM #1
Host 4 (192.168.1.21): RTX 3090 - Main LLM #2 (load balanced)
Host 5 (192.168.1.22): RTX 4070 - Embeddings
Host 6 (192.168.1.23): RTX 2060 - Reranking
```

## CLI Commands

```bash
# Interactive chat
python -m morgan chat

# Ask a single question
python -m morgan ask "What is Python?"

# Learn from documents
python -m morgan learn ./documents

# Check system health
python -m morgan health

# Start web interface
python -m morgan serve
```

## Health Check

Check all providers:

```bash
python -m morgan.utils.health
```

Output:
```
Morgan System Health
============================================================
Overall Status: [OK] HEALTHY

[OK] LLM
    Endpoint: http://192.168.1.20:11434/v1
    Mode: single
    Model: qwen2.5:32b-instruct-q4_K_M

[OK] EMBEDDING
    Endpoint: http://192.168.1.22:11434
    Mode: remote
    Model: nomic-embed-text

[OK] VECTOR_DB
    Endpoint: http://localhost:6333
    Collections: 2
```

## Project Structure

```
morgan-rag/
├── morgan/
│   ├── config/           # Settings and configuration
│   ├── infrastructure/   # Distributed LLM, embeddings, reranking
│   ├── services/         # LLM, embedding, and service factory
│   ├── emotional/        # Emotion detection engine
│   ├── memory/           # Conversation memory
│   ├── search/           # Multi-stage search pipeline
│   ├── cli/              # Command-line interface
│   └── core/             # Main assistant logic
├── .env.example          # Configuration template
├── docker-compose.yml    # Docker deployment
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container build
```

## Requirements

- **Python:** 3.11+
- **GPU:** Optional (for local embedding/reranking)
- **Ollama:** For LLM serving
- **Qdrant:** Vector database

## Performance Targets

- Embeddings: <200ms batch
- Search + rerank: <500ms
- Simple queries: 1-2s
- Complex reasoning: 5-10s (acceptable for quality)

## License

MIT License

## Version

2.0.0-alpha
