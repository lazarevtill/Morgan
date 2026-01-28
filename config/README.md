# Morgan Configuration

This directory contains modular YAML configuration files for each Morgan system component.

## Configuration Files

| File | Purpose |
|------|---------|
| `llm.yaml` | LLM service (models, endpoints, temperature) |
| `embeddings.yaml` | Embedding service (models, dimensions, batch size) |
| `reranking.yaml` | Reranking service (models, weights, fallbacks) |
| `qdrant.yaml` | Vector database (host, port, collections) |
| `redis.yaml` | Cache and session storage |
| `server.yaml` | Web server and API settings |
| `memory.yaml` | Conversation memory and learning |
| `intelligence.yaml` | Emotional intelligence and reasoning |
| `search.yaml` | Multi-stage search pipeline |
| `distributed.yaml` | Multi-host GPU cluster settings |

## Quick Start

1. Copy example files if needed:
   ```bash
   cp config/*.yaml.example config/*.yaml
   ```

2. Edit the files for your environment:
   - For single-host setup: Edit `llm.yaml`, `qdrant.yaml`
   - For distributed setup: Edit `distributed.yaml` with your host IPs

3. Environment variables override YAML settings with prefixes:
   - `MORGAN_LLM_*` - LLM settings
   - `MORGAN_EMBEDDING_*` - Embedding settings
   - `MORGAN_QDRANT_*` - Qdrant settings
   - `MORGAN_REDIS_*` - Redis settings

## Common Configurations

### Single Host (Development)

```yaml
# llm.yaml
main:
  endpoint: "http://localhost:11434/v1"
  model: "qwen2.5:7b-instruct"  # Smaller model for development

# qdrant.yaml
connection:
  host: "localhost"
  port: 6333
```

### Distributed (Production)

```yaml
# distributed.yaml
settings:
  enabled: true

hosts:
  - host_id: "llm-1"
    address: "192.168.1.20"
    port: 11434
    role: "main_llm"
```

## Environment Variable Overrides

All settings can be overridden via environment variables:

```bash
# LLM
export MORGAN_LLM_ENDPOINT="http://192.168.1.20:11434/v1"
export MORGAN_LLM_MODEL="qwen2.5:32b-instruct-q4_K_M"
export MORGAN_LLM_TEMPERATURE="0.7"

# Embeddings
export MORGAN_EMBEDDING_ENDPOINT="http://192.168.1.22:11434/v1"
export MORGAN_EMBEDDING_MODEL="qwen3-embedding:4b"

# Qdrant
export MORGAN_QDRANT_HOST="192.168.1.10"
export MORGAN_QDRANT_PORT="6333"

# Redis
export MORGAN_REDIS_HOST="192.168.1.10"
export MORGAN_REDIS_PORT="6379"
```

## Docker Compose

For Docker deployments, mount the config directory:

```yaml
services:
  morgan:
    volumes:
      - ./config:/app/config:ro
```

## Validation

Run config validation:

```bash
python -c "from morgan.config import load_config; load_config().validate()"
```
