# Configuration Guide

This guide explains how to configure Morgan Server for different deployment scenarios.

## Configuration Sources

Morgan Server supports multiple configuration sources with the following precedence (highest to lowest):

1. **Environment variables** (highest precedence)
2. **Configuration files** (YAML, JSON, .env)
3. **Default values** (lowest precedence)

## Configuration Files

### YAML Configuration

Create a `config.yaml` file:

```yaml
# Server settings
host: "0.0.0.0"
port: 8080
workers: 4

# LLM settings
llm_provider: "ollama"
llm_endpoint: "http://localhost:11434"
llm_model: "gemma3"
llm_api_key: null  # Optional

# Vector database settings
vector_db_url: "http://localhost:6333"
vector_db_api_key: null  # Optional

# Embedding settings
embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_device: "cpu"
embedding_endpoint: null  # Required for remote providers
embedding_api_key: null  # Optional

# Cache settings
cache_dir: "./data/cache"
cache_size_mb: 1000

# Logging settings
log_level: "INFO"
log_format: "json"

# Performance settings
max_concurrent_requests: 100
request_timeout_seconds: 60
session_timeout_minutes: 60
```

Load with:
```bash
morgan-server --config config.yaml
```

### JSON Configuration

Create a `config.json` file:

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "llm_provider": "ollama",
  "llm_endpoint": "http://localhost:11434",
  "llm_model": "gemma3",
  "vector_db_url": "http://localhost:6333",
  "embedding_provider": "local",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "log_level": "INFO"
}
```

### .env Configuration

Create a `.env` file:

```bash
MORGAN_HOST=0.0.0.0
MORGAN_PORT=8080
MORGAN_LLM_PROVIDER=ollama
MORGAN_LLM_ENDPOINT=http://localhost:11434
MORGAN_LLM_MODEL=gemma3
MORGAN_VECTOR_DB_URL=http://localhost:6333
MORGAN_EMBEDDING_PROVIDER=local
MORGAN_LOG_LEVEL=INFO
```

## Environment Variables

All configuration options can be set via environment variables with the `MORGAN_` prefix:

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_HOST` | `0.0.0.0` | Server host address |
| `MORGAN_PORT` | `8080` | Server port (1-65535) |
| `MORGAN_WORKERS` | `4` | Number of worker processes |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_LLM_PROVIDER` | `ollama` | LLM provider (`ollama`, `openai-compatible`) |
| `MORGAN_LLM_ENDPOINT` | `http://localhost:11434` | LLM service endpoint URL |
| `MORGAN_LLM_MODEL` | `gemma3` | LLM model name |
| `MORGAN_LLM_API_KEY` | `null` | API key for LLM service (optional) |

### Vector Database Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_VECTOR_DB_URL` | `http://localhost:6333` | Qdrant vector database URL |
| `MORGAN_VECTOR_DB_API_KEY` | `null` | Vector database API key (optional) |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_EMBEDDING_PROVIDER` | `local` | Embedding provider (`local`, `ollama`, `openai-compatible`) |
| `MORGAN_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name |
| `MORGAN_EMBEDDING_DEVICE` | `cpu` | Device for local embeddings (`cpu`, `cuda`, `mps`) |
| `MORGAN_EMBEDDING_ENDPOINT` | `null` | Remote embedding endpoint (required for remote providers) |
| `MORGAN_EMBEDDING_API_KEY` | `null` | API key for remote embeddings (optional) |

### Cache Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_CACHE_DIR` | `./data/cache` | Cache directory path |
| `MORGAN_CACHE_SIZE_MB` | `1000` | Maximum cache size in MB |

### Logging Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `MORGAN_LOG_FORMAT` | `json` | Log format (`json`, `text`) |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MORGAN_MAX_CONCURRENT_REQUESTS` | `100` | Maximum concurrent requests |
| `MORGAN_REQUEST_TIMEOUT_SECONDS` | `60` | Request timeout in seconds |
| `MORGAN_SESSION_TIMEOUT_MINUTES` | `60` | Session timeout in minutes |

## Configuration Validation

Morgan Server validates configuration at startup and will fail fast with clear error messages if configuration is invalid.

### Required Configuration

The following must be configured (either explicitly or via defaults):

- ✅ `llm_endpoint` - LLM service endpoint
- ✅ `vector_db_url` - Vector database URL
- ✅ `cache_dir` - Cache directory (must be writable)

### Validation Rules

**Port:**
- Must be between 1 and 65535

**LLM Provider:**
- Must be `ollama` or `openai-compatible`

**LLM Endpoint:**
- Must start with `http://` or `https://`
- Cannot be empty

**Vector DB URL:**
- Must start with `http://` or `https://`
- Cannot be empty

**Embedding Provider:**
- Must be `local`, `ollama`, or `openai-compatible`

**Embedding Device:**
- Must be `cpu`, `cuda`, or `mps`

**Embedding Endpoint:**
- Required when `embedding_provider` is `ollama` or `openai-compatible`
- Must start with `http://` or `https://`

**Log Level:**
- Must be `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`

**Log Format:**
- Must be `json` or `text`

**Numeric Values:**
- `workers` must be >= 1
- `cache_size_mb` must be >= 1
- `max_concurrent_requests` must be >= 1
- `request_timeout_seconds` must be >= 1
- `session_timeout_minutes` must be >= 1

## Example Configurations

### Development (Local Everything)

```yaml
host: "127.0.0.1"  # Localhost only
port: 8080
llm_provider: "ollama"
llm_endpoint: "http://localhost:11434"
llm_model: "gemma3"
vector_db_url: "http://localhost:6333"
embedding_provider: "local"
embedding_model: "all-MiniLM-L6-v2"
embedding_device: "cpu"
log_level: "DEBUG"
log_format: "text"
```

### Production (Self-Hosted)

```yaml
host: "0.0.0.0"
port: 8080
workers: 8
llm_provider: "ollama"
llm_endpoint: "http://ollama-server:11434"
llm_model: "gemma3"
vector_db_url: "http://qdrant-server:6333"
embedding_provider: "ollama"
embedding_endpoint: "http://ollama-server:11434"
embedding_model: "qwen3-embedding"
cache_dir: "/var/cache/morgan"
cache_size_mb: 5000
log_level: "INFO"
log_format: "json"
max_concurrent_requests: 200
```

### Production (Cloud Services)

```yaml
host: "0.0.0.0"
port: 8080
workers: 4
llm_provider: "openai-compatible"
llm_endpoint: "https://api.openai.com/v1"
llm_model: "gpt-4"
llm_api_key: "${OPENAI_API_KEY}"
vector_db_url: "https://qdrant-cloud.example.com:6333"
vector_db_api_key: "${QDRANT_API_KEY}"
embedding_provider: "openai-compatible"
embedding_endpoint: "https://api.openai.com/v1"
embedding_model: "text-embedding-3-small"
embedding_api_key: "${OPENAI_API_KEY}"
log_level: "INFO"
log_format: "json"
```

### Docker Compose

```yaml
version: '3.8'

services:
  morgan-server:
    image: morgan-server:latest
    ports:
      - "8080:8080"
    environment:
      - MORGAN_HOST=0.0.0.0
      - MORGAN_PORT=8080
      - MORGAN_LLM_PROVIDER=ollama
      - MORGAN_LLM_ENDPOINT=http://ollama:11434
      - MORGAN_LLM_MODEL=gemma3
      - MORGAN_VECTOR_DB_URL=http://qdrant:6333
      - MORGAN_EMBEDDING_PROVIDER=ollama
      - MORGAN_EMBEDDING_ENDPOINT=http://ollama:11434
      - MORGAN_EMBEDDING_MODEL=qwen3-embedding
      - MORGAN_LOG_LEVEL=INFO
      - MORGAN_LOG_FORMAT=json
    volumes:
      - morgan-cache:/app/data/cache
    depends_on:
      - ollama
      - qdrant

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

volumes:
  morgan-cache:
  ollama-models:
  qdrant-data:
```

## Configuration Precedence Example

Given:

**config.yaml:**
```yaml
port: 8080
llm_model: "gemma3"
```

**Environment:**
```bash
export MORGAN_PORT=9000
```

**Result:**
- `port` = `9000` (from environment, highest precedence)
- `llm_model` = `"gemma3"` (from config file)
- `host` = `"0.0.0.0"` (from default)

## Troubleshooting

### Configuration Not Loading

**Problem:** Configuration file not being read

**Solution:**
- Check file path is correct
- Verify file format (YAML, JSON, .env)
- Check file permissions
- Use `--config` flag explicitly

### Environment Variables Not Working

**Problem:** Environment variables not overriding config

**Solution:**
- Ensure variables have `MORGAN_` prefix
- Check variable names match exactly (case-insensitive)
- Verify variables are exported: `export MORGAN_PORT=8080`
- Check for typos in variable names

### Validation Errors

**Problem:** Server fails to start with validation error

**Solution:**
- Read error message carefully
- Check configuration values against validation rules
- Verify URLs start with `http://` or `https://`
- Ensure numeric values are in valid ranges
- Check enum values match exactly

### Cache Directory Errors

**Problem:** Cannot create cache directory

**Solution:**
- Check directory permissions
- Ensure parent directories exist
- Use absolute path if relative path fails
- Check disk space

## Security Considerations

### API Keys

- Never commit API keys to version control
- Use environment variables for sensitive data
- Use secrets management in production (e.g., Kubernetes Secrets, AWS Secrets Manager)

### Network Security

- Bind to `127.0.0.1` for local-only access
- Use `0.0.0.0` only when needed for remote access
- Use HTTPS for remote endpoints
- Configure firewall rules appropriately

### File Permissions

- Restrict cache directory permissions: `chmod 700 data/cache`
- Protect configuration files: `chmod 600 config.yaml`
- Run server as non-root user in production

## Further Reading

- [Embedding Configuration Guide](./EMBEDDING_CONFIGURATION.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [API Documentation](./API.md)
