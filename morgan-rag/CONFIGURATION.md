# Morgan v2-0.0.1 Configuration Guide

Complete guide to configuring Morgan RAG system for development, testing, and production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Files](#environment-files)
3. [Configuration Categories](#configuration-categories)
4. [Docker Compose Configurations](#docker-compose-configurations)
5. [Development Tools Configuration](#development-tools-configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Development Setup

```bash
# Copy development environment file
cp .env.development .env

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f morgan
```

### Production Setup

```bash
# Copy production environment file
cp .env.production .env

# IMPORTANT: Edit .env and change all passwords and secrets!
nano .env

# Start production services
docker-compose -f docker-compose.prod.yml up -d
```

### Testing

```bash
# Run all tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Run specific test suite
docker-compose -f docker-compose.test.yml run --rm unit-tests

# Run with coverage
docker-compose -f docker-compose.test.yml run --rm coverage-tests
```

---

## Environment Files

Morgan provides three pre-configured environment templates:

### `.env.example`
Complete reference with all available options, descriptions, and examples. Use this as documentation.

### `.env.development`
Development-friendly configuration with:
- Debug mode enabled
- Verbose logging
- Auto-reload on code changes
- Permissive security (localhost only!)
- Smaller resource limits
- Local model caching

### `.env.production`
Production-ready configuration with:
- Security hardened
- All secrets require replacement
- Optimized performance settings
- Restricted CORS
- Monitoring enabled
- TLS/SSL ready

**Never commit `.env` files to version control!**

---

## Configuration Categories

### 1. LLM Configuration

Controls the Language Model service and API connections.

```bash
# LLM API endpoint (OpenAI compatible)
LLM_BASE_URL=https://gpt.lazarev.cloud/ollama/v1

# API authentication key
LLM_API_KEY=your_api_key_here

# Model selection
LLM_MODEL=llama3.1:8b              # Primary model
LLM_CHAT_MODEL=llama3.1:8b         # Chat-specific model
LLM_EMBEDDING_MODEL=qwen3-embedding:latest  # Embeddings model
```

**Supported LLM Providers:**
- Ollama (local or remote)
- OpenAI API
- Azure OpenAI
- Any OpenAI-compatible API

**Model Recommendations:**
- **Small/Fast**: `llama3.1:8b`, `mistral:7b`
- **Balanced**: `gemma3:12b`, `llama3.1:70b`
- **High Quality**: `llama3.1:405b`, `gpt-4`

### 2. Database Configuration

#### PostgreSQL (Structured Data)

```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=morgan
POSTGRES_USER=morgan
POSTGRES_PASSWORD=CHANGE_ME_IN_PRODUCTION

# Alternative: Use connection URL
DATABASE_URL=postgresql://user:password@host:port/database
```

**Production Best Practices:**
- Use strong passwords (32+ characters)
- Enable SSL/TLS connections
- Regular backups (automated)
- Connection pooling
- Monitoring and alerting

#### Redis (Caching & Sessions)

```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=                    # Optional, recommended for production
REDIS_DB=0

# Alternative: Use connection URL
REDIS_URL=redis://[:password@]host:port/db

# Cache settings
MORGAN_CACHE_SIZE=1000             # In-memory cache size
MORGAN_CACHE_TTL=3600              # Cache TTL in seconds
```

**Cache Strategy:**
- Development: Small cache (100-500 items)
- Production: Large cache (5000+ items)
- TTL: 1-2 hours for most use cases

#### Qdrant (Vector Database)

```bash
QDRANT_HOST=qdrant
QDRANT_PORT=6333                   # HTTP API
QDRANT_GRPC_PORT=6334              # gRPC API

# Alternative: Use URL
QDRANT_URL=http://qdrant:6333

# Security
QDRANT_API_KEY=                    # Optional, recommended for production

# Collections
QDRANT_DEFAULT_COLLECTION=morgan_knowledge
QDRANT_MEMORY_COLLECTION=morgan_memory

# Logging
QDRANT_LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
```

### 3. Embedding Configuration

Controls how documents are converted to vector embeddings.

```bash
# Primary embedding model (remote, high quality)
EMBEDDING_MODEL=qwen3-embedding:latest

# Fallback model (local, fast)
EMBEDDING_LOCAL_MODEL=all-MiniLM-L6-v2

# Force remote even in dev mode
EMBEDDING_FORCE_REMOTE=false

# Batch processing
EMBEDDING_BATCH_SIZE=100           # 50-100 for dev, 100-200 for prod

# Hardware acceleration
EMBEDDING_DEVICE=cpu               # cpu, cuda, mps

# Quality improvements
EMBEDDING_USE_INSTRUCTIONS=true    # +22% relevance improvement
```

**Embedding Model Comparison:**

| Model | Dimensions | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| `qwen3-embedding:latest` | 1024 | Excellent | Medium | Production |
| `nomic-embed-text` | 768 | Good | Fast | General use |
| `all-MiniLM-L6-v2` | 384 | Fair | Very Fast | Development |
| `all-mpnet-base-v2` | 768 | Good | Medium | Balanced |

### 4. System Configuration

```bash
# Data storage
MORGAN_DATA_DIR=/app/data          # Docker: /app/data, Local: ./data
MORGAN_CONFIG_DIR=/app/config      # Configuration directory

# Logging
MORGAN_LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
MORGAN_DEBUG=false                 # Enable debug mode

# LLM limits
MORGAN_MAX_CONTEXT=8192            # Match your model's context window
MORGAN_MAX_RESPONSE_TOKENS=2048    # Maximum response length

# Performance
MORGAN_WORKERS=4                   # 2-4 dev, 4-8 production
MORGAN_MAX_SEARCH_RESULTS=50       # Maximum search results
MORGAN_DEFAULT_SEARCH_RESULTS=10   # Default search results
```

**Log Levels:**
- `DEBUG`: Verbose logging for development (slowest)
- `INFO`: Standard logging for production
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors and critical issues
- `CRITICAL`: Only critical failures

### 5. Security Settings

```bash
# API authentication
MORGAN_API_KEY=                    # Generate with: openssl rand -hex 32

# CORS (Cross-Origin Resource Sharing)
MORGAN_CORS_ORIGINS=*              # Development: *, Production: specific domains

# Feature flags
MORGAN_ALLOW_FILE_UPLOAD=true
MORGAN_ALLOW_URL_INGESTION=true
MORGAN_ALLOW_CODE_EXECUTION=false  # DANGER: Only enable if absolutely needed

# Session management
MORGAN_SESSION_SECRET=CHANGE_ME    # Generate with: openssl rand -hex 32
```

**Security Checklist for Production:**

- [ ] Change all default passwords
- [ ] Generate strong API keys (32+ chars)
- [ ] Restrict CORS to specific domains
- [ ] Disable code execution
- [ ] Enable HTTPS/TLS
- [ ] Use environment-specific secrets
- [ ] Enable rate limiting
- [ ] Set up firewall rules
- [ ] Regular security audits

### 6. Web Interface & API

```bash
# Server configuration
MORGAN_HOST=0.0.0.0                # 0.0.0.0: all interfaces, 127.0.0.1: localhost only
MORGAN_PORT=8080                   # Web UI port
MORGAN_API_PORT=8000               # REST API port

# Features
MORGAN_WEB_ENABLED=true            # Enable web interface
MORGAN_DOCS_ENABLED=true           # API documentation (disable in production)
MORGAN_REQUEST_LOGGING=false       # Log all requests (enable in production)
```

**Port Usage:**
- `8080`: Web interface (Streamlit/Gradio)
- `8000`: REST API (FastAPI)
- `9000`: Prometheus metrics (monitoring)
- `6333`: Qdrant HTTP API
- `6379`: Redis
- `5432`: PostgreSQL

### 7. Document Ingestion

```bash
# Chunking strategy
MORGAN_CHUNK_SIZE=1000             # Characters per chunk
MORGAN_CHUNK_OVERLAP=200           # Overlap for context preservation

# Upload limits
MORGAN_MAX_FILE_SIZE=100           # Maximum file size in MB

# Supported formats
MORGAN_SUPPORTED_TYPES=pdf,docx,txt,md,html,py,js,ts,go,java,cpp,c,h,json,yaml,yml

# Web scraping
MORGAN_SCRAPE_DEPTH=3              # Maximum crawl depth
MORGAN_SCRAPE_DELAY=1              # Delay between requests (seconds)
MORGAN_SCRAPE_TIMEOUT=30           # Request timeout (seconds)
```

**Chunking Guidelines:**

| Chunk Size | Overlap | Use Case |
|-----------|---------|----------|
| 500-800 | 100-150 | Code, technical docs |
| 1000-1500 | 200-300 | General documents, articles |
| 2000-3000 | 400-600 | Long-form content, books |

### 8. Memory & Learning

```bash
# Conversation memory
MORGAN_MEMORY_ENABLED=true
MORGAN_MEMORY_MAX_CONVERSATIONS=1000
MORGAN_MEMORY_MAX_TURNS_PER_CONVERSATION=100

# Learning system
MORGAN_LEARNING_ENABLED=true
MORGAN_FEEDBACK_WEIGHT=0.1         # 0.0-1.0, how much to weight user feedback

# Knowledge graph
MORGAN_KNOWLEDGE_GRAPH_ENABLED=true
MORGAN_ENTITY_EXTRACTION_ENABLED=true
```

### 9. Feature Flags

```bash
# Core features
MORGAN_ENABLE_MEMORY=true          # Enable conversation memory
MORGAN_ENABLE_TOOLS=true           # Enable tool use/function calling
MORGAN_ENABLE_PROMETHEUS=true      # Enable metrics collection
```

### 10. Monitoring & Analytics

```bash
# Metrics collection
MORGAN_METRICS_ENABLED=true
MORGAN_METRICS_PORT=9000           # Prometheus metrics endpoint

# Health checks
MORGAN_HEALTH_CHECK_INTERVAL=30    # Health check interval (seconds)

# Data retention
MORGAN_ANALYTICS_RETENTION=90      # Days to retain analytics data
```

**Monitoring Stack:**
- Prometheus: Metrics collection
- Grafana: Visualization
- Logs: JSON structured logging

### 11. CUDA & GPU Configuration

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0             # Single GPU: 0, Multi: 0,1,2,3, CPU: -1

# CUDA paths
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# Optimization
CUDA_LAUNCH_BLOCKING=0             # 0: async (fast), 1: sync (debug)
TORCH_USE_CUDA_DSA=1               # Enable device-side assertions
```

**GPU Recommendations:**
- TTS Service: Requires GPU (CSM model)
- STT Service: Requires GPU (Faster Whisper)
- Embeddings: CPU acceptable for small batches
- LLM: CPU if using remote API

### 12. HuggingFace Configuration

```bash
# Authentication
HF_TOKEN=your_token_here           # Get from: https://huggingface.co/settings/tokens
HUGGINGFACE_HUB_TOKEN=your_token_here

# Cache directories
HF_HOME=/root/.cache/huggingface
HF_HUB_CACHE=/root/.cache/huggingface/hub
HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
```

**When You Need HF Token:**
- Downloading private models
- Accessing gated models (e.g., Llama)
- Higher rate limits
- Early access to new models

### 13. Development Settings

```bash
# Development mode
MORGAN_DEV_MODE=false              # Enable development features
MORGAN_AUTO_RELOAD=false           # Auto-reload on code changes

# Python configuration
PYTHONPATH=/app
PYTHONUNBUFFERED=1                 # Disable output buffering
PYTHONDONTWRITEBYTECODE=1          # Don't create .pyc files

# Proxy configuration
BEHIND_PROXY=false                 # Set true if behind reverse proxy
```

---

## Docker Compose Configurations

### Standard Configuration (`docker-compose.yml`)

Basic setup with all core services. Suitable for:
- Local development
- Small-scale deployments
- Testing

**Services:**
- Morgan RAG
- Qdrant (vector DB)
- Redis (cache)
- PostgreSQL (optional)

**Usage:**
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build
```

### Development Configuration (`docker-compose.dev.yml`)

Development-optimized setup with:
- Source code volume mounts (hot reload)
- Debug mode enabled
- Verbose logging
- Local model caching
- Development tools

**Usage:**
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Follow logs
docker-compose -f docker-compose.dev.yml logs -f morgan

# Rebuild after dependency changes
docker-compose -f docker-compose.dev.yml up -d --build
```

**Features:**
- Auto-reload on code changes
- Mounted source directories
- Debug port exposed (5678)
- Smaller resource limits

### Production Configuration (`docker-compose.prod.yml`)

Production-ready setup with:
- Security hardening
- Resource limits
- Monitoring stack
- High availability
- Automated backups

**Services:**
- Morgan RAG (optimized)
- Qdrant (with persistence)
- Redis (with authentication)
- PostgreSQL (with backups)
- Nginx (reverse proxy)
- Prometheus (monitoring)
- Grafana (dashboards)

**Usage:**
```bash
# IMPORTANT: Configure .env.production first!
cp .env.production .env
nano .env  # Change all REPLACE_WITH_* values

# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Backup databases
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U morgan morgan > backup.sql
```

**Resource Limits (Production):**
- Morgan: 8 CPU, 8GB RAM
- Qdrant: 2 CPU, 4GB RAM
- Redis: 1 CPU, 2GB RAM
- PostgreSQL: 2 CPU, 2GB RAM

### Testing Configuration (`docker-compose.test.yml`)

Isolated testing environment with:
- Ephemeral databases
- Parallel test execution
- Coverage reporting
- Security scanning

**Test Runners:**
- `test-runner`: All tests
- `unit-tests`: Unit tests only
- `integration-tests`: Integration tests
- `coverage-tests`: Tests with coverage
- `lint`: Code linting
- `type-check`: Type checking
- `security-scan`: Security scanning

**Usage:**
```bash
# Run all tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Run specific test suite
docker-compose -f docker-compose.test.yml run --rm unit-tests

# Run with coverage
docker-compose -f docker-compose.test.yml run --rm coverage-tests

# Lint code
docker-compose -f docker-compose.test.yml run --rm lint

# Type check
docker-compose -f docker-compose.test.yml run --rm type-check

# Security scan
docker-compose -f docker-compose.test.yml run --rm security-scan

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

---

## Development Tools Configuration

### Ruff Linting (`config/ruff.toml`)

Fast Python linter and formatter.

**Key Settings:**
- Line length: 120 characters
- Target: Python 3.11+
- Rules: pycodestyle, pyflakes, isort, pyupgrade, bugbear, and more

**Usage:**
```bash
# Lint code
ruff check morgan tests

# Auto-fix issues
ruff check --fix morgan tests

# Format code
ruff format morgan tests

# Check formatting
ruff format --check morgan tests
```

**Pre-commit Hook:**
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Pytest Testing (`config/pytest.ini`)

Comprehensive test configuration.

**Features:**
- Verbose output
- Coverage reporting
- Asyncio support
- Custom markers
- Structured logging

**Usage:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py

# Run with coverage
pytest --cov=morgan --cov-report=html

# Run marked tests
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

**Custom Markers:**
- `unit`: Fast, isolated unit tests
- `integration`: Integration tests (requires services)
- `slow`: Slow tests
- `llm`: Requires LLM service
- `cuda`: Requires GPU

### MyPy Type Checking (`config/mypy.ini`)

Static type checker for Python.

**Features:**
- Python 3.11 target
- Incremental mode
- Third-party stubs
- Per-module overrides

**Usage:**
```bash
# Check all files
mypy morgan

# Check specific file
mypy morgan/embeddings.py

# Generate HTML report
mypy --html-report reports morgan
```

---

## Best Practices

### Environment Management

1. **Never commit `.env` files**
   ```bash
   # Add to .gitignore
   .env
   .env.*
   !.env.example
   ```

2. **Use environment-specific files**
   - Development: `.env.development`
   - Staging: `.env.staging`
   - Production: `.env.production`

3. **Generate strong secrets**
   ```bash
   # API keys (32 chars)
   openssl rand -hex 32

   # Session secrets (64 chars)
   openssl rand -hex 64

   # Passwords (recommended: password manager)
   ```

4. **Use Docker secrets in production**
   ```yaml
   secrets:
     postgres_password:
       file: ./secrets/postgres_password.txt
   ```

### Security Hardening

1. **Database Security**
   - Strong passwords (32+ characters)
   - Limit network access
   - Enable SSL/TLS
   - Regular backups
   - Access logging

2. **API Security**
   - Enable API key authentication
   - Rate limiting
   - CORS restrictions
   - Input validation
   - SQL injection prevention

3. **Container Security**
   - Run as non-root user
   - Read-only file systems
   - Security scanning
   - Regular updates
   - Resource limits

### Performance Optimization

1. **Caching Strategy**
   ```bash
   # Development
   MORGAN_CACHE_SIZE=100
   MORGAN_CACHE_TTL=600

   # Production
   MORGAN_CACHE_SIZE=5000
   MORGAN_CACHE_TTL=7200
   ```

2. **Worker Scaling**
   ```bash
   # Development
   MORGAN_WORKERS=2

   # Production (CPU cores * 2)
   MORGAN_WORKERS=8
   ```

3. **Database Connection Pooling**
   ```python
   # PostgreSQL
   max_connections=100
   shared_buffers=256MB

   # Redis
   maxclients=10000
   ```

### Monitoring

1. **Enable Metrics**
   ```bash
   MORGAN_METRICS_ENABLED=true
   MORGAN_METRICS_PORT=9000
   ```

2. **Prometheus Scraping**
   ```yaml
   scrape_configs:
     - job_name: 'morgan'
       static_configs:
         - targets: ['morgan:9000']
   ```

3. **Grafana Dashboards**
   - Request rate
   - Response time
   - Error rate
   - Cache hit rate
   - Resource usage

### Backup Procedures

1. **PostgreSQL Backups**
   ```bash
   # Daily backup
   docker-compose exec postgres pg_dump -U morgan morgan > backup_$(date +%Y%m%d).sql

   # Restore
   docker-compose exec -T postgres psql -U morgan morgan < backup.sql
   ```

2. **Qdrant Backups**
   ```bash
   # Snapshot
   curl -X POST http://localhost:6333/collections/morgan_knowledge/snapshots

   # Download snapshot
   curl -o snapshot.tar http://localhost:6333/collections/morgan_knowledge/snapshots/{snapshot_name}
   ```

3. **Volume Backups**
   ```bash
   # Backup volume
   docker run --rm -v morgan_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data.tar.gz /data

   # Restore volume
   docker run --rm -v morgan_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_data.tar.gz -C /
   ```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused Errors

**Symptom:** Services can't connect to each other

**Solution:**
```bash
# Check network
docker network ls
docker network inspect morgan-network

# Restart services
docker-compose restart

# Check service names match
# Use service names from docker-compose.yml, not localhost
QDRANT_URL=http://qdrant:6333  # ✓ Correct
QDRANT_URL=http://localhost:6333  # ✗ Wrong in Docker
```

#### 2. Permission Denied Errors

**Symptom:** Cannot write to volumes

**Solution:**
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) ./data ./logs

# Or run with user
docker-compose up -d --user $(id -u):$(id -g)
```

#### 3. Out of Memory Errors

**Symptom:** Services crash with OOM

**Solution:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Reduce workers
MORGAN_WORKERS=2

# Reduce cache
MORGAN_CACHE_SIZE=500

# Add resource limits
docker-compose -f docker-compose.prod.yml up -d
```

#### 4. Slow Embedding Performance

**Symptom:** Slow document ingestion

**Solution:**
```bash
# Use GPU if available
EMBEDDING_DEVICE=cuda

# Increase batch size
EMBEDDING_BATCH_SIZE=200

# Use faster model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Enable remote embeddings
EMBEDDING_FORCE_REMOTE=true
```

#### 5. Database Connection Pool Exhausted

**Symptom:** "too many connections" error

**Solution:**
```bash
# Increase PostgreSQL connections
max_connections=200

# Reduce Morgan workers
MORGAN_WORKERS=4

# Add connection pooling
# Use pgbouncer or similar
```

### Debug Mode

Enable verbose logging:
```bash
MORGAN_LOG_LEVEL=DEBUG
MORGAN_DEBUG=true
QDRANT_LOG_LEVEL=DEBUG
```

View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f morgan

# Last 100 lines
docker-compose logs --tail=100 morgan
```

### Health Checks

Check service health:
```bash
# Morgan
curl http://localhost:8080/health

# Qdrant
curl http://localhost:6333/health

# Redis
docker-compose exec redis redis-cli ping

# PostgreSQL
docker-compose exec postgres pg_isready -U morgan
```

---

## Additional Resources

- [Morgan Documentation](./README.md)
- [API Documentation](http://localhost:8080/docs) (when running)
- [Deployment Guide](./DEPLOYMENT.md)
- [Contributing Guide](./CONTRIBUTING.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/morgan/issues
- Documentation: https://morgan-docs.example.com
- Community: https://discord.gg/morgan

---

**Last Updated:** 2025-01-08
**Version:** v2-0.0.1
