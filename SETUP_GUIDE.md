# Morgan v2-0.0.1 - Complete Setup Guide

**Version**: 2.0.0-alpha.1
**Last Updated**: 2025-11-08
**Status**: Production-Ready Core (85% Complete)

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Prerequisites](#prerequisites)
4. [Installation Methods](#installation-methods)
5. [Environment Configuration](#environment-configuration)
6. [Database Setup](#database-setup)
7. [Service Configuration](#service-configuration)
8. [First-Time Setup](#first-time-setup)
9. [Verification Steps](#verification-steps)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

---

## Overview

Morgan is a human-first AI assistant with emotional intelligence, built on RAG (Retrieval-Augmented Generation) principles. This guide will walk you through setting up Morgan v2-0.0.1 from scratch.

### What You'll Set Up

- **Core RAG System**: Document ingestion, vector search, and intelligent retrieval
- **Emotional Intelligence**: Emotion detection, empathy engine, and adaptive responses
- **Learning System**: Pattern recognition and user preference learning
- **Memory System**: Conversation history and relationship tracking
- **Vector Database**: Qdrant for semantic search
- **Caching Layer**: Redis for performance optimization
- **CLI**: Interactive command-line tools

---

## System Requirements

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|------------|-------|
| **OS** | Linux, macOS, Windows 10+ | WSL2 recommended for Windows |
| **Python** | 3.11 or higher | Python 3.12 recommended |
| **RAM** | 8 GB | 16 GB recommended for production |
| **Storage** | 10 GB free | More for large document collections |
| **CPU** | 4 cores | 8+ cores recommended |

### Recommended Requirements (Production)

| Component | Requirement | Purpose |
|-----------|------------|---------|
| **RAM** | 16-32 GB | Better embedding performance |
| **Storage** | 100+ GB SSD | Fast vector database operations |
| **CPU** | 8+ cores | Parallel processing |
| **GPU** | NVIDIA GPU (optional) | Faster embeddings (CUDA support) |
| **Network** | 100 Mbps+ | For distributed setup |

### GPU Support (Optional)

For faster embeddings and LLM inference:

| GPU | VRAM | Use Case |
|-----|------|----------|
| RTX 3060 | 12 GB | Development |
| RTX 3090 | 24 GB | Production (single node) |
| RTX 4070+ | 8+ GB | Embeddings |
| A100 | 40-80 GB | Enterprise production |

---

## Prerequisites

### 1. Install Python 3.11+

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

#### macOS
```bash
brew install python@3.11
```

#### Windows
Download from [python.org](https://www.python.org/downloads/) or use Windows Store.

### 2. Install Docker & Docker Compose

#### Linux
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
sudo apt install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from docker.com
```

#### Windows
Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)

### 3. Install Git
```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git

# Windows - download from git-scm.com
```

### 4. Install Additional System Dependencies

#### Ubuntu/Debian
```bash
sudo apt install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    libpq-dev \
    redis-tools
```

#### macOS
```bash
brew install \
    poppler \
    tesseract \
    postgresql \
    redis
```

---

## NetBird VPN Setup

### Important: NetBird VPN is Required

NetBird VPN is required for accessing the **private Nexus repository** (nexus.in.lazarev.cloud) used for installing Morgan dependencies. Without NetBird, dependency installation will fail.

### Why NetBird?

- **Private Registry Access**: Nexus repository is only accessible through the NetBird VPN
- **Secure Communication**: All package downloads are encrypted and secured
- **Network Isolation**: Internal services are protected behind the VPN

### 1. Get a NetBird Setup Key

You'll need a NetBird setup key to connect to the VPN. Contact your system administrator to obtain:
- **NetBird Setup Key**: A unique token for authentication
- **VPN Management URL**: https://vpn.lazarev.cloud (standard for Morgan infrastructure)

### 2. Install NetBird

#### Linux (Ubuntu/Debian)

```bash
# Download and run installer
curl -fsSL https://pkgs.netbird.io/install.sh | sh

# Verify installation
netbird version
```

#### macOS

```bash
# Install via Homebrew
brew install netbird

# Or download from: https://releases.netbird.io/
```

#### Windows

Download installer from: https://releases.netbird.io/

### 3. Connect to VPN

```bash
# Connect to NetBird VPN
netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-setup-key>

# Example:
# netbird up --management-url https://vpn.lazarev.cloud --setup-key AKIA2E5F7B9D1C3F6H8J
```

### 4. Verify Connection

```bash
# Check VPN status
netbird status

# Expected output:
# NetBird daemon is running
# Connected to management:  true
# Connected to signal:      true
# Connected to relay:       true
# Management Address: https://vpn.lazarev.cloud:33073
# Signal Address: signal.netbird.io:10000
# Relay Address: relay.netbird.io
# Interface name: wt0
# ...
# IP: 100.xx.xx.xx

# Verify you can reach Nexus
curl -I https://nexus.in.lazarev.cloud
# Should return a response (not connection refused)
```

### 5. Troubleshooting NetBird Connection

#### Connection Refused
```bash
# If you get "connection refused" to Nexus:

# 1. Check NetBird is running
netbird status

# 2. Verify setup key is correct
# 3. Check internet connection
# 4. Try reconnecting
netbird down
netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-setup-key>

# 5. Check NetBird logs (if using systemd)
sudo journalctl -u netbird -f
```

#### Persistent Issues
```bash
# Restart NetBird service
sudo systemctl restart netbird  # Linux with systemd

# Or restart the service manually on other platforms
netbird down
netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-setup-key>
```

### Keep VPN Connected During Setup

**Important**: Keep NetBird VPN connected throughout the entire installation process. The following commands require Nexus access:

1. Creating Python virtual environment
2. Installing dependencies (`pip install -r requirements.txt`)
3. Building Docker images
4. Running tests

If you disconnect during installation, you'll need to restart the process.

---

## Installation Methods

Choose one of the following installation methods:

### Method 1: Docker Compose (Recommended for Quick Start)

**Pros**: Fastest setup, isolated environment, includes all services
**Cons**: Less control, requires Docker

```bash
# 1. Clone repository
git clone <repository-url> morgan
cd morgan
git checkout v2-0.0.1

# 2. Navigate to morgan-rag directory
cd morgan-rag

# 3. Copy environment template
cp .env.example .env

# 4. Edit .env file (see Environment Configuration section)
nano .env

# 5. Start all services
docker compose up -d

# 6. Check status
docker compose ps

# 7. View logs
docker compose logs -f morgan
```

### Method 2: Local Installation (Recommended for Development)

**Pros**: Full control, easier debugging, faster iteration
**Cons**: Manual setup of dependencies

```bash
# 1. Clone repository
git clone <repository-url> morgan
cd morgan
git checkout v2-0.0.1
cd morgan-rag

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template
cp .env.example .env

# 6. Edit .env file
nano .env

# 7. Start external services (Qdrant, Redis)
# Option A: Using Docker
docker compose up -d qdrant redis

# Option B: Using local installations
# Start Qdrant (separate terminal)
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant:latest

# Start Redis (separate terminal)
redis-server

# 8. Verify installation
python -m morgan health
```

### Method 3: Production Docker Build

**Pros**: Production-optimized, multi-stage build, smaller image
**Cons**: Requires Docker, longer build time

```bash
# 1. Clone and setup
git clone <repository-url> morgan
cd morgan
git checkout v2-0.0.1
cd morgan-rag

# 2. Build production image
docker build -t morgan-rag:v2.0.0 -f Dockerfile .

# 3. Copy and configure environment
cp .env.example .env
nano .env

# 4. Start with docker compose
docker compose -f docker-compose.yml up -d

# 5. Or run standalone
docker run -d \
    --name morgan-rag \
    -p 8080:8080 \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    --env-file .env \
    morgan-rag:v2.0.0
```

---

## Environment Configuration

### 1. Copy Environment Template

```bash
cd morgan-rag
cp .env.example .env
```

### 2. Configure Required Settings

Edit `.env` file with your specific configuration:

#### LLM Configuration (Required)

```bash
# Your LLM endpoint (OpenAI compatible)
# Examples:
# - OpenAI: https://api.openai.com/v1
# - Ollama: http://localhost:11434/v1
# - Custom: https://your-llm-endpoint.com/v1
LLM_BASE_URL=https://gpt.lazarev.cloud/ollama/v1

# API key for your LLM service
LLM_API_KEY=sk-your-api-key-here

# Model to use
LLM_MODEL=llama3.1:8b

# Optional: Alternative models for different tasks
LLM_CHAT_MODEL=llama3.1:8b
LLM_EMBEDDING_MODEL=nomic-embed-text
```

#### Vector Database (Qdrant)

```bash
# Qdrant connection
# - Docker: http://localhost:6333
# - Docker Compose: http://qdrant:6333
# - Remote: https://your-qdrant-instance.com
QDRANT_URL=http://localhost:6333

# Optional API key for Qdrant (leave empty for local)
QDRANT_API_KEY=

# Collection names
QDRANT_DEFAULT_COLLECTION=morgan_knowledge
QDRANT_MEMORY_COLLECTION=morgan_memory
```

#### Embedding Configuration

```bash
# Primary embedding model (remote or local)
EMBEDDING_MODEL=nomic-embed-text

# Local fallback model (when primary unavailable)
EMBEDDING_LOCAL_MODEL=all-MiniLM-L6-v2

# Device for local embeddings (cpu, cuda, mps)
EMBEDDING_DEVICE=cpu

# Batch size for embedding operations
EMBEDDING_BATCH_SIZE=100

# Use instruction prefixes for better relevance
EMBEDDING_USE_INSTRUCTIONS=true
```

#### System Configuration

```bash
# Data directory (where Morgan stores data)
MORGAN_DATA_DIR=./data

# Log level (DEBUG, INFO, WARNING, ERROR)
MORGAN_LOG_LEVEL=INFO

# Enable debug mode
MORGAN_DEBUG=false

# Maximum context length for LLM
MORGAN_MAX_CONTEXT=8192

# Maximum tokens for responses
MORGAN_MAX_RESPONSE_TOKENS=2048
```

#### Performance Settings

```bash
# Number of worker processes
MORGAN_WORKERS=4

# Cache size (number of items to cache in memory)
MORGAN_CACHE_SIZE=1000

# Redis cache TTL (seconds)
MORGAN_CACHE_TTL=3600

# Search result limits
MORGAN_MAX_SEARCH_RESULTS=50
MORGAN_DEFAULT_SEARCH_RESULTS=10
```

#### Security Settings

```bash
# API key for Morgan API (leave empty for no auth)
MORGAN_API_KEY=

# Allowed CORS origins (comma-separated)
MORGAN_CORS_ORIGINS=*

# Feature flags
MORGAN_ALLOW_FILE_UPLOAD=true
MORGAN_ALLOW_URL_INGESTION=true
MORGAN_ALLOW_CODE_EXECUTION=false
```

#### Optional: Redis Cache

```bash
# Redis for caching (leave empty to use in-memory cache)
# - Docker Compose: redis://redis:6379
# - Local: redis://localhost:6379
REDIS_URL=redis://localhost:6379
```

### 3. Validate Configuration

```bash
# Check configuration
python -m morgan config validate

# View current configuration
python -m morgan config show
```

---

## Database Setup

### 1. Qdrant Vector Database

#### Option A: Docker (Recommended)

```bash
# Start Qdrant with Docker
docker run -d \
    --name morgan-qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant:latest

# Verify it's running
curl http://localhost:6333/health
```

#### Option B: Docker Compose

Already included in `docker-compose.yml`:

```bash
docker compose up -d qdrant
```

#### Option C: Standalone Installation

```bash
# Download Qdrant binary (Linux)
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Run Qdrant
./qdrant --config-path ./config/production.yaml
```

#### Initialize Collections

```bash
# Initialize Qdrant collections for Morgan
python -m morgan init-db

# Or manually with setup script
cd morgan-rag
python setup_collections.py
```

### 2. Redis Cache (Optional but Recommended)

#### Option A: Docker

```bash
docker run -d \
    --name morgan-redis \
    -p 6379:6379 \
    -v $(pwd)/redis_data:/data \
    redis:latest redis-server --appendonly yes
```

#### Option B: Local Installation

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Verify
redis-cli ping  # Should return "PONG"
```

### 3. PostgreSQL (Optional - for advanced features)

```bash
# Create database
sudo -u postgres psql
CREATE DATABASE morgan_db;
CREATE USER morgan_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE morgan_db TO morgan_user;
\q

# Update .env
DATABASE_URL=postgresql://morgan_user:secure_password@localhost:5432/morgan_db
```

---

## Service Configuration

### 1. Create Data Directories

```bash
cd morgan-rag
mkdir -p data logs config knowledge

# Set permissions
chmod 755 data logs config knowledge
```

### 2. Configure Logging

Create `config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/morgan.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  morgan:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

### 3. Configure Web Server (Optional)

Create `config/server.yaml`:

```yaml
host: 0.0.0.0
port: 8080
workers: 4
timeout: 60
keepalive: 5
max_requests: 1000
max_requests_jitter: 100

# SSL/TLS (for production)
ssl:
  enabled: false
  cert_file: /path/to/cert.pem
  key_file: /path/to/key.pem
```

---

## First-Time Setup

### 1. Initialize Morgan

```bash
# Initialize system (creates directories, databases, config)
python -m morgan init

# Expected output:
# ✓ Created data directories
# ✓ Initialized Qdrant collections
# ✓ Created default configuration
# ✓ System ready!
```

### 2. Run Health Check

```bash
# Check system health
python -m morgan health

# Expected output:
# ✓ LLM Connection: OK
# ✓ Qdrant Connection: OK
# ✓ Redis Connection: OK
# ✓ Embeddings: OK
# ✓ System: Ready
```

### 3. Ingest Sample Data (Optional)

```bash
# Create sample knowledge base
python morgan-rag/create_sample_data.py

# Or manually add documents
python -m morgan learn ./docs --collection documentation
python -m morgan learn ./code --type code --collection codebase
```

### 4. Test Basic Functionality

```bash
# Test basic query
python -m morgan ask "What is Morgan?"

# Start interactive chat
python -m morgan chat

# Test web interface
python -m morgan serve
# Open browser to http://localhost:8080
```

---

## Verification Steps

### 1. Component Health Checks

```bash
# Detailed system status
python -m morgan status --detailed

# Check individual components
python -m morgan health --component llm
python -m morgan health --component qdrant
python -m morgan health --component redis
python -m morgan health --component embeddings
```

### 2. Test Document Ingestion

```bash
# Create test document
echo "Morgan is an AI assistant with emotional intelligence." > test_doc.txt

# Ingest document
python -m morgan learn test_doc.txt

# Verify ingestion
python -m morgan ask "What is Morgan?"
# Should return relevant information from test_doc.txt
```

### 3. Test Embeddings

```bash
# Test embedding generation
python -c "
from morgan.vector_db.embeddings import get_embedding_service

service = get_embedding_service()
text = 'Test embedding generation'
embedding = service.embed_text(text)
print(f'✓ Generated embedding with dimension: {len(embedding)}')
print(f'✓ Expected dimension: 768')
"
```

### 4. Test Vector Database

```bash
# Test Qdrant connection
python -c "
from qdrant_client import QdrantClient

client = QdrantClient(url='http://localhost:6333')
collections = client.get_collections().collections
print(f'✓ Connected to Qdrant')
print(f'✓ Collections: {[c.name for c in collections]}')
"
```

### 5. Test LLM Connection

```bash
# Test LLM endpoint
python -c "
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)

response = client.chat.completions.create(
    model=os.getenv('LLM_MODEL'),
    messages=[{'role': 'user', 'content': 'Say hello'}]
)

print(f'✓ LLM Response: {response.choices[0].message.content}')
"
```

### 6. Performance Benchmarks

```bash
# Run performance tests
python -m morgan benchmark --component search
python -m morgan benchmark --component embeddings
python -m morgan benchmark --component llm

# Expected results:
# - Search: < 200ms for typical query
# - Embeddings: < 100ms per text (batch)
# - LLM: Depends on endpoint (typically 1-5s)
```

---

## Troubleshooting

### Common Issues

#### 1. "Connection refused" to Qdrant

**Symptom**: `ConnectionError: Cannot connect to Qdrant at http://localhost:6333`

**Solutions**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check Qdrant logs
docker logs morgan-qdrant

# Restart Qdrant
docker restart morgan-qdrant

# Check Qdrant health
curl http://localhost:6333/health
```

#### 2. LLM Connection Errors

**Symptom**: `Failed to connect to LLM endpoint`

**Solutions**:
```bash
# Verify LLM endpoint is accessible
curl -I $LLM_BASE_URL

# Test with OpenAI client
python -c "
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)
print('Testing connection...')
response = client.chat.completions.create(
    model=os.getenv('LLM_MODEL'),
    messages=[{'role': 'user', 'content': 'test'}],
    max_tokens=5
)
print('✓ Connection successful')
"

# Check .env configuration
grep LLM_ .env
```

#### 3. Embedding Errors

**Symptom**: `Failed to generate embeddings`

**Solutions**:
```bash
# Check embedding model availability
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✓ Embedding model loaded successfully')
"

# Check CUDA availability (if using GPU)
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# Switch to CPU if GPU issues
# In .env: EMBEDDING_DEVICE=cpu
```

#### 4. Redis Connection Issues

**Symptom**: `Redis connection failed`

**Solutions**:
```bash
# Check if Redis is running
redis-cli ping

# Check Redis logs
docker logs morgan-redis

# Restart Redis
docker restart morgan-redis

# Test connection
redis-cli -h localhost -p 6379 ping
```

#### 5. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'pydantic_settings'`

**Solutions**:
```bash
# Install missing dependencies
pip install pydantic-settings

# Reinstall all requirements
pip install -r requirements.txt --force-reinstall

# Verify installation
pip list | grep pydantic
```

#### 6. Permission Errors

**Symptom**: `PermissionError: [Errno 13] Permission denied`

**Solutions**:
```bash
# Fix data directory permissions
chmod -R 755 data logs config

# Fix Docker volume permissions
docker compose down
sudo chown -R $USER:$USER qdrant_data redis_data
docker compose up -d
```

#### 7. Out of Memory

**Symptom**: `MemoryError` or system slowdown

**Solutions**:
```bash
# Reduce batch sizes in .env
EMBEDDING_BATCH_SIZE=32  # Default: 100
MORGAN_WORKERS=2  # Default: 4
MORGAN_CACHE_SIZE=500  # Default: 1000

# Monitor memory usage
python -m morgan status --detailed

# Use GPU for embeddings to free RAM
EMBEDDING_DEVICE=cuda  # If GPU available
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# In .env
MORGAN_DEBUG=true
MORGAN_LOG_LEVEL=DEBUG

# Run with verbose output
python -m morgan chat --verbose
```

### Getting Help

If you encounter issues not covered here:

1. Check logs: `tail -f logs/morgan.log`
2. Review error messages carefully
3. Consult documentation: `docs/`
4. Check GitHub issues: [GitHub Issues](https://github.com/your-repo/issues)
5. Join community: [Discord/Slack/Forum]

---

## Next Steps

### For Development

1. **Explore Examples**: Check `morgan-rag/examples/` for code samples
2. **Read Documentation**: See `docs/` for architecture details
3. **Run Tests**: `pytest tests/`
4. **Add Your Data**: `python -m morgan learn ./your-docs`

### For Production

1. **Review Security**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. **Configure Monitoring**: Set up Prometheus/Grafana
3. **Set Up Backups**: Configure automated backups
4. **Performance Tuning**: Optimize for your workload

### Learning Morgan

```bash
# Interactive tutorial
python -m morgan tutorial

# View all CLI commands
python -m morgan --help

# Explore specific command help
python -m morgan learn --help
python -m morgan chat --help
python -m morgan serve --help
```

### Distributed Setup

For multi-host GPU deployment:
- See [JARVIS_SETUP_GUIDE.md](JARVIS_SETUP_GUIDE.md) for self-hosted LLM setup
- See [DISTRIBUTED_SETUP_GUIDE.md](DISTRIBUTED_SETUP_GUIDE.md) for multi-host architecture

---

## Additional Resources

- **README**: [morgan-rag/README.md](morgan-rag/README.md)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Documentation**: [http://localhost:8080/docs](http://localhost:8080/docs) (when server is running)
- **Architecture Docs**: [docs/](docs/)
- **Examples**: [morgan-rag/examples/](morgan-rag/examples/)

---

**Setup Complete!** You're now ready to use Morgan. Start with:

```bash
python -m morgan chat
```

Have questions? Check the troubleshooting section or open an issue on GitHub.
