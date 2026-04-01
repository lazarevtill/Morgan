# Morgan Docker Deployment

Docker configurations for deploying Morgan in various setups.

## Quick Start (Single Machine)

```bash
# From repository root
cd docker

# Copy env template once
cp .env.example .env

# Start stack (server + redis + qdrant; remote Ollama from .env)
docker compose up -d

# Optional monitoring (Prometheus + Grafana)
docker compose --profile monitoring up -d

# Run CLI from container
docker compose run --rm morgan-cli chat
```

## Configuration

### Environment Variables

Copy and customize the environment file:

```bash
cp .env.example .env
# Edit with your values
vi .env
```

**Key variables:**
- `MORGAN_LLM_ENDPOINT` - Primary remote Ollama endpoint
- `MORGAN_EMBEDDING_ENDPOINT` - Remote Ollama endpoint for smaller/embedding models
- `MORGAN_LLM_MODEL` - Main LLM model name
- `MORGAN_ENABLE_CHANNELS` - enable Telegram/Discord/Synology adapters
- `HF_TOKEN` - Hugging Face API token (for gated model downloads)

### Hugging Face Token

Some models require authentication to download (gated models). To use them:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to your `.env` file:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Models that may require HF_TOKEN:
- Some Jina AI models (embeddings, rerankers)
- Gated community models
- Private organization models

**Note:** Most public models (like `all-MiniLM-L6-v2`, `nomic-embed-text`) 
work without a token.

### Distributed Architecture Configuration

Morgan uses YAML files for distributed architecture configuration.
Config files are mounted at `/app/config/` in the container.

**Config files:**
- `config/distributed.yaml` - Default single-machine config
- `config/distributed.6host.yaml` - Template for 6-host setup

**Customization:**

```bash
# Create local override
cp config/distributed.yaml config/distributed.local.yaml

# Edit with your IP addresses
vi config/distributed.local.yaml

# Set environment variable
export MORGAN_DISTRIBUTED_CONFIG=/app/config/distributed.local.yaml
```

## Deployment Options

### 1. Single Machine (Development)

Uses `docker-compose.yml`:
- Morgan Server
- Morgan CLI image (run via `docker compose run`)
- Ollama
- Redis (session/cache)
- Qdrant (vector DB)
- Prometheus/Grafana (optional)

```bash
docker compose up -d
```

### 2. Distributed (6-Host Production)

Uses `docker-compose.distributed.yml` with Docker Swarm:

**Architecture:**
| Host | IP | Role | Components |
|------|-----|------|------------|
| 1 | 192.168.1.10 | Core | Morgan, Qdrant, Redis |
| 2 | 192.168.1.11 | Background | Prometheus, Grafana |
| 3 | 192.168.1.20 | LLM Primary | Ollama (RTX 3090) |
| 4 | 192.168.1.21 | LLM Secondary | Ollama (RTX 3090) |
| 5 | 192.168.1.22 | Embeddings | Ollama (RTX 4070) |
| 6 | 192.168.1.23 | Reranking | Reranking Service (RTX 2060) |

**Setup:**

```bash
# Initialize Swarm on Host 1
docker swarm init --advertise-addr 192.168.1.10

# Join other hosts to swarm
docker swarm join --token <token> 192.168.1.10:2377

# Label nodes
docker node update --label-add morgan.role=core host1
docker node update --label-add morgan.role=background host2
docker node update --label-add morgan.role=llm-primary host3
docker node update --label-add morgan.role=llm-secondary host4
docker node update --label-add morgan.role=embeddings host5
docker node update --label-add morgan.role=reranking host6

# Copy and customize config
cp config/distributed.6host.yaml config/distributed.local.yaml
vi config/distributed.local.yaml

# Deploy stack
docker stack deploy -c docker-compose.distributed.yml morgan
```

## Services

### Morgan Server (Port 8080)

Main API server with endpoints:
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
- `POST /ask` - Question answering
- `GET /docs` - API documentation

### Morgan CLI (Containerized)

Run CLI commands through compose:

```bash
# interactive chat
docker compose run --rm morgan-cli chat

# one-off question
docker compose run --rm morgan-cli ask "hello morgan"
```

### Remote Ollama Endpoints

The stack uses external Ollama hosts configured in `.env`:

- Primary LLM host: `http://192.168.100.233:11434`
- Smaller/embedding host: `http://192.168.100.222:11434`

### Qdrant (Ports 6333, 6334)

Vector database:
- REST API: `http://localhost:6333`
- gRPC: `localhost:6334`
- Dashboard: `http://localhost:6333/dashboard`

### Redis (Port 6379)

Session and cache storage:
- Connect: `redis://localhost:6379`

### Prometheus (Port 9090) [Optional]

Metrics collection:
- Dashboard: `http://localhost:9090`

### Grafana (Port 3000) [Optional]

Metrics visualization:
- Dashboard: `http://localhost:3000`
- Default credentials: admin/morgan

## Health Checks

```bash
# Morgan Server
curl http://localhost:8080/health

# Qdrant
curl http://localhost:6333/healthz

# Redis
redis-cli ping
```

## Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f morgan-server

# Distributed deployment
docker service logs morgan_morgan-core -f
```

## Volumes

Data is persisted in Docker volumes:
- `morgan-cache` - Embedding and response cache
- `morgan-data` - Application data
- `morgan-models` - **Model weights cache** (see below)
- `redis-data` - Redis persistence
- `qdrant-storage` - Vector database storage
- `reranking-models` - Reranking model cache (distributed setup)

## Model Caching

**Model weights are downloaded once and cached persistently** to avoid 
re-downloading on each container restart.

### How It Works

1. On first startup, models are downloaded from Hugging Face
2. Models are saved to the `morgan-models` volume (`/app/models` in container)
3. On subsequent starts, cached models are loaded instantly

### Configuration

In `config/distributed.yaml`:

```yaml
model_cache:
  # Base directory for model weights
  base_dir: "/app/models"
  # sentence-transformers models (embeddings, rerankers)
  sentence_transformers_home: "/app/models/sentence-transformers"
  # Hugging Face models
  hf_home: "/app/models/huggingface"
  # Preload models on startup
  preload_on_startup: true
```

### Environment Variables

These are set automatically by the configuration:
- `SENTENCE_TRANSFORMERS_HOME` - sentence-transformers cache
- `HF_HOME` - Hugging Face cache
- `TRANSFORMERS_CACHE` - Transformers cache
- `MODEL_CACHE_DIR` - General model cache (for reranking service)

### First Run

The first startup may take several minutes while app services initialize.

**Ensure models exist on your remote Ollama hosts before starting:**
```bash
# Smaller/embedding node (192.168.100.222)
ollama pull qwen3-embedding:latest

# Primary LLM node (192.168.100.233)
ollama pull qwen3.5:35b
```

**Auto-downloaded Models:**
- Reranking models (CrossEncoder): ~100MB
- Fallback embedding models (sentence-transformers): ~90MB

### Clearing Cache

To re-download models, remove the volume:

```bash
# Stop containers
docker compose down

# Remove model cache volume
docker volume rm morgan-models

# Restart (will re-download)
docker compose up -d
```

## Troubleshooting

### Connection Refused to LLM

Ensure your remote Ollama hosts are reachable:
```bash
# Primary LLM endpoint
curl http://192.168.100.233:11434/api/tags

# Smaller/embedding endpoint
curl http://192.168.100.222:11434/api/tags
```

### GPU Not Detected

For GPU hosts, ensure NVIDIA Container Toolkit is installed:
```bash
# Install
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Memory Issues

Adjust container memory limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

## License

```
Copyright 2025 Morgan AI Assistant Contributors
Licensed under the Apache License, Version 2.0
```

See [LICENSE](../LICENSE) for the full license text.