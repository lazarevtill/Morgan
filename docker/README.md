# Morgan Docker Deployment

This directory contains Docker configuration for deploying Morgan in a containerized environment.

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Remote Ollama instance running (or other OpenAI-compatible LLM endpoint)

### Basic Deployment

1. **Start the services:**

```bash
cd docker
docker-compose up -d
```

This will start:
- Morgan Server (port 8080)
- Qdrant vector database (ports 6333, 6334)

2. **Check service health:**

```bash
docker-compose ps
```

All services should show "healthy" status.

3. **View logs:**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f morgan-server
```

4. **Stop services:**

```bash
docker-compose down
```

### With Monitoring (Optional)

To include Prometheus monitoring:

```bash
docker-compose --profile monitoring up -d
```

Access Prometheus at: http://localhost:9090

## Configuration

### Environment Variables

Create a `.env` file in the `docker/` directory to customize configuration:

```bash
# LLM Configuration (required - remote Ollama)
MORGAN_LLM_ENDPOINT=http://your-ollama-host:11434
MORGAN_LLM_MODEL=gemma3
MORGAN_LLM_API_KEY=  # Optional for self-hosted

# Embedding Configuration
MORGAN_EMBEDDING_PROVIDER=local  # Options: local, ollama, openai-compatible
MORGAN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MORGAN_EMBEDDING_DEVICE=cpu  # For local embeddings
MORGAN_EMBEDDING_ENDPOINT=  # Required for remote providers
MORGAN_EMBEDDING_API_KEY=  # Optional for remote providers

# Cache Configuration
MORGAN_CACHE_SIZE_MB=1000

# Logging Configuration
MORGAN_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
MORGAN_LOG_FORMAT=json  # json or text

# Performance Configuration
MORGAN_MAX_CONCURRENT=100
MORGAN_REQUEST_TIMEOUT=60
MORGAN_SESSION_TIMEOUT=60

# Vector Database Configuration
MORGAN_VECTOR_DB_API_KEY=  # Optional
```

### Using Remote Ollama

The default configuration assumes Ollama is running on the host machine. To connect to it from Docker:

**On Linux:**
```bash
MORGAN_LLM_ENDPOINT=http://172.17.0.1:11434
```

**On macOS/Windows:**
```bash
MORGAN_LLM_ENDPOINT=http://host.docker.internal:11434
```

**On a remote server:**
```bash
MORGAN_LLM_ENDPOINT=http://your-server-ip:11434
```

## Services

### Morgan Server

- **Port:** 8080
- **Health Check:** http://localhost:8080/health
- **API Docs:** http://localhost:8080/docs
- **Metrics:** http://localhost:8080/metrics

### Qdrant Vector Database

- **HTTP Port:** 6333
- **gRPC Port:** 6334
- **Dashboard:** http://localhost:6333/dashboard
- **Health Check:** http://localhost:6333/healthz

### Prometheus (Optional)

- **Port:** 9090
- **Dashboard:** http://localhost:9090

## Data Persistence

Data is persisted in named Docker volumes:

- `morgan-cache`: Application cache
- `morgan-data`: Application data
- `morgan-qdrant-storage`: Vector database storage
- `morgan-qdrant-snapshots`: Vector database snapshots
- `morgan-prometheus-data`: Prometheus metrics (if enabled)

### Backup Data

```bash
# Backup Qdrant data
docker run --rm -v morgan-qdrant-storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant-backup.tar.gz -C /data .

# Backup Morgan data
docker run --rm -v morgan-data:/data -v $(pwd):/backup alpine tar czf /backup/morgan-backup.tar.gz -C /data .
```

### Restore Data

```bash
# Restore Qdrant data
docker run --rm -v morgan-qdrant-storage:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/qdrant-backup.tar.gz"

# Restore Morgan data
docker run --rm -v morgan-data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/morgan-backup.tar.gz"
```

## Troubleshooting

### Service Won't Start

1. Check logs:
```bash
docker-compose logs morgan-server
```

2. Verify configuration:
```bash
docker-compose config
```

3. Check if ports are available:
```bash
netstat -an | grep -E '8080|6333|9090'
```

### Can't Connect to Ollama

1. Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

2. Check Docker network connectivity:
```bash
docker-compose exec morgan-server curl http://host.docker.internal:11434/api/tags
```

3. Update `MORGAN_LLM_ENDPOINT` in `.env` file with correct host

### Qdrant Connection Issues

1. Check Qdrant health:
```bash
curl http://localhost:6333/healthz
```

2. View Qdrant logs:
```bash
docker-compose logs qdrant
```

### High Memory Usage

1. Reduce cache size in `.env`:
```bash
MORGAN_CACHE_SIZE_MB=500
```

2. Limit concurrent requests:
```bash
MORGAN_MAX_CONCURRENT=50
```

3. Use CPU for embeddings instead of GPU:
```bash
MORGAN_EMBEDDING_DEVICE=cpu
```

## Production Deployment

For production deployments:

1. **Use specific image tags** instead of `latest`
2. **Set resource limits** in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

3. **Enable monitoring** with Prometheus
4. **Set up log aggregation** (e.g., ELK stack)
5. **Configure backups** for volumes
6. **Use secrets management** for API keys
7. **Enable HTTPS** with reverse proxy (nginx, traefik)
8. **Set up health check monitoring** and alerting

## Scaling

To scale the Morgan server:

```bash
docker-compose up -d --scale morgan-server=3
```

Note: You'll need to add a load balancer (nginx, traefik) in front of multiple server instances.

## Updating

To update to a new version:

```bash
# Pull latest images
docker-compose pull

# Restart services
docker-compose up -d

# Remove old images
docker image prune -f
```

## Cleanup

To completely remove all services and data:

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: This deletes all data!)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```
