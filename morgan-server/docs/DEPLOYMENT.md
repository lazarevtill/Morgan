# Deployment Guide

This guide covers deploying Morgan Server in various environments, from local development to production deployments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Bare Metal Deployment](#bare-metal-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB free space
- OS: Linux, macOS, or Windows

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 50+ GB free space (for models and cache)
- OS: Linux (Ubuntu 20.04+, Debian 11+, or similar)

### Software Requirements

**For Docker Deployment:**
- Docker Engine 20.10+
- Docker Compose 2.0+

**For Bare Metal Deployment:**
- Python 3.11+
- pip 23.0+
- Ollama (or other self-hosted LLM)
- Qdrant vector database

## Deployment Options

Morgan Server can be deployed in several ways:

1. **Docker Compose** (Recommended) - Complete stack with all dependencies
2. **Docker** - Server only, external dependencies
3. **Bare Metal** - Direct installation on host system
4. **Kubernetes** - For large-scale deployments (advanced)

## Docker Deployment

### Option 1: Docker Compose (Recommended)

This is the easiest way to deploy Morgan with all dependencies.

#### Step 1: Clone or Download Configuration

```bash
# Create deployment directory
mkdir morgan-deployment
cd morgan-deployment

# Download docker-compose.yml
curl -O https://raw.githubusercontent.com/your-repo/morgan/main/docker/docker-compose.yml
```

#### Step 2: Configure Environment

Create a `.env` file:

```bash
# LLM Configuration (required - remote Ollama)
MORGAN_LLM_ENDPOINT=http://ollama:11434
MORGAN_LLM_MODEL=gemma3

# Embedding Configuration
MORGAN_EMBEDDING_PROVIDER=local
MORGAN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MORGAN_EMBEDDING_DEVICE=cpu

# Logging
MORGAN_LOG_LEVEL=INFO
MORGAN_LOG_FORMAT=json

# Performance
MORGAN_MAX_CONCURRENT=100
MORGAN_REQUEST_TIMEOUT=60
```

#### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f morgan-server
```

#### Step 4: Initialize Ollama

```bash
# Pull the LLM model
docker-compose exec ollama ollama pull gemma3

# Verify model is available
docker-compose exec ollama ollama list
```

#### Step 5: Verify Deployment

```bash
# Check server health
curl http://localhost:8080/health

# Check API documentation
open http://localhost:8080/docs
```

#### Step 6: Connect with Client

```bash
# Install client
pip install morgan-cli

# Configure client
export MORGAN_SERVER_URL=http://localhost:8080

# Start chatting
morgan chat
```

### Option 2: Docker (Server Only)

If you have external Ollama and Qdrant instances:

#### Step 1: Build Image

```bash
# Clone repository
git clone https://github.com/your-repo/morgan.git
cd morgan

# Build server image
docker build -t morgan-server -f docker/Dockerfile.server .
```

#### Step 2: Run Container

```bash
docker run -d \
  --name morgan-server \
  -p 8080:8080 \
  -e MORGAN_LLM_ENDPOINT=http://your-ollama-host:11434 \
  -e MORGAN_LLM_MODEL=gemma3 \
  -e MORGAN_VECTOR_DB_URL=http://your-qdrant-host:6333 \
  -e MORGAN_EMBEDDING_PROVIDER=local \
  -e MORGAN_LOG_LEVEL=INFO \
  -v morgan-cache:/app/data/cache \
  morgan-server
```

#### Step 3: Verify

```bash
# Check container status
docker ps

# Check logs
docker logs -f morgan-server

# Test health endpoint
curl http://localhost:8080/health
```

### Docker Networking

#### Connecting to Host Services

**On Linux:**
```bash
# Use host network mode
docker run --network host morgan-server

# Or use host.docker.internal (Docker 20.10+)
-e MORGAN_LLM_ENDPOINT=http://172.17.0.1:11434
```

**On macOS/Windows:**
```bash
-e MORGAN_LLM_ENDPOINT=http://host.docker.internal:11434
```

#### Custom Network

```bash
# Create network
docker network create morgan-net

# Run services on same network
docker run --network morgan-net --name qdrant qdrant/qdrant
docker run --network morgan-net --name ollama ollama/ollama
docker run --network morgan-net \
  -e MORGAN_LLM_ENDPOINT=http://ollama:11434 \
  -e MORGAN_VECTOR_DB_URL=http://qdrant:6333 \
  morgan-server
```

## Bare Metal Deployment

### Prerequisites

Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip build-essential
```

**macOS:**
```bash
brew install python@3.11
```

**Windows:**
Download and install Python 3.11+ from python.org

### Step 1: Install Dependencies

#### Install Qdrant

**Using Docker:**
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Using Binary:**
```bash
# Download Qdrant
wget https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Run Qdrant
./qdrant
```

#### Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.com/download

**Start Ollama and pull model:**
```bash
ollama serve &
ollama pull gemma3
```

### Step 2: Install Morgan Server

```bash
# Clone repository
git clone https://github.com/your-repo/morgan.git
cd morgan/morgan-server

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install server
pip install -e .
```

### Step 3: Configure Server

Create `config.yaml`:

```yaml
host: "0.0.0.0"
port: 8080
workers: 4

llm_provider: "ollama"
llm_endpoint: "http://localhost:11434"
llm_model: "gemma3"

vector_db_url: "http://localhost:6333"

embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_device: "cpu"

cache_dir: "./data/cache"
cache_size_mb: 1000

log_level: "INFO"
log_format: "json"
```

### Step 4: Run Server

**Development:**
```bash
python -m morgan_server --config config.yaml
```

**Production (with Uvicorn):**
```bash
uvicorn morgan_server.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --log-level info
```

### Step 5: Create Systemd Service (Linux)

Create `/etc/systemd/system/morgan-server.service`:

```ini
[Unit]
Description=Morgan Server
After=network.target

[Service]
Type=simple
User=morgan
Group=morgan
WorkingDirectory=/opt/morgan/morgan-server
Environment="PATH=/opt/morgan/morgan-server/venv/bin"
ExecStart=/opt/morgan/morgan-server/venv/bin/uvicorn morgan_server.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable morgan-server
sudo systemctl start morgan-server
sudo systemctl status morgan-server
```

## Production Considerations

### Security

#### Network Security

1. **Firewall Configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 8080/tcp  # Morgan Server
sudo ufw allow 6333/tcp  # Qdrant (if remote access needed)
sudo ufw enable
```

2. **Bind to Localhost:**
For local-only access, set `host: "127.0.0.1"` in configuration.

3. **Reverse Proxy:**
Use nginx or traefik for HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name morgan.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### File Permissions

```bash
# Create dedicated user
sudo useradd -r -s /bin/false morgan

# Set ownership
sudo chown -R morgan:morgan /opt/morgan

# Restrict permissions
chmod 700 /opt/morgan/data
chmod 600 /opt/morgan/config.yaml
```

#### Secrets Management

Never store secrets in configuration files:

```bash
# Use environment variables
export MORGAN_LLM_API_KEY="your-secret-key"

# Or use secrets management
# - Kubernetes Secrets
# - AWS Secrets Manager
# - HashiCorp Vault
```

### Performance Optimization

#### Resource Limits

**Docker:**
```yaml
services:
  morgan-server:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

**Systemd:**
```ini
[Service]
MemoryLimit=8G
CPUQuota=400%
```

#### Tuning

1. **Worker Processes:**
```yaml
workers: 8  # 2x CPU cores
```

2. **Concurrent Requests:**
```yaml
max_concurrent_requests: 200
```

3. **Cache Size:**
```yaml
cache_size_mb: 5000  # Adjust based on available RAM
```

4. **Embedding Device:**
```yaml
embedding_device: "cuda"  # Use GPU if available
```

### High Availability

#### Load Balancing

Use nginx or HAProxy to distribute load:

```nginx
upstream morgan_backend {
    least_conn;
    server morgan-1:8080;
    server morgan-2:8080;
    server morgan-3:8080;
}

server {
    listen 80;
    location / {
        proxy_pass http://morgan_backend;
    }
}
```

#### Health Checks

Configure health check monitoring:

```yaml
# Docker Compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Backup and Recovery

#### Data Backup

**Qdrant:**
```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/morgan/snapshots

# Download snapshot
curl http://localhost:6333/collections/morgan/snapshots/snapshot-name \
  -o backup.snapshot

# Restore snapshot
curl -X PUT http://localhost:6333/collections/morgan/snapshots/upload \
  -F 'snapshot=@backup.snapshot'
```

**Cache:**
```bash
# Backup cache directory
tar czf cache-backup.tar.gz data/cache/

# Restore
tar xzf cache-backup.tar.gz
```

#### Automated Backups

Create backup script:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/morgan"

# Backup Qdrant
docker exec qdrant curl -X POST http://localhost:6333/snapshots
docker cp qdrant:/qdrant/snapshots/latest.snapshot \
  $BACKUP_DIR/qdrant-$DATE.snapshot

# Backup cache
tar czf $BACKUP_DIR/cache-$DATE.tar.gz /opt/morgan/data/cache

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.snapshot" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

Schedule with cron:
```bash
# Run daily at 2 AM
0 2 * * * /opt/morgan/backup.sh
```

## Monitoring and Maintenance

### Health Monitoring

#### Basic Health Check

```bash
# Simple health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/api/status
```

#### Prometheus Metrics

```bash
# Scrape metrics
curl http://localhost:8080/metrics
```

Configure Prometheus (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'morgan-server'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Log Management

#### Log Rotation

**Using logrotate:**

Create `/etc/logrotate.d/morgan`:

```
/var/log/morgan/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 morgan morgan
    sharedscripts
    postrotate
        systemctl reload morgan-server
    endscript
}
```

#### Centralized Logging

**Using Docker:**
```yaml
services:
  morgan-server:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Using syslog:**
```yaml
logging:
  driver: "syslog"
  options:
    syslog-address: "tcp://logserver:514"
    tag: "morgan-server"
```

### Updates and Maintenance

#### Updating Morgan Server

**Docker:**
```bash
# Pull latest image
docker-compose pull morgan-server

# Restart with new image
docker-compose up -d morgan-server

# Cleanup old images
docker image prune -f
```

**Bare Metal:**
```bash
# Backup current installation
cp -r /opt/morgan /opt/morgan.backup

# Pull updates
cd /opt/morgan
git pull

# Update dependencies
source venv/bin/activate
pip install -e . --upgrade

# Restart service
sudo systemctl restart morgan-server
```

#### Database Maintenance

**Qdrant Optimization:**
```bash
# Optimize collections
curl -X POST http://localhost:6333/collections/morgan/optimize
```

**Cache Cleanup:**
```bash
# Clear old cache entries
curl -X DELETE http://localhost:8080/api/memory/cleanup
```

## Troubleshooting

### Server Won't Start

**Check logs:**
```bash
# Docker
docker-compose logs morgan-server

# Systemd
sudo journalctl -u morgan-server -f

# Direct
tail -f /var/log/morgan/server.log
```

**Common issues:**

1. **Port already in use:**
```bash
# Find process using port
sudo lsof -i :8080
# Kill process or change port
```

2. **Configuration error:**
```bash
# Validate configuration
python -m morgan_server --check-config
```

3. **Permission denied:**
```bash
# Fix permissions
sudo chown -R morgan:morgan /opt/morgan
chmod 700 /opt/morgan/data
```

### Connection Issues

**Can't connect to Ollama:**
```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Check from container
docker exec morgan-server curl http://host.docker.internal:11434/api/tags
```

**Can't connect to Qdrant:**
```bash
# Test Qdrant
curl http://localhost:6333/healthz

# Check collections
curl http://localhost:6333/collections
```

### Performance Issues

**High memory usage:**
```bash
# Reduce cache size
export MORGAN_CACHE_SIZE_MB=500

# Reduce concurrent requests
export MORGAN_MAX_CONCURRENT=50
```

**Slow responses:**
```bash
# Check metrics
curl http://localhost:8080/metrics | grep response_time

# Check component status
curl http://localhost:8080/api/status
```

**Database slow:**
```bash
# Optimize Qdrant
curl -X POST http://localhost:6333/collections/morgan/optimize

# Check Qdrant metrics
curl http://localhost:6333/metrics
```

### Data Issues

**Lost conversations:**
```bash
# Check memory stats
curl http://localhost:8080/api/memory/stats

# Search memory
curl "http://localhost:8080/api/memory/search?query=test&limit=10"
```

**Vector search not working:**
```bash
# Check Qdrant collections
curl http://localhost:6333/collections

# Check collection info
curl http://localhost:6333/collections/morgan
```

## Further Reading

- [Configuration Guide](./CONFIGURATION.md) - Detailed configuration options
- [Embedding Configuration](./EMBEDDING_CONFIGURATION.md) - Embedding setup
- [API Documentation](./API.md) - API reference
- [Migration Guide](../../MIGRATION.md) - Migrating from old system

## Support

For additional help:
- Check server logs for error messages
- Review configuration against examples
- Test individual components (Ollama, Qdrant)
- Check GitHub issues for similar problems
