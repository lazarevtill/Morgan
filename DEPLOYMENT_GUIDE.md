# Morgan v2-0.0.1 - Production Deployment Guide

**Version**: 2.0.0-alpha.1
**Last Updated**: 2025-11-08
**Target**: Production Environments

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Options](#architecture-options)
3. [Production Environment Setup](#production-environment-setup)
4. [Security Hardening](#security-hardening)
5. [Performance Tuning](#performance-tuning)
6. [High Availability Setup](#high-availability-setup)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Disaster Recovery](#backup--disaster-recovery)
9. [Scaling Strategies](#scaling-strategies)
10. [Maintenance & Updates](#maintenance--updates)
11. [Production Checklist](#production-checklist)

---

## Overview

This guide covers production deployment of Morgan v2-0.0.1, including security hardening, performance optimization, monitoring, and scaling strategies.

### Production Readiness Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core RAG System | ✅ Production Ready | 95%+ complete |
| Vector Database | ✅ Production Ready | Qdrant fully supported |
| CLI Interface | ✅ Production Ready | All commands implemented |
| Emotional AI | 🟡 Beta | Missing 2 modules (see V2_IMPLEMENTATION_STATUS_REPORT.md) |
| Learning System | 🟡 Beta | Missing consolidation module |
| Web API | ✅ Production Ready | FastAPI with health checks |

**Overall Status**: ✅ **Production Ready** for core RAG functionality with emotional AI in beta.

---

## Architecture Options

### Option 1: Single-Node Deployment

**Best for**: Small to medium deployments, development, testing

```
┌─────────────────────────────────────────┐
│         Single Server                    │
│  ┌──────────┐  ┌─────────┐  ┌────────┐ │
│  │  Morgan  │  │ Qdrant  │  │ Redis  │ │
│  │  (API)   │  │ (Vector)│  │(Cache) │ │
│  └──────────┘  └─────────┘  └────────┘ │
│                                         │
│  Resources: 16GB RAM, 8 CPU, 100GB SSD  │
└─────────────────────────────────────────┘
```

**Specifications**:
- RAM: 16-32 GB
- CPU: 8-16 cores
- Storage: 100-500 GB SSD
- Expected Load: 100-500 requests/hour

### Option 2: Multi-Service Deployment

**Best for**: Medium to large deployments, production use

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Morgan     │  │   Qdrant     │  │    Redis     │
│   Service    │  │   Cluster    │  │   Cluster    │
│              │  │              │  │              │
│ Load Balancer│  │ 3+ Nodes     │  │ Master+Slave │
│ 2+ Instances │  │ Replication  │  │ Sentinel     │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                    Network Layer
```

**Specifications**:
- Morgan: 2+ instances, 8GB RAM each
- Qdrant: 3+ nodes, 16GB RAM each
- Redis: Master + 2 replicas, 8GB RAM each
- Load Balancer: Nginx or HAProxy
- Expected Load: 1K-10K requests/hour

### Option 3: Distributed GPU Deployment

**Best for**: Self-hosted LLM, maximum performance, multi-host GPU setups

See [DISTRIBUTED_SETUP_GUIDE.md](DISTRIBUTED_SETUP_GUIDE.md) for complete multi-host GPU architecture.

```
┌──────────────────────────────────────────────────────┐
│               Morgan Orchestrator                     │
│                (Main Application)                     │
└────────┬──────────┬──────────┬─────────┬─────────────┘
         │          │          │         │
    ┌────▼────┐┌───▼────┐┌────▼───┐┌────▼────┐
    │ Host 1  ││ Host 2 ││ Host 3 ││ Host 4  │
    │RTX 3090 ││RTX 3090││RTX 4070││RTX 2060 │
    │Main LLM ││Backup  ││Embed   ││Rerank   │
    └─────────┘└────────┘└────────┘└─────────┘
```

---

## NetBird VPN Prerequisite

### Required for Deployment

Before deploying Morgan to production, ensure **NetBird VPN is configured** on your deployment machine. This is required for:

- **Accessing Nexus Registry**: Private Docker images and Python packages are stored in the Nexus repository at `nexus.in.lazarev.cloud`
- **Internal Service Communication**: Production servers may need to access internal resources via the VPN
- **Deployment Automation**: CI/CD pipelines use NetBird to authenticate and pull dependencies

### Setup Steps

1. **Get NetBird Setup Key**
   - Contact your infrastructure team for a NetBird setup key
   - Management URL: `https://vpn.lazarev.cloud`

2. **Install NetBird**
   ```bash
   # Linux
   curl -fsSL https://pkgs.netbird.io/install.sh | sh

   # macOS
   brew install netbird

   # Windows
   # Download from https://releases.netbird.io/
   ```

3. **Connect to VPN**
   ```bash
   netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-setup-key>
   ```

4. **Verify Connection**
   ```bash
   netbird status  # Should show "Connected to management: true"
   curl -I https://nexus.in.lazarev.cloud  # Should return HTTP response
   ```

### Keep NetBird Connected

**Important**: Keep NetBird VPN connected throughout the entire deployment process. All deployment steps that access the Nexus registry, pull Docker images, or connect to internal services require VPN access.

---

## Production Environment Setup

### 1. Infrastructure Provisioning

#### Cloud Deployment (AWS Example)

```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Create subnets (public and private)
aws ec2 create-subnet --vpc-id vpc-xxx --cidr-block 10.0.1.0/24  # Public
aws ec2 create-subnet --vpc-id vpc-xxx --cidr-block 10.0.2.0/24  # Private

# Create security groups
aws ec2 create-security-group \
    --group-name morgan-api \
    --description "Morgan API Server" \
    --vpc-id vpc-xxx

# Launch EC2 instances
aws ec2 run-instances \
    --image-id ami-xxx \
    --instance-type t3.xlarge \
    --count 2 \
    --subnet-id subnet-xxx \
    --security-group-ids sg-xxx \
    --key-name your-key \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=morgan-api}]'
```

#### Docker Swarm Deployment

```bash
# Initialize Swarm
docker swarm init --advertise-addr <manager-ip>

# Add worker nodes
# On each worker:
docker swarm join --token <token> <manager-ip>:2377

# Deploy stack
docker stack deploy -c docker-compose.prod.yml morgan
```

#### Kubernetes Deployment

```yaml
# morgan-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morgan-api
  namespace: morgan
spec:
  replicas: 3
  selector:
    matchLabels:
      app: morgan-api
  template:
    metadata:
      labels:
        app: morgan-api
        version: v2.0.0
    spec:
      containers:
      - name: morgan
        image: morgan-rag:v2.0.0
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9000
          name: metrics
        env:
        - name: LLM_BASE_URL
          valueFrom:
            secretKeyRef:
              name: morgan-secrets
              key: llm-base-url
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: morgan-secrets
              key: llm-api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: morgan-api-service
  namespace: morgan
spec:
  selector:
    app: morgan-api
  ports:
  - name: api
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9000
    targetPort: 9000
  type: LoadBalancer
```

### 2. Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  morgan:
    image: morgan-rag:v2.0.0
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
        max_attempts: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      - MORGAN_ENV=production
      - MORGAN_DEBUG=false
      - MORGAN_LOG_LEVEL=INFO
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LLM_API_KEY=${LLM_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    secrets:
      - llm_api_key
      - session_secret
    volumes:
      - morgan_data:/app/data:rw
      - morgan_logs:/app/logs:rw
    networks:
      - frontend
      - backend
    healthcheck:
      test: ["CMD", "python", "-m", "morgan", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

  qdrant:
    image: qdrant/qdrant:latest
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.labels.storage==ssd
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__CLUSTER__ENABLED=true
    volumes:
      - qdrant_data:/qdrant/storage:rw
    networks:
      - backend

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    command: >
      redis-server
      --appendonly yes
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data:rw
    networks:
      - backend

  nginx:
    image: nginx:alpine
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 512M
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_cache:/var/cache/nginx:rw
    networks:
      - frontend
    depends_on:
      - morgan

  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus:rw
    networks:
      - backend

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana:rw
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    networks:
      - backend
      - frontend
    depends_on:
      - prometheus

secrets:
  llm_api_key:
    external: true
  session_secret:
    external: true

volumes:
  morgan_data:
    driver: local
  morgan_logs:
    driver: local
  qdrant_data:
    driver: local
  redis_data:
    driver: local
  nginx_cache:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay
    internal: true
```

---

## Security Hardening

### 1. API Security

#### Enable Authentication

```bash
# Generate strong API key
openssl rand -hex 32

# In .env
MORGAN_API_KEY=your-generated-api-key-here
```

#### Configure CORS

```bash
# In .env - restrict to your domains
MORGAN_CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

#### Rate Limiting

Create `config/rate_limits.yaml`:

```yaml
global:
  requests_per_minute: 100
  requests_per_hour: 1000

endpoints:
  /api/chat:
    requests_per_minute: 20
    requests_per_hour: 200
  /api/learn:
    requests_per_minute: 10
    requests_per_hour: 50
  /api/search:
    requests_per_minute: 50
    requests_per_hour: 500
```

### 2. Network Security

#### Firewall Configuration

```bash
# Ubuntu/Debian with UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Block direct access to internal services
# Only allow from localhost or private network
sudo ufw deny 6333/tcp   # Qdrant
sudo ufw deny 6379/tcp   # Redis
```

#### SSL/TLS Configuration

```bash
# Generate SSL certificate with Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot certonly --nginx -d api.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

#### Nginx SSL Configuration

```nginx
# /etc/nginx/sites-available/morgan
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req zone=api_limit burst=20 nodelay;
    limit_req_status 429;

    location / {
        proxy_pass http://morgan-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /metrics {
        deny all;  # Only accessible from monitoring
        allow 10.0.0.0/8;
    }
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# Upstream backend
upstream morgan-backend {
    least_conn;
    server morgan-1:8080 max_fails=3 fail_timeout=30s;
    server morgan-2:8080 max_fails=3 fail_timeout=30s;
}
```

### 3. Secrets Management

#### Using Docker Secrets

```bash
# Create secrets
echo "your-llm-api-key" | docker secret create llm_api_key -
echo "your-session-secret" | docker secret create session_secret -
echo "your-redis-password" | docker secret create redis_password -

# List secrets
docker secret ls
```

#### Using HashiCorp Vault (Advanced)

```bash
# Start Vault
vault server -dev

# Store secrets
vault kv put secret/morgan/llm api_key=sk-xxx
vault kv put secret/morgan/db password=xxx

# Retrieve in application
vault kv get secret/morgan/llm
```

### 4. Database Security

#### Qdrant Security

```yaml
# qdrant-config.yaml
service:
  api_key: ${QDRANT_API_KEY}

storage:
  # Enable encryption at rest
  encryption:
    enabled: true
    key_path: /etc/qdrant/encryption.key
```

#### Redis Security

```bash
# In redis.conf
requirepass your-strong-redis-password
rename-command CONFIG ""
rename-command FLUSHALL ""
maxmemory-policy allkeys-lru
```

### 5. Container Security

```dockerfile
# Production Dockerfile with security hardening
FROM python:3.11-slim as builder

# Security: Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ... build steps ...

FROM python:3.11-slim

# Security: Install only runtime dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Security: Create non-root user
RUN groupadd -r morgan && \
    useradd -r -g morgan -u 1000 morgan && \
    mkdir -p /app/data /app/logs && \
    chown -R morgan:morgan /app

# Security: Copy only necessary files
COPY --from=builder --chown=morgan:morgan /opt/venv /opt/venv
COPY --chown=morgan:morgan . /app

# Security: Switch to non-root user
USER morgan

# Security: Read-only root filesystem
# Use volumes for writable directories
VOLUME ["/app/data", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m morgan health || exit 1

CMD ["python", "-m", "morgan", "serve"]
```

---

## Performance Tuning

### 1. Application Optimization

#### Environment Configuration

```bash
# .env.production
# Performance settings
MORGAN_WORKERS=8  # 2x CPU cores
MORGAN_CACHE_SIZE=5000
MORGAN_CACHE_TTL=7200

# Embedding optimization
EMBEDDING_BATCH_SIZE=200
EMBEDDING_DEVICE=cuda  # If GPU available

# Search optimization
MORGAN_MAX_SEARCH_RESULTS=100
MORGAN_DEFAULT_SEARCH_RESULTS=20

# LLM optimization
MORGAN_MAX_CONTEXT=4096  # Reduce if not needed
MORGAN_MAX_RESPONSE_TOKENS=1024
```

#### Python Optimization

```python
# config/optimization.py
import os

# Use optimized BLAS
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# PyTorch optimization
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# Enable JIT compilation
if hasattr(torch.jit, 'enable_onednn_fusion'):
    torch.jit.enable_onednn_fusion(True)
```

### 2. Database Optimization

#### Qdrant Tuning

```yaml
# qdrant-config.yaml
storage:
  # Use mmap for better performance
  mmap_threshold_kb: 10000

  # Optimize for throughput
  performance:
    max_optimization_threads: 8

  # Indexing configuration
  hnsw_config:
    m: 16  # Connections per layer
    ef_construct: 100  # Construction quality
```

#### Redis Optimization

```bash
# redis.conf
# Memory optimization
maxmemory 4gb
maxmemory-policy allkeys-lru

# Performance
save ""  # Disable RDB snapshots for pure cache
appendonly yes
appendfsync everysec

# Networking
tcp-backlog 511
timeout 300
tcp-keepalive 300
```

### 3. Embedding Performance

```python
# config/embeddings.py
from sentence_transformers import SentenceTransformer

# Load model with optimizations
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cuda',  # Use GPU
)

# Enable model quantization
model.half()  # FP16 for 2x speedup

# Enable TorchScript
model = torch.jit.script(model)

# Batch processing
embeddings = model.encode(
    texts,
    batch_size=256,  # Larger batches for GPU
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
)
```

### 4. Caching Strategy

```python
# Implement multi-layer caching
from functools import lru_cache
import redis

# L1: In-memory LRU cache
@lru_cache(maxsize=1000)
def get_embedding_cached(text: str):
    return generate_embedding(text)

# L2: Redis cache
redis_client = redis.Redis(host='redis', port=6379)

def get_with_redis_cache(key: str):
    # Check Redis
    cached = redis_client.get(key)
    if cached:
        return pickle.loads(cached)

    # Generate and cache
    value = expensive_operation(key)
    redis_client.setex(key, 3600, pickle.dumps(value))
    return value
```

### 5. Load Balancing

```nginx
# nginx.conf
upstream morgan-backend {
    least_conn;  # Route to least busy server

    server morgan-1:8080 weight=3 max_fails=3 fail_timeout=30s;
    server morgan-2:8080 weight=3 max_fails=3 fail_timeout=30s;
    server morgan-3:8080 weight=2 max_fails=3 fail_timeout=30s;

    keepalive 32;  # Keep connections alive
}

# Connection pooling
proxy_http_version 1.1;
proxy_set_header Connection "";
```

---

## High Availability Setup

### 1. Multi-Instance Deployment

```yaml
# docker-compose.ha.yml
services:
  morgan-1:
    image: morgan-rag:v2.0.0
    environment:
      - INSTANCE_ID=morgan-1
    networks:
      - backend

  morgan-2:
    image: morgan-rag:v2.0.0
    environment:
      - INSTANCE_ID=morgan-2
    networks:
      - backend

  morgan-3:
    image: morgan-rag:v2.0.0
    environment:
      - INSTANCE_ID=morgan-3
    networks:
      - backend
```

### 2. Qdrant Clustering

```bash
# Node 1 (Leader)
docker run -d \
    --name qdrant-1 \
    -e QDRANT__CLUSTER__ENABLED=true \
    -e QDRANT__CLUSTER__P2P__PORT=6335 \
    -p 6333:6333 \
    qdrant/qdrant:latest

# Node 2 (Follower)
docker run -d \
    --name qdrant-2 \
    -e QDRANT__CLUSTER__ENABLED=true \
    -e QDRANT__CLUSTER__P2P__PORT=6335 \
    -e QDRANT__CLUSTER__P2P__BOOTSTRAP=http://qdrant-1:6335 \
    qdrant/qdrant:latest
```

### 3. Redis Sentinel (HA)

```bash
# docker-compose.redis-ha.yml
services:
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes

  redis-slave-1:
    image: redis:7-alpine
    command: redis-server --slaveof redis-master 6379 --appendonly yes

  redis-slave-2:
    image: redis:7-alpine
    command: redis-server --slaveof redis-master 6379 --appendonly yes

  redis-sentinel-1:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./redis/sentinel.conf:/etc/redis/sentinel.conf
```

---

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'morgan'
    static_configs:
      - targets: ['morgan-1:9000', 'morgan-2:9000', 'morgan-3:9000']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant-1:6333', 'qdrant-2:6333']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
```

### 2. Grafana Dashboards

Create dashboards for:

1. **System Overview**
   - Request rate
   - Error rate
   - Latency (p50, p95, p99)
   - Active connections

2. **RAG Performance**
   - Search latency
   - Embedding generation time
   - Vector database operations
   - Cache hit rate

3. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network traffic

4. **Business Metrics**
   - Documents ingested
   - Conversations
   - User satisfaction (from feedback)
   - Token usage

### 3. Application Metrics

```python
# morgan/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter(
    'morgan_requests_total',
    'Total requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'morgan_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

# RAG metrics
search_duration = Histogram(
    'morgan_search_duration_seconds',
    'Search operation duration'
)

embedding_duration = Histogram(
    'morgan_embedding_duration_seconds',
    'Embedding generation duration'
)

cache_hit_rate = Gauge(
    'morgan_cache_hit_rate',
    'Cache hit rate percentage'
)

# Business metrics
documents_ingested = Counter(
    'morgan_documents_ingested_total',
    'Total documents ingested'
)

conversations_started = Counter(
    'morgan_conversations_started_total',
    'Total conversations started'
)
```

### 4. Logging Strategy

```yaml
# config/logging.yaml
version: 1
formatters:
  json:
    format: '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "trace_id": "%(trace_id)s"}'

handlers:
  console:
    class: logging.StreamHandler
    formatter: json

  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/morgan/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    formatter: json

  error_file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/morgan/error.log
    maxBytes: 104857600
    backupCount: 10
    formatter: json
    level: ERROR

loggers:
  morgan:
    level: INFO
    handlers: [console, file, error_file]
```

### 5. Alert Rules

```yaml
# monitoring/alerts.yml
groups:
  - name: morgan_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(morgan_requests_total{status="500"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, morgan_request_duration_seconds) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile latency > 5s"

      - alert: QdrantDown
        expr: up{job="qdrant"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Qdrant instance down"

      - alert: LowCacheHitRate
        expr: morgan_cache_hit_rate < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 50%"
```

---

## Backup & Disaster Recovery

### 1. Backup Strategy

#### Qdrant Backups

```bash
#!/bin/bash
# scripts/backup-qdrant.sh

BACKUP_DIR="/backups/qdrant"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
curl -X POST "http://localhost:6333/collections/morgan_knowledge/snapshots" \
    -H "Content-Type: application/json"

# Download snapshot
curl "http://localhost:6333/collections/morgan_knowledge/snapshots/latest" \
    -o "${BACKUP_DIR}/morgan_knowledge_${DATE}.snapshot"

# Compress
gzip "${BACKUP_DIR}/morgan_knowledge_${DATE}.snapshot"

# Upload to S3
aws s3 cp \
    "${BACKUP_DIR}/morgan_knowledge_${DATE}.snapshot.gz" \
    "s3://your-backup-bucket/qdrant/"

# Cleanup old backups (keep last 30 days)
find ${BACKUP_DIR} -name "*.snapshot.gz" -mtime +30 -delete
```

#### Redis Backups

```bash
#!/bin/bash
# scripts/backup-redis.sh

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Trigger save
redis-cli BGSAVE

# Wait for save to complete
while [ $(redis-cli LASTSAVE) -eq $LASTSAVE ]; do
    sleep 1
done

# Copy dump
cp /var/lib/redis/dump.rdb "${BACKUP_DIR}/dump_${DATE}.rdb"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/dump_${DATE}.rdb" "s3://your-backup-bucket/redis/"

# Cleanup
find ${BACKUP_DIR} -name "dump_*.rdb" -mtime +7 -delete
```

#### Application Data Backups

```bash
#!/bin/bash
# scripts/backup-data.sh

BACKUP_DIR="/backups/morgan"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data directory
tar -czf "${BACKUP_DIR}/data_${DATE}.tar.gz" /app/data

# Backup configuration
tar -czf "${BACKUP_DIR}/config_${DATE}.tar.gz" /app/config

# Upload to S3
aws s3 sync ${BACKUP_DIR} "s3://your-backup-bucket/morgan/"

# Cleanup
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +30 -delete
```

### 2. Automated Backups

```bash
# Add to crontab
crontab -e

# Daily Qdrant backup at 2 AM
0 2 * * * /opt/morgan/scripts/backup-qdrant.sh

# Hourly Redis backup
0 * * * * /opt/morgan/scripts/backup-redis.sh

# Weekly full backup on Sunday at 3 AM
0 3 * * 0 /opt/morgan/scripts/backup-data.sh
```

### 3. Disaster Recovery

#### Restore Qdrant

```bash
#!/bin/bash
# scripts/restore-qdrant.sh

SNAPSHOT_FILE=$1

# Upload snapshot
curl -X POST "http://localhost:6333/collections/morgan_knowledge/snapshots/upload" \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@${SNAPSHOT_FILE}"

# Restore from snapshot
curl -X PUT "http://localhost:6333/collections/morgan_knowledge/snapshots/restore" \
    -H "Content-Type: application/json" \
    -d "{\"location\": \"${SNAPSHOT_FILE}\"}"
```

#### Restore Redis

```bash
#!/bin/bash
# scripts/restore-redis.sh

DUMP_FILE=$1

# Stop Redis
systemctl stop redis

# Restore dump
cp ${DUMP_FILE} /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb

# Start Redis
systemctl start redis
```

### 4. Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: < 4 hours
2. **RPO (Recovery Point Objective)**: < 24 hours

**Recovery Steps**:

1. Provision new infrastructure (if needed)
2. Restore Qdrant from latest snapshot (30 min)
3. Restore Redis from backup (10 min)
4. Deploy Morgan application (20 min)
5. Verify all services (30 min)
6. Update DNS/Load balancer (10 min)

**Total Estimated Recovery Time**: 2 hours

---

## Scaling Strategies

### 1. Horizontal Scaling

```bash
# Scale Morgan instances
docker service scale morgan=5

# Kubernetes
kubectl scale deployment morgan-api --replicas=5

# Verify scaling
kubectl get pods -l app=morgan-api
```

### 2. Vertical Scaling

```yaml
# Increase resources per instance
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

### 3. Auto-Scaling

```yaml
# Kubernetes HPA (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: morgan-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: morgan-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### 4. Database Scaling

#### Qdrant Sharding

```python
# Implement collection sharding
collections = [
    f"morgan_knowledge_shard_{i}"
    for i in range(4)
]

# Route to shard based on hash
shard_id = hash(document_id) % len(collections)
collection = collections[shard_id]
```

#### Read Replicas

```python
# Configure read replicas
qdrant_writer = QdrantClient(url="http://qdrant-master:6333")
qdrant_readers = [
    QdrantClient(url=f"http://qdrant-replica-{i}:6333")
    for i in range(3)
]

# Load balance reads
def search(query):
    reader = random.choice(qdrant_readers)
    return reader.search(collection, query)
```

---

## Maintenance & Updates

### 1. Zero-Downtime Deployment

```bash
# Rolling update with Kubernetes
kubectl set image deployment/morgan-api \
    morgan=morgan-rag:v2.0.1 \
    --record

# Monitor rollout
kubectl rollout status deployment/morgan-api

# Rollback if issues
kubectl rollout undo deployment/morgan-api
```

### 2. Database Migrations

```bash
# Create migration script
cat > migrations/001_add_new_collection.py << 'EOF'
from qdrant_client import QdrantClient

def migrate():
    client = QdrantClient(url=os.getenv('QDRANT_URL'))

    # Create new collection
    client.create_collection(
        collection_name="morgan_new_feature",
        vectors_config={"size": 768, "distance": "Cosine"}
    )

    print("✓ Migration complete")

if __name__ == "__main__":
    migrate()
EOF

# Run migration
python migrations/001_add_new_collection.py
```

### 3. Health Checks

```python
# morgan/health.py
from typing import Dict

async def health_check() -> Dict[str, str]:
    """Comprehensive health check"""
    checks = {}

    # Check LLM
    try:
        await llm_client.test()
        checks['llm'] = 'ok'
    except Exception as e:
        checks['llm'] = f'error: {e}'

    # Check Qdrant
    try:
        qdrant_client.get_collections()
        checks['qdrant'] = 'ok'
    except Exception as e:
        checks['qdrant'] = f'error: {e}'

    # Check Redis
    try:
        redis_client.ping()
        checks['redis'] = 'ok'
    except Exception as e:
        checks['redis'] = f'error: {e}'

    return checks
```

---

## Production Checklist

### Pre-Deployment

- [ ] Load testing completed (target: 1000 req/hour)
- [ ] Security audit passed
- [ ] SSL/TLS certificates configured
- [ ] Secrets properly managed
- [ ] Backups configured and tested
- [ ] Monitoring and alerting set up
- [ ] Documentation updated
- [ ] Disaster recovery plan documented
- [ ] Team trained on deployment procedures

### Deployment

- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Verify monitoring and logs
- [ ] Perform smoke tests
- [ ] Deploy to production (rolling update)
- [ ] Monitor for errors/performance issues
- [ ] Verify all services healthy
- [ ] Test failover mechanisms

### Post-Deployment

- [ ] Monitor metrics for 24 hours
- [ ] Review logs for errors/warnings
- [ ] Verify backups are running
- [ ] Check alert thresholds
- [ ] Update runbooks
- [ ] Conduct team retrospective
- [ ] Document any issues encountered

---

## Additional Resources

- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Architecture**: [docs/architecture/](docs/architecture/)
- **API Documentation**: `/docs` endpoint when running
- **Monitoring Dashboards**: See `monitoring/grafana/`
- **Runbooks**: [docs/runbooks/](docs/runbooks/)

---

**Production deployment complete!** Monitor your deployment and refer to this guide for maintenance and scaling.

For issues or questions, consult the troubleshooting section or open a GitHub issue.
