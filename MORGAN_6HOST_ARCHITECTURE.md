# Morgan - 6-Host Distributed Architecture

## Your Complete Hardware Setup

**6 Hosts Total:**
- **Host 1-2 (CPU Only)**: i9 11th/13th gen, 64GB RAM (no GPU)
- **Host 3 (GPU)**: i9 + RTX 3090 (24GB VRAM) + 32GB+ RAM
- **Host 4 (GPU)**: i9 + RTX 3090 (24GB VRAM) + 32GB+ RAM
- **Host 5 (GPU)**: i9 + RTX 4070 (8GB VRAM) + 32GB+ RAM
- **Host 6 (GPU)**: i9 + RTX 2060 (6GB VRAM) + 32GB+ RAM

---

## Optimized Architecture

### Recommended Allocation

```
┌────────────────────────────────────────────────────────────────┐
│                       Morgan Core Orchestrator                  │
│                    Runs on Host 1 (CPU) or PC                   │
│                   - Request routing                             │
│                   - Context management                          │
│                   - Memory coordination                         │
└──────┬─────────┬─────────┬─────────┬─────────┬─────────────────┘
       │         │         │         │         │
   ┌───▼───┐ ┌──▼───┐ ┌───▼───┐ ┌───▼───┐ ┌──▼───┐
   │ Host3 │ │ Host4│ │ Host5 │ │ Host6 │ │Host1-2│
   │3090 #1│ │3090#2│ │ 4070  │ │ 2060  │ │CPU-Only
   │       │ │      │ │       │ │       │ │       │
   │Main   │ │Main  │ │Embed  │ │Rerank │ │Heavy  │
   │LLM    │ │LLM   │ │+ Fast │ │+Utils │ │CPU    │
   │       │ │      │ │LLM    │ │       │ │Tasks  │
   └───────┘ └──────┘ └───────┘ └───────┘ └───────┘
```

---

## Recommended Role Assignment

### GPU Hosts

**Host 3 (RTX 3090 #1) - Primary Reasoning**
```yaml
Role: Main LLM Instance #1
Model: Qwen2.5-32B-Instruct (Q4_K_M)
Purpose: Complex reasoning, detailed responses
Port: 11434
Expected Load: 40% of inference requests
```

**Host 4 (RTX 3090 #2) - Secondary Reasoning**
```yaml
Role: Main LLM Instance #2 (Load balanced)
Model: Qwen2.5-32B-Instruct (Q4_K_M)
Purpose: Load balancing + failover
Port: 11434
Expected Load: 40% of inference requests
```

**Host 5 (RTX 4070) - Embeddings + Fast Queries**
```yaml
Role: Embeddings + Fast LLM
Models:
  - nomic-embed-text (embeddings)
  - Qwen2.5-7B-Instruct (fast responses)
Purpose: RAG embeddings + simple queries
Port: 11434
Expected Load: 100% of embeddings, 20% of queries
```

**Host 6 (RTX 2060) - Reranking + Utilities**
```yaml
Role: Reranking + Small Models
Models:
  - CrossEncoder (reranking)
  - Small utility models
Purpose: Search result reranking, classification
Port: 8080 (FastAPI service)
Expected Load: 100% of reranking requests
```

### CPU-Only Hosts

**Host 1 (CPU) - Morgan Core**
```yaml
Role: Main Orchestrator + Heavy Processing
Services:
  - Morgan core assistant
  - Request routing & load balancing
  - Context aggregation
  - Memory processing
  - Document ingestion
  - Vector database (Qdrant)
  - Redis cache
Resources: 64GB RAM ideal for:
  - Large context buffering
  - Document processing
  - Vector database
  - Caching
```

**Host 2 (CPU) - Background Services**
```yaml
Role: Background Processing + Monitoring
Services:
  - Background task scheduler
  - Proactive monitoring
  - Health checks
  - Metrics collection
  - Log aggregation
  - Backup services
Resources: 64GB RAM for:
  - Async task processing
  - Pattern analysis
  - Long-running jobs
  - System monitoring
```

---

## Architecture Benefits

### Why This Setup is Excellent

✅ **Performance**
- 2x RTX 3090s for heavy reasoning (load balanced)
- Dedicated embedding host (RTX 4070)
- Separate reranking (RTX 2060)
- 2x CPU hosts for orchestration & processing

✅ **Reliability**
- Automatic failover (3090 #1 → 3090 #2)
- Health monitoring on separate host
- No single point of failure
- Graceful degradation

✅ **Scalability**
- Easy to add more hosts
- Independent scaling per role
- Can run multiple models simultaneously
- Horizontal scaling for CPU-intensive tasks

✅ **Resource Optimization**
- GPUs focused on inference only
- CPU hosts handle heavy document processing
- 64GB RAM perfect for vector database & caching
- Distributed load prevents bottlenecks

✅ **Flexibility**
- Can test different models per host
- A/B testing capabilities
- Easy model updates (one host at a time)
- Specialized hosts for specialized tasks

---

## Network Configuration

### Recommended Network Topology

```
Router/Switch
    ├── Host 1 (CPU) - 192.168.1.10 - Morgan Core
    ├── Host 2 (CPU) - 192.168.1.11 - Background Services
    ├── Host 3 (3090) - 192.168.1.20 - Main LLM #1
    ├── Host 4 (3090) - 192.168.1.21 - Main LLM #2
    ├── Host 5 (4070) - 192.168.1.22 - Embeddings + Fast LLM
    └── Host 6 (2060) - 192.168.1.23 - Reranking
```

### Required Network
- **Bandwidth**: 1 Gbps minimum (10 Gbps ideal)
- **Latency**: <5ms between hosts (same local network)
- **Firewall**: Open required ports (11434, 8080, 6333, 6379)

---

## Service Ports

```
# LLM Services (Ollama)
Host 3 (3090 #1): 11434 - Main LLM
Host 4 (3090 #2): 11434 - Main LLM
Host 5 (4070):    11434 - Embeddings + Fast LLM

# Reranking Service
Host 6 (2060):    8080 - Reranking API

# Morgan Core Services (Host 1)
Port 6333 - Qdrant Vector DB
Port 6379 - Redis Cache
Port 8000 - Morgan API
Port 8080 - Web Interface

# Monitoring (Host 2)
Port 9090 - Prometheus Metrics
Port 3000 - Grafana Dashboard
Port 8081 - Health Check API
```

---

## Detailed Setup Per Host

### Host 1 (CPU) - Morgan Core

```bash
# Install Morgan and dependencies
cd /opt
git clone <morgan-repo>
cd Morgan/morgan-rag

# Install Python dependencies
pip install -r requirements.txt

# Install Qdrant (Vector Database)
docker pull qdrant/qdrant
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Install Redis (Cache)
docker pull redis:alpine
docker run -d -p 6379:6379 redis:alpine

# Configure Morgan
cp .env.example .env
nano .env  # Configure endpoints (see below)

# Start Morgan
python -m morgan.cli.app
```

**Morgan .env Configuration:**
```env
# Main LLM Endpoints (Load Balanced)
LLM_ENDPOINTS=http://192.168.1.20:11434/v1,http://192.168.1.21:11434/v1
LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
LLM_LOAD_BALANCING=round_robin

# Fast LLM (Host 5)
LLM_FAST_ENDPOINT=http://192.168.1.22:11434/v1
LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M

# Embeddings (Host 5)
EMBEDDING_ENDPOINT=http://192.168.1.22:11434/v1
EMBEDDING_MODEL=nomic-embed-text

# Reranking (Host 6)
RERANKING_ENDPOINT=http://192.168.1.23:8080/rerank

# Local Services
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

---

### Host 2 (CPU) - Background Services

```bash
# Install monitoring stack
docker-compose up -d prometheus grafana

# Install Morgan background services
pip install morgan[background]

# Start background scheduler
python -m morgan.background.service

# Start health monitor
python -m morgan.monitoring.health_monitor
```

---

### Host 3 & 4 (RTX 3090) - Main LLM

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5:32b-instruct-q4_K_M

# Start Ollama (listen on all interfaces)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Verify GPU usage
nvidia-smi  # Should show ~20-22GB VRAM used
```

**Systemd Service:**
```bash
sudo tee /etc/systemd/system/ollama.service << EOF
[Unit]
Description=Ollama LLM Service
After=network.target

[Service]
Type=simple
User=$USER
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

---

### Host 5 (RTX 4070) - Embeddings + Fast LLM

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct-q5_K_M

# Start Ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Verify GPU usage
nvidia-smi  # Should show ~6-7GB VRAM used
```

---

### Host 6 (RTX 2060) - Reranking

```bash
# Install Python and dependencies
pip install sentence-transformers torch fastapi uvicorn

# Create reranking service (see DISTRIBUTED_SETUP_GUIDE.md)
# Start service
python reranking_service.py

# Verify GPU usage
nvidia-smi  # Should show ~2-3GB VRAM used
```

---

## Load Distribution Examples

### Scenario 1: User Asks Complex Question

```
User Query → Host 1 (Morgan Core)
    ↓
Router selects endpoint → Host 3 (3090 #1) [Round-robin]
    ↓
LLM generates response (8 seconds)
    ↓
Response returned → User
```

### Scenario 2: Multiple Concurrent Queries

```
Query 1 → Host 1 → Host 3 (3090 #1) - Processing
Query 2 → Host 1 → Host 4 (3090 #2) - Processing
Query 3 → Host 1 → Host 5 (4070 Fast) - Processing (simple query)
```

**Result**: 3x throughput vs single GPU!

### Scenario 3: Document Ingestion

```
Documents → Host 1 (Morgan Core)
    ↓
1. Text extraction (Host 1 CPU) - Parallel processing
    ↓
2. Chunking (Host 1 CPU) - 64GB RAM handles large docs
    ↓
3. Embedding generation → Host 5 (4070)
    ↓ Batch of 100 chunks
4. Vector storage (Host 1 Qdrant)
```

**Result**: CPU hosts handle heavy processing, GPU focused on embeddings!

### Scenario 4: RAG Search

```
Query → Host 1
    ↓
1. Embed query → Host 5 (4070) - 100ms
    ↓
2. Vector search (Host 1 Qdrant) - 50ms
    ↓
3. Retrieve top 100 chunks - 100ms
    ↓
4. Rerank top 20 → Host 6 (2060) - 300ms
    ↓
5. Generate response → Host 3 or 4 (3090) - 8s
    ↓
Response → User
```

**Total**: ~8.5s (excellent for quality-over-speed!)

---

## Resource Utilization

### Normal Operation

**GPU Hosts:**
- Host 3 (3090 #1): 60-80% utilization, ~20GB VRAM
- Host 4 (3090 #2): 40-60% utilization, ~20GB VRAM (load balanced)
- Host 5 (4070): 40-60% utilization, ~6GB VRAM
- Host 6 (2060): 20-40% utilization, ~2GB VRAM

**CPU Hosts:**
- Host 1: 30-50% CPU, 40-50GB RAM (Qdrant, cache, Morgan core)
- Host 2: 10-30% CPU, 10-20GB RAM (background tasks)

### Peak Load (Multiple Concurrent Requests)

**GPU Hosts:**
- Host 3 & 4 (3090s): 90-95% utilization (both serving)
- Host 5 (4070): 70-80% utilization (embeddings + fast queries)
- Host 6 (2060): 60-70% utilization (reranking)

**CPU Hosts:**
- Host 1: 60-70% CPU, 50-60GB RAM
- Host 2: 40-50% CPU (background processing)

---

## Performance Estimates

### Throughput

**Concurrent Capacity:**
- 2 complex reasoning queries (1 per 3090)
- 3-4 simple queries (4070)
- 10+ embedding requests (4070 batch)
- 20+ reranking requests (2060 batch)

**Total Effective Throughput:**
- ~4-5 queries per minute (complex)
- ~15-20 queries per minute (simple)
- Unlimited if using smart routing (fast/slow model selection)

### Latency

- Simple queries: 1-2s (4070 fast model)
- Complex queries: 5-10s (3090 main model)
- Embeddings: <200ms batch
- Reranking: <300ms batch
- Network overhead: +10-30ms per hop

### Uptime & Reliability

- **Availability**: 99.9% (automatic failover)
- **MTBF**: Very high (multiple redundant hosts)
- **Recovery**: <10s (health checks + failover)

---

## Cost & Power Efficiency

### Power Consumption

**Idle:**
- 4x GPUs idle: ~200W
- 6x i9 CPUs idle: ~150W
- Total: ~350W idle

**Under Load:**
- 4x GPUs full: ~1000W (350+350+200+100)
- 6x i9 CPUs full: ~500W
- Total: ~1500W peak

**Average (typical usage):**
- ~600-800W continuous

**Cost** (at $0.12/kWh):
- ~$50-60/month continuous operation
- Comparable to mid-tier cloud LLM costs!

---

## Advantages Over Cloud

### Cost (Annual)

**Your Setup (Self-Hosted):**
- Hardware: Owned (one-time cost)
- Power: ~$600-720/year
- **Total**: <$1,000/year

**Cloud (ChatGPT Plus/Claude Pro equivalent):**
- $20/month × 12 = $240/year (individual)
- API costs: ~$2,000-5,000/year (heavy usage)
- **Total**: $2,000-5,000/year

**Savings**: $1,000-4,000/year after first year!

### Privacy & Control

- ✅ All data stays on your hardware
- ✅ No external API calls
- ✅ Full model customization
- ✅ No usage limits
- ✅ Works offline

---

## Next Steps

1. **Network Setup** - Configure all hosts on same network
2. **Install Services** - Follow setup guide per host
3. **Test Connectivity** - Verify all endpoints reachable
4. **Load Test** - Benchmark performance
5. **Deploy Morgan** - Run distributed setup

**Estimated Setup Time**: 1 day (2 hours per host + testing)

---

**Your 6-host setup is ideal for a production-grade, self-hosted intelligent assistant!**

**Last Updated**: November 2, 2025
**Architecture**: 6-Host Distributed (4 GPU + 2 CPU)
