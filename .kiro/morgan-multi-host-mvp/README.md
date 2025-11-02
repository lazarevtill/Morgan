# Morgan Multi-Host MVP - Project Overview (V2 Integration)

> **Status**: Planning Complete - Updated for v2-0.0.1 Branch
> **Version**: 2.0.0
> **Last Updated**: 2025-11-02

---

## 📋 Project Summary

This directory contains the complete specification for deploying **Morgan V2 AI Assistant** across **6 hosts** with different hardware configurations. Morgan V2 includes:

- **Advanced RAG** (Retrieval-Augmented Generation) with hierarchical search
- **Emotion Detection** (11 specialized modules)
- **Empathy Engine** (5 modules for emotional support)
- **Learning & Adaptation** (6 modules for continuous improvement)
- **Multi-Stage Search** (coarse → medium → fine)
- **Distributed Inference** across GPU and CPU hosts

---

## 🎯 MVP Goals

1. **6-Host Architecture**: Distribute Morgan V2 services across specialized hardware
2. **GPU Optimization**: Schedule LLM, embeddings, and reranking across 4 GPU hosts
3. **CPU Services**: Run RAG coordination, emotion detection, empathy, learning on 2 CPU hosts
4. **High Availability**: Implement failover between redundant GPU hosts
5. **Centralized Management**: Use Consul, Traefik, PostgreSQL, Redis, Qdrant
6. **V2 Features**: Full deployment of RAG, emotion, empathy, and learning modules

---

## 🏗️ Your 6-Host Hardware Configuration

| Host | Hardware | Role | Services |
|------|----------|------|----------|
| **Host 1** | i9 + 64GB RAM (CPU) | Primary Orchestrator | Core RAG, Qdrant, PostgreSQL, Redis, Emotion, Empathy, Traefik, Consul |
| **Host 2** | i9 + 64GB RAM (CPU) | Secondary Orchestrator | Core RAG backup, Learning, PostgreSQL standby, Redis, Monitoring, Consul |
| **Host 3** | i9 + RTX 3090 (24GB) | Main LLM #1 | Qwen2.5-32B-Instruct (Q4_K_M), Consul |
| **Host 4** | i9 + RTX 3090 (24GB) | Main LLM #2 | Qwen2.5-32B-Instruct (Q4_K_M - load balanced) |
| **Host 5** | i9 + RTX 4070 (8GB) | Embeddings + Fast LLM | nomic-embed-text, Qwen2.5-7B-Instruct |
| **Host 6** | i9 + RTX 2060 (6GB) | Reranking + Utils | CrossEncoder, utility models |

---

## 🏛️ High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    External Clients                            │
│              (Web UI, API Clients, Mobile Apps)                │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│              Traefik API Gateway (Host 1:8080)                 │
│       Load Balancing | TLS | Rate Limiting | WebSocket        │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Host 1     │    │   Host 2     │    │  Hosts 3-6   │
│ Primary CPU  │    │Secondary CPU │    │  GPU Cluster │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Core RAG     │    │ Core RAG     │    │ Host 3:      │
│ Qdrant       │    │ Learning     │    │  Qwen 32B #1 │
│ PostgreSQL   │    │ PostgreSQL   │    │              │
│ Redis        │    │ Redis        │    │ Host 4:      │
│ Emotion      │    │ Monitoring   │    │  Qwen 32B #2 │
│ Empathy      │    │ Prometheus   │    │              │
│ Traefik      │    │ Grafana      │    │ Host 5:      │
│ Consul       │    │ Loki         │    │  Embeddings  │
│              │    │ Consul       │    │              │
│              │    │              │    │ Host 6:      │
│              │    │              │    │  Reranking   │
└──────────────┘    └──────────────┘    └──────────────┘

        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    Consul Service
                     Discovery
                    (3-node cluster)
```

---

## 📁 Document Structure

| Document | Purpose | Lines | Status |
|----------|---------|-------|--------|
| **[requirements.md](./requirements.md)** | 12 functional requirements for V2 deployment | 389 | ✅ Updated for V2 |
| **[design.md](./design.md)** | Architecture for 6-host V2 deployment | TBD | 🔄 In Progress |
| **[tasks.md](./tasks.md)** | 100+ implementation tasks for V2 | TBD | ⏳ Pending |
| **README.md** (this file) | Project overview and quick reference | - | ✅ Updated for V2 |

---

## 🔑 Key V2 Features Distribution

### Host 1 (Primary CPU) - 64GB RAM
**Services**:
- ✅ **Core RAG Orchestrator**: Coordinates all RAG queries
- ✅ **Emotion Detection** (11 modules): Analyzer, Classifier, Detector, Intensity, Patterns, Triggers, Recovery, Memory, Context, Tracker
- ✅ **Empathy Engine** (5 modules): Generator, Mirror, Support, Tone, Validator
- ✅ **Document Ingestion**: Processes 100+ docs/minute
- ✅ **Qdrant Vector Database**: Stores hierarchical embeddings (coarse, medium, fine)
- ✅ **PostgreSQL Primary**: Conversations, emotional memory, learning data
- ✅ **Redis Cluster Node 1**: Conversation caching
- ✅ **Traefik API Gateway**: Unified entry point
- ✅ **Consul Server**: Service discovery leader

**Why Host 1**:
- 64GB RAM ideal for large vector database and document processing
- CPU-intensive emotion and empathy analysis
- Centralized orchestration logic

### Host 2 (Secondary CPU) - 64GB RAM
**Services**:
- ✅ **Learning & Adaptation Engine** (6 modules): Engine, Adaptation, Preferences, Patterns, Feedback
- ✅ **Background Processing**: Proactive monitoring, long-running jobs
- ✅ **PostgreSQL Standby**: Streaming replication from Host 1
- ✅ **Redis Cluster Node 2**: High availability caching
- ✅ **Monitoring Stack**:
  - Prometheus: Metrics collection
  - Grafana: Visualization dashboards
  - Loki: Centralized logging
- ✅ **Consul Agent**: Service discovery participant

**Why Host 2**:
- 64GB RAM for intensive pattern analysis and monitoring data
- Backup for critical orchestration services
- Dedicated monitoring reduces load on Host 1

### Host 3 (RTX 3090 #1) - 24GB VRAM
**Services**:
- ✅ **Ollama + Qwen2.5-32B-Instruct (Q4_K_M)**: Primary LLM inference
- ✅ **Consul Agent**: Health reporting with GPU metrics

**Model Details**:
- Quantization: Q4_K_M (fits in 24GB VRAM)
- Context: 32K tokens
- Performance: ~20-40 tokens/second
- Load: 40% of total LLM requests

**Why Host 3**:
- 24GB VRAM allows full 32B model
- High-end GPU for complex reasoning
- Primary inference workload

### Host 4 (RTX 3090 #2) - 24GB VRAM
**Services**:
- ✅ **Ollama + Qwen2.5-32B-Instruct (Q4_K_M)**: Secondary LLM (load balanced)
- ✅ **Consul Agent**: Health reporting

**Model Details**:
- Identical to Host 3 for seamless failover
- Load: 40% of total LLM requests
- Automatic failover if Host 3 fails

**Why Host 4**:
- Redundancy for high availability
- Load balancing for parallel requests
- Failover target for Host 3

### Host 5 (RTX 4070) - 8GB VRAM
**Services**:
- ✅ **Embedding Service**: nomic-embed-text (all RAG embeddings)
- ✅ **Fast LLM**: Qwen2.5-7B-Instruct (simple queries)
- ✅ **Consul Agent**: Health reporting

**Workload**:
- 100% of embedding generation (100+ embeddings/minute)
- 20% of simple LLM queries (non-RAG)
- Batch processing for efficiency

**Why Host 5**:
- 8GB VRAM sufficient for embedding model + 7B LLM
- NVIDIA Ampere architecture (RTX 40 series) optimized for throughput
- Dedicated embeddings prevent interference with main LLM

### Host 6 (RTX 2060) - 6GB VRAM
**Services**:
- ✅ **Reranking Service**: CrossEncoder (BGE-reranker-large)
- ✅ **Utility Models**: Classification, small tasks
- ✅ **Consul Agent**: Health reporting

**Workload**:
- 100% of reranking requests (top-100 results)
- Small model inference for utilities

**Why Host 6**:
- 6GB VRAM sufficient for reranking model
- Older GPU (Turing architecture) still effective for reranking
- Dedicated reranking improves RAG quality without impacting main LLM

---

## 🚀 V2 Feature Highlights

### 1. Advanced RAG System
- **Hierarchical Search**: Coarse → Medium → Fine (3-stage)
- **Multi-Format Support**: PDF, DOCX, MD, HTML, TXT, Code
- **Intelligent Chunking**: Semantic-aware text splitting
- **Batch Processing**: 100+ documents/minute
- **Reciprocal Rank Fusion**: Combines results from multiple searches
- **Caching**: Redis-based caching of search results (1-hour TTL)

### 2. Emotion Detection (11 Modules)
- **Analyzer**: Multi-dimensional emotion analysis
- **Classifier**: Categorizes emotions (joy, sadness, anger, fear, etc.)
- **Detector**: Real-time emotion detection in text
- **Intensity**: Measures emotion strength (0-1 scale)
- **Patterns**: Identifies behavioral patterns
- **Triggers**: Detects emotional triggers
- **Recovery**: Supports emotional recovery
- **Memory**: Stores emotional history
- **Context**: Context-aware emotion analysis
- **Tracker**: Tracks emotions over time
- **Validator**: Validates emotion detection accuracy

### 3. Empathy Engine (5 Modules)
- **Generator**: Creates empathetic responses
- **Mirror**: Mirrors user's emotional state
- **Support**: Provides crisis support
- **Tone**: Matches emotional tone
- **Validator**: Validates empathy quality

### 4. Learning & Adaptation (6 Modules)
- **Engine**: Core learning orchestration
- **Adaptation**: Adapts to user preferences
- **Preferences**: Learns and stores preferences
- **Patterns**: Discovers patterns in behavior
- **Feedback**: Processes user feedback
- **Continuous Improvement**: Evolves responses over time

---

## 📊 Current vs. Target State

### Current State (v2-0.0.1 Branch - Single Host)

- ✅ V2 modules fully implemented (RAG, emotion, empathy, learning)
- ✅ 67,287 lines of Python code
- ✅ Hierarchical search working
- ✅ Emotion detection functional
- ❌ All services on one host
- ❌ No distributed deployment
- ❌ No GPU load balancing
- ❌ No high availability

### Target State (Multi-Host V2 MVP)

- ✅ V2 modules deployed across 6 hosts
- ✅ GPU workloads distributed (LLM, embeddings, reranking)
- ✅ CPU workloads optimized (RAG, emotion, empathy, learning)
- ✅ High availability with failover (Host 3 ↔ Host 4)
- ✅ Service discovery with Consul
- ✅ Centralized monitoring (Prometheus, Grafana, Loki)
- ✅ Load balancing across GPU hosts
- ✅ Shared data layer (PostgreSQL, Redis, Qdrant)

---

## 📈 Success Criteria

The V2 MVP is considered **complete** when:

- [ ] ✅ Morgan operates across all 6 hosts with proper service distribution
- [ ] ✅ RAG hierarchical search works with distributed embeddings
- [ ] ✅ Emotion detection processes messages in <200ms on Host 1
- [ ] ✅ Empathy engine generates responses in <300ms on Host 1
- [ ] ✅ Learning system adapts based on user feedback on Host 2
- [ ] ✅ LLM inference load-balanced between Hosts 3-4 (RTX 3090)
- [ ] ✅ Embedding generation runs on Host 5 (RTX 4070)
- [ ] ✅ Reranking processes results on Host 6 (RTX 2060)
- [ ] ✅ Services automatically discover each other via Consul
- [ ] ✅ Failover works when Host 3 fails (→ Host 4)
- [ ] ✅ Grafana dashboard shows all 6 hosts with V2 module metrics
- [ ] ✅ E2E tests pass for RAG + emotion + empathy + learning
- [ ] ✅ System survives chaos tests (random host failures)
- [ ] ✅ P95 latency <2s for RAG-enhanced queries

---

## 🛠️ Implementation Timeline

**Estimated Total**: 4-6 weeks (1 developer)

**Week 1-2**: Service Discovery & Configuration
- Deploy Consul cluster across Hosts 1, 2, 3
- Implement service registration for V2 modules
- Configure distributed configuration with Consul KV

**Week 3**: Data Layer & API Gateway
- Set up PostgreSQL replication (Host 1 → Host 2)
- Deploy Redis cluster (Hosts 1, 2, 3)
- Deploy Qdrant on Host 1
- Configure Traefik on Host 1

**Week 4**: GPU Service Deployment
- Deploy Ollama + Qwen 32B on Hosts 3-4
- Deploy embedding service on Host 5
- Deploy reranking service on Host 6
- Implement GPU load balancing

**Week 5**: V2 Module Distribution
- Deploy RAG orchestrator on Hosts 1-2
- Deploy emotion detection on Host 1
- Deploy empathy engine on Host 1
- Deploy learning system on Host 2
- Test distributed RAG workflow

**Week 6**: Monitoring, Testing & Documentation
- Deploy Prometheus, Grafana, Loki on Host 2
- Create Grafana dashboards for V2 metrics
- Run load tests and chaos tests
- Write operational runbooks

---

## 📝 Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-11-02 | Initial multi-host specification | Claude Code |
| 2.0.0 | 2025-11-02 | Updated for v2-0.0.1 branch with RAG, emotion, empathy, learning | Claude Code |

---

## 🔗 Related Documents

- [Morgan V2 README](../../../morgan-rag/README.md)
- [6-Host Architecture Guide](../../../MORGAN_6HOST_ARCHITECTURE.md)
- [V2 Transformation Summary](../../../MORGAN_TRANSFORMATION_SUMMARY.md)
- [Distributed Setup Guide](../../../DISTRIBUTED_SETUP_GUIDE.md)

---

**Next Steps**:
1. Review updated requirements.md (12 requirements for V2)
2. Review/approve design.md (will be updated next with 6-host V2 architecture)
3. Review tasks.md (will include V2-specific deployment tasks)
4. Begin implementation with Phase 1 (Consul Service Discovery)

---

**Status**: ✅ Requirements updated for V2 - Ready for design phase
