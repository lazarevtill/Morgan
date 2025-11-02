# Morgan Multi-Host MVP - Requirements Document (V2 Integration)

> **Version**: 2.0.0
> **Status**: Updated for v2-0.0.1 Branch
> **Last Updated**: 2025-11-02
> **Feature**: Multi-Host Self-Hosted AI Assistant with RAG, Empathy, and Learning

---

## Introduction

Morgan V2 is envisioned as a fully self-hosted, distributed AI assistant with advanced RAG (Retrieval-Augmented Generation), emotion detection, empathy engine, and continuous learning capabilities. The system leverages **6 hosts** with different hardware configurations:

- **2 CPU-only hosts**: i9 11th/13th gen, 64GB RAM (orchestration, heavy processing)
- **2 RTX 3090 hosts**: i9 + 24GB VRAM (main LLM reasoning, load balanced)
- **1 RTX 4070 host**: i9 + 8GB VRAM (embeddings + fast LLM)
- **1 RTX 2060 host**: i9 + 6GB VRAM (reranking + utilities)

This requirements document defines the MVP functionality needed to deploy Morgan V2's advanced features across this multi-host infrastructure while maintaining high availability, low latency, and optimal resource utilization.

---

## Requirements

### 1. Multi-Host Architecture with V2 Modules

**User Story:** As a system administrator, I want Morgan V2 to operate across 6 hosts, so that I can leverage specialized hardware for RAG, emotion detection, empathy, and learning features.

#### Acceptance Criteria

1. WHEN the system is deployed THEN Morgan SHALL distribute services across all 6 hosts
2. WHEN Host 3-4 (RTX 3090) are available THEN Morgan SHALL schedule main LLM inference (Qwen2.5-32B-Instruct) with load balancing
3. WHEN Host 5 (RTX 4070) is available THEN Morgan SHALL schedule embedding generation (nomic-embed-text) and fast LLM queries
4. WHEN Host 6 (RTX 2060) is available THEN Morgan SHALL schedule reranking and utility models
5. WHEN Host 1-2 (CPU) are available THEN Morgan SHALL schedule:
   - Core orchestration and RAG coordination
   - Document ingestion and processing
   - Vector database (Qdrant)
   - PostgreSQL and Redis
   - Emotion detection and empathy engine
   - Learning and adaptation services
6. IF a GPU service fails on one host THEN Morgan SHALL maintain service availability through redundancy or failover to alternative GPU hosts
7. WHEN RAG queries are processed THEN Morgan SHALL use hierarchical search (coarse → medium → fine) across distributed embeddings

---

### 2. Advanced RAG System Deployment

**User Story:** As a user, I want Morgan to intelligently retrieve information from ingested documents using hierarchical search, so that I receive accurate, context-aware responses.

#### Acceptance Criteria

1. WHEN documents are ingested THEN Morgan SHALL:
   - Process PDFs, DOCX, MD, HTML, TXT, and code files
   - Use intelligent semantic-aware chunking
   - Generate hierarchical embeddings (coarse, medium, fine) on Host 5 (RTX 4070)
   - Store vectors in Qdrant on Host 1 (CPU)
2. WHEN a query is processed THEN Morgan SHALL:
   - Perform coarse search across all domains
   - Refine with medium-grained search
   - Execute fine-grained search for precision
   - Use Reciprocal Rank Fusion for result merging
   - Rerank results on Host 6 (RTX 2060)
3. WHEN batch processing documents THEN Morgan SHALL achieve >100 docs/minute throughput on Host 1 (CPU)
4. WHEN searching 10K documents THEN Morgan SHALL return results within 500ms
5. IF embedding generation is slow THEN Morgan SHALL queue requests and utilize GPU batch processing

---

### 3. Emotion Detection and Empathy Engine Distribution

**User Story:** As a user, I want Morgan to detect my emotional state and respond empathetically, so that I feel understood and supported.

#### Acceptance Criteria

1. WHEN a user message is received THEN Morgan SHALL:
   - Analyze emotional content using 11 emotion detection modules on Host 1 (CPU)
   - Classify emotions (joy, sadness, anger, fear, surprise, etc.)
   - Measure emotion intensity
   - Detect triggers and patterns
2. WHEN emotional state is detected THEN Morgan SHALL:
   - Generate empathetic responses using 5 empathy modules on Host 1 (CPU)
   - Mirror emotional tone appropriately
   - Provide validation and support
   - Detect crisis situations and respond accordingly
3. WHEN emotion processing load is high THEN Morgan SHALL distribute emotion detection across Host 1 and Host 2 (CPU)
4. IF crisis is detected THEN Morgan SHALL prioritize empathy engine and provide immediate support resources
5. WHEN emotion analysis is complete THEN Morgan SHALL store emotional memory for future reference

---

### 4. Learning and Adaptation System

**User Story:** As a user, I want Morgan to learn from my interactions and adapt to my preferences, so that responses improve over time.

#### Acceptance Criteria

1. WHEN a conversation occurs THEN Morgan SHALL:
   - Track user preferences using learning engine on Host 2 (CPU)
   - Identify behavioral patterns
   - Store feedback for continuous improvement
2. WHEN feedback is provided THEN Morgan SHALL:
   - Integrate feedback into learning model
   - Adjust response generation strategies
   - Update user preference profiles in PostgreSQL
3. WHEN patterns are discovered THEN Morgan SHALL:
   - Adapt communication style
   - Personalize future responses
   - Learn domain-specific knowledge
4. IF user preferences conflict THEN Morgan SHALL prioritize most recent feedback
5. WHEN adaptation occurs THEN Morgan SHALL maintain privacy and store learning data securely

---

### 5. GPU Workload Distribution and Load Balancing

**User Story:** As a system administrator, I want GPU resources optimally allocated across hosts, so that inference performance is maximized.

#### Acceptance Criteria

1. WHEN main LLM requests arrive THEN Morgan SHALL:
   - Load balance between Host 3 and Host 4 (RTX 3090) using round-robin or least-connection
   - Monitor GPU utilization on both hosts
   - Route to less-loaded host if one exceeds 80% utilization
2. WHEN embedding requests arrive THEN Morgan SHALL:
   - Route all embedding generation to Host 5 (RTX 4070)
   - Batch requests for GPU efficiency (batch size: 32-64)
   - Fall back to Host 6 (RTX 2060) if Host 5 is unavailable
3. WHEN reranking requests arrive THEN Morgan SHALL:
   - Route all reranking to Host 6 (RTX 2060)
   - Process in batches for efficiency
   - Fall back to CPU on Host 1 if Host 6 is unavailable
4. WHEN GPU VRAM is exhausted THEN Morgan SHALL:
   - Queue requests and provide estimated wait time
   - Use quantized models (Q4_K_M) to fit in VRAM
   - Gracefully degrade to CPU with user notification
5. WHEN failover occurs THEN Morgan SHALL:
   - Automatically retry requests on healthy GPU hosts
   - Maintain request context and conversation state
   - Alert administrators of service degradation

---

### 6. Distributed Service Discovery (Consul Integration)

**User Story:** As a developer, I want services to automatically discover each other across 6 hosts, so that configuration is minimal and failover is automatic.

#### Acceptance Criteria

1. WHEN a service starts THEN it SHALL register itself with Consul (3-node cluster on Hosts 1, 2, 3)
2. WHEN a service needs GPU inference THEN it SHALL:
   - Query Consul for healthy LLM services on Hosts 3-4
   - Query Consul for embedding services on Host 5
   - Query Consul for reranking services on Host 6
3. WHEN a GPU service becomes unavailable THEN Consul SHALL:
   - Remove it from service registry within 30 seconds
   - Update routing to remaining healthy instances
4. WHEN multiple instances of a service exist THEN Consul SHALL:
   - Provide load balancing information with GPU utilization metrics
   - Support service-specific metadata (VRAM, model loaded, etc.)
5. WHEN a service endpoint changes THEN dependent services SHALL automatically discover the new endpoint

---

### 7. Shared Data Layer for V2 Features

**User Story:** As a user, I want my conversation history, emotional memory, learning data, and RAG knowledge to be available regardless of which host processes my request.

#### Acceptance Criteria

1. WHEN conversation data is stored THEN it SHALL be persisted to PostgreSQL on Host 1 with async replication to Host 2
2. WHEN semantic memory is created THEN Morgan SHALL:
   - Store embeddings in Qdrant on Host 1
   - Store conversation context in Redis cluster (Hosts 1, 2, 3)
   - Store emotional memory in PostgreSQL
   - Store learning patterns in PostgreSQL
3. WHEN RAG documents are ingested THEN Morgan SHALL:
   - Store document chunks in PostgreSQL
   - Store hierarchical embeddings in Qdrant
   - Cache frequently accessed chunks in Redis
4. WHEN database connection fails THEN Morgan SHALL:
   - Cache writes locally on CPU hosts
   - Sync when connectivity is restored
   - Maintain eventual consistency
5. WHEN vector search is performed THEN Morgan SHALL:
   - Use Qdrant collections: `documents_coarse`, `documents_medium`, `documents_fine`
   - Cache search results in Redis with 1-hour TTL
   - Invalidate cache when new documents are ingested

---

### 8. API Gateway for Unified Access

**User Story:** As a client application, I want a single API endpoint to access all Morgan V2 features, so that I don't need to know which host handles my request.

#### Acceptance Criteria

1. WHEN clients connect to Morgan THEN they SHALL use Traefik API gateway on Host 1:8080
2. WHEN a text query arrives THEN Traefik SHALL:
   - Route to Core orchestrator on Host 1 or Host 2
   - Maintain sticky sessions for conversation context
3. WHEN a document upload request arrives THEN Traefik SHALL:
   - Route to document ingestion service on Host 1
   - Stream large files without buffering entire file in memory
4. WHEN WebSocket connections are established THEN Traefik SHALL:
   - Maintain persistent connections with sticky sessions
   - Support real-time emotional feedback streaming
5. WHEN API rate limiting is configured THEN Traefik SHALL:
   - Enforce 100 requests/minute per client for text queries
   - Enforce 10 requests/minute per client for document ingestion
   - Enforce 50 requests/minute per client for RAG searches

---

### 9. Health Monitoring and Observability for V2

**User Story:** As a system administrator, I want comprehensive health monitoring across all 6 hosts and V2 modules, so that I can detect and resolve issues proactively.

#### Acceptance Criteria

1. WHEN services run THEN they SHALL report health every 30 seconds including:
   - GPU utilization (NVIDIA-SMI metrics for Hosts 3-6)
   - VRAM usage and available memory
   - Model loading status
   - Request queue depth
   - Emotion detection processing time
   - RAG search latency
   - Learning system status
2. WHEN viewing Grafana dashboard THEN administrators SHALL see:
   - All 6 hosts with CPU, RAM, GPU, VRAM metrics
   - All services color-coded by health status
   - RAG performance metrics (search latency, precision, recall)
   - Emotion detection accuracy and processing time
   - Learning system adaptation rate
   - Request throughput per host
3. WHEN anomalies are detected THEN Morgan SHALL:
   - Alert via configured channels (Slack, email, webhook)
   - Trigger automatic recovery procedures
   - Log detailed diagnostics for troubleshooting
4. WHEN logs are generated THEN Morgan SHALL:
   - Aggregate to Loki on Host 2
   - Include structured metadata (host, service, user_id, emotion, search_results)
   - Retain logs for 30 days with compression
5. WHEN metrics exceed thresholds THEN Prometheus SHALL:
   - GPU utilization >90% for 5 minutes → Alert
   - RAG search latency >1 second → Alert
   - Emotion detection queue >100 requests → Alert
   - Learning system adaptation failure → Alert

---

### 10. Platform-Specific Optimization

**User Story:** As a system administrator, I want Morgan to optimize for each host's hardware, so that performance is maximized.

#### Acceptance Criteria

1. WHEN running on Hosts 3-4 (RTX 3090) THEN Morgan SHALL:
   - Load Qwen2.5-32B-Instruct (Q4_K_M quantization)
   - Use CUDA 12.4 with cuBLAS for matrix operations
   - Enable tensor parallelism for 24GB VRAM
   - Monitor temperature and throttle if needed
2. WHEN running on Host 5 (RTX 4070) THEN Morgan SHALL:
   - Load nomic-embed-text for embeddings
   - Load Qwen2.5-7B-Instruct for fast queries
   - Use CUDA 12.4 with optimized batch processing
   - Prioritize throughput over latency for embeddings
3. WHEN running on Host 6 (RTX 2060) THEN Morgan SHALL:
   - Load CrossEncoder reranking model
   - Use CUDA 12.4 with minimal VRAM overhead
   - Fall back to CPU if VRAM insufficient
4. WHEN running on Hosts 1-2 (CPU) THEN Morgan SHALL:
   - Use AVX2/AVX-512 optimizations for CPU inference (if needed)
   - Leverage 64GB RAM for large vector database caching
   - Use multi-threading for document processing (16+ threads)
   - Optimize PostgreSQL and Redis for high throughput

---

### 11. High Availability and Failover for V2

**User Story:** As a user, I want Morgan to remain available even if individual hosts or services fail, so that I can rely on it for critical tasks.

#### Acceptance Criteria

1. WHEN Host 3 (RTX 3090 #1) fails THEN Morgan SHALL:
   - Automatically failover to Host 4 (RTX 3090 #2) within 10 seconds
   - Maintain conversation context from PostgreSQL
   - Notify user of temporary slowdown if load is high
2. WHEN Host 1 (Primary CPU) fails THEN Morgan SHALL:
   - Failover orchestration to Host 2 (Secondary CPU)
   - Promote PostgreSQL standby to primary
   - Maintain Redis cluster with 2/3 nodes
   - Restore full functionality within 60 seconds
3. WHEN Qdrant becomes unavailable THEN Morgan SHALL:
   - Use fallback keyword search (PostgreSQL full-text search)
   - Degrade RAG quality gracefully
   - Alert administrators and attempt to restart Qdrant
4. WHEN all GPU hosts are unavailable THEN Morgan SHALL:
   - Fall back to CPU-based inference on Hosts 1-2 (degraded performance)
   - Use smaller quantized models (Q4_0 or GGUF)
   - Disable real-time features (emotion detection, empathy)
   - Queue GPU-required requests for later processing
5. WHEN a host is shut down for maintenance THEN Morgan SHALL:
   - Drain connections gracefully (60-second timeout)
   - Redistribute services to remaining hosts
   - Maintain service availability throughout

---

### 12. Security and Privacy for V2

**User Story:** As a privacy-conscious user, I want all my data (conversations, emotions, learning data) to be secure and encrypted, so that my information cannot be compromised.

#### Acceptance Criteria

1. WHEN services communicate across hosts THEN Morgan SHALL use mutual TLS (mTLS) with certificate rotation every 90 days
2. WHEN storing emotional memory THEN Morgan SHALL encrypt sensitive data at rest (AES-256)
3. WHEN transferring user data THEN Morgan SHALL encrypt in transit (TLS 1.3)
4. WHEN API keys are stored THEN Morgan SHALL use Consul KV with encryption
5. WHEN rate limiting is enforced THEN Morgan SHALL protect against abuse and DDoS

---

## Non-Functional Requirements

### Performance

1. **Latency**:
   - Text responses (simple): <1 second
   - Text responses (RAG-enhanced): <2 seconds
   - Embedding generation: <500ms per document
   - Emotion detection: <200ms per message
   - Reranking: <300ms for top-100 results

2. **Throughput**:
   - Main LLM (Hosts 3-4): 20 requests/minute combined
   - Embeddings (Host 5): 100 embeddings/minute
   - Document ingestion: >100 docs/minute
   - Concurrent users: 20+

### Scalability

1. The system SHALL support adding GPU/CPU hosts without service interruption
2. The system SHALL scale to 50+ service instances across all hosts

### Reliability

1. The system SHALL achieve 99.5% uptime
2. The system SHALL automatically recover from transient failures within 60 seconds

### Maintainability

1. The system SHALL provide centralized logging and monitoring
2. The system SHALL support zero-downtime updates

---

## Acceptance Criteria Summary

The V2 MVP is considered complete when:

1. ✅ Morgan operates across all 6 hosts with optimized GPU allocation
2. ✅ RAG system processes documents with hierarchical search
3. ✅ Emotion detection and empathy engine respond in <200ms
4. ✅ Learning system adapts based on user feedback
5. ✅ Services automatically discover each other via Consul
6. ✅ Failover works between RTX 3090 hosts (3↔4)
7. ✅ Grafana dashboard shows all hosts and V2 module metrics
8. ✅ E2E tests pass for RAG, emotion, empathy, learning
9. ✅ System survives GPU host failure chaos tests
10. ✅ P95 latency <2s for RAG-enhanced queries

---

## Out of Scope for MVP

- Mobile client applications
- Multi-region deployment across geographic locations
- Advanced GPU scheduling with multi-GPU per host
- Kubernetes orchestration (using Docker Swarm)
- Real-time voice processing (TTS/STT)
- Integration with external services (Home Assistant, etc.)

---

**Document Status**: Ready for design phase (V2 integration)
**Next Steps**: Update design document with V2 module distribution across 6-host architecture
