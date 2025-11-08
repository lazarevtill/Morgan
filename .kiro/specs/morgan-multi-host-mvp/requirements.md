# Morgan Multi-Host MVP - Requirements Document (V4 - Cloud-Native MicroK8s)

> **Version**: 4.0.0
> **Status**: Refactored for MicroK8s Cloud-Native Architecture
> **Last Updated**: 2025-11-02
> **Feature**: Cloud-Native Multi-Host AI Assistant with RAG, Empathy, and Learning (CLI-Only MVP)

---

## Introduction

Morgan V4 is a cloud-native, self-hosted AI assistant with advanced RAG (Retrieval-Augmented Generation), emotion detection, empathy engine, and continuous learning capabilities. The MVP focuses on a **CLI-based interface** using Click (https://click.palletsprojects.com/) for simplicity and automation.

The system is designed as a cloud-native application deployed on **MicroK8s clusters** across any number of available nodes (minimum 1, recommended 3-7) with varying hardware configurations. MicroK8s provides lightweight Kubernetes orchestration with built-in service mesh, ingress, and storage capabilities.

### Supported Hardware Configurations

**GPU Hosts** (NVIDIA CUDA 12.4+):
- Any NVIDIA GPU with CUDA compute capability 7.0+
- VRAM: 6GB minimum, 8GB+ recommended, 12GB+ optimal
- CPU: Intel i7 9th gen / i9 11th gen / i9 13th gen
- RAM: 32GB minimum, 64GB recommended

**CPU-Only Hosts**:
- CPU: Intel i7 9th gen / i9 11th gen / i9 13th gen
- RAM: 32GB minimum, 64GB recommended

**ARM64 Hosts** (macOS M1):
- Apple Silicon M1/M2/M3
- RAM: 16GB minimum, 32GB+ recommended
- No GPU acceleration (CPU inference only)

### MVP Interface

**CLI Only** (Click-based):
- `morgan chat` - Interactive chat session
- `morgan ingest <path>` - Ingest documents for RAG
- `morgan query "<question>"` - One-shot query
- `morgan status` - System health and node status
- `morgan nodes list` - List all detected nodes and services
- `morgan config` - Configure system settings via Kubernetes ConfigMaps

**Web UI**: Out of scope for MVP (future enhancement)

This requirements document defines the MVP functionality needed to deploy Morgan V4's advanced features across a cloud-native MicroK8s infrastructure while maintaining high availability, low latency, and optimal resource utilization.

---

## Glossary

- **Morgan_System**: The complete cloud-native AI assistant platform including all services, databases, and infrastructure components deployed on MicroK8s
- **Node**: A physical or virtual machine running MicroK8s as part of the Morgan cluster
- **GPU_Node**: A node equipped with NVIDIA GPU hardware capable of CUDA 12.4+ acceleration with GPU operator support
- **CPU_Node**: A node without GPU hardware (x86_64), optimized for CPU-intensive processing
- **ARM64_Node**: A node with ARM64 architecture (e.g., Apple Silicon M1), CPU-only
- **Pod**: A containerized microservice running as part of the Morgan system in Kubernetes
- **MicroK8s**: The lightweight Kubernetes distribution providing container orchestration and service discovery
- **RAG_System**: The Retrieval-Augmented Generation subsystem for document processing and semantic search deployed as Kubernetes Deployments
- **Emotion_Engine**: The subsystem responsible for detecting and analyzing emotional content in user messages (11 modules) deployed as StatefulSets
- **Empathy_Engine**: The subsystem that generates emotionally appropriate responses (5 modules) deployed as Deployments
- **Learning_System**: The subsystem that adapts Morgan's behavior based on feedback (6 modules) deployed as StatefulSets
- **LLM_Service**: The Large Language Model inference service (Ollama with Qwen2.5 or similar) deployed as Deployments with GPU node affinity and NFS storage for models
- **Embedding_Service**: The text embedding generation service (qwen3-embedding:latest via Ollama) deployed as Deployments with GPU node affinity
- **Reranking_Service**: The document reranking service using Jina reranking models via HuggingFace Transformers or sentence-transformers deployed as Deployments
- **Qdrant**: The vector database system deployed as StatefulSets with persistent volumes
- **Istio_Gateway**: The service mesh ingress gateway for routing client requests (MicroK8s Istio addon)
- **Click_CLI**: The Python Click framework for building the command-line interface
- **MinIO**: S3-compatible object storage deployed as StatefulSets with persistent volumes for files (documents, audio, artifacts)
- **PostgreSQL**: Relational database deployed as StatefulSets with persistent volumes for structured data (conversations, emotions, learning). Uses tables/indexes/constraints ONLY, no functions/triggers
- **SQLAlchemy**: Python ORM for database operations (all business logic in application code)
- **MicroK8s_Registry**: Built-in container registry addon for local image storage
- **GPU_Addon**: MicroK8s GPU addon providing NVIDIA GPU support and device plugins for GPU nodes
- **NFS_Storage**: Network File System for shared model storage across nodes (for Ollama models)
- **Ollama_Service**: Ollama LLM inference service deployed as Deployments in MicroK8s cluster (stateless with NFS model storage)
- **Prometheus**: Monitoring addon for basic metrics collection (optional for MVP)
- **Nexus_Proxy**: External network proxy server providing cached access to repositories and registries
- **HF_Proxy**: Hugging Face model proxy at https://nexus.in.lazarev.cloud/repository/hf-proxy/
- **PyPI_Proxy**: Python package proxy at https://nexus.in.lazarev.cloud/repository/pypi-proxy/
- **Harbor_Registry**: External Docker registry with proxies for Docker Hub and GitHub

---

## Requirements

### 1. CLI-Based User Interface (MVP Primary Interface)

**User Story:** As a user, I want to interact with Morgan through a simple CLI interface, so that I can chat, ingest documents, and query the system without needing a web browser.

#### Acceptance Criteria

1. WHEN Morgan CLI is installed, THE Morgan_System SHALL provide the following commands:
   - `morgan chat` - Start interactive chat session
   - `morgan query "<question>"` - Ask single question and get response
   - `morgan ingest <path>` - Ingest documents (PDF, MD, TXT, etc.) for RAG
   - `morgan status` - Show system health, host status, GPU utilization
   - `morgan hosts list` - List all detected hosts and running services
   - `morgan config set <key> <value>` - Configure system settings
2. WHEN using `morgan chat`, THE Morgan_System SHALL:
   - Display conversation history in terminal
   - Support multi-line input (Ctrl+D to send)
   - Show emotional tone detected (optional, via colored output)
   - Provide exit command (`/exit`, `/quit`, or Ctrl+C)
3. WHEN using `morgan ingest <path>`, THE Morgan_System SHALL:
   - Accept file path or directory path
   - Show progress bar for document processing
   - Display summary (X documents processed, Y chunks created, Z embeddings generated)
   - Report any errors with file names
4. WHEN using `morgan status` THEN system SHALL display:
   - All hosts with CPU%, RAM%, GPU% (if applicable), VRAM%
   - All services with health status (✓ healthy, ✗ unhealthy, ⚠ degraded)
   - Consul cluster status
   - PostgreSQL, Redis, Qdrant status
5. WHEN using `morgan hosts list` THEN system SHALL display table:
   - Host name, IP, OS, CPU, RAM, GPU (if any), services running
6. CLI SHALL use Click framework (https://click.palletsprojects.com/) for all commands
7. THE Morgan_System SHALL use local network proxies for all external dependencies to ensure fast and local-first experience

---



### 2. Local Network Infrastructure and Proxies

**User Story:** As a system administrator, I want Morgan V4 to use local network proxies for all external dependencies, so that deployment is fast, reliable, and reduces external bandwidth usage.

#### Acceptance Criteria

1. WHEN downloading Hugging Face models, THE Morgan_System SHALL use HF_Proxy at https://nexus.in.lazarev.cloud/repository/hf-proxy/
2. WHEN installing Python packages, THE Morgan_System SHALL use PyPI_Proxy at https://nexus.in.lazarev.cloud/repository/pypi-proxy/
3. WHEN pulling Docker images from Docker Hub, THE Morgan_System SHALL use Harbor_Registry proxy at harbor.in.lazarev.cloud/proxy/
4. WHEN pulling Docker images from GitHub Container Registry, THE Morgan_System SHALL use Harbor_Registry proxy at harbor.in.lazarev.cloud/gh-proxy/
5. WHEN installing Debian packages, THE Morgan_System SHALL use Debian_Proxy at https://nexus.in.lazarev.cloud/repository/debian-proxy/ and https://nexus.in.lazarev.cloud/repository/debian-security/
6. WHEN installing Ubuntu packages, THE Morgan_System SHALL use Ubuntu_Proxy at https://nexus.in.lazarev.cloud/repository/ubuntu-group/
7. WHEN proxy services are unavailable, THE Morgan_System SHALL fall back to direct internet access with warning message
8. THE Morgan_System SHALL configure all services to prioritize local proxies for dependency resolution

---

### 3. Cloud-Native Multi-Node Architecture

**User Story:** As a system administrator, I want Morgan V4 to automatically detect available nodes and distribute services based on hardware capabilities using Kubernetes-native scheduling, so that I can deploy on any number of machines without manual configuration.

#### Acceptance Criteria

1. WHEN the Morgan_System is deployed THEN Morgan_System SHALL:
   - Auto-detect all available nodes via MicroK8s cluster node discovery
   - Identify node capabilities using Kubernetes node labels (GPU type/VRAM, CPU type, RAM, architecture)
   - Dynamically assign pods based on detected hardware using node selectors and affinity rules
2. WHEN GPU_Node instances are detected THEN Morgan_System SHALL:
   - Query GPU properties via MicroK8s GPU addon and set Kubernetes node labels (nvidia.com/gpu.product, nvidia.com/gpu.memory)
   - Deploy Ollama Deployments to nodes with highest VRAM using node affinity and NFS storage for models
   - Deploy LLM_Service Deployments to nodes with highest VRAM using node affinity
   - Deploy Embedding_Service Deployments (qwen3-embedding via Ollama) to nodes with moderate VRAM (8GB+) using node selectors
   - Deploy Reranking_Service Deployments (Jina models via HuggingFace Transformers) to nodes with minimal VRAM (6GB+) using node selectors
3. WHEN CPU_Node instances are detected THEN Morgan_System SHALL:
   - Set Kubernetes node labels for RAM capacity and CPU type
   - Deploy RAG_System and Qdrant StatefulSets to nodes with highest RAM using node affinity
   - Deploy Emotion_Engine, Empathy_Engine, Learning_System to available CPU nodes using node selectors
4. WHEN ARM64_Node instances (macOS M1) are detected THEN Morgan_System SHALL:
   - Set Kubernetes node labels for ARM64 architecture (kubernetes.io/arch=arm64)
   - Deploy CPU-only services using multi-arch container images
   - Use node selectors to avoid GPU-dependent services on ARM64 nodes
5. IF minimum node count (<1) is not met THEN Morgan_System SHALL:
   - Fail gracefully with error message during MicroK8s cluster initialization
6. WHEN new nodes are added dynamically THEN Morgan_System SHALL:
   - Auto-join MicroK8s cluster within 60 seconds via cluster join tokens
   - Redistribute pods using Kubernetes scheduler and horizontal pod autoscaler
   - Maintain zero-downtime during rebalancing via rolling updates and pod disruption budgets

---

### 4. Advanced RAG System Deployment with Ollama Integration

**User Story:** As a user, I want Morgan to intelligently retrieve information from ingested documents using hierarchical search, with embedding generation distributed across available GPU nodes and LLM inference via Ollama.

#### Acceptance Criteria

1. WHEN documents are ingested (via `morgan ingest`) THEN Morgan_System SHALL:
   - Process PDFs, DOCX, MD, HTML, TXT, and code files
   - Use intelligent semantic-aware chunking
   - Generate hierarchical embeddings (coarse, medium, fine) on available GPU_Node or CPU_Node
   - Store vectors in Qdrant StatefulSet on node with highest available RAM
   - Display progress in CLI with progress bar
2. WHEN embedding generation is requested THEN Morgan_System SHALL:
   - Route to Ollama Deployment running qwen3-embedding:latest on GPU_Node with moderate VRAM (8GB+) if available
   - Fall back to GPU_Node with minimal VRAM (6GB) if needed
   - Fall back to CPU_Node if no GPU available (degraded performance)
   - Batch requests for GPU efficiency (batch size: 32-64)
3. WHEN a RAG query is processed (via `morgan query` or `morgan chat`) THEN Morgan_System SHALL:
   - Perform coarse search across all domains
   - Refine with medium-grained search
   - Execute fine-grained search for precision
   - Use Reciprocal Rank Fusion for result merging
   - Rerank results on available Reranking_Service using Jina models via HuggingFace Transformers (GPU or CPU)
   - Send final prompt to Ollama Deployment for LLM inference
   - Display source attribution in CLI output
4. WHEN Ollama models are deployed THEN Morgan_System SHALL:
   - Store models on NFS shared storage accessible by all GPU_Node instances
   - Load models (Qwen2.5-32B, Qwen2.5-7B, qwen3-embedding:latest) from NFS storage
   - Share model files across multiple Ollama Deployment replicas (stateless pods)
   - Cache frequently used models in local node storage for performance
5. WHEN Jina reranking models are deployed THEN Morgan_System SHALL:
   - Use HuggingFace Transformers or sentence-transformers libraries for model loading
   - Deploy as standard Python Deployments with model caching
   - Support both GPU and CPU inference for reranking tasks
5. WHEN batch processing documents THEN Morgan_System SHALL:
   - Achieve >100 docs/minute on CPU_Node with 64GB RAM
   - Achieve >200 docs/minute on GPU_Node with 8GB+ VRAM
6. WHEN searching 10K documents THEN Morgan_System SHALL return results within 500ms on CPU_Node or 300ms on GPU_Node

---

### 5. Emotion Detection and Empathy Engine Distribution (Dynamic)

**User Story:** As a user, I want Morgan to detect my emotional state and respond empathetically, with processing distributed across available CPU hosts.

#### Acceptance Criteria

1. WHEN a user message is received (via `morgan chat` or `morgan query`) THEN Morgan_System SHALL:
   - Analyze emotional content using 11 emotion detection modules
   - Route to CPU_Host with highest available RAM and CPU capacity
   - Classify emotions (joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
   - Measure emotion intensity (0-1 scale)
   - Detect triggers and patterns
   - Optionally display emotion in CLI (colored output: red for anger, blue for sadness, green for joy)
2. WHEN emotional state is detected THEN Morgan_System SHALL:
   - Generate empathetic responses using 5 empathy modules
   - Mirror emotional tone appropriately
   - Provide validation and support
   - Detect crisis situations and respond accordingly
3. WHEN emotion processing load is high THEN Morgan_System SHALL:
   - Distribute emotion detection across multiple CPU_Host instances
   - Use load balancing (round-robin or least-connection)
   - Queue requests if all instances are saturated
4. IF crisis is detected THEN Morgan_System SHALL:
   - Prioritize empathy engine processing
   - Provide immediate support resources in CLI output
   - Alert configured emergency contacts (if configured)
5. WHEN emotion analysis is complete THEN Morgan_System SHALL store emotional memory in PostgreSQL for future reference

---

### 6. Learning and Adaptation System (Dynamic)

**User Story:** As a user, I want Morgan to learn from my interactions and adapt to my preferences, with learning processing distributed across available hosts.

#### Acceptance Criteria

1. WHEN a conversation occurs THEN Morgan_System SHALL:
   - Track user preferences using Learning_System on available CPU_Host
   - Identify behavioral patterns
   - Store feedback for continuous improvement in PostgreSQL
2. WHEN feedback is provided (via CLI command `morgan feedback <rating> "<comment>"`) THEN Morgan_System SHALL:
   - Integrate feedback into learning model within 5 seconds
   - Adjust response generation strategies
   - Update user preference profiles in PostgreSQL
3. WHEN patterns are discovered THEN Morgan_System SHALL:
   - Adapt communication style
   - Personalize future responses
   - Learn domain-specific knowledge
4. IF user preferences conflict THEN Morgan_System SHALL prioritize most recent feedback (temporal decay)
5. WHEN adaptation occurs THEN Morgan_System SHALL:
   - Maintain privacy and encrypt learning data at rest (AES-256)
   - Store learning data securely in PostgreSQL

---

### 7. GPU Workload Distribution and Load Balancing (Cloud-Native)

**User Story:** As a system administrator, I want GPU resources optimally allocated across all available GPU nodes using Kubernetes-native scheduling, so that inference performance is maximized.

#### Acceptance Criteria

1. WHEN main LLM requests arrive THEN Morgan_System SHALL:
   - Use Kubernetes Services (ollama-service.morgan.svc.cluster.local) for service discovery
   - Load balance using Kubernetes Service load balancing across Ollama Deployment replicas
   - Route to healthy pods automatically via Kubernetes Service endpoints
2. WHEN embedding requests arrive THEN Morgan_System SHALL:
   - Route to Ollama Deployment running qwen3-embedding:latest on GPU_Node with 8GB+ VRAM if available
   - Fall back to GPU_Node with 6GB VRAM if 8GB+ not available
   - Fall back to CPU_Node if no GPU_Node available (with performance warning in CLI)
   - Batch requests for GPU efficiency (batch size: 32-64)
3. WHEN reranking requests arrive THEN Morgan_System SHALL:
   - Route to available Jina Reranking_Service pods (GPU or CPU) via Kubernetes Services
   - Use HuggingFace Transformers or sentence-transformers for model inference
   - Prefer GPU_Node with lowest VRAM usage using node affinity
   - Fall back to CPU_Node if no GPU_Node available
   - Process in batches for efficiency
4. WHEN GPU VRAM is exhausted on any GPU_Node THEN Morgan_System SHALL:
   - Queue requests and display estimated wait time in CLI
   - Use quantized models (Q4_K_M, Q4_0, GGUF) to fit in available VRAM
   - Gracefully degrade to CPU with user notification in CLI
5. WHEN failover occurs THEN Morgan_System SHALL:
   - Automatically retry requests on healthy GPU_Node pods via Kubernetes Services
   - Maintain request context and conversation state via PostgreSQL StatefulSets

---

### 8. Cloud-Native Service Discovery (Kubernetes-Native)

**User Story:** As a developer, I want services to automatically discover each other across any number of nodes using Kubernetes-native service discovery, so that configuration is minimal and failover is automatic.

#### Acceptance Criteria

1. WHEN a Pod starts THEN MicroK8s SHALL:
   - Register pod with Kubernetes Service registry within 30 seconds
   - Configure health check endpoint via Kubernetes readiness probes
2. WHEN a service needs GPU inference THEN service SHALL:
   - Use Kubernetes DNS to discover LLM_Service instances (llm-service.morgan.svc.cluster.local)
   - Use Kubernetes DNS to discover Embedding_Service instances (embedding-service.morgan.svc.cluster.local)
   - Use Kubernetes DNS to discover Reranking_Service instances (reranking-service.morgan.svc.cluster.local)
3. WHEN a GPU_Node pod becomes unavailable THEN MicroK8s SHALL:
   - Remove pod from Service endpoints within 30 seconds (3 failed readiness checks)
   - Trigger pod restart or reschedule to healthy nodes via ReplicaSets
4. WHEN multiple replicas of a service exist THEN MicroK8s SHALL:
   - Provide automatic load balancing via Kubernetes Services
   - Support pod placement constraints based on node selectors

---

### 9. Cloud-Native Data Layer for V4 Features

**User Story:** As a user, I want my conversation history, emotional memory, learning data, and RAG knowledge to be available regardless of which node processes my request.

#### Acceptance Criteria - PostgreSQL (Code-Only Logic)

1. WHEN PostgreSQL schema is designed THEN Morgan_System SHALL:
   - Use tables, indexes, foreign keys, and basic constraints ONLY (PRIMARY KEY, FOREIGN KEY, UNIQUE, NOT NULL, CHECK)
   - Store ALL business logic in application code (Python services), NOT in database
   - NEVER create stored procedures, functions, or triggers in PostgreSQL
   - Use SQLAlchemy or similar ORM for all database operations
   - Enforce referential integrity via foreign keys, but implement cascade logic in application code
2. WHEN conversation data is stored THEN Morgan_System SHALL:
   - Persist to PostgreSQL StatefulSet on designated primary node (highest RAM CPU_Node)
   - Configure async streaming replication to standby StatefulSet if available
   - Store in tables: `conversations`, `messages`, `conversation_metadata`
   - Implement soft deletes via `deleted_at` column (not DB triggers)
   - Handle timestamp creation/updates in application code (not DB defaults beyond CURRENT_TIMESTAMP)
3. WHEN emotional memory is stored THEN Morgan_System SHALL:
   - Store in tables: `emotional_states`, `emotion_patterns`, `emotion_triggers`
   - Link to `messages` table via foreign key
   - Calculate emotion aggregations in application code (not DB views or functions)
4. WHEN learning data is stored THEN Morgan_System SHALL:
   - Store in tables: `user_preferences`, `behavioral_patterns`, `feedback`
   - Implement preference decay/prioritization logic in Learning_System service code
   - Use PostgreSQL for storage only, not processing
5. WHEN RAG document metadata is stored THEN Morgan_System SHALL:
   - Store in tables: `documents`, `document_chunks`, `chunk_metadata`
   - Store file paths/URLs to MinIO S3 buckets (not file content in PostgreSQL)
   - Implement document versioning logic in application code

#### Acceptance Criteria - MinIO (S3-Compatible File Storage)

1. WHEN MinIO is deployed THEN Morgan_System SHALL:
   - Deploy MinIO as StatefulSet with persistent volumes on primary CPU_Node
   - Support external MinIO via configurable Service/Ingress (e.g., `http://minio.morgan.svc.cluster.local:9000`)
   - Configure S3-compatible client (boto3 or MinIO Python SDK) via Kubernetes ConfigMaps
   - Use TLS for external MinIO connections via Istio service mesh
2. WHEN documents are ingested (via `morgan ingest`) THEN Morgan_System SHALL:
   - Upload original files to MinIO bucket `morgan-documents` via Kubernetes Jobs
   - Store file metadata in PostgreSQL `documents` table (file_path, bucket, object_key, size, mime_type, upload_timestamp)
   - Generate unique object keys (e.g., `{user_id}/{upload_date}/{uuid}_{filename}`)
   - Support file types: PDF, DOCX, MD, HTML, TXT, code files
3. WHEN processed chunks are stored THEN Morgan_System SHALL:
   - Store chunk text in PostgreSQL `document_chunks` table (for fast retrieval)
   - Optionally store processed artifacts (images, tables) in MinIO bucket `morgan-artifacts`
   - Link chunks to original documents via foreign key
4. WHEN audio files are processed (future TTS/STT) THEN Morgan_System SHALL:
   - Upload audio files to MinIO bucket `morgan-audio`
   - Store audio metadata in PostgreSQL (file_path, duration, format, sample_rate)
5. WHEN file retrieval is needed THEN Morgan_System SHALL:
   - Retrieve file metadata from PostgreSQL
   - Download file from MinIO using object_key via Kubernetes Services
   - Cache frequently accessed files in Redis StatefulSet (configurable, max 100MB per file)
   - Generate pre-signed URLs for direct file access (1-hour expiry)
6. WHEN MinIO is unavailable THEN Morgan_System SHALL:
   - Queue upload operations for later retry using Kubernetes Jobs
   - Use cached file data from Redis if available
   - Display warning in CLI about degraded file storage

#### Acceptance Criteria - Semantic Memory and Caching

1. WHEN semantic memory is created THEN Morgan_System SHALL:
   - Store embeddings in Qdrant StatefulSet on node with highest available RAM
   - Store conversation context in Redis StatefulSet cluster (minimum 3 replicas if available, else single instance)
   - Store emotional memory metadata in PostgreSQL StatefulSet
   - Store learning patterns metadata in PostgreSQL StatefulSet
2. WHEN RAG embeddings are generated THEN Morgan_System SHALL:
   - Store hierarchical embeddings in Qdrant StatefulSet (collections: `documents_coarse`, `documents_medium`, `documents_fine`)
   - Store chunk metadata in PostgreSQL `document_chunks` table
   - Cache frequently accessed chunks in Redis StatefulSet with 1-hour TTL
3. WHEN database connection fails THEN Morgan_System SHALL:
   - Cache writes locally on CPU_Node pods (local SQLite for temporary storage via emptyDir volumes)
   - Sync when connectivity is restored (eventual consistency)
   - Maintain write-ahead log for durability using persistent volumes
4. WHEN vector search is performed THEN Morgan_System SHALL:
   - Use Qdrant StatefulSet collections with hierarchical search
   - Cache search results in Redis StatefulSet with 1-hour TTL
   - Invalidate cache when new documents are ingested via Kubernetes Events

---



### 11. Cloud-Native Health Monitoring and Observability for V4

**User Story:** As a system administrator, I want comprehensive health monitoring across all nodes using cloud-native observability tools, so that I can detect and resolve issues proactively.

#### Acceptance Criteria

1. WHEN services run THEN services SHALL report health every 30 seconds including:
   - GPU utilization and VRAM usage (for GPU_Node)
   - CPU utilization and RAM usage
   - Model loading status
   - Request queue depth
2. WHEN running `morgan status` command THEN CLI SHALL display:
   - All Kubernetes nodes with CPU%, RAM%, GPU% (if applicable), VRAM%
   - All pods color-coded by health status (✓ green, ⚠ yellow, ✗ red)
   - Kubernetes deployment replica status and node placement
3. WHEN logs are generated THEN Morgan_System SHALL:
   - Store logs locally on each node
   - Include structured metadata (node, pod, user_id)

---

### 12. Cloud-Native Platform-Specific Optimization

**User Story:** As a system administrator, I want Morgan to optimize for each node's hardware using Kubernetes resource management, so that performance is maximized across diverse configurations.

#### Acceptance Criteria

1. WHEN running on GPU_Node with high VRAM (12GB+) THEN Morgan_System SHALL:
   - Deploy Ollama Deployment with largest available LLM model (Qwen2.5-32B-Instruct Q4_K_M or larger)
   - Use CUDA 12.4 with cuBLAS for matrix operations via MicroK8s GPU addon
   - Mount NFS storage for shared model access across stateless replicas
   - Enable GPU resource requests and limits in Kubernetes manifests
2. WHEN running on GPU_Node with moderate VRAM (8-16GB) THEN Morgan_System SHALL:
   - Deploy Ollama Deployment with embedding models (qwen3-embedding:latest) and/or medium LLM (Qwen2.5-7B-Instruct)
   - Use CUDA 12.4 with optimized batch processing
   - Share model storage via NFS across multiple GPU nodes
3. WHEN running on GPU_Node with minimal VRAM (6GB) THEN Morgan_System SHALL:
   - Deploy Jina reranking models via HuggingFace Transformers or sentence-transformers Deployments
   - Use MicroK8s GPU addon for GPU resource allocation
   - Fall back to CPU if VRAM insufficient via node affinity rules
4. WHEN running on CPU_Node with high RAM (64GB) THEN Morgan_System SHALL:
   - Host Qdrant vector database StatefulSet with large cache
   - Host PostgreSQL primary StatefulSet with generous shared_buffers (16GB)
   - Host Redis StatefulSet with large maxmemory (8GB)
   - Run RAG_System, Emotion_Engine, Empathy_Engine Deployments
5. WHEN running on CPU_Node with moderate RAM (32GB) THEN Morgan_System SHALL:
   - Host smaller Qdrant collections or PostgreSQL standby StatefulSets
   - Run Learning_System Deployments or background processing Jobs
6. WHEN running on ARM64_Node (macOS M1) THEN Morgan_System SHALL:
   - Use ARM64-compatible multi-arch container images
   - Run CPU-only services (LLM proxy, lightweight coordination) via Deployments
   - Avoid GPU-dependent services

---

### 13. Cloud-Native High Availability and Failover for V4

**User Story:** As a user, I want Morgan to remain available even if individual nodes or pods fail using Kubernetes-native resilience, so that I can rely on it for critical tasks.

#### Acceptance Criteria

1. WHEN any GPU_Node with LLM_Service fails THEN Morgan_System SHALL:
   - Automatically failover to next available GPU_Node with LLM_Service within 10 seconds via Kubernetes Services
   - Maintain conversation context from PostgreSQL StatefulSet with persistent volumes
2. WHEN primary CPU_Node fails THEN Morgan_System SHALL:
   - Failover orchestration to secondary CPU_Node via Kubernetes scheduler
   - Restore full functionality within 60 seconds via readiness probes
3. WHEN Qdrant becomes unavailable THEN Morgan_System SHALL:
   - Use fallback keyword search (PostgreSQL full-text search)
   - Display warning in CLI about degraded RAG performance
   - Kubernetes SHALL automatically restart Qdrant StatefulSet on healthy nodes
4. WHEN all GPU_Node instances are unavailable THEN Morgan_System SHALL:
   - Fall back to CPU-based inference on CPU_Node pods (degraded performance)
   - Use smaller quantized models (Q4_0, Q2_K, GGUF)
   - Display warning in CLI about degraded performance
5. WHEN a node is shut down for maintenance THEN Morgan_System SHALL:
   - Kubernetes SHALL automatically redistribute pods to remaining healthy nodes
   - Maintain service availability throughout maintenance window via rolling updates

---



## Non-Functional Requirements

### Performance (Dynamic Scaling)

1. **Latency**:
   - Text responses (simple): <1 second on GPU_Host, <2 seconds on CPU_Host
   - Text responses (RAG-enhanced): <2 seconds on GPU_Host, <3 seconds on CPU_Host
   - Embedding generation: <500ms per document on GPU_Host, <2s on CPU_Host
   - Emotion detection: <200ms per message on CPU_Host with 64GB RAM
   - Reranking: <300ms for top-100 results on GPU_Host, <1s on CPU_Host

2. **Throughput** (scales with host count):
   - Main LLM: 10-20 requests/minute per GPU_Host
   - Embeddings: 50-100 embeddings/minute per GPU_Host, 10-20 per CPU_Host
   - Document ingestion: >100 docs/minute per CPU_Host with 64GB RAM
   - Concurrent users: 5-10 per host via CLI

### Scalability

1. The Morgan_System SHALL support adding GPU_Host or CPU_Host without service interruption
2. The Morgan_System SHALL scale from 1 host (single-node) to 10+ hosts (distributed)
3. The Morgan_System SHALL automatically rebalance services when hosts are added/removed

### Reliability

1. The Morgan_System SHALL achieve 99.5% uptime with 2+ hosts (HA configuration)
2. The Morgan_System SHALL automatically recover from transient failures within 60 seconds

### Maintainability

1. The Morgan_System SHALL provide centralized logging and monitoring
2. The Morgan_System SHALL support zero-downtime updates via rolling deployment

---

## Minimum Hardware Requirements

### Minimum Deployment (Single Node MicroK8s)

- CPU: Intel i7 9th gen or i9 11th/13th gen
- RAM: 32GB (minimum for MicroK8s + Morgan workloads)
- GPU: Optional (NVIDIA with 6GB+ VRAM and CUDA 12.4+ support)
- Storage: 100GB SSD (for container images, persistent volumes, etcd)
- OS: Ubuntu 22.04+ (recommended), Debian 11+, Windows 11 (with WSL2), or macOS (M1+)
- MicroK8s: Version 1.28+ with required addons (dns, storage, registry, gpu if GPU present)
- Network: Access to local proxy infrastructure (nexus.in.lazarev.cloud, harbor.in.lazarev.cloud)

### Recommended Deployment (3+ Node MicroK8s Cluster)

**Primary CPU Node** (Control Plane + Worker):
- CPU: i9 11th/13th gen
- RAM: 64GB
- Storage: 500GB SSD (for etcd, container images, persistent volumes)
- Role: Kubernetes control plane, Qdrant StatefulSet, PostgreSQL StatefulSet, Redis StatefulSet, RAG orchestration, Emotion, Empathy
- MicroK8s: Control plane node with basic addons (dns, storage, registry)
- NFS: NFS server for shared model storage (can be on same node or external)

**Secondary CPU Node** (Worker):
- CPU: i9 11th/13th gen or i7 9th gen
- RAM: 32GB minimum, 64GB recommended
- Storage: 250GB SSD
- Role: Learning system, Monitoring (Prometheus/Grafana), PostgreSQL standby, Redis cluster node
- MicroK8s: Worker node

**GPU Node(s)** (Worker):
- CPU: i9 11th/13th gen or i7 9th gen
- RAM: 32GB minimum
- GPU: NVIDIA with CUDA 12.4+ support
  - 12GB VRAM: Main LLM StatefulSets (Qwen 32B)
  - 8-16GB VRAM: Embeddings + Fast LLM Deployments
  - 6GB VRAM: Reranking Deployments
- Storage: 100GB SSD
- MicroK8s: Worker node with GPU addon and NVIDIA drivers
- NFS: NFS client for accessing shared model storage

**ARM64 Node (macOS M1)** (optional Worker):
- CPU: Apple Silicon M1/M2/M3
- RAM: 16GB minimum
- Storage: 100GB SSD
- Role: CPU-only services, LLM proxy Deployments
- MicroK8s: Worker node with ARM64 support

---

## Acceptance Criteria Summary

The V4 Cloud-Native MVP is considered complete when:

1. ✅ Morgan operates across all available nodes (minimum 1, tested up to 7) in MicroK8s cluster
2. ✅ All services use local network proxies for fast dependency resolution via proxy infrastructure
3. ✅ CLI interface (`morgan chat`, `morgan ingest`, `morgan query`, `morgan status`) works correctly with Kubernetes Services
4. ✅ Services auto-detect and utilize GPU capabilities (CUDA 12.4+) via MicroK8s GPU addon with full NVIDIA support
5. ✅ RAG system processes documents with hierarchical search via `morgan ingest` using Deployments and Jobs with Ollama integration
6. ✅ Emotion detection and empathy engine respond in <200ms on CPU_Node via optimized Deployments
7. ✅ Learning system adapts based on user feedback via `morgan feedback` using StatefulSets
8. ✅ Services automatically discover each other via Kubernetes DNS
9. ✅ Failover works between available GPU_Node instances via Kubernetes Services and readiness probes
10. ✅ `morgan status` command shows all nodes with dynamic service metrics via Kubernetes metrics API
11. ✅ E2E tests pass for RAG, emotion, empathy, learning across multi-node MicroK8s setup
12. ✅ System survives random node failure chaos tests via Kubernetes resilience features
13. ✅ P95 latency <2s for RAG-enhanced queries on GPU_Node, <3s on CPU_Node
14. ✅ System scales from 1 node to 7+ nodes without configuration changes via Kubernetes scheduler

---

## Out of Scope for MVP

- **Web UI**: MVP uses CLI only (Click-based interface), web UI is future enhancement
- **WebSocket/Real-time streaming**: HTTP-only for MVP, WebSocket for future
- **Mobile client applications**: Future enhancement
- **Multi-region deployment**: Geographic distribution not in MVP scope (single MicroK8s cluster)
- **Advanced GPU scheduling**: Multi-GPU per node not supported (single GPU per node only with max of 12GB for now)
- **Full Kubernetes features**: Using MicroK8s with essential addons only, advanced Kubernetes features are future enhancements
- **Real-time voice processing**: TTS/STT services not in MVP (future enhancement)
- **Integration with external services**: Home Assistant, etc. are future enhancements
- **Advanced service mesh features**: Service mesh is future enhancement, MVP uses basic Kubernetes Services
- **Multi-cluster deployment**: Single MicroK8s cluster only, federation is future enhancement

---

**Document Status**: Ready for design phase (Cloud-Native Multi-Node V4.0 - MicroK8s CLI-Only)
**Next Steps**: Update design document with MicroK8s architecture and cloud-native deployment patterns
