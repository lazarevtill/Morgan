# Morgan Multi-Host MVP - Design Document

> **Version**: 4.0.0
> **Status**: Updated for Cloud-Native MicroK8s Architecture
> **Last Updated**: 2025-11-02
> **Feature**: Cloud-Native Multi-Host AI Assistant with RAG, Empathy, and Learning (CLI-Only MVP)

---

## Overview

This design document outlines the architecture for Morgan V4: a cloud-native, self-hosted AI assistant platform with advanced capabilities including RAG (Retrieval-Augmented Generation), emotion detection, empathy engine, and continuous learning. The MVP focuses on a **CLI-only interface** using MicroK8s for container orchestration and service management.

### Design Goals

1. **Cloud-Native Architecture**: Use MicroK8s for lightweight Kubernetes orchestration with built-in service mesh, ingress, and storage capabilities
2. **Flexible Node Allocation**: Support any number of nodes (minimum 1, recommended 3-7) with dynamic pod distribution based on hardware capabilities
3. **CLI-First Interface**: Use Python Click framework for all user interactions (no WebUI for MVP)
4. **Code-Only Database Logic**: PostgreSQL for structured data with ALL business logic in application code (no stored procedures, functions, or triggers)
5. **S3-Compatible File Storage**: MinIO for document storage (local or remote)
6. **Local Network Infrastructure**: Use local network proxies for all external dependencies (nexus.in.lazarev.cloud, harbor.in.lazarev.cloud)
7. **Kubernetes-Native Service Discovery**: Use Kubernetes Services and DNS for service discovery instead of external tools
8. **Ollama Integration**: Use Ollama for LLM inference with shared NFS model storage across GPU nodes
9. **Performance Targets**: <1s text responses on GPU, <2s RAG-enhanced responses, <200ms emotion detection
10. **OS-Agnostic**: Support Windows 11, Debian-based Linux, and macOS M1 with platform-specific optimizations
11. **Self-Healing**: Automatic pod recovery and failover via Kubernetes
12. **Developer-Friendly**: Maintain familiar deployment experience with Kubernetes manifests

---

## Architecture Design

### System Architecture Diagram

```mermaid
graph TB
    subgraph "CLI Client"
        CLI[Morgan CLI<br/>Python Click<br/>HTTPx + Rich]
    end

    subgraph "MicroK8s Cluster Orchestration"
        K8S[MicroK8s Cluster<br/>Service Discovery<br/>Load Balancing<br/>Health Checks<br/>Config Management]
        ISTIO[Istio Service Mesh<br/>Optional Gateway]
    end

    subgraph "Dynamic Node Pool (1-7+ nodes)"
        direction LR

        subgraph "GPU_Node_1 (12GB VRAM)"
            H1_OLLAMA[Ollama Deployment<br/>Qwen2.5-32B<br/>CUDA 12.4 + NFS]
            H1_K8S_WORKER[MicroK8s Worker<br/>GPU Addon]
        end

        subgraph "GPU_Node_2 (8GB VRAM)"
            H2_OLLAMA_EMBED[Ollama Deployment<br/>qwen3-embedding<br/>CUDA 12.4 + NFS]
            H2_OLLAMA_FAST[Ollama Deployment<br/>Qwen2.5-7B<br/>CUDA 12.4 + NFS]
            H2_K8S_WORKER[MicroK8s Worker<br/>GPU Addon]
        end

        subgraph "GPU_Node_N (6GB VRAM)"
            HN_RERANK[Reranking Deployment<br/>Jina Models<br/>HuggingFace Transformers]
            HN_K8S_WORKER[MicroK8s Worker<br/>GPU Addon]
        end

        subgraph "CPU_Node_1 (Primary - 64GB RAM)"
            HC1_CORE[Core RAG Deployment x2]
            HC1_EMOTION[Emotion Engine Deployment<br/>11 modules]
            HC1_EMPATHY[Empathy Engine Deployment<br/>5 modules]
            HC1_POSTGRES[PostgreSQL StatefulSet]
            HC1_QDRANT[Qdrant StatefulSet]
            HC1_REDIS[Redis StatefulSet]
            HC1_MINIO[MinIO StatefulSet<br/>Optional Local]
            HC1_K8S_CONTROL[MicroK8s Control Plane]
        end

        subgraph "CPU_Node_2 (Secondary - 32GB RAM)"
            HC2_LEARNING[Learning System Deployment<br/>6 modules]
            HC2_POSTGRES_STANDBY[PostgreSQL Standby StatefulSet]
            HC2_REDIS2[Redis StatefulSet Node 2]
            HC2_MONITORING[Prometheus + Grafana<br/>Optional Addon]
            HC2_K8S_WORKER[MicroK8s Worker]
        end

        subgraph "ARM64_Node (macOS M1 - Optional)"
            ARM_CORE[Core Proxy Deployment<br/>ARM64]
            ARM_LLM[LLM Proxy Deployment<br/>ARM64]
            ARM_K8S_WORKER[MicroK8s Worker<br/>ARM64]
        end

        subgraph "NFS Storage (Shared)"
            NFS[NFS Server<br/>Ollama Models<br/>Shared across GPU nodes]
        end
    end

    subgraph "External Services"
        MINIO_REMOTE[MinIO Remote<br/>Optional<br/>192.168.101.5:9000]
    end

    CLI -->|Kubernetes Services| K8S
    CLI -->|Optional Istio Gateway| ISTIO

    ISTIO --> K8S
    K8S --> HC1_CORE

    HC1_CORE -->|K8s Services| H1_OLLAMA
    HC1_CORE -->|K8s Services| H2_OLLAMA_EMBED
    HC1_CORE -->|K8s Services| H2_OLLAMA_FAST
    HC1_CORE -->|K8s Services| HN_RERANK
    HC1_CORE -->|K8s Services| HC1_EMOTION
    HC1_CORE -->|K8s Services| HC1_EMPATHY
    HC1_CORE -->|K8s Services| HC2_LEARNING

    HC1_CORE --> HC1_POSTGRES
    HC1_CORE --> HC1_QDRANT
    HC1_CORE --> HC1_REDIS
    HC1_CORE --> HC1_MINIO
    HC1_CORE --> MINIO_REMOTE

    HC1_POSTGRES -->|Streaming Replication| HC2_POSTGRES_STANDBY
    HC1_REDIS -->|Cluster| HC2_REDIS2

    H1_OLLAMA --> NFS
    H2_OLLAMA_EMBED --> NFS
    H2_OLLAMA_FAST --> NFS

    K8S --> HC2_MONITORING
    HC1_CORE --> HC2_MONITORING
```

### Data Flow Diagram

```mermaid
sequenceDiagram
    participant CLI as Morgan CLI
    participant K8S as Kubernetes DNS
    participant Core as Core RAG Orchestrator
    participant Ollama as Ollama Service
    participant Embed as Embedding Service (Ollama)
    participant Rerank as Reranking Service
    participant Emotion as Emotion Engine
    participant Empathy as Empathy Engine
    participant Learning as Learning System
    participant PG as PostgreSQL
    participant Qdrant as Qdrant Vector DB
    participant Redis as Redis Cache
    participant MinIO as MinIO S3

    CLI->>K8S: Resolve core-service.morgan.svc.cluster.local
    K8S-->>CLI: Load balanced endpoint
    CLI->>Core: POST /api/chat {"text": "What did I say about AI?"}

    Core->>Redis: GET conversation:user123
    Redis-->>Core: Cache hit -> context

    Core->>Emotion: POST /analyze {"text": "..."}
    Emotion-->>Core: {emotion: "curiosity", intensity: 0.7}

    Core->>K8S: Resolve embedding-service.morgan.svc.cluster.local
    K8S-->>Core: ollama-embedding pod endpoint

    Core->>Embed: POST /api/embeddings {"input": "What did I say about AI?"}
    Embed-->>Core: vector[768]

    Core->>Qdrant: Search in documents_coarse
    Qdrant-->>Core: Top 100 candidates

    Core->>Qdrant: Search in documents_medium (filtered)
    Qdrant-->>Core: Top 50 candidates

    Core->>K8S: Resolve reranking-service.morgan.svc.cluster.local
    K8S-->>Core: reranking pod endpoint

    Core->>Rerank: POST /rerank {query, candidates}
    Rerank-->>Core: Top 10 reranked results

    Core->>PG: SELECT chunks WHERE chunk_id IN (...)
    PG-->>Core: Full chunk text

    Core->>K8S: Resolve ollama-service.morgan.svc.cluster.local
    K8S-->>Core: ollama main LLM pod endpoint

    Core->>Ollama: POST /api/generate {prompt + RAG context + emotion}
    Ollama-->>Core: "You mentioned AI ethics in our previous chat..."

    Core->>Empathy: POST /enhance {text, emotion}
    Empathy-->>Core: Enhanced empathetic response

    Core->>PG: INSERT INTO messages (conversation_id, role, content, ...)
    Core->>PG: INSERT INTO emotional_states (message_id, emotion_type, ...)

    Core->>Redis: SET conversation:user123 (updated context)

    Core->>Learning: POST /adapt {user_id, feedback_implicit}
    Learning-->>Core: OK

    Core-->>CLI: JSON {text, emotion, sources: [doc1, doc2]}
```

---

## Cloud-Native MicroK8s Infrastructure

**Design Rationale**: MicroK8s provides a lightweight, production-ready Kubernetes distribution that simplifies deployment while maintaining cloud-native benefits. It includes built-in addons for DNS, storage, registry, and GPU support, eliminating the need for external orchestration tools.

### MicroK8s Architecture Benefits

1. **Lightweight Kubernetes**: Minimal resource overhead compared to full Kubernetes distributions
2. **Built-in Addons**: DNS, storage, registry, GPU support, and optional service mesh
3. **Multi-Architecture Support**: Native support for x86_64 and ARM64 (macOS M1)
4. **Simplified Networking**: Automatic pod-to-pod communication and service discovery
5. **Persistent Storage**: Built-in storage classes for StatefulSets
6. **GPU Integration**: Native NVIDIA GPU support via GPU addon

## Local Network Proxy Infrastructure

**Design Rationale**: To ensure fast, reliable, and local-first deployment, all external dependencies are routed through local network proxies. This reduces external bandwidth usage, improves deployment speed, and provides better reliability.

### Proxy Configuration

**Hugging Face Models**:
- **Proxy URL**: https://nexus.in.lazarev.cloud/repository/hf-proxy/
- **Usage**: All non ollama model downloads (Qwen2.5, nomic-embed-text, CrossEncoder models)
- **Fallback**: Direct Hugging Face Hub if proxy unavailable

**Python Packages**:
- **Proxy URL**: https://nexus.in.lazarev.cloud/repository/pypi-proxy/
- **Usage**: All pip installations in Docker containers and virtual environments
- **Configuration**: `pip config set global.index-url https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple`

**Docker Images**:
- **Docker Hub Proxy**: harbor.in.lazarev.cloud/proxy/
- **GitHub Container Registry Proxy**: harbor.in.lazarev.cloud/gh-proxy/
- **Usage**: All Docker image pulls
- **Configuration**: Docker daemon.json registry mirrors

**System Packages**:
- **Debian Proxy**: https://nexus.in.lazarev.cloud/repository/debian-proxy/
- **Debian Security**: https://nexus.in.lazarev.cloud/repository/debian-security/
- **Ubuntu Proxy**: https://nexus.in.lazarev.cloud/repository/ubuntu-group/
- **Usage**: All apt package installations in containers

### Implementation

**MicroK8s Configuration**:

```bash
# Configure MicroK8s to use proxy registries
microk8s config set registry-mirrors.docker.io harbor.in.lazarev.cloud/proxy/
microk8s config set insecure-registries harbor.in.lazarev.cloud

# Enable required addons
microk8s enable dns storage registry
microk8s enable gpu  # On GPU nodes only
```

**Dockerfile Template**:

```dockerfile
# Use proxy for base images
FROM harbor.in.lazarev.cloud/proxy/python:3.11-slim

# Configure apt sources for proxy
RUN echo "deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ bookworm main" > /etc/apt/sources.list && \
    echo "deb https://nexus.in.lazarev.cloud/repository/debian-security/ bookworm-security main" >> /etc/apt/sources.list

# Configure pip for proxy
RUN pip config set global.index-url https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple && \
    pip config set global.trusted-host nexus.in.lazarev.cloud

# Install dependencies via proxy
COPY requirements.txt .
RUN pip install -r requirements.txt

# Configure Hugging Face for proxy
ENV HF_ENDPOINT=https://nexus.in.lazarev.cloud/repository/hf-proxy/
```

**Kubernetes Deployment with Proxy Configuration**:

```yaml
# kubernetes/core-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: core-rag-orchestrator
  namespace: morgan
spec:
  replicas: 2
  selector:
    matchLabels:
      app: core-rag
  template:
    metadata:
      labels:
        app: core-rag
    spec:
      containers:
      - name: core-rag
        image: harbor.in.lazarev.cloud/morgan/core:latest
        env:
        - name: HF_ENDPOINT
          value: "https://nexus.in.lazarev.cloud/repository/hf-proxy/"
        - name: PYPI_INDEX_URL
          value: "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      nodeSelector:
        node-type: cpu
```

**Service Configuration**:

```python
# shared/config/proxy_config.py
import os
from typing import Optional

class ProxyConfig:
    """Configuration for local network proxies"""
    
    HF_PROXY = "https://nexus.in.lazarev.cloud/repository/hf-proxy/"
    PYPI_PROXY = "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
    DOCKER_PROXY = "harbor.in.lazarev.cloud/proxy/"
    DEBIAN_PROXY = "https://nexus.in.lazarev.cloud/repository/debian-proxy/"
    
    @classmethod
    def get_hf_endpoint(cls) -> str:
        """Get Hugging Face endpoint with proxy fallback"""
        return os.getenv("HF_ENDPOINT", cls.HF_PROXY)
    
    @classmethod
    def configure_huggingface(cls):
        """Configure Hugging Face to use local proxy"""
        os.environ["HF_ENDPOINT"] = cls.get_hf_endpoint()
        
        # Fallback to official HF if proxy fails
        try:
            import requests
            response = requests.get(f"{cls.HF_PROXY}/api/models", timeout=5)
            if response.status_code != 200:
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        except:
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
```

**Kubernetes ConfigMap for Proxy Settings**:

```yaml
# kubernetes/proxy-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: proxy-config
  namespace: morgan
data:
  HF_ENDPOINT: "https://nexus.in.lazarev.cloud/repository/hf-proxy/"
  PYPI_INDEX_URL: "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
  DOCKER_REGISTRY_MIRROR: "harbor.in.lazarev.cloud/proxy/"
  DEBIAN_PROXY: "https://nexus.in.lazarev.cloud/repository/debian-proxy/"
  UBUNTU_PROXY: "https://nexus.in.lazarev.cloud/repository/ubuntu-group/"
```

**Ollama Model Download with Proxy**:

```python
# services/ollama/model_manager.py
import requests
import os
from shared.config.proxy_config import ProxyConfig

class OllamaModelManager:
    def __init__(self, ollama_host: str = "localhost:11434"):
        self.ollama_host = ollama_host
        ProxyConfig.configure_huggingface()
    
    def pull_model(self, model_name: str):
        """Pull model via Ollama with proxy configuration"""
        try:
            # Configure Ollama to use proxy for model downloads
            response = requests.post(
                f"http://{self.ollama_host}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    print(f"Pulling {model_name}: {line.decode()}")
                    
            return True
        except Exception as e:
            print(f"Model pull failed: {e}")
            return False
    
    def list_models(self):
        """List available models in Ollama"""
        try:
            response = requests.get(f"http://{self.ollama_host}/api/tags")
            return response.json()
        except Exception as e:
            print(f"Failed to list models: {e}")
            return {"models": []}
```

**NFS Storage for Shared Models**:

```yaml
# kubernetes/nfs-storage.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ollama-models-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: nfs-server.morgan.svc.cluster.local
    path: /exports/ollama-models
  persistentVolumeReclaimPolicy: Retain

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-models-pvc
  namespace: morgan
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
```

---

## Component Design

### 1. CLI Interface (Morgan CLI)

**Responsibilities**:
- User interaction via command-line interface
- Connect directly to Kubernetes Services (or Istio Gateway if enabled)
- Display rich terminal output with progress bars, colored text, tables
- Handle user input (interactive chat, one-shot queries, file paths)

**Technology Stack**:
- **Click**: Command-line framework (https://click.palletsprojects.com/)
- **HTTPx**: Async HTTP client for API calls
- **Rich**: Terminal formatting (progress bars, tables, colored output)
- **Prompt Toolkit**: Interactive chat input with multi-line support

**Commands**:

```python
# morgan/cli/main.py
import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

@click.group()
@click.version_option(version="3.2.0")
def cli():
    """Morgan AI Assistant CLI - Multi-Host Distributed AI with RAG, Empathy, and Learning"""
    pass

@cli.command()
@click.option('--user-id', default='default', help='User ID for conversation')
@click.option('--emotion/--no-emotion', default=True, help='Show emotional tone in responses')
def chat(user_id, emotion):
    """Start interactive chat session with multi-line input support"""
    console = Console()
    console.print("[bold green]Morgan Chat Session[/bold green]")
    console.print("Type your message and press Ctrl+D to send, or /exit to quit")
    # Use Prompt Toolkit for multi-line input
    # Connect to core.service.consul:8000 or kong:8000
    # Display responses with emotion coloring if enabled
    # Support commands: /exit, /quit, /clear, /history

@cli.command()
@click.argument('question')
@click.option('--user-id', default='default')
@click.option('--json', 'output_json', is_flag=True, help='Output response as JSON')
@click.option('--emotion', is_flag=True, help='Show detected emotional state')
def query(question, user_id, output_json, emotion):
    """Ask a single question and get response (stateless)"""
    # POST to /api/chat with one-shot query
    # Display response and exit with appropriate exit code
    # Support --json flag for machine-readable output

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--user-id', default='default')
@click.option('--recursive/--no-recursive', default=True, help='Process directories recursively')
def ingest(path, user_id, recursive):
    """Ingest documents for RAG (PDF, MD, TXT, DOCX, HTML)"""
    console = Console()
    with Progress() as progress:
        # Upload files to MinIO via Core service
        # Show progress bar for chunking and embedding
        # Display summary: X docs, Y chunks, Z embeddings
        # Report errors with file names

@cli.command()
@click.option('--watch', is_flag=True, help='Auto-refresh every 5 seconds')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def status(watch, output_json):
    """Show system health and host status"""
    # Query /api/status endpoint
    # Display table with all hosts, services, metrics
    # Show: Host, CPU%, RAM%, GPU%, VRAM%, Services, Health
    # Support --watch flag for continuous monitoring

@cli.group()
def hosts():
    """Manage and monitor hosts"""
    pass

@hosts.command('list')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def hosts_list(output_json):
    """List all detected hosts and their capabilities"""
    # Query Consul catalog and KV for host capabilities
    # Display table: Host, IP, OS, Arch, CPU, RAM, GPU, VRAM, Services
    # Show service allocation and health status

@hosts.command('capabilities')
@click.argument('hostname', required=False)
def hosts_capabilities(hostname):
    """Show detailed hardware capabilities for host(s)"""
    # Query Consul KV for detailed host capabilities
    # Show GPU models, CUDA versions, compute capabilities
    # If hostname provided, show only that host

@cli.command()
@click.option('--rating', type=click.IntRange(1, 5), help='Rating from 1-5')
@click.option('--comment', help='Feedback comment')
@click.option('--conversation-id', help='Specific conversation ID')
def feedback(rating, comment, conversation_id):
    """Provide feedback on interactions for learning system"""
    # POST to /api/feedback
    # Link to last conversation if no conversation_id provided
    # Store in PostgreSQL via Core service
    # Display confirmation

@cli.group()
def config():
    """Configure system settings via Consul KV"""
    pass

@config.command('get')
@click.argument('key')
def config_get(key):
    """Get configuration value"""
    # GET from Consul KV /v1/kv/morgan/config/{key}
    # Support dot notation for nested keys

@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set configuration value"""
    # PUT to Consul KV /v1/kv/morgan/config/{key}
    # Validate values (e.g., temperature 0-2, max_tokens > 0)
    # Support dot notation for nested keys

@config.command('list')
@click.option('--prefix', default='', help='Filter by key prefix')
def config_list(prefix):
    """List all configuration keys and values"""
    # GET from Consul KV /v1/kv/morgan/config/?recurse
    # Display as table: Key, Value, Type
    # Support filtering by prefix

@config.command('delete')
@click.argument('key')
@click.confirmation_option(prompt='Are you sure you want to delete this config?')
def config_delete(key):
    """Delete configuration key"""
    # DELETE from Consul KV /v1/kv/morgan/config/{key}
    # Require confirmation for safety

# Hidden debug commands
@cli.group(hidden=True)
def debug():
    """Debug and troubleshooting commands"""
    pass

@debug.command()
def consul():
    """Show Consul cluster status"""
    # Query Consul /v1/status/leader and /v1/catalog/nodes
    # Display cluster health and node status

@debug.command()
def services():
    """Show all registered services"""
    # Query Consul /v1/catalog/services
    # Display service registry with health checks
```

**Connection Logic**:

```python
# morgan/cli/client.py
import httpx
from kubernetes import client, config
from typing import Optional

class MorganClient:
    def __init__(self, base_url: Optional[str] = None):
        # MVP: Direct connection to MicroK8s
        self.base_url = base_url or self._discover_core_via_k8s()

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            follow_redirects=True,
            timeout=30.0,
        )

    def _discover_core_via_k8s(self) -> str:
        """Discover Core service via Kubernetes API (no auth for MVP)"""
        try:
            # Try to load kubeconfig from default location
            config.load_kube_config()
            v1 = client.CoreV1Api()
            
            # Get core service endpoints
            service = v1.read_namespaced_service(
                name="core-service",
                namespace="morgan"
            )
            
            # For MVP, use NodePort or port-forward
            # In production, would use proper ingress
            cluster_ip = service.spec.cluster_ip
            port = service.spec.ports[0].port
            
            return f"http://{cluster_ip}:{port}"
            
        except Exception as e:
            print(f"Failed to discover via K8s API: {e}")
            # Fallback to direct connection (assumes port-forward or NodePort)
            microk8s_ip = os.getenv("MICROK8S_IP", "localhost")
            return f"http://{microk8s_ip}:8000"

    def _get_microk8s_node_ip(self) -> str:
        """Get MicroK8s node IP for direct connection"""
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            
            nodes = v1.list_node()
            for node in nodes.items:
                for address in node.status.addresses:
                    if address.type == "InternalIP":
                        return address.address
            
            return "localhost"
        except:
            return "localhost"

    async def chat(self, text: str, user_id: str) -> dict:
        response = await self.client.post(
            "/api/chat",
            json={"text": text, "user_id": user_id}
        )
        return response.json()

    async def ingest(self, file_path: str, user_id: str):
        with open(file_path, 'rb') as f:
            response = await self.client.post(
                "/api/ingest",
                files={"file": f},
                data={"user_id": user_id}
            )
        return response.json()

    async def status(self) -> dict:
        response = await self.client.get("/api/status")
        return response.json()
```

---

### 2. Kubernetes-Native Service Discovery

**Responsibilities**:
- Service registration and health monitoring via Kubernetes Services
- DNS-based service discovery (`*.svc.cluster.local`)
- Configuration management via ConfigMaps and Secrets
- Load balancing and failover via Kubernetes Services

**MicroK8s Setup**:
- **Control Plane**: Primary CPU node runs MicroK8s control plane
- **Worker Nodes**: All other nodes join as workers
- **DNS Addon**: Built-in CoreDNS for service discovery
- **Storage Addon**: Built-in storage classes for persistent volumes

**Configuration Example**:

```bash
# Initialize MicroK8s cluster on primary CPU node
microk8s install --channel=1.28/stable
microk8s enable dns storage registry

# Generate join token for worker nodes
microk8s add-node

# On each worker node
microk8s join <control-plane-ip>:25000/<token>

# Enable GPU addon on GPU nodes only
microk8s enable gpu  # On GPU nodes
```

**Kubernetes Service Definition**:

```yaml
# kubernetes/core-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: core-service
  namespace: morgan
spec:
  selector:
    app: core-rag
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
  namespace: morgan
spec:
  selector:
    app: ollama-main
  ports:
  - name: http
    port: 11434
    targetPort: 11434
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
  namespace: morgan
spec:
  selector:
    app: ollama-embedding
  ports:
  - name: http
    port: 11434
    targetPort: 11434
  type: ClusterIP
```

**Kubernetes Service Discovery Client**:

```python
# shared/utils/k8s_client.py
from kubernetes import client, config
import socket
import os

class KubernetesServiceDiscovery:
    def __init__(self, namespace: str = "morgan"):
        # Load in-cluster config if running in pod, otherwise load kubeconfig
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.namespace = namespace
        self.pod_name = os.getenv("HOSTNAME", socket.gethostname())

    def get_service_url(self, service_name: str) -> str:
        """Get service URL using Kubernetes DNS"""
        # Use Kubernetes DNS: service-name.namespace.svc.cluster.local
        return f"http://{service_name}.{self.namespace}.svc.cluster.local"

    def get_service_endpoints(self, service_name: str) -> list:
        """Get all endpoints for a service"""
        try:
            endpoints = self.v1.read_namespaced_endpoints(
                name=service_name,
                namespace=self.namespace
            )
            
            instances = []
            if endpoints.subsets:
                for subset in endpoints.subsets:
                    for address in subset.addresses:
                        for port in subset.ports:
                            instances.append({
                                'address': address.ip,
                                'port': port.port,
                                'ready': True
                            })
            
            return instances
        except Exception as e:
            print(f"Failed to get endpoints for {service_name}: {e}")
            return []

    def is_service_healthy(self, service_name: str) -> bool:
        """Check if service has healthy endpoints"""
        endpoints = self.get_service_endpoints(service_name)
        return len(endpoints) > 0

    def wait_for_service(self, service_name: str, timeout: int = 300):
        """Wait for service to become available"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_service_healthy(service_name):
                return True
            time.sleep(5)
        
        return False
```

**DNS Resolution**:

```python
# Example: Resolving services via Kubernetes DNS
import socket

# Resolve core-service.morgan.svc.cluster.local
core_ip = socket.gethostbyname("core-service.morgan.svc.cluster.local")
# Returns IP of one healthy Core service instance (load balanced)

# Use in HTTP client
import httpx
client = httpx.AsyncClient(base_url="http://core-service.morgan.svc.cluster.local:8000")

# Alternative: Use service discovery client
from shared.utils.k8s_client import KubernetesServiceDiscovery

k8s = KubernetesServiceDiscovery()
core_url = k8s.get_service_url("core-service")
client = httpx.AsyncClient(base_url=f"{core_url}:8000")
```

---

### 3. Istio Service Mesh (Optional for MVP)

**Responsibilities** (If Deployed):
- Unified entry point for all client requests via Istio Gateway
- Automatic service discovery via Kubernetes Services
- Load balancing across Core service instances
- Rate limiting and authentication policies
- Request/response logging and metrics

**MVP Alternative**: Direct Kubernetes Service access (`core-service.morgan.svc.cluster.local:8000`) without Istio.

**Configuration** (For Production Deployment):

```bash
# Enable Istio addon in MicroK8s
microk8s enable community
microk8s enable istio
```

**Istio Gateway Configuration**:

```yaml
# kubernetes/istio-gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: morgan-gateway
  namespace: morgan
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - morgan.local

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: morgan-vs
  namespace: morgan
spec:
  hosts:
  - morgan.local
  gateways:
  - morgan-gateway
  http:
  - match:
    - uri:
        prefix: /api/chat
    route:
    - destination:
        host: core-service.morgan.svc.cluster.local
        port:
          number: 8000
  - match:
    - uri:
        prefix: /api/ingest
    route:
    - destination:
        host: core-service.morgan.svc.cluster.local
        port:
          number: 8000
  - match:
    - uri:
        prefix: /api/status
    route:
    - destination:
        host: core-service.morgan.svc.cluster.local
        port:
          number: 8000
```

**Rate Limiting Policy**:

```yaml
# kubernetes/rate-limit-policy.yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: morgan-rate-limit
  namespace: morgan
spec:
  selector:
    matchLabels:
      app: core-rag
  rules:
  - to:
    - operation:
        methods: ["POST"]
        paths: ["/api/chat"]
    when:
    - key: source.ip
      values: ["*"]
```

---

### 4. Dynamic Pod Allocation Logic

**Responsibilities**:
- Detect all available nodes via Kubernetes API
- Query node capabilities (GPU type, VRAM, CPU, RAM, architecture) via node labels
- Assign pods based on hardware capabilities using node selectors and affinity
- Rebalance pods when nodes are added/removed via Kubernetes scheduler

**Node Detection Algorithm**:

```python
# core/allocation/node_detector.py
from kubernetes import client
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class NodeCapabilities:
    node_name: str
    ip: str
    os: str  # "linux", "windows", "darwin"
    arch: str  # "amd64", "arm64"
    cpu_cores: int
    ram_gb: int

    # GPU info (None if no GPU)
    gpu_type: str = None  # "nvidia", None
    gpu_model: str = None  # "RTX 3090", "RTX 3060", etc.
    vram_gb: int = None
    cuda_version: str = None

class NodeDetector:
    def __init__(self):
        self.v1 = client.CoreV1Api()

    def detect_all_nodes(self) -> List[NodeCapabilities]:
        """Detect all nodes in Kubernetes cluster"""
        nodes = self.v1.list_node()

        node_caps = []
        for node in nodes.items:
            caps = self._parse_node_capabilities(node)
            node_caps.append(caps)

        return node_caps

    def _parse_node_capabilities(self, node) -> NodeCapabilities:
        """Parse node capabilities from Kubernetes node object"""
        labels = node.metadata.labels or {}
        status = node.status
        
        # Extract basic info
        node_name = node.metadata.name
        ip = self._get_node_ip(status.addresses)
        os = labels.get("kubernetes.io/os", "linux")
        arch = labels.get("kubernetes.io/arch", "amd64")
        
        # Extract resources
        capacity = status.capacity
        cpu_cores = int(capacity.get("cpu", "0"))
        memory_ki = capacity.get("memory", "0Ki")
        ram_gb = self._parse_memory(memory_ki)
        
        # Extract GPU info from labels (set by GPU addon)
        gpu_type = None
        gpu_model = None
        vram_gb = None
        cuda_version = None
        
        if "nvidia.com/gpu" in capacity:
            gpu_type = "nvidia"
            gpu_model = labels.get("nvidia.com/gpu.product", "Unknown")
            vram_memory = labels.get("nvidia.com/gpu.memory", "0")
            vram_gb = int(vram_memory.replace("Mi", "")) // 1024 if vram_memory else None
            cuda_version = labels.get("nvidia.com/cuda.driver.major", "Unknown")

        return NodeCapabilities(
            node_name=node_name,
            ip=ip,
            os=os,
            arch=arch,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_type=gpu_type,
            gpu_model=gpu_model,
            vram_gb=vram_gb,
            cuda_version=cuda_version
        )

    def _get_node_ip(self, addresses) -> str:
        """Extract node IP from addr"""
        for addr in addresses:
            if addr.type == "InternalIP":
                return addr.address
        return "unknown"

    def _parse_memory(self, memory_str: str) -> int:
        """Parse Kubernetes memory string to GB"""
        if memory_str.endswith("Ki"):
            return int(memory_str[:-2]) // (1024 * 1024)
        elif memory_str.endswith("Mi"):
            return int(memory_str[:-2]) // 1024
        elif memory_str.endswith("Gi"):
            return int(memory_str[:-2])
        return 0
```

**Pod Allocation Algorithm**:

```python
# core/allocation/pod_allocator.py
from typing import List, Dict
from dataclasses import dataclass
from kubernetes import client

@dataclass
class PodAllocation:
    deployment_name: str
    node_selector: Dict[str, str]
    replicas: int
    resources: Dict
    affinity: Dict = None

class PodAllocator:
    def __init__(self, nodes: List[NodeCapabilities]):
        self.nodes = nodes
        self.allocations = []

    def allocate_pods(self) -> List[PodAllocation]:
        """Allocate pods based on node capabilities"""

        # Sort nodes by capability
        gpu_nodes = [n for n in self.nodes if n.gpu_type == "nvidia"]
        cpu_nodes = [n for n in self.nodes if n.gpu_type is None and n.arch == "amd64"]
        arm_nodes = [n for n in self.nodes if n.arch == "arm64"]

        # Sort GPU nodes by VRAM (descending)
        gpu_nodes_sorted = sorted(gpu_nodes, key=lambda n: n.vram_gb or 0, reverse=True)

        # Sort CPU nodes by RAM (descending)
        cpu_nodes_sorted = sorted(cpu_nodes, key=lambda n: n.ram_gb, reverse=True)

        # Allocate GPU services
        self._allocate_gpu_pods(gpu_nodes_sorted)

        # Allocate CPU services
        self._allocate_cpu_pods(cpu_nodes_sorted)

        # Allocate ARM services
        self._allocate_arm_pods(arm_nodes)

        return self.allocations

    def _allocate_gpu_pods(self, gpu_nodes: List[NodeCapabilities]):
        """Allocate GPU-dependent pods"""

        if not gpu_nodes:
            print("WARNING: No GPU nodes detected. GPU services will not be available.")
            return

        # Main LLM → highest VRAM GPU node
        if gpu_nodes and gpu_nodes[0].vram_gb >= 12:
            self.allocations.append(PodAllocation(
                deployment_name="ollama-main",
                node_selector={"node-type": "gpu", "gpu-vram": "high"},
                replicas=1,
                resources={
                    "requests": {"nvidia.com/gpu": 1, "memory": "8Gi"},
                    "limits": {"nvidia.com/gpu": 1, "memory": "16Gi"}
                }
            ))

        # Embedding service → moderate VRAM GPU node (8GB+)
        embed_nodes = [n for n in gpu_nodes if n.vram_gb >= 8]
        if embed_nodes:
            self.allocations.append(PodAllocation(
                deployment_name="ollama-embedding",
                node_selector={"node-type": "gpu", "gpu-vram": "medium"},
                replicas=1,
                resources={
                    "requests": {"nvidia.com/gpu": 1, "memory": "4Gi"},
                    "limits": {"nvidia.com/gpu": 1, "memory": "8Gi"}
                }
            ))

        # Fast LLM → moderate VRAM GPU node
        if len(embed_nodes) > 1:
            self.allocations.append(PodAllocation(
                deployment_name="ollama-fast",
                node_selector={"node-type": "gpu", "gpu-vram": "medium"},
                replicas=1,
                resources={
                    "requests": {"nvidia.com/gpu": 1, "memory": "4Gi"},
                    "limits": {"nvidia.com/gpu": 1, "memory": "8Gi"}
                }
            ))

        # Reranking service → minimal VRAM GPU node (6GB+)
        rerank_nodes = [n for n in gpu_nodes if n.vram_gb >= 6]
        if rerank_nodes:
            self.allocations.append(PodAllocation(
                deployment_name="reranking-service",
                node_selector={"node-type": "gpu", "gpu-vram": "low"},
                replicas=1,
                resources={
                    "requests": {"nvidia.com/gpu": 1, "memory": "2Gi"},
                    "limits": {"nvidia.com/gpu": 1, "memory": "4Gi"}
                }
            ))

    def _allocate_cpu_pods(self, cpu_nodes: List[NodeCapabilities]):
        """Allocate CPU-based pods"""

        if not cpu_nodes:
            raise Exception("ERROR: At least one CPU node required for core services")

        # Core RAG orchestrator
        self.allocations.append(PodAllocation(
            deployment_name="core-rag-orchestrator",
            node_selector={"node-type": "cpu", "ram-tier": "high"},
            replicas=2,
            resources={
                "requests": {"memory": "2Gi", "cpu": "1000m"},
                "limits": {"memory": "4Gi", "cpu": "2000m"}
            }
        ))

        # PostgreSQL primary
        self.allocations.append(PodAllocation(
            deployment_name="postgresql-primary",
            node_selector={"node-type": "cpu", "ram-tier": "high"},
            replicas=1,
            resources={
                "requests": {"memory": "4Gi", "cpu": "1000m"},
                "limits": {"memory": "8Gi", "cpu": "2000m"}
            }
        ))

        # Qdrant vector database
        self.allocations.append(PodAllocation(
            deployment_name="qdrant",
            node_selector={"node-type": "cpu", "ram-tier": "high"},
            replicas=1,
            resources={
                "requests": {"memory": "4Gi", "cpu": "1000m"},
                "limits": {"memory": "8Gi", "cpu": "2000m"}
            }
        ))

        # Redis cluster
        self.allocations.append(PodAllocation(
            deployment_name="redis",
            node_selector={"node-type": "cpu"},
            replicas=3,  # Redis cluster
            resources={
                "requests": {"memory": "1Gi", "cpu": "500m"},
                "limits": {"memory": "2Gi", "cpu": "1000m"}
            }
        ))

        # MinIO (if local)
        self.allocations.append(PodAllocation(
            deployment_name="minio",
            node_selector={"node-type": "cpu"},
            replicas=1,
            resources={
                "requests": {"memory": "1Gi", "cpu": "500m"},
                "limits": {"memory": "2Gi", "cpu": "1000m"}
            }
        ))

        # Emotion engine
        self.allocations.append(PodAllocation(
            deployment_name="emotion-engine",
            node_selector={"node-type": "cpu"},
            replicas=1,
            resources={
                "requests": {"memory": "2Gi", "cpu": "1000m"},
                "limits": {"memory": "4Gi", "cpu": "2000m"}
            }
        ))

        # Learning system
        self.allocations.append(PodAllocation(
            deployment_name="learning-system",
            node_selector={"node-type": "cpu"},
            replicas=1,
            resources={
                "requests": {"memory": "2Gi", "cpu": "1000m"},
                "limits": {"memory": "4Gi", "cpu": "2000m"}
            }
        ))

    def _allocate_arm_pods(self, arm_nodes: List[NodeCapabilities]):
        """Allocate ARM64 services (macOS M1)"""

        if not arm_nodes:
            return  # Optional

        arm_node = arm_nodes[0]

        # Core proxy (ARM64-compatible)
        self.allocations.append(PodAllocation(
            deployment_name="core-proxy-arm64",
            node_selector={"node-type": "cpu", "arch": "arm64"},
            replicas=1,
            resources={
                "requests": {"memory": "1Gi", "cpu": "500m"},
                "limits": {"memory": "2Gi", "cpu": "1000m"}
            }
        ))

        # LLM proxy (ARM64-compatible)
        self.allocations.append(PodAllocation(
            deployment_name="llm-proxy-arm64",
            node_selector={"node-type": "cpu", "arch": "arm64"},
            replicas=1,
            resources={
                "requests": {"memory": "1Gi", "cpu": "500m"},
                "limits": {"memory": "2Gi", "cpu": "1000m"}
            }
        ))
```

---

### 5. PostgreSQL Schema Design (Code-Only Logic)

**Responsibilities**:
- Store structured data (conversations, emotions, learning, documents)
- Use tables, indexes, foreign keys, and basic constraints ONLY
- ALL business logic in application code (Python services)
- NO stored procedures, functions, or triggers

**Database Architecture**:
- **Primary Instance**: On primary CPU_Host (highest RAM)
- **Standby Instance**: On secondary CPU_Host (if available) with async streaming replication
- **Connection Pooling**: SQLAlchemy with asyncpg driver
- **Migrations**: Alembic for schema versioning

**Schema Definition**:

```sql
-- Morgan V2 PostgreSQL Schema
-- Version: 3.0.0
-- All business logic in application code (Python)
-- No stored procedures, functions, or triggers

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Conversations table
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_deleted_at ON conversations(deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);

-- Messages table
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX idx_messages_role ON messages(role);

-- Emotional states table (linked to messages)
CREATE TABLE emotional_states (
    emotion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL,
    emotion_type VARCHAR(50) NOT NULL,
    intensity FLOAT NOT NULL CHECK (intensity >= 0 AND intensity <= 1),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);

CREATE INDEX idx_emotional_states_message_id ON emotional_states(message_id);
CREATE INDEX idx_emotional_states_emotion_type ON emotional_states(emotion_type);
CREATE INDEX idx_emotional_states_detected_at ON emotional_states(detected_at DESC);

-- Emotion patterns table (aggregated emotion trends)
CREATE TABLE emotion_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    first_detected TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_emotion_patterns_user_id ON emotion_patterns(user_id);
CREATE INDEX idx_emotion_patterns_pattern_type ON emotion_patterns(pattern_type);

-- Emotion triggers table
CREATE TABLE emotion_triggers (
    trigger_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    trigger_text TEXT NOT NULL,
    triggered_emotion VARCHAR(50) NOT NULL,
    intensity FLOAT NOT NULL CHECK (intensity >= 0 AND intensity <= 1),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_emotion_triggers_user_id ON emotion_triggers(user_id);
CREATE INDEX idx_emotion_triggers_detected_at ON emotion_triggers(detected_at DESC);

-- Documents table (RAG system - file metadata)
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    bucket VARCHAR(255) NOT NULL DEFAULT 'morgan-documents',
    object_key VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(255),
    upload_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_bucket_key ON documents(bucket, object_key);
CREATE INDEX idx_documents_upload_timestamp ON documents(upload_timestamp DESC);

-- Document chunks table (RAG system - text chunks for embedding)
CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_document_chunks_chunk_index ON document_chunks(document_id, chunk_index);

-- User preferences table (Learning system)
CREATE TABLE user_preferences (
    preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    preference_key VARCHAR(255) NOT NULL,
    preference_value JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(user_id, preference_key)
);

CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX idx_user_preferences_preference_key ON user_preferences(preference_key);

-- Behavioral patterns table (Learning system)
CREATE TABLE behavioral_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    first_observed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_observed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    observation_count INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_behavioral_patterns_user_id ON behavioral_patterns(user_id);
CREATE INDEX idx_behavioral_patterns_pattern_type ON behavioral_patterns(pattern_type);
CREATE INDEX idx_behavioral_patterns_confidence ON behavioral_patterns(confidence DESC);

-- Feedback table (Learning system)
CREATE TABLE feedback (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,
    message_id UUID,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);

CREATE INDEX idx_feedback_conversation_id ON feedback(conversation_id);
CREATE INDEX idx_feedback_message_id ON feedback(message_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);

-- Comments explaining code-only logic approach:
COMMENT ON TABLE conversations IS 'Conversation metadata. Soft deletes handled in application code via deleted_at column.';
COMMENT ON TABLE messages IS 'Individual messages in conversations. Timestamp updates handled in application code.';
COMMENT ON TABLE emotional_states IS 'Detected emotions linked to messages. All emotion analysis in Python emotion engine.';
COMMENT ON TABLE emotion_patterns IS 'Aggregated emotion patterns. Pattern detection and aggregation in Python application code.';
COMMENT ON TABLE documents IS 'Document file metadata. File storage in MinIO S3. Document processing in Python RAG system.';
COMMENT ON TABLE document_chunks IS 'Text chunks for embedding. Chunking logic in Python application code.';
COMMENT ON TABLE user_preferences IS 'User preferences for personalization. Preference learning in Python learning system.';
COMMENT ON TABLE behavioral_patterns IS 'Detected behavioral patterns. Pattern mining in Python application code.';
COMMENT ON TABLE feedback IS 'User feedback for continuous learning. Feedback processing in Python learning system.';
```

**SQLAlchemy ORM Models**:

```python
# shared/models/database.py
from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(TIMESTAMP, nullable=True, index=True)
    metadata = Column(JSONB, default={})

    # Relationships
    messages = relationship("Message", back_populates="conversation")
    feedback = relationship("Feedback", back_populates="conversation")

    def soft_delete(self):
        """Soft delete - set deleted_at timestamp (application logic)"""
        self.deleted_at = datetime.utcnow()

class Message(Base):
    __tablename__ = 'messages'

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.conversation_id'), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)
    metadata = Column(JSONB, default={})

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name='check_role'),
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    emotional_states = relationship("EmotionalState", back_populates="message")
    feedback = relationship("Feedback", back_populates="message")

class EmotionalState(Base):
    __tablename__ = 'emotional_states'

    emotion_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.message_id'), nullable=False)
    emotion_type = Column(String(50), nullable=False, index=True)
    intensity = Column(Float, nullable=False)
    detected_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)
    metadata = Column(JSONB, default={})

    __table_args__ = (
        CheckConstraint("intensity >= 0 AND intensity <= 1", name='check_intensity'),
    )

    # Relationships
    message = relationship("Message", back_populates="emotional_states")

class Document(Base):
    __tablename__ = 'documents'

    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
    bucket = Column(String(255), nullable=False, default='morgan-documents')
    object_key = Column(String(1024), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(255), nullable=True)
    upload_timestamp = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)
    metadata = Column(JSONB, default={})

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.document_id'), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    metadata = Column(JSONB, default={})

    # Relationships
    document = relationship("Document", back_populates="chunks")

class UserPreference(Base):
    __tablename__ = 'user_preferences'

    preference_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    preference_key = Column(String(255), nullable=False, index=True)
    preference_value = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB, default={})

class Feedback(Base):
    __tablename__ = 'feedback'

    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.conversation_id'), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.message_id'), nullable=True)
    rating = Column(Integer, nullable=True, index=True)
    comment = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)
    metadata = Column(JSONB, default={})

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name='check_rating'),
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="feedback")
    message = relationship("Message", back_populates="feedback")
```

**Database Client**:

```python
# shared/database/client.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

class DatabaseClient:
    def __init__(self, connection_string: str):
        # Example: postgresql+asyncpg://morgan:password@postgres.service.consul:5432/morgan
        self.engine = create_async_engine(
            connection_string,
            echo=False,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True  # Verify connections before use
        )

        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        async with self.async_session() as session:
            yield session

    async def close(self):
        """Close database engine"""
        await self.engine.dispose()
```

**PostgreSQL Replication Configuration**:

```yaml
# docker-compose-postgres.yml (Primary CPU_Host)
services:
  postgres-primary:
    image: postgres:17-alpine
    container_name: morgan-postgres-primary
    environment:
      - POSTGRES_DB=morgan
      - POSTGRES_USER=morgan
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_REPLICATION_USER=replicator
      - POSTGRES_REPLICATION_PASSWORD=${REPLICATION_PASSWORD}
    volumes:
      - postgres-primary-data:/var/lib/postgresql/data
      - ./postgres-primary.conf:/etc/postgresql/postgresql.conf:ro
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    networks:
      - morgan-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U morgan"]
      interval: 30s
      timeout: 5s
      retries: 3

# postgres-primary.conf
# Streaming replication configuration
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB
max_replication_slots = 3
synchronous_commit = off  # Async replication for performance
```

```yaml
# docker-compose-postgres-standby.yml (Secondary CPU_Host)
services:
  postgres-standby:
    image: postgres:17-alpine
    container_name: morgan-postgres-standby
    environment:
      - POSTGRES_DB=morgan
      - POSTGRES_USER=morgan
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - PGDATA=/var/lib/postgresql/data/pgdata
    volumes:
      - postgres-standby-data:/var/lib/postgresql/data
      - ./postgres-standby.conf:/etc/postgresql/postgresql.conf:ro
      - ./recovery.signal:/var/lib/postgresql/data/pgdata/recovery.signal
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "5433:5432"
    networks:
      - morgan-net
    depends_on:
      - postgres-primary
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U morgan"]
      interval: 30s
      timeout: 5s
      retries: 3

# postgres-standby.conf (template - use envsubst or init script to expand)
hot_standby = on
# Note: Replace REPLICATION_PASSWORD_VALUE with actual password via init script
# Example init script: envsubst < postgres-standby.conf.template > postgres-standby.conf
primary_conninfo = 'host=postgres-primary port=5432 user=replicator password=REPLICATION_PASSWORD_VALUE'

# Alternative: Use .pgpass file for password management
# Create /root/.pgpass with: postgres-primary:5432:*:replicator:actual_password
```

---

### 6. MinIO S3 Storage Architecture

**Responsibilities**:
- Store original uploaded files (documents for RAG)
- Store processed artifacts (images, tables extracted from documents)
- Store audio files (future TTS/STT)
- Support local deployment (Docker) or remote MinIO instance

**Deployment Options**:

**Option A: Local MinIO (Docker)**:

```yaml
# docker-compose-minio.yml (Primary CPU_Host)
services:
  minio:
    image: minio/minio:latest
    container_name: morgan-minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=morgan
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"  # S3 API
      - "9001:9001"  # Web Console
    volumes:
      - minio-data:/data
    networks:
      - morgan-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO client for bucket initialization
  minio-init:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      sleep 5;
      mc alias set myminio http://minio:9000 morgan ${MINIO_ROOT_PASSWORD};
      mc mb --ignore-existing myminio/morgan-documents;
      mc mb --ignore-existing myminio/morgan-artifacts;
      mc mb --ignore-existing myminio/morgan-audio;
      mc policy set download myminio/morgan-documents;
      mc policy set download myminio/morgan-artifacts;
      mc policy set download myminio/morgan-audio;
      exit 0;
      "
    networks:
      - morgan-net
```

**Option B: Remote MinIO**:

```yaml
# .env configuration
MINIO_ENDPOINT=http://192.168.101.5:9000
MINIO_ACCESS_KEY=morgan
MINIO_SECRET_KEY=<secret>
MINIO_USE_SSL=false  # Set to true for HTTPS
```

**Bucket Structure**:

```
morgan-documents/
  ├── {user_id}/
  │   ├── {upload_date}/
  │   │   ├── {uuid}_{filename}.pdf
  │   │   ├── {uuid}_{filename}.docx
  │   │   └── {uuid}_{filename}.md

morgan-artifacts/
  ├── {user_id}/
  │   ├── {document_id}/
  │   │   ├── images/
  │   │   │   ├── image_001.png
  │   │   │   └── image_002.jpg
  │   │   └── tables/
  │   │       ├── table_001.csv
  │   │       └── table_002.json

morgan-audio/
  ├── {user_id}/
  │   ├── {session_id}/
  │   │   ├── input_{timestamp}.wav
  │   │   └── output_{timestamp}.wav
```

**MinIO Client**:

```python
# shared/storage/minio_client.py
from minio import Minio
from minio.error import S3Error
from typing import BinaryIO, Optional
import uuid
from datetime import datetime, timedelta

class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        use_ssl: bool = False
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=use_ssl
        )

        # Ensure buckets exist
        self._ensure_buckets()

    def _ensure_buckets(self):
        """Ensure required buckets exist"""
        buckets = ['morgan-documents', 'morgan-artifacts', 'morgan-audio']

        for bucket in buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def upload_document(
        self,
        file_data: BinaryIO,
        filename: str,
        user_id: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Upload document to MinIO"""

        # Generate object key: {user_id}/{date}/{uuid}_{filename}
        upload_date = datetime.utcnow().strftime("%Y-%m-%d")
        file_uuid = str(uuid.uuid4())
        object_key = f"{user_id}/{upload_date}/{file_uuid}_{filename}"

        # Upload to MinIO
        self.client.put_object(
            bucket_name="morgan-documents",
            object_name=object_key,
            data=file_data,
            length=-1,  # Unknown size, stream until EOF
            content_type=content_type,
            part_size=10*1024*1024  # 10MB parts
        )

        return object_key

    def download_document(self, object_key: str) -> bytes:
        """Download document from MinIO"""
        try:
            response = self.client.get_object(
                bucket_name="morgan-documents",
                object_name=object_key
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            raise Exception(f"Failed to download {object_key}: {e}")

    def get_presigned_url(
        self,
        object_key: str,
        bucket: str = "morgan-documents",
        expiry: int = 3600
    ) -> str:
        """Generate pre-signed URL for direct file access"""
        url = self.client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_key,
            expires=timedelta(seconds=expiry)
        )
        return url

    def delete_document(self, object_key: str, bucket: str = "morgan-documents"):
        """Delete document from MinIO"""
        self.client.remove_object(bucket, object_key)

    def list_documents(self, user_id: str, prefix: str = "") -> list:
        """List documents for user"""
        objects = self.client.list_objects(
            bucket_name="morgan-documents",
            prefix=f"{user_id}/{prefix}",
            recursive=True
        )

        return [obj.object_name for obj in objects]
```

**Integration with PostgreSQL**:

```python
# core/rag/document_ingestion.py
from shared.storage.minio_client import MinIOClient
from shared.database.client import DatabaseClient
from shared.models.database import Document, DocumentChunk
import uuid

class DocumentIngestion:
    def __init__(self, minio_client: MinIOClient, db_client: DatabaseClient):
        self.minio = minio_client
        self.db = db_client

    async def ingest_document(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        mime_type: str
    ) -> uuid.UUID:
        """Ingest document: upload to MinIO + store metadata in PostgreSQL"""

        # 1. Upload to MinIO
        object_key = self.minio.upload_document(
            file_data=file_data,
            filename=filename,
            user_id=user_id,
            content_type=mime_type
        )

        # 2. Store metadata in PostgreSQL
        async with self.db.get_session() as session:
            document = Document(
                user_id=user_id,
                file_path=filename,
                bucket="morgan-documents",
                object_key=object_key,
                file_size=len(file_data),
                mime_type=mime_type
            )

            session.add(document)
            await session.commit()

            return document.document_id

    async def process_document_chunks(
        self,
        document_id: uuid.UUID,
        chunks: list
    ):
        """Store document chunks in PostgreSQL"""

        async with self.db.get_session() as session:
            for idx, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_text=chunk_text,
                    chunk_index=idx,
                    token_count=len(chunk_text.split())  # Simple token count
                )
                session.add(chunk)

            await session.commit()
```

---

## Business Process

### Process 1: CLI Chat Interaction with RAG and Emotion Detection

```mermaid
flowchart TD
    A[User runs: morgan chat] --> B[CLI loads MorganClient]
    B --> C[CLI resolves core.service.consul]
    C --> D[CLI sends POST /api/chat]

    D --> E[Core RAG Orchestrator receives request]
    E --> F[Load conversation context from Redis]
    F --> G{Cache hit?}
    G -->|Yes| H[Use cached context]
    G -->|No| I[Load from PostgreSQL]
    I --> H

    H --> J[Send to Emotion Engine]
    J --> K[Emotion Engine analyzes text]
    K --> L[Detect emotion type + intensity]
    L --> M[Store emotional_state in PostgreSQL]

    M --> N[Generate query embedding via Embedding Service]
    N --> O[Search Qdrant coarse collection]
    O --> P[Search Qdrant medium collection]
    P --> Q[Search Qdrant fine collection]
    Q --> R[Merge results with RRF]

    R --> S[Rerank top candidates via Reranking Service]
    S --> T[Load chunk text from PostgreSQL]

    T --> U[Build RAG-enhanced prompt]
    U --> V[Send to Main LLM Service]
    V --> W[LLM generates response]

    W --> X[Send to Empathy Engine]
    X --> Y[Empathy Engine enhances response]

    Y --> Z[Store message + emotional_state in PostgreSQL]
    Z --> AA[Update conversation cache in Redis]
    AA --> AB[Send to Learning System for adaptation]

    AB --> AC[Return response to CLI]
    AC --> AD[CLI displays response with emotion color]
    AD --> AE{User continues?}
    AE -->|Yes| D
    AE -->|No| AF[Exit chat]
```

### Process 2: Document Ingestion via CLI

```mermaid
flowchart TD
    A[User runs: morgan ingest /path/to/docs] --> B[CLI loads file list]
    B --> C[CLI sends POST /api/ingest with file]

    C --> D[Core RAG Orchestrator receives file]
    D --> E[Upload original file to MinIO]
    E --> F[Store document metadata in PostgreSQL]

    F --> G[Extract text from document]
    G --> H[Perform intelligent chunking]
    H --> I[Store chunks in PostgreSQL document_chunks]

    I --> J[Generate embeddings via Embedding Service]
    J --> K[Batch embed chunks 32-64 at a time]
    K --> L[Store embeddings in Qdrant coarse collection]
    L --> M[Store embeddings in Qdrant medium collection]
    M --> N[Store embeddings in Qdrant fine collection]

    N --> O[Return ingestion summary to CLI]
    O --> P[CLI displays: X docs, Y chunks, Z embeddings]
    P --> Q{More files?}
    Q -->|Yes| C
    Q -->|No| R[Complete ingestion]
```

### Process 3: Dynamic Service Failover (Kubernetes-Native)

```mermaid
flowchart TD
    A[Pod Running] --> B[Liveness Probe Fails 3x]
    B --> C[Kubelet terminates pod]
    C --> D[Deployment controller creates new pod]

    A --> E[Readiness Probe Fails]
    E --> F[Kubernetes removes pod from Service endpoints]
    F --> G[Service routes traffic to other ready pods]

    G --> H{Other ready pods available?}

    H -->|Yes| I[Traffic continues via healthy pods]
    H -->|No| J{Service type?}

    J -->|GPU LLM| K[Fall back to CPU LLM or queue]
    J -->|Embedding| L[Fall back to CPU embedding degraded]
    J -->|Qdrant| M[Use PostgreSQL full-text search fallback]

    I --> N[Continue processing requests]
    K --> N
    L --> N
    M --> N

    D --> O{New pod starts successfully?}

    O -->|Yes| P[Readiness probe passes]
    P --> Q[Pod added to Service endpoints]
    Q --> R[Pod joins healthy pool]

    O -->|No - CrashLoopBackOff| S[Alert via Prometheus/Grafana]
    S --> T[Manual intervention required]

    R --> N
```

**Key Kubernetes Mechanisms:**
- **Liveness Probes**: Kubelet automatically restarts failed pods
- **Readiness Probes**: Service only routes to pods passing readiness checks
- **Deployments**: Maintain desired replica count, auto-restart failed pods
- **Service Discovery**: Native DNS (service-name.namespace.svc.cluster.local)
- **Load Balancing**: Built-in via Service resources across healthy pods

---

## Error Handling Strategy

### 1. Network Failures

**Scenario**: Host loses network connectivity to Consul

**Handling**:
1. Services detect Consul unavailability
2. Use cached service endpoints from last successful query (TTL: 5 minutes)
3. Queue database write operations locally (local SQLite)
4. When network restores:
   - Re-register with Consul
   - Flush queued operations to PostgreSQL
   - Sync cached data

### 2. Service Failures

**Scenario**: Embedding service crashes on GPU_Host_2

**Handling**:
1. Consul health check fails (3x within 90 seconds)
2. Consul removes service from registry
3. Core RAG Orchestrator queries Consul for healthy Embedding instances
4. If no healthy instances:
   - Fall back to CPU-based embedding (slower)
   - Display warning in CLI: "GPU embedding unavailable, using CPU (degraded performance)"
5. Docker Swarm/restart policy restarts failed container
6. Service re-registers when healthy

### 3. Database Failures

**Scenario**: PostgreSQL primary goes down

**Handling**:
1. Connection errors detected by Core service
2. If PostgreSQL standby exists:
   - Promote standby to primary (manual or via Patroni)
   - Update Consul DNS to point to new primary
   - Services reconnect automatically
3. If no standby:
   - Queue write operations locally
   - Alert administrator
   - Return degraded service (read-only from Redis cache)

### 4. MinIO Failures

**Scenario**: MinIO becomes unavailable

**Handling**:
1. Document upload/download operations fail
2. Queue upload operations for retry (max 100 pending)
3. Use cached file data from Redis if available
4. Display warning in CLI: "File storage degraded, uploads will retry automatically"
5. When MinIO restores:
   - Flush queued uploads
   - Resume normal operation

### 5. GPU VRAM Exhaustion

**Scenario**: Main LLM service exhausts 12GB VRAM

**Handling**:
1. LLM service detects CUDA OOM error
2. Unload current model
3. Load quantized model (Q4_K_M → Q4_0 → Q2_K)
4. If still exhausted:
   - Fall back to Fast LLM on GPU_Host_2 (7B model)
   - Or fall back to CPU inference (very slow)
5. Display warning in CLI: "Using smaller model due to memory constraints"

---

## Testing Strategy

### Unit Tests

- Test each service independently with mocked dependencies
- Test PostgreSQL ORM models and queries
- Test MinIO client upload/download
- Test Consul service discovery
- Test CLI command parsing

**Example**:

```python
# tests/unit/test_database.py
import pytest
from shared.models.database import Conversation, Message

@pytest.mark.asyncio
async def test_soft_delete_conversation(db_session):
    """Test soft delete sets deleted_at timestamp"""
    conv = Conversation(user_id="test_user")
    db_session.add(conv)
    await db_session.commit()

    # Soft delete
    conv.soft_delete()
    await db_session.commit()

    assert conv.deleted_at is not None
```

### Integration Tests

- Deploy full stack in Docker Compose
- Test service discovery via Consul
- Test data persistence (PostgreSQL + MinIO + Qdrant)
- Test RAG pipeline end-to-end
- Test emotion detection + empathy engine

**Example**:

```python
# tests/integration/test_rag_pipeline.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_document_ingestion_and_query():
    """Test full RAG pipeline"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # 1. Ingest document
        with open("tests/fixtures/sample.pdf", "rb") as f:
            response = await client.post(
                "/api/ingest",
                files={"file": f},
                data={"user_id": "test_user"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["chunks_created"] > 0

        # 2. Query document
        response = await client.post(
            "/api/chat",
            json={
                "text": "What does the document say about X?",
                "user_id": "test_user"
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert "sources" in result
        assert len(result["sources"]) > 0
```

### Multi-Host Simulation

- Use Docker Compose with multiple docker-compose.yml files
- Simulate network latency with `tc` (traffic control)
- Test service allocation across hosts
- Test failover scenarios

### Load Testing

- Use `locust` to generate concurrent requests
- Measure latency and throughput under load
- Test with varying host counts (1, 3, 5, 7 hosts)
- Identify bottlenecks

**Example**:

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class MorganUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        self.client.post(
            "/api/chat",
            json={
                "text": "Hello Morgan, how are you?",
                "user_id": "load_test_user"
            }
        )

    @task
    def query(self):
        self.client.post(
            "/api/chat",
            json={
                "text": "What is in my documents?",
                "user_id": "load_test_user"
            }
        )
```

### Chaos Engineering

- Randomly kill services with `chaos-monkey`
- Test automatic recovery and data integrity
- Test network partition scenarios
- Verify no data loss during failures

---

## Deployment Strategy

### Phase 1: Single-Host Baseline

```bash
# Deploy all services on one host
docker-compose -f docker-compose-single-host.yml up -d
```

- All services on one host
- PostgreSQL single instance
- Redis single instance
- Qdrant single instance
- MinIO local
- Consul single server
- Validate all functionality

### Phase 2: Add Service Discovery

```bash
# Deploy Consul cluster
docker-compose -f docker-compose-consul.yml up -d

# Update services to register with Consul
# Test service discovery without distributing services
```

### Phase 3: Multi-Host with Docker Compose

```bash
# On each host, deploy host-specific compose file
# Host 1 (Primary CPU):
docker-compose -f docker-compose-host1-cpu-primary.yml up -d

# Host 2 (GPU 12GB):
docker-compose -f docker-compose-host2-gpu-12gb.yml up -d

# Host 3 (GPU 8GB):
docker-compose -f docker-compose-host3-gpu-8gb.yml up -d

# Services discover each other via Consul
```

### Phase 4: Docker Swarm Orchestration (Optional)

```bash
# Initialize Swarm on primary host
docker swarm init --advertise-addr 192.168.1.10

# Join workers
docker swarm join --token <token> 192.168.1.10:2377

# Label nodes
docker node update --label-add gpu=nvidia --label-add vram=12 host2
docker node update --label-add gpu=nvidia --label-add vram=8 host3
docker node update --label-add gpu=false --label-add ram=64 host1

# Deploy stack
docker stack deploy -c docker-stack.yml morgan
```

### Phase 5: Monitoring and Observability

```bash
# Deploy monitoring stack on secondary CPU host
docker-compose -f docker-compose-monitoring.yml up -d

# Access Grafana: http://host2:3000
# Access Prometheus: http://host2:9090
```

---

## Security Considerations

### Kubernetes Ingress & Gateway (MVP Approach)

```yaml
# Kubernetes Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: morgan-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
    - hosts:
        - api.morgan.local
      secretName: morgan-tls
  rules:
    - host: api.morgan.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: core-service
                port:
                  number: 8000
```

### Secure Secrets Management (Kubernetes Secrets)

**MVP Approach**: Use Kubernetes Secrets with RBAC and optional encryption at rest.

```yaml
# Create secret from file or literal
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
  namespace: morgan
type: Opaque
data:
  username: <base64-encoded>
  password: <base64-encoded>

---
# Enable encryption at rest in MicroK8s
# Edit /var/snap/microk8s/current/args/kube-apiserver
# Add: --encryption-provider-config=/path/to/encryption-config.yaml
```

**Accessing Secrets in Services**:

```yaml
# Mount secrets as environment variables
apiVersion: v1
kind: Pod
metadata:
  name: core-service
spec:
  containers:
    - name: core
      env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
```

**Secret Rotation**:
- Use external-secrets-operator for automated secret rotation from Vault (optional)
- Manual rotation: Update secret and restart pods to pick up new values
- Enable audit logging to track secret access

### Data Encryption

- **At Rest**: PostgreSQL data encrypted via LUKS disk encryption
- **In Transit**: TLS 1.3 for all HTTP communication
- **MinIO**: Use TLS for remote MinIO connections
- **Emotional Data**: Encrypt sensitive emotion data in PostgreSQL (AES-256)

---

## Performance Optimizations

### 1. Connection Pooling

```python
# PostgreSQL connection pool (SQLAlchemy)
engine = create_async_engine(
    connection_string,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

### 2. Caching Strategy

- **L1 Cache**: In-memory LRU cache per service (5 minutes TTL)
- **L2 Cache**: Redis cluster (1 hour TTL)
- **L3 Cache**: PostgreSQL (permanent)

### 3. GPU Memory Management

```python
# services/llm/service.py
import torch

def optimize_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
```

### 4. Batch Processing

- Embedding generation: Batch size 32-64
- Reranking: Batch size 100
- Database writes: Batch insert/update

---

## Monitoring and Observability

### Metrics to Track

1. **Service Metrics**:
   - Request count, latency (p50, p95, p99)
   - Error rate (4xx, 5xx)
   - Queue depth
   - GPU utilization (CUDA services)
   - VRAM usage

2. **Infrastructure Metrics**:
   - Host CPU, RAM, disk usage
   - Network throughput between hosts
   - Container restart count

3. **Business Metrics**:
   - Active conversations
   - Messages per conversation
   - RAG search precision/recall
   - Emotion detection accuracy
   - Learning adaptation rate

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "Morgan Multi-Host Overview",
    "panels": [
      {
        "title": "Service Health by Host",
        "targets": [
          {
            "expr": "up{job='morgan-*'}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_duty_cycle",
            "legendFormat": "{{hostname}}"
          }
        ]
      },
      {
        "title": "RAG Search Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_search_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

---

## Deployment Configurations

### Single-Host Docker Compose

```yaml
# docker-compose-single-host.yml
version: '3.8'

services:
  consul:
    image: hashicorp/consul:1.17
    command: agent -server -bootstrap -ui -client=0.0.0.0
    ports:
      - "8500:8500"
      - "8600:8600/udp"

  postgres:
    image: postgres:17-alpine
    environment:
      POSTGRES_DB: morgan
      POSTGRES_USER: morgan
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant-data:/qdrant/storage

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: morgan
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio-data:/data

  core:
    image: morgan-core:latest
    depends_on:
      - consul
      - postgres
      - redis
      - qdrant
      - minio
    environment:
      - CONSUL_HTTP_ADDR=consul:8500
    ports:
      - "8000:8000"

  # Add LLM, embedding, emotion, empathy, learning services...
```

### Multi-Host Docker Swarm Stack

```yaml
# docker-stack.yml
version: '3.8'

services:
  core:
    image: harbor.in.lazarev.cloud/morgan/core:latest
    deploy:
      mode: replicated
      replicas: 2
      placement:
        constraints:
          - node.labels.role==cpu
          - node.labels.ram==64
    environment:
      - CONSUL_HTTP_ADDR=consul.service.consul:8500
    networks:
      - morgan-overlay

  llm-main:
    image: harbor.in.lazarev.cloud/morgan/llm-main:latest
    deploy:
      mode: replicated
      replicas: 1
      placement:
        constraints:
          - node.labels.gpu==nvidia
          - node.labels.vram==24
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    networks:
      - morgan-overlay

  embedding:
    image: harbor.in.lazarev.cloud/morgan/embedding:latest
    deploy:
      mode: replicated
      replicas: 1
      placement:
        constraints:
          - node.labels.gpu==nvidia
          - node.labels.vram>=8
    networks:
      - morgan-overlay

networks:
  morgan-overlay:
    driver: overlay
    attachable: true
```

---

## Next Steps

1. ✅ Review and approve this design document
2. Create implementation plan document (tasks.md)
3. Set up Consul cluster on primary host
4. Implement CLI with Click framework
5. Implement PostgreSQL schema with Alembic migrations
6. Implement MinIO client and document ingestion
7. Implement service allocation logic
8. Deploy multi-host setup
9. Implement monitoring stack
10. Test failover scenarios
11. Document operational runbooks

---

**Document Status**: Ready for review and task planning
**Dependencies**: Requires requirements.md V3.2.0 approval
**Major Changes from V1.0.0**:
- Removed fixed 6-host architecture → flexible 1-7+ hosts
- Removed WebUI → CLI-only (Click framework)
- Added PostgreSQL code-only logic (no stored procedures/functions/triggers)
- Added MinIO S3 storage for files
- Kong API gateway now optional for MVP (direct Consul DNS access)
- Updated CUDA requirement: 12.4+ (from 13.0.1)
- Added comprehensive service allocation algorithm
- Added detailed PostgreSQL schema with SQLAlchemy ORM
- Added MinIO deployment configurations and client
- Added CLI architecture and commands
- Updated all diagrams for flexible host count
