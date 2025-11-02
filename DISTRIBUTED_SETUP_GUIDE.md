# Morgan JARVIS - Distributed Multi-Host Setup Guide

## Your Distributed Hardware Configuration

**4 Separate Hosts:**
- **Host 1**: 1x RTX 3090 (24GB), i9 CPU, 64GB RAM
- **Host 2**: 1x RTX 3090 (24GB), i9 CPU, 64GB RAM
- **Host 3**: 1x RTX 4070 (8GB), i9 CPU, 64GB RAM
- **Host 4**: 1x RTX 2060 (6GB), i9 CPU, 64GB RAM

**Network:** All hosts on same local network

---

## Distributed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Morgan JARVIS                          │
│                    (Main Orchestrator)                       │
│                   Runs on any host or PC                     │
└────────────┬────────────┬────────────┬──────────────────────┘
             │            │            │
    ┌────────▼──────┐ ┌──▼──────────┐ ┌▼────────────────┐
    │   Host 1      │ │   Host 2    │ │   Host 3        │
    │ RTX 3090 #1   │ │ RTX 3090 #2 │ │  RTX 4070       │
    │ Main LLM      │ │ Main LLM    │ │  Embeddings +   │
    │ (Primary)     │ │ (Backup)    │ │  Fast LLM       │
    └───────────────┘ └─────────────┘ └─────────────────┘
                                              │
                                       ┌──────▼─────────┐
                                       │   Host 4       │
                                       │  RTX 2060      │
                                       │  Reranking     │
                                       └────────────────┘
```

---

## Setup Strategy

### Option A: Load Balancing (Recommended)
Run same model on both RTX 3090s, distribute load:
- **Host 1 (3090)**: Primary LLM instance
- **Host 2 (3090)**: Secondary LLM instance (load balancing)
- **Host 3 (4070)**: Embeddings + Fast LLM
- **Host 4 (2060)**: Reranking

### Option B: Model Sharding
Distribute different models across hosts:
- **Host 1 (3090)**: Main reasoning LLM (Qwen2.5-32B)
- **Host 2 (3090)**: Code/specialized LLM (CodeQwen-32B)
- **Host 3 (4070)**: Embeddings + Fast LLM
- **Host 4 (2060)**: Reranking

### Option C: Pipeline Processing
Sequential processing across hosts:
- **Host 1**: Embedding generation
- **Host 2**: Vector search + retrieval
- **Host 3**: LLM generation
- **Host 4**: Reranking + final response

**Recommendation**: **Option A** - Simple, fault-tolerant, good performance

---

## Step 1: Network Setup

### Configure Network Access

On each host, ensure they can communicate:

```bash
# Test connectivity from main host to all GPU hosts
ping host1.local  # or IP: 192.168.1.10
ping host2.local  # or IP: 192.168.1.11
ping host3.local  # or IP: 192.168.1.12
ping host4.local  # or IP: 192.168.1.13

# Ensure ports are open (run on each GPU host)
# Ollama default: 11434
# Custom ports: 11435, 11436, 11437
sudo ufw allow 11434/tcp
sudo ufw allow 11435/tcp
```

### Create Host Configuration File

```bash
# /etc/hosts on main orchestrator
192.168.1.10  host1-gpu3090-1
192.168.1.11  host2-gpu3090-2
192.168.1.12  host3-gpu4070
192.168.1.13  host4-gpu2060
```

---

## Step 2: Install Ollama on Each Host

### Host 1 (RTX 3090 #1) - Main LLM

```bash
# SSH to Host 1
ssh user@host1-gpu3090-1

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull main reasoning model
ollama pull qwen2.5:32b-instruct-q4_K_M

# Start Ollama on all network interfaces
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Or as systemd service
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

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

### Host 2 (RTX 3090 #2) - Secondary LLM

```bash
# SSH to Host 2
ssh user@host2-gpu3090-2

# Same setup as Host 1
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:32b-instruct-q4_K_M

# Start on port 11434 (same as Host 1 for load balancing)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Host 3 (RTX 4070) - Embeddings + Fast LLM

```bash
# SSH to Host 3
ssh user@host3-gpu4070

# Install and configure
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Pull fast LLM
ollama pull qwen2.5:7b-instruct-q5_K_M

# Start Ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Host 4 (RTX 2060) - Reranking

```bash
# SSH to Host 4
ssh user@host4-gpu2060

# Install Python and dependencies (no Ollama needed)
pip install sentence-transformers torch

# Create reranking service script
cat > /home/user/reranking_service.py << 'EOF'
#!/usr/bin/env python3
"""
Local Reranking Service

Runs a FastAPI service for reranking search results.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import uvicorn

app = FastAPI(title="Morgan Reranking Service")

# Load model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]
    ranked_indices: List[int]

@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """Rerank documents for query"""
    try:
        # Create pairs
        pairs = [[request.query, doc] for doc in request.documents]

        # Get scores
        scores = model.predict(pairs).tolist()

        # Sort by score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return RerankResponse(
            scores=scores,
            ranked_indices=ranked_indices
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ms-marco-MiniLM-L-6-v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

chmod +x /home/user/reranking_service.py

# Run as systemd service
sudo tee /etc/systemd/system/reranking.service << EOF
[Unit]
Description=Morgan Reranking Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/user
ExecStart=/usr/bin/python3 /home/user/reranking_service.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable reranking
sudo systemctl start reranking
```

---

## Step 3: Configure Morgan for Distributed Setup

### Update `.env` on Main Orchestrator

```env
# ============================================================================
# DISTRIBUTED MULTI-HOST CONFIGURATION
# ============================================================================

# Main LLM Instances (Load Balanced)
LLM_ENDPOINTS=http://host1-gpu3090-1:11434/v1,http://host2-gpu3090-2:11434/v1
LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
LLM_LOAD_BALANCING=round_robin  # or: random, least_loaded

# Fast LLM Instance (Host 3)
LLM_FAST_ENDPOINT=http://host3-gpu4070:11434/v1
LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M

# Embeddings (Host 3)
EMBEDDING_ENDPOINT=http://host3-gpu4070:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BATCH_SIZE=32

# Reranking (Host 4)
RERANKING_ENABLED=true
RERANKING_ENDPOINT=http://host4-gpu2060:8080/rerank
RERANKING_BATCH_SIZE=20

# Fallback Configuration
ENABLE_FALLBACK=true
FALLBACK_ON_TIMEOUT=true
TIMEOUT_SECONDS=30

# Health Monitoring
HEALTH_CHECK_INTERVAL=60
ENABLE_AUTO_FAILOVER=true

# Performance
MAX_CONCURRENT_REQUESTS=4
REQUEST_TIMEOUT=60
```

---

## Step 4: Implement Distributed Infrastructure

### Create Distributed LLM Client

```python
# morgan/infrastructure/distributed_llm.py
"""
Distributed LLM Client for Multi-Host Setup

Manages LLM requests across multiple hosts with:
- Load balancing
- Failover
- Health monitoring
"""

import random
from typing import List, Optional
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
import httpx
from enum import Enum

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"


@dataclass
class LLMEndpoint:
    """LLM endpoint configuration"""
    url: str
    model: str
    healthy: bool = True
    response_times: List[float] = None
    error_count: int = 0

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []


class DistributedLLMClient:
    """
    Distributed LLM client with load balancing and failover.

    Example:
        >>> client = DistributedLLMClient(
        ...     endpoints=[
        ...         "http://host1:11434/v1",
        ...         "http://host2:11434/v1"
        ...     ],
        ...     model="qwen2.5:32b-instruct-q4_K_M"
        ... )
        >>> response = await client.generate("What is Python?")
    """

    def __init__(
        self,
        endpoints: List[str],
        model: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        api_key: str = "ollama"
    ):
        """
        Initialize distributed LLM client.

        Args:
            endpoints: List of LLM endpoint URLs
            model: Model name (must be available on all endpoints)
            strategy: Load balancing strategy
            api_key: API key (default: "ollama" for Ollama)
        """
        self.endpoints = [
            LLMEndpoint(url=url, model=model)
            for url in endpoints
        ]
        self.model = model
        self.strategy = strategy
        self.api_key = api_key
        self.current_index = 0

        logger.info(f"Initialized DistributedLLMClient with {len(self.endpoints)} endpoints")
        logger.info(f"Strategy: {strategy}, Model: {model}")

    def _select_endpoint(self) -> LLMEndpoint:
        """Select endpoint based on load balancing strategy"""
        healthy_endpoints = [e for e in self.endpoints if e.healthy]

        if not healthy_endpoints:
            logger.error("No healthy endpoints available!")
            # Try to use any endpoint as fallback
            return self.endpoints[0]

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round-robin through healthy endpoints
            endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
            self.current_index += 1
            return endpoint

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_endpoints)

        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select endpoint with lowest average response time
            return min(
                healthy_endpoints,
                key=lambda e: sum(e.response_times[-10:]) / len(e.response_times[-10:])
                if e.response_times else 0
            )

        return healthy_endpoints[0]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ):
        """
        Generate response using distributed LLMs.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Enable streaming

        Returns:
            Generated text or stream
        """
        import time

        # Try up to 3 times (primary + 2 fallbacks)
        for attempt in range(3):
            endpoint = self._select_endpoint()

            try:
                logger.info(f"Attempt {attempt + 1}: Using endpoint {endpoint.url}")

                client = AsyncOpenAI(
                    base_url=endpoint.url,
                    api_key=self.api_key,
                    timeout=httpx.Timeout(60.0)
                )

                start_time = time.time()

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )

                elapsed = time.time() - start_time

                # Track response time
                endpoint.response_times.append(elapsed)
                endpoint.error_count = 0  # Reset error count on success

                logger.info(f"✓ Success in {elapsed:.2f}s using {endpoint.url}")

                if stream:
                    return response
                else:
                    return response.choices[0].message.content

            except Exception as e:
                logger.error(f"✗ Error with endpoint {endpoint.url}: {e}")
                endpoint.error_count += 1

                # Mark as unhealthy after 3 consecutive errors
                if endpoint.error_count >= 3:
                    endpoint.healthy = False
                    logger.warning(f"Marked {endpoint.url} as unhealthy")

                if attempt == 2:
                    raise  # Re-raise on last attempt

                continue

    async def health_check(self):
        """Check health of all endpoints"""
        import asyncio

        async def check_endpoint(endpoint: LLMEndpoint):
            try:
                client = AsyncOpenAI(
                    base_url=endpoint.url,
                    api_key=self.api_key,
                    timeout=httpx.Timeout(5.0)
                )

                # Simple test query
                await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )

                endpoint.healthy = True
                endpoint.error_count = 0
                logger.info(f"✓ {endpoint.url} is healthy")

            except Exception as e:
                endpoint.healthy = False
                logger.error(f"✗ {endpoint.url} is unhealthy: {e}")

        # Check all endpoints in parallel
        await asyncio.gather(*[
            check_endpoint(endpoint)
            for endpoint in self.endpoints
        ])

    def get_stats(self):
        """Get statistics for all endpoints"""
        return {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": sum(1 for e in self.endpoints if e.healthy),
            "endpoints": [
                {
                    "url": e.url,
                    "healthy": e.healthy,
                    "error_count": e.error_count,
                    "avg_response_time": (
                        sum(e.response_times[-10:]) / len(e.response_times[-10:])
                        if e.response_times else 0
                    )
                }
                for e in self.endpoints
            ]
        }
```

---

## Step 5: Test Distributed Setup

### Test Script

```python
# test_distributed.py
#!/usr/bin/env python3
"""Test distributed Morgan JARVIS setup"""

import asyncio
from morgan.infrastructure.distributed_llm import DistributedLLMClient

async def main():
    # Initialize client
    client = DistributedLLMClient(
        endpoints=[
            "http://host1-gpu3090-1:11434/v1",
            "http://host2-gpu3090-2:11434/v1"
        ],
        model="qwen2.5:32b-instruct-q4_K_M"
    )

    # Health check
    print("Running health check...")
    await client.health_check()
    print(client.get_stats())

    # Test queries
    queries = [
        "What is Docker?",
        "Explain Kubernetes.",
        "What is Python?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = await client.generate(query)
        print(f"Response: {response[:100]}...")

    # Final stats
    print("\nFinal stats:")
    print(client.get_stats())

if __name__ == "__main__":
    asyncio.run(main())
```

Run test:
```bash
python test_distributed.py
```

---

## Step 6: Monitoring Distributed Setup

### Create Monitoring Dashboard

```python
# monitor_hosts.py
#!/usr/bin/env python3
"""Monitor all GPU hosts"""

import httpx
import asyncio
from rich.console import Console
from rich.table import Table
from rich.live import Live
import time

console = Console()

ENDPOINTS = {
    "Host 1 (3090)": "http://host1-gpu3090-1:11434",
    "Host 2 (3090)": "http://host2-gpu3090-2:11434",
    "Host 3 (4070)": "http://host3-gpu4070:11434",
    "Host 4 (2060)": "http://host4-gpu2060:8080/health",
}

async def check_endpoint(name: str, url: str):
    """Check if endpoint is alive"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            start = time.time()
            response = await client.get(url)
            latency = (time.time() - start) * 1000  # ms

            return {
                "name": name,
                "status": "✓ Online" if response.status_code == 200 else "✗ Error",
                "latency_ms": f"{latency:.0f}ms",
                "healthy": True
            }
    except Exception as e:
        return {
            "name": name,
            "status": "✗ Offline",
            "latency_ms": "N/A",
            "healthy": False
        }

async def monitor_loop():
    """Continuous monitoring loop"""
    while True:
        # Check all endpoints
        results = await asyncio.gather(*[
            check_endpoint(name, url)
            for name, url in ENDPOINTS.items()
        ])

        # Create table
        table = Table(title="Morgan JARVIS - Host Status")
        table.add_column("Host", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Latency", style="yellow")

        for result in results:
            table.add_row(
                result["name"],
                result["status"],
                result["latency_ms"]
            )

        # Display
        console.clear()
        console.print(table)
        console.print("\n[italic]Press Ctrl+C to exit[/italic]")

        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")
```

Run monitoring:
```bash
pip install rich httpx
python monitor_hosts.py
```

---

## Architecture Benefits

### Advantages of Distributed Setup

✅ **Fault Tolerance**
- If one host fails, others continue working
- Automatic failover to healthy hosts
- No single point of failure

✅ **Scalability**
- Easy to add more hosts
- Horizontal scaling as needed
- Independent model upgrades per host

✅ **Flexibility**
- Different models on different hosts
- Specialized hosts for specialized tasks
- Can run different versions for testing

✅ **Resource Utilization**
- All GPUs fully utilized
- CPU and RAM distributed
- Network bandwidth distributed

---

## Performance Expectations

### Distributed vs Single-Host

**Distributed (Your Setup):**
- Latency: +10-50ms (network overhead)
- Throughput: 2x (two 3090s serving independently)
- Fault tolerance: High
- Scalability: Excellent

**Single-Host (Original Plan):**
- Latency: Lower (no network)
- Throughput: 1x (tensor parallelism on 2 GPUs = 1 model)
- Fault tolerance: None
- Scalability: Limited

**Verdict:** Distributed is **better** for your use case!

---

## Next Steps

1. **Setup all 4 hosts** following this guide
2. **Test connectivity** between hosts
3. **Implement distributed client** in Morgan
4. **Run benchmarks** to validate performance
5. **Proceed to Phase 2** - Reasoning engine

---

**Updated:** November 2, 2025
**Architecture:** Distributed Multi-Host (4 hosts, 4 GPUs)
**Status:** Ready for implementation
