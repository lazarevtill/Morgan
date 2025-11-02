# Morgan → JARVIS: Self-Hosted Setup Guide

## Hardware Configuration

**Your Setup:**
- 2x RTX 3090 (24GB each = 48GB total)
- 1x RTX 4070 (8GB)
- 1x RTX 2060 (6GB)
- 4x i9 systems with 64GB RAM each

**GPU Allocation:**
```
GPU 0 (RTX 3090 #1) + GPU 1 (RTX 3090 #2): Main reasoning LLM
GPU 2 (RTX 4070): Embeddings + Fast LLM
GPU 3 (RTX 2060): Reranking + Utilities
```

---

## Step 1: Install Ollama

### On Linux/WSL:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### On Windows (Native):
Download from: https://ollama.com/download/windows

---

## Step 2: Pull Models

### Main Reasoning LLM (for RTX 3090s)
```bash
# Qwen2.5-32B-Instruct with Q4_K_M quantization (~19GB)
ollama pull qwen2.5:32b-instruct-q4_K_M

# Alternative if above unavailable:
ollama pull qwen2.5:32b-instruct

# Test it
ollama run qwen2.5:32b-instruct-q4_K_M "Explain chain-of-thought reasoning in 3 steps."
```

### Fast Response LLM (for RTX 4070)
```bash
# Qwen2.5-7B for quick queries (~4.4GB)
ollama pull qwen2.5:7b-instruct-q5_K_M

# Alternative:
ollama pull qwen2.5:7b-instruct

# Test it
ollama run qwen2.5:7b-instruct-q5_K_M "What is Docker?"
```

### Embedding Model (for RTX 4070)
```bash
# Nomic Embed Text v1.5 (best for RAG)
ollama pull nomic-embed-text

# Alternative (Qwen embedding)
# ollama pull qwen2.5-embed:7b

# Test embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "The sky is blue"
}'
```

---

## Step 3: Configure Multi-GPU Setup

### Start Ollama with Multi-GPU (Tensor Parallelism)

**On System with 2x RTX 3090:**
```bash
# Set CUDA devices to use both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Start Ollama with 2 GPUs
OLLAMA_NUM_GPU=2 ollama serve --host 0.0.0.0:11434

# In another terminal, test it
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:32b-instruct-q4_K_M",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Start Embeddings Server (RTX 4070)

**On System with RTX 4070:**
```bash
# Use GPU 2 (or adjust based on your setup)
export CUDA_VISIBLE_DEVICES=2

# Start on different port
OLLAMA_NUM_GPU=1 ollama serve --host 0.0.0.0:11435

# Test embeddings endpoint
curl http://localhost:11435/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Test embedding"
}'
```

### Alternative: Single System Multi-Service

If all GPUs are on one system:
```bash
# Terminal 1: Main LLM on GPU 0+1
CUDA_VISIBLE_DEVICES=0,1 OLLAMA_NUM_GPU=2 ollama serve --host 0.0.0.0:11434

# Terminal 2: Fast LLM + Embeddings on GPU 2
CUDA_VISIBLE_DEVICES=2 OLLAMA_NUM_GPU=1 ollama serve --host 0.0.0.0:11435
```

---

## Step 4: Install Python Dependencies

```bash
cd morgan-rag

# Install additional dependencies for self-hosted setup
pip install sentence-transformers  # For local reranking
pip install html2text              # For HTML to Markdown
pip install psutil                 # For system monitoring

# Verify transformers version
pip install transformers>=4.35.0

# Update requirements
pip freeze > requirements.txt
```

---

## Step 5: Configure Morgan for Self-Hosted

### Update `.env` File

```bash
# Edit morgan-rag/.env
nano morgan-rag/.env
```

Add these settings:
```env
# ============================================================================
# SELF-HOSTED CONFIGURATION
# ============================================================================

# Main Reasoning LLM (RTX 3090s)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7

# Fast Response LLM (RTX 4070) - Optional
LLM_FAST_BASE_URL=http://localhost:11435/v1
LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M

# Embeddings (RTX 4070)
EMBEDDING_BASE_URL=http://localhost:11435/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
EMBEDDING_BATCH_SIZE=32
EMBEDDING_FORCE_LOCAL=true

# Local Fallback Embeddings
EMBEDDING_LOCAL_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Reranking (RTX 2060 or CPU)
RERANKING_ENABLED=true
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_DEVICE=cuda:3
RERANKING_BATCH_SIZE=16

# HTML Processing (Local)
HTML_TO_MARKDOWN_LIBRARY=html2text

# Vector Database (Qdrant - existing)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# System
DATA_DIR=./data
LOG_LEVEL=INFO
DEBUG=false

# Performance
WORKERS=4
CACHE_SIZE=1000
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

---

## Step 6: Test the Setup

### Test Main LLM
```bash
cd morgan-rag
python -c "
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

response = client.chat.completions.create(
    model='qwen2.5:32b-instruct-q4_K_M',
    messages=[{'role': 'user', 'content': 'Explain quantum computing in one sentence.'}]
)

print(response.choices[0].message.content)
print(f'Time: ~5-10s expected')
"
```

### Test Embeddings
```bash
python -c "
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11435/v1',
    api_key='ollama'
)

response = client.embeddings.create(
    model='nomic-embed-text',
    input='Hello world'
)

print(f'Embedding dimension: {len(response.data[0].embedding)}')
print('Expected: 768')
"
```

### Test Local Reranking
```bash
python -c "
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [
    ['What is Python?', 'Python is a programming language.'],
    ['What is Python?', 'The sky is blue.']
]

scores = model.predict(pairs)
print(f'Relevance scores: {scores}')
print('Higher score = more relevant')
"
```

---

## Step 7: Benchmark Performance

### Create Benchmark Script
```bash
cat > benchmark_setup.py << 'EOF'
#!/usr/bin/env python3
"""Benchmark self-hosted JARVIS setup"""
import time
from openai import OpenAI

# Initialize clients
llm_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
embed_client = OpenAI(base_url='http://localhost:11435/v1', api_key='ollama')

def benchmark_llm():
    """Benchmark main LLM"""
    queries = [
        "What is Docker?",  # Simple
        "Explain step-by-step how to deploy a web app with Docker.",  # Complex
        "Compare and contrast REST vs GraphQL APIs with pros and cons."  # Reasoning
    ]

    for query in queries:
        start = time.time()
        response = llm_client.chat.completions.create(
            model='qwen2.5:32b-instruct-q4_K_M',
            messages=[{'role': 'user', 'content': query}]
        )
        elapsed = time.time() - start

        print(f"\nQuery: {query[:50]}...")
        print(f"Response time: {elapsed:.2f}s")
        print(f"Tokens: ~{len(response.choices[0].message.content.split())}")

def benchmark_embeddings():
    """Benchmark embeddings"""
    texts = ["Test text"] * 10  # Batch of 10

    start = time.time()
    response = embed_client.embeddings.create(
        model='nomic-embed-text',
        input=texts
    )
    elapsed = time.time() - start

    print(f"\nEmbedding batch (10 texts): {elapsed:.3f}s")
    print(f"Target: <200ms")

if __name__ == "__main__":
    print("=== JARVIS Setup Benchmark ===\n")
    print("Testing LLM (Qwen2.5-32B)...")
    benchmark_llm()

    print("\n\nTesting Embeddings (Nomic)...")
    benchmark_embeddings()

    print("\n\n=== Targets ===")
    print("Simple queries: <2s")
    print("Complex queries: 5-10s")
    print("Embeddings batch: <200ms")
EOF

python benchmark_setup.py
```

**Expected Results:**
- Simple queries: 1-3s ✓
- Complex queries: 5-12s ✓
- Embeddings batch: 100-300ms ✓

---

## Step 8: Monitor GPU Usage

### Install Monitoring Tools
```bash
# Install nvtop (GPU monitor)
sudo apt install nvtop

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

### Monitor During Inference
```bash
# Terminal 1: Run query
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
response = client.chat.completions.create(
    model='qwen2.5:32b-instruct-q4_K_M',
    messages=[{'role': 'user', 'content': 'Write a story about AI'}]
)
print(response.choices[0].message.content)
"

# Terminal 2: Monitor GPUs
nvidia-smi -l 1
```

**Expected GPU Usage:**
- GPU 0+1 (RTX 3090s): 70-90% utilization during inference
- GPU Memory: ~20-22GB out of 24GB per card
- GPU 2 (RTX 4070): 40-60% during embeddings

---

## Step 9: Create Systemd Services (Optional)

### Main LLM Service
```bash
sudo tee /etc/systemd/system/ollama-main.service << EOF
[Unit]
Description=Ollama Main LLM Service
After=network.target

[Service]
Type=simple
User=$USER
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_NUM_GPU=2"
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama-main
sudo systemctl start ollama-main
```

### Embeddings Service
```bash
sudo tee /etc/systemd/system/ollama-embed.service << EOF
[Unit]
Description=Ollama Embeddings Service
After=network.target

[Service]
Type=simple
User=$USER
Environment="CUDA_VISIBLE_DEVICES=2"
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_HOST=0.0.0.0:11435"
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama-embed
sudo systemctl start ollama-embed
```

---

## Step 10: Verify Morgan Integration

### Test Morgan with New Setup
```bash
cd morgan-rag

# Run Morgan CLI
python -m morgan.cli.app
```

### Test Query
```
You: Explain Docker in 3 steps.

Morgan: [Should respond using local Qwen2.5-32B model]
```

**Expected behavior:**
- Response in 5-10 seconds
- No external API calls
- GPU utilization visible in `nvidia-smi`

---

## Troubleshooting

### Issue: "Connection refused" to Ollama
```bash
# Check if Ollama is running
ps aux | grep ollama

# Check port
netstat -tulpn | grep 11434

# Restart Ollama
pkill ollama
CUDA_VISIBLE_DEVICES=0,1 OLLAMA_NUM_GPU=2 ollama serve
```

### Issue: "Model not found"
```bash
# List available models
ollama list

# Pull model again
ollama pull qwen2.5:32b-instruct-q4_K_M
```

### Issue: "Out of memory"
```bash
# Check GPU memory
nvidia-smi

# Use smaller model or more quantization
ollama pull qwen2.5:14b-instruct-q4_K_M

# Update .env
LLM_MODEL=qwen2.5:14b-instruct-q4_K_M
```

### Issue: Slow inference (>15s)
```bash
# Check both GPUs are being used
nvidia-smi

# Verify tensor parallelism
echo $OLLAMA_NUM_GPU  # Should be 2

# Check model quantization
ollama show qwen2.5:32b-instruct-q4_K_M
```

---

## Next Steps

Once this setup is complete:
1. ✅ Self-hosted inference working
2. ✅ Performance benchmarks acceptable
3. ✅ GPU utilization optimized

**Proceed to Phase 2: Implement Multi-Step Reasoning**

See `PHASE_2_REASONING.md` for next steps.
