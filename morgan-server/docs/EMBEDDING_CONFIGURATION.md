# Embedding Configuration Guide

This guide explains how to configure embedding generation in Morgan Server. Morgan supports three embedding providers: local, remote Ollama, and OpenAI-compatible APIs.

## Overview

Embeddings are vector representations of text used for semantic search in the Knowledge Engine. Morgan Server provides flexible embedding configuration to support different deployment scenarios.

## Embedding Providers

### 1. Local Embeddings (Default)

Uses sentence-transformers locally for embedding generation.

#### Configuration

```yaml
embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_device: "cpu"  # Options: "cpu", "cuda", "mps"
```

#### Environment Variables

```bash
export MORGAN_EMBEDDING_PROVIDER="local"
export MORGAN_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export MORGAN_EMBEDDING_DEVICE="cpu"
```

#### Pros
- ✅ No external dependencies
- ✅ Fast for small batches
- ✅ No API costs
- ✅ Works offline
- ✅ Complete data privacy

#### Cons
- ❌ Requires local compute resources
- ❌ Limited to sentence-transformers models
- ❌ May be slower for large batches
- ❌ GPU required for optimal performance with large models

#### Recommended Models

| Model | Dimensions | Performance | Use Case |
|-------|-----------|-------------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | General purpose, default |
| `all-mpnet-base-v2` | 768 | Balanced | Better quality, slower |
| `multi-qa-mpnet-base-dot-v1` | 768 | Balanced | Question answering |
| `all-MiniLM-L12-v2` | 384 | Medium | Better than L6, still fast |

#### Device Selection

- **CPU**: Works everywhere, slower
- **CUDA**: NVIDIA GPUs, fastest
- **MPS**: Apple Silicon (M1/M2/M3), fast on Mac

### 2. Remote Ollama Embeddings

Uses a remote Ollama instance for embeddings.

#### Configuration

```yaml
embedding_provider: "ollama"
embedding_model: "qwen3-embedding"
embedding_endpoint: "http://ollama-server:11434"
# embedding_api_key: "optional-api-key"  # If your Ollama requires auth
```

#### Environment Variables

```bash
export MORGAN_EMBEDDING_PROVIDER="ollama"
export MORGAN_EMBEDDING_MODEL="qwen3-embedding"
export MORGAN_EMBEDDING_ENDPOINT="http://ollama-server:11434"
# export MORGAN_EMBEDDING_API_KEY="your-api-key"  # Optional
```

#### Pros
- ✅ Offload compute to remote server
- ✅ Access to Ollama's embedding models
- ✅ Can use GPU on remote server
- ✅ Self-hosted (no cloud dependencies)
- ✅ Complete data privacy

#### Cons
- ❌ Requires network connectivity
- ❌ Additional infrastructure to manage
- ❌ Network latency

#### Recommended Models

| Model | Dimensions | Context Length | Use Case |
|-------|-----------|----------------|----------|
| `qwen3-embedding` | 768 | 8192 | General purpose, high quality |
| `mxbai-embed-large` | 1024 | 512 | High quality, shorter context |
| `snowflake-arctic-embed` | 1024 | 512 | Retrieval tasks |

#### Setup Remote Ollama

1. **Install Ollama on remote server:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull embedding model:**
   ```bash
   ollama pull qwen3-embedding
   ```

3. **Configure Ollama to accept remote connections:**
   ```bash
   # Set environment variable
   export OLLAMA_HOST=0.0.0.0:11434
   
   # Or edit systemd service
   sudo systemctl edit ollama
   # Add: Environment="OLLAMA_HOST=0.0.0.0:11434"
   sudo systemctl restart ollama
   ```

4. **Test connection:**
   ```bash
   curl http://your-server:11434/api/embeddings -d '{
     "model": "qwen3-embedding",
     "prompt": "test"
   }'
   ```

### 3. OpenAI-Compatible Embeddings

Uses any OpenAI-compatible embedding API (OpenAI, Azure OpenAI, etc.).

#### Configuration

```yaml
embedding_provider: "openai-compatible"
embedding_model: "text-embedding-3-small"
embedding_endpoint: "https://api.openai.com/v1"
embedding_api_key: "your-api-key"
```

#### Environment Variables

```bash
export MORGAN_EMBEDDING_PROVIDER="openai-compatible"
export MORGAN_EMBEDDING_MODEL="text-embedding-3-small"
export MORGAN_EMBEDDING_ENDPOINT="https://api.openai.com/v1"
export MORGAN_EMBEDDING_API_KEY="your-api-key"
```

#### Pros
- ✅ Access to cloud-based embedding models
- ✅ No local compute required
- ✅ High-quality embeddings
- ✅ Scalable
- ✅ No infrastructure management

#### Cons
- ❌ Requires API key and costs money
- ❌ Network dependency
- ❌ Data sent to external service
- ❌ Rate limits may apply

#### Recommended Models

**OpenAI:**

| Model | Dimensions | Cost (per 1M tokens) | Use Case |
|-------|-----------|---------------------|----------|
| `text-embedding-3-small` | 1536 | $0.02 | Cost-effective, good quality |
| `text-embedding-3-large` | 3072 | $0.13 | Highest quality |
| `text-embedding-ada-002` | 1536 | $0.10 | Legacy, still good |

**Azure OpenAI:**

Same models as OpenAI, but deployed in Azure. Use Azure endpoint:

```yaml
embedding_endpoint: "https://your-resource.openai.azure.com"
```

## Configuration Validation

Morgan Server validates embedding configuration at startup:

### Local Provider
- ✅ `embedding_model` must be specified
- ✅ `embedding_device` must be one of: `cpu`, `cuda`, `mps`
- ✅ No endpoint required

### Remote Providers (Ollama, OpenAI-compatible)
- ✅ `embedding_endpoint` is **required**
- ✅ `embedding_endpoint` must start with `http://` or `https://`
- ✅ `embedding_api_key` is optional (but recommended for OpenAI)

### Invalid Configuration Examples

```yaml
# ❌ INVALID: Remote provider without endpoint
embedding_provider: "ollama"
embedding_model: "qwen3-embedding"
# Missing: embedding_endpoint

# ❌ INVALID: Invalid endpoint format
embedding_provider: "ollama"
embedding_endpoint: "localhost:11434"  # Missing http://

# ❌ INVALID: Invalid device
embedding_provider: "local"
embedding_device: "gpu"  # Should be "cuda"

# ❌ INVALID: Invalid provider
embedding_provider: "huggingface"  # Not supported
```

## Performance Considerations

### Local Embeddings

**CPU Performance:**
- Small models (384 dim): ~100-500 docs/sec
- Large models (768+ dim): ~50-200 docs/sec

**GPU Performance (CUDA):**
- Small models: ~1000-5000 docs/sec
- Large models: ~500-2000 docs/sec

**Optimization Tips:**
- Use batch processing for multiple documents
- Use smaller models for faster processing
- Use GPU when available
- Cache embeddings to avoid recomputation

### Remote Embeddings

**Network Latency:**
- Local network: ~1-10ms
- Internet: ~50-200ms

**Throughput:**
- Depends on remote server capacity
- Ollama: Limited by server GPU/CPU
- OpenAI: Rate limited by API tier

**Optimization Tips:**
- Use batch requests when possible
- Implement retry logic for network failures
- Monitor API rate limits
- Consider caching frequently used embeddings

## Migration Between Providers

### Switching Providers

When switching embedding providers, you may need to re-embed your documents:

1. **Backup your data:**
   ```bash
   # Backup Qdrant collection
   curl -X POST http://localhost:6333/collections/documents/snapshots
   ```

2. **Update configuration:**
   ```yaml
   # Change from local to remote
   embedding_provider: "ollama"
   embedding_endpoint: "http://ollama-server:11434"
   embedding_model: "qwen3-embedding"
   ```

3. **Re-embed documents:**
   - Option A: Delete and re-ingest all documents
   - Option B: Use migration script (if available)

### Dimension Compatibility

⚠️ **Important:** Different models produce different embedding dimensions. Qdrant collections are dimension-specific.

If switching models with different dimensions:
1. Create a new Qdrant collection with the new dimension
2. Re-ingest all documents
3. Update application to use new collection

## Troubleshooting

### Local Embeddings

**Problem:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:**
```bash
pip install sentence-transformers
```

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
- Switch to CPU: `embedding_device: "cpu"`
- Use smaller model: `all-MiniLM-L6-v2`
- Reduce batch size

**Problem:** Model download fails

**Solution:**
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Remote Ollama

**Problem:** `Connection refused` to Ollama endpoint

**Solution:**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Check firewall allows connections
- Verify `OLLAMA_HOST` is set correctly

**Problem:** Model not found

**Solution:**
```bash
# Pull model on Ollama server
ollama pull qwen3-embedding
```

**Problem:** Slow performance

**Solution:**
- Check network latency
- Ensure Ollama server has GPU
- Use batch requests
- Consider local embeddings

### OpenAI-Compatible

**Problem:** `401 Unauthorized`

**Solution:**
- Verify API key is correct
- Check API key has embedding permissions
- Ensure endpoint URL is correct

**Problem:** `429 Rate Limit Exceeded`

**Solution:**
- Implement exponential backoff
- Reduce request rate
- Upgrade API tier
- Consider caching

**Problem:** High costs

**Solution:**
- Use `text-embedding-3-small` instead of `large`
- Implement caching
- Consider local or Ollama embeddings

## Best Practices

### Development
- Use **local embeddings** for development
- Small models for fast iteration
- CPU is fine for small datasets

### Production (Self-Hosted)
- Use **remote Ollama** with GPU server
- Centralize embedding generation
- Monitor performance and costs

### Production (Cloud)
- Use **OpenAI-compatible** for simplicity
- Implement caching to reduce costs
- Monitor API usage and costs
- Set up alerts for rate limits

### Hybrid Approach
- Use **local** for development
- Use **Ollama** for staging
- Use **OpenAI** for production
- Keep configuration flexible

## Example Configurations

### Development Setup

```yaml
# Fast, local, no dependencies
embedding_provider: "local"
embedding_model: "all-MiniLM-L6-v2"
embedding_device: "cpu"
```

### Self-Hosted Production

```yaml
# Centralized, GPU-accelerated
embedding_provider: "ollama"
embedding_model: "qwen3-embedding"
embedding_endpoint: "http://embedding-server:11434"
```

### Cloud Production

```yaml
# Scalable, managed
embedding_provider: "openai-compatible"
embedding_model: "text-embedding-3-small"
embedding_endpoint: "https://api.openai.com/v1"
embedding_api_key: "${OPENAI_API_KEY}"
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  morgan-server:
    image: morgan-server:latest
    environment:
      - MORGAN_EMBEDDING_PROVIDER=ollama
      - MORGAN_EMBEDDING_MODEL=qwen3-embedding
      - MORGAN_EMBEDDING_ENDPOINT=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama-models:/root/.ollama
    command: serve

volumes:
  ollama-models:
```

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Ollama Embedding Models](https://ollama.com/library?sort=popular&q=embed)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
