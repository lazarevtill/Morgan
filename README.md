# Morgan AI Assistant

Morgan is a self-hosted, distributed personal AI assistant with emotional intelligence, knowledge management, and natural conversation capabilities. Built with a clean modular architecture for privacy-focused deployment.

## ✨ Features

- **🧠 Emotional Intelligence** - Understands emotions, responds empathetically, builds relationships
- **📚 Knowledge Engine** - RAG system with vector database and semantic search
- **🎯 Personalization** - User profiles, preferences, and conversation memory
- **🏠 Self-Hosted** - Run entirely on your own hardware, no external APIs required
- **⚡ Distributed** - Scale across multiple hosts with load balancing and failover
- **🔒 Privacy First** - All data stays on your hardware

## 🏗️ Architecture

Morgan consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                        Morgan Stack                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ morgan-cli  │  │morgan-server│  │     morgan-rag      │  │
│  │   (TUI)     │◄─┤   (API)     │◄─┤  (Core Intelligence)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                              │               │
│                          ┌───────────────────┼───────────────┤
│                          │                   │               │
│  ┌─────────────┐  ┌──────▼──────┐  ┌────────▼────────┐     │
│  │   Ollama    │  │   Qdrant    │  │     Redis       │     │
│  │   (LLM)     │  │ (Vector DB) │  │   (Cache)       │     │
│  └─────────────┘  └─────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description | Status |
|-----------|-------------|--------|
| **morgan-rag** | Core intelligence: services, emotional intelligence, memory, search | ✅ Active |
| **morgan-server** | FastAPI server with REST/WebSocket API | ✅ Active |
| **morgan-cli** | Terminal UI client | ✅ Active |
| **docker** | Docker deployment configurations | ✅ Active |

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/your-repo/morgan.git
cd morgan

# Start services
cd docker
cp env.example .env
# Edit .env with your settings
docker-compose up -d

# Pull LLM model
docker-compose exec ollama ollama pull qwen2.5:7b

# Install client
pip install -e morgan-cli

# Start chatting
export MORGAN_SERVER_URL=http://localhost:8080
morgan chat
```

### Manual Installation

**1. Start Dependencies:**

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start Ollama and pull model
ollama serve &
ollama pull qwen2.5:7b
```

**2. Install Server:**

```bash
cd morgan-server
pip install -e .
python -m morgan_server
```

**3. Install Client:**

```bash
cd morgan-cli
pip install -e .
morgan chat
```

## 📁 Project Structure

```
Morgan/
├── morgan-rag/              # Core RAG intelligence
│   └── morgan/
│       ├── services/        # Unified service layer
│       │   ├── llm/         # LLM service (single + distributed)
│       │   ├── embeddings/  # Embedding service
│       │   └── reranking/   # Reranking service
│       ├── intelligence/    # Emotional intelligence
│       ├── memory/          # Conversation memory
│       ├── search/          # Multi-stage search
│       ├── learning/        # Pattern learning
│       └── ...
│
├── morgan-server/           # FastAPI server
├── morgan-cli/              # Terminal client
├── docker/                  # Docker configs
├── shared/                  # Shared utilities
└── archive/                 # Archived deprecated code
```

## 🔧 Configuration

Morgan supports multiple configuration methods:

1. **Environment variables** (highest precedence)
2. **Configuration files** (YAML, JSON, .env)
3. **Default values** (lowest precedence)

### Example Configuration

```yaml
# Server settings
host: "0.0.0.0"
port: 8080

# LLM settings
llm_provider: "ollama"
llm_endpoint: "http://localhost:11434"
llm_model: "qwen2.5:7b"

# Vector database
vector_db_url: "http://localhost:6333"

# Embedding settings
embedding_model: "qwen3-embedding:4b"
embedding_dimensions: 2048
```

## 💻 Usage

### Python API

```python
from morgan.services import (
    get_llm_service,
    get_embedding_service,
    get_reranking_service,
)

# Get service instances
llm = get_llm_service()
embeddings = get_embedding_service()

# Generate response
response = llm.generate("What is Python?")
print(response.content)

# Async generation
response = await llm.agenerate("Explain Docker")

# Embeddings
embedding = embeddings.encode("Document text")
```

### CLI

```bash
# Start chat
morgan chat

# Check health
morgan health

# Learn from document
morgan learn /path/to/document.pdf
```

### REST API

```bash
# Chat
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Morgan!"}'

# Health check
curl http://localhost:8080/health
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [claude.md](./claude.md) | Complete project context for AI assistants |
| [DOCUMENTATION.md](./DOCUMENTATION.md) | Documentation index |
| [MIGRATION.md](./MIGRATION.md) | Migration guide |
| [docker/README.md](./docker/README.md) | Docker deployment guide |
| [morgan-server/README.md](./morgan-server/README.md) | Server documentation |
| [morgan-cli/README.md](./morgan-cli/README.md) | CLI documentation |

## 🖥️ Hardware Requirements

### Minimum (Single Host)
- **CPU:** 4+ cores
- **RAM:** 16GB
- **GPU:** 8GB VRAM (for local LLM)
- **Disk:** 50GB free space

### Recommended (Distributed)
- **Host 1-2:** CPU hosts (i9, 64GB RAM) for orchestration
- **Host 3-4:** GPU hosts (RTX 3090) for main LLM
- **Host 5:** GPU host (RTX 4070) for embeddings
- **Host 6:** GPU host (RTX 2060) for reranking

## 🛠️ Development

### Running Tests

```bash
# Server tests
cd morgan-server && pytest

# RAG tests
cd morgan-rag && pytest

# Client tests
cd morgan-cli && pytest
```

### Code Quality

```bash
# Format
black morgan_server morgan_cli

# Lint
ruff check .

# Type check
mypy morgan_server
```

## 📋 Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Infrastructure & Services | ✅ 83% Complete |
| Phase 2 | Multi-Step Reasoning | ⏳ Planned |
| Phase 3 | Proactive Features | ⏳ Planned |
| Phase 4 | Enhanced Context | ⏳ Planned |
| Phase 5 | Production Polish | ⏳ Planned |

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines (coming soon).

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Documentation:** [DOCUMENTATION.md](./DOCUMENTATION.md)
- **Project Context:** [claude.md](./claude.md)
- **Migration Guide:** [MIGRATION.md](./MIGRATION.md)

---

**Morgan** - Your private, emotionally intelligent AI companion.
