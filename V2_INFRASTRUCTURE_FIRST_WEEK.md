# Morgan v2 - Week 1: Infrastructure Setup
## Foundation for Complete Emotional AI System

> **Goal**: Set up infrastructure for FULL v2-0.0.1 system
> **NOT**: A simple chatbot, but foundation for complete emotional AI
> **Duration**: Week 1-2

---

## What We're Preparing For

This week sets up infrastructure for:
- ✅ 11 emotion detection modules
- ✅ 5 empathy modules
- ✅ 6 learning modules
- ✅ Advanced RAG with reranking
- ✅ Multi-layer memory system
- ✅ Relationship tracking

**Week 1 is about getting all services ready**, not building a minimal chatbot.

---

## Day 1-2: Project Structure & Dependencies

### Directory Structure
```bash
mkdir -p morgan-rag/morgan/{
  cli,                    # CLI interface
  core,                   # Assistant orchestration
  emotions,               # 11 emotion modules
  empathy,                # 5 empathy modules
  learning,               # 6 learning modules
  memory,                 # Memory processing
  search,                 # Multi-stage search
  storage,                # Vector, profile, memory storage
  services,               # Embedding, LLM services
  jina,                   # Reranking service
  companion,              # Relationship tracking
  config,                 # Configuration
  utils                   # Utilities
}

mkdir -p morgan-rag/tests/{
  unit,
  integration,
  e2e
}
```

### Core Dependencies

**requirements.txt**:
```ini
# ============================================
# Core Framework
# ============================================
click>=8.1.0                    # CLI framework
rich>=13.0.0                    # Terminal formatting
httpx>=0.27.0                   # Async HTTP client
pydantic>=2.0.0                 # Data validation
pydantic-settings>=2.0.0        # Settings management
python-dotenv>=1.0.0            # Environment variables

# ============================================
# LLM & Embeddings
# ============================================
openai>=1.0.0                   # Ollama integration
sentence-transformers>=2.0.0    # Local embedding fallback
transformers>=4.36.0            # Jina reranker
torch>=2.1.0                    # PyTorch for models
accelerate>=0.25.0              # GPU acceleration

# ============================================
# RAG & Document Processing
# ============================================
qdrant-client>=1.7.0            # Vector database
langchain>=0.1.0                # Document processing
langchain-community>=0.0.10     # Community loaders
pypdf>=3.0.0                    # PDF parsing
python-docx>=1.0.0              # DOCX parsing
beautifulsoup4>=4.12.0          # HTML parsing
markdown>=3.5.0                 # Markdown parsing
chardet>=5.2.0                  # Character encoding detection

# ============================================
# Storage & Database
# ============================================
sqlalchemy>=2.0.0               # ORM
asyncpg>=0.29.0                 # Async PostgreSQL
aiosqlite>=0.19.0               # Async SQLite (fallback)
redis>=5.0.0                    # Redis client
aiofiles>=23.0.0                # Async file operations

# ============================================
# Data Processing
# ============================================
numpy>=1.24.0                   # Numerical operations
pandas>=2.0.0                   # Data analysis
scikit-learn>=1.3.0             # ML utilities
python-dateutil>=2.8.0          # Date parsing

# ============================================
# Monitoring & Logging
# ============================================
structlog>=23.2.0               # Structured logging
prometheus-client>=0.19.0       # Metrics
psutil>=5.9.0                   # System monitoring

# ============================================
# Development
# ============================================
pytest>=7.4.0                   # Testing
pytest-asyncio>=0.21.0          # Async testing
pytest-cov>=4.1.0               # Coverage
black>=23.12.0                  # Formatting
ruff>=0.1.0                     # Linting
mypy>=1.7.0                     # Type checking
```

### Environment Configuration

**.env.example**:
```bash
# ============================================
# LLM Configuration
# ============================================
MORGAN_LLM_PROVIDER=ollama
MORGAN_LLM_BASE_URL=http://localhost:11434
MORGAN_LLM_MODEL=qwen2.5:32b           # Or qwen2.5:7b for lower resource

# ============================================
# Embedding Configuration
# ============================================
MORGAN_EMBEDDING_PROVIDER=ollama
MORGAN_EMBEDDING_MODEL=qwen3-embedding:latest
MORGAN_EMBEDDING_BATCH_SIZE=32
MORGAN_EMBEDDING_DIMENSIONS=1024

# ============================================
# Reranking Configuration
# ============================================
MORGAN_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
MORGAN_RERANKER_DEVICE=cuda              # or cpu
MORGAN_RERANKER_BATCH_SIZE=16

# ============================================
# Vector Database (Qdrant)
# ============================================
MORGAN_QDRANT_HOST=localhost
MORGAN_QDRANT_PORT=6333
MORGAN_QDRANT_GRPC_PORT=6334
MORGAN_QDRANT_API_KEY=                   # Optional
MORGAN_QDRANT_COLLECTION_PREFIX=morgan_

# ============================================
# PostgreSQL (Optional - fallback to SQLite)
# ============================================
MORGAN_POSTGRES_HOST=localhost
MORGAN_POSTGRES_PORT=5432
MORGAN_POSTGRES_DB=morgan
MORGAN_POSTGRES_USER=morgan
MORGAN_POSTGRES_PASSWORD=

# ============================================
# Redis (Optional - in-memory fallback)
# ============================================
MORGAN_REDIS_HOST=localhost
MORGAN_REDIS_PORT=6379
MORGAN_REDIS_DB=0
MORGAN_REDIS_PASSWORD=

# ============================================
# Emotion Detection Settings
# ============================================
MORGAN_EMOTION_DETECTION_ENABLED=true
MORGAN_EMOTION_MIN_CONFIDENCE=0.6
MORGAN_EMOTION_CONTEXT_WINDOW=10         # Messages
MORGAN_EMOTION_MEMORY_RETENTION_DAYS=90

# ============================================
# Empathy Settings
# ============================================
MORGAN_EMPATHY_ENABLED=true
MORGAN_EMPATHY_TONE_ADJUSTMENT=true
MORGAN_EMPATHY_VALIDATION_LEVEL=moderate  # low, moderate, high

# ============================================
# Learning Settings
# ============================================
MORGAN_LEARNING_ENABLED=true
MORGAN_LEARNING_RATE=0.1
MORGAN_LEARNING_FEEDBACK_WEIGHT=0.7
MORGAN_LEARNING_ADAPTATION_SPEED=moderate # slow, moderate, fast

# ============================================
# Memory Settings
# ============================================
MORGAN_MEMORY_SHORT_TERM_SIZE=50         # Messages
MORGAN_MEMORY_LONG_TERM_ENABLED=true
MORGAN_MEMORY_CONSOLIDATION_INTERVAL=3600 # Seconds
MORGAN_MEMORY_RETENTION_DAYS=365

# ============================================
# RAG Settings
# ============================================
MORGAN_RAG_ENABLED=true
MORGAN_RAG_CHUNK_SIZE=512
MORGAN_RAG_CHUNK_OVERLAP=50
MORGAN_RAG_TOP_K_COARSE=20
MORGAN_RAG_TOP_K_MEDIUM=10
MORGAN_RAG_TOP_K_FINE=5
MORGAN_RAG_RERANK_TOP_K=3

# ============================================
# System Settings
# ============================================
MORGAN_LOG_LEVEL=INFO
MORGAN_LOG_FORMAT=json                   # json or text
MORGAN_DATA_DIR=~/.morgan
MORGAN_CACHE_DIR=~/.morgan/cache
MORGAN_MAX_WORKERS=4
```

### Docker Compose for Services

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  # Ollama for LLM & Embeddings
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL (optional)
  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: morgan
      POSTGRES_USER: morgan
      POSTGRES_PASSWORD: morgan_dev_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis (optional)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  qdrant_data:
  ollama_data:
  postgres_data:
  redis_data:
```

---

## Day 3: Service Setup & Verification

### Start Services
```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check Qdrant
curl http://localhost:6333/collections

# Check Ollama
curl http://localhost:11434/api/tags
```

### Pull Models
```bash
# LLM model
docker exec -it morgan-ollama ollama pull qwen2.5:7b
# or for more capable (requires more VRAM)
docker exec -it morgan-ollama ollama pull qwen2.5:32b

# Embedding model
docker exec -it morgan-ollama ollama pull qwen3-embedding:latest

# Verify models
docker exec -it morgan-ollama ollama list
```

### Setup Qdrant Collections
**File**: `morgan-rag/scripts/setup_qdrant.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# Create collections for hierarchical embeddings
collections = {
    "morgan_knowledge_coarse": 1024,   # Document-level
    "morgan_knowledge_medium": 1024,   # Section-level
    "morgan_knowledge_fine": 1024,     # Chunk-level
    "morgan_memory_emotional": 1024,   # Emotional memories
    "morgan_memory_conversation": 1024, # Conversation history
}

for name, dimension in collections.items():
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"✓ Created collection: {name}")
    except Exception as e:
        print(f"✗ Collection {name} may already exist: {e}")
```

Run:
```bash
python morgan-rag/scripts/setup_qdrant.py
```

---

## Day 4: Configuration System

**File**: `morgan/config/settings.py`

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from pathlib import Path


class LLMSettings(BaseSettings):
    """LLM configuration"""
    provider: Literal['ollama', 'openai'] = 'ollama'
    base_url: str = 'http://localhost:11434'
    model: str = 'qwen2.5:7b'
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingSettings(BaseSettings):
    """Embedding configuration"""
    provider: Literal['ollama', 'local'] = 'ollama'
    model: str = 'qwen3-embedding:latest'
    batch_size: int = 32
    dimensions: int = 1024


class RerankerSettings(BaseSettings):
    """Reranking configuration"""
    model: str = 'jinaai/jina-reranker-v2-base-multilingual'
    device: Literal['cuda', 'cpu'] = 'cuda'
    batch_size: int = 16


class QdrantSettings(BaseSettings):
    """Qdrant configuration"""
    host: str = 'localhost'
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    collection_prefix: str = 'morgan_'


class EmotionSettings(BaseSettings):
    """Emotion detection configuration"""
    detection_enabled: bool = True
    min_confidence: float = 0.6
    context_window: int = 10
    memory_retention_days: int = 90


class EmpathySettings(BaseSettings):
    """Empathy configuration"""
    enabled: bool = True
    tone_adjustment: bool = True
    validation_level: Literal['low', 'moderate', 'high'] = 'moderate'


class LearningSettings(BaseSettings):
    """Learning configuration"""
    enabled: bool = True
    learning_rate: float = 0.1
    feedback_weight: float = 0.7
    adaptation_speed: Literal['slow', 'moderate', 'fast'] = 'moderate'


class MemorySettings(BaseSettings):
    """Memory configuration"""
    short_term_size: int = 50
    long_term_enabled: bool = True
    consolidation_interval: int = 3600
    retention_days: int = 365


class RAGSettings(BaseSettings):
    """RAG configuration"""
    enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_coarse: int = 20
    top_k_medium: int = 10
    top_k_fine: int = 5
    rerank_top_k: int = 3


class Settings(BaseSettings):
    """Main Morgan configuration"""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='MORGAN_',
        case_sensitive=False,
        extra='ignore'
    )

    # Component settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    emotion: EmotionSettings = Field(default_factory=EmotionSettings)
    empathy: EmpathySettings = Field(default_factory=EmpathySettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)

    # System settings
    log_level: str = 'INFO'
    log_format: Literal['json', 'text'] = 'json'
    data_dir: Path = Field(default_factory=lambda: Path.home() / '.morgan')
    cache_dir: Path = Field(default_factory=lambda: Path.home() / '.morgan' / 'cache')
    max_workers: int = 4


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)
```

---

## Day 5: Basic CLI & Service Verification

**File**: `morgan/cli/app.py`

```python
import click
from rich.console import Console
from rich.table import Table
from rich import box
from morgan.config.settings import settings
import httpx

console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Morgan v2 - Emotional AI Assistant with Advanced RAG"""
    pass


@cli.command()
def health():
    """Check all services status"""

    table = Table(title="Morgan v2 Services Health", box=box.ROUNDED)
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # Check Ollama
    try:
        response = httpx.get(f"{settings.llm.base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            table.add_row(
                "Ollama (LLM)",
                "[green]✓ Healthy[/green]",
                f"Models: {', '.join(model_names[:3])}"
            )
        else:
            table.add_row("Ollama (LLM)", "[red]✗ Error[/red]", f"HTTP {response.status_code}")
    except Exception as e:
        table.add_row("Ollama (LLM)", "[red]✗ Down[/red]", str(e)[:50])

    # Check Qdrant
    try:
        response = httpx.get("http://localhost:6333/collections", timeout=5.0)
        if response.status_code == 200:
            collections = response.json().get('result', {}).get('collections', [])
            table.add_row(
                "Qdrant (Vector DB)",
                "[green]✓ Healthy[/green]",
                f"Collections: {len(collections)}"
            )
        else:
            table.add_row("Qdrant (Vector DB)", "[red]✗ Error[/red]", f"HTTP {response.status_code}")
    except Exception as e:
        table.add_row("Qdrant (Vector DB)", "[red]✗ Down[/red]", str(e)[:50])

    # Check Configuration
    table.add_row(
        "Configuration",
        "[green]✓ Loaded[/green]",
        f"Data: {settings.data_dir}"
    )

    # Feature flags
    features = []
    if settings.emotion.detection_enabled:
        features.append("Emotions")
    if settings.empathy.enabled:
        features.append("Empathy")
    if settings.learning.enabled:
        features.append("Learning")
    if settings.rag.enabled:
        features.append("RAG")

    table.add_row(
        "Features Enabled",
        "[green]✓ Active[/green]",
        ", ".join(features)
    )

    console.print(table)


@cli.command()
def info():
    """Show Morgan v2 system information"""

    console.print("\n[bold cyan]Morgan v2.0.0 - Emotional AI Assistant[/bold cyan]\n")

    console.print("[bold]Capabilities:[/bold]")
    console.print("  • 11 Emotion Detection Modules")
    console.print("  • 5 Empathy Processing Modules")
    console.print("  • 6 Learning & Adaptation Modules")
    console.print("  • Advanced RAG with Hierarchical Search")
    console.print("  • Multi-Layer Memory System")
    console.print("  • Companion Relationship Tracking")

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  LLM Model: {settings.llm.model}")
    console.print(f"  Embedding Model: {settings.embedding.model}")
    console.print(f"  Reranker: {settings.reranker.model}")
    console.print(f"  Data Directory: {settings.data_dir}")

    console.print("\n[bold]Status:[/bold]")
    console.print(f"  Emotions: {'[green]Enabled[/green]' if settings.emotion.detection_enabled else '[dim]Disabled[/dim]'}")
    console.print(f"  Empathy: {'[green]Enabled[/green]' if settings.empathy.enabled else '[dim]Disabled[/dim]'}")
    console.print(f"  Learning: {'[green]Enabled[/green]' if settings.learning.enabled else '[dim]Disabled[/dim]'}")
    console.print(f"  RAG: {'[green]Enabled[/green]' if settings.rag.enabled else '[dim]Disabled[/dim]'}")

    console.print()


if __name__ == '__main__':
    cli()
```

**Test it**:
```bash
cd morgan-rag
python -m morgan health
python -m morgan info
```

---

## Week 1 Success Criteria

By end of Week 1, you should have:

✅ **Services Running**
- [ ] Qdrant vector database
- [ ] Ollama with LLM model (qwen2.5:7b or 32b)
- [ ] Ollama with embedding model (qwen3-embedding)
- [ ] PostgreSQL (optional)
- [ ] Redis (optional)

✅ **Collections Created**
- [ ] knowledge_coarse
- [ ] knowledge_medium
- [ ] knowledge_fine
- [ ] memory_emotional
- [ ] memory_conversation

✅ **Configuration Working**
- [ ] All settings load from .env
- [ ] Feature flags work
- [ ] Directories created

✅ **CLI Functional**
- [ ] `morgan health` shows all services
- [ ] `morgan info` shows configuration
- [ ] No errors when running commands

✅ **Models Downloaded**
- [ ] LLM model pulled
- [ ] Embedding model pulled
- [ ] Reranker model will download on first use

---

## Next Week: Emotion Detection

With infrastructure ready, Week 2 begins implementing:
- Emotion analyzer
- Emotion classifier
- Emotion intensity detector
- Emotion context tracker
- And 7 more emotion modules...

See **V2_FULL_SYSTEM_PLAN.md** for complete roadmap.

---

**Status**: Week 1 Infrastructure Setup
**Goal**: Solid foundation for complete emotional AI system
**NOT**: A working chatbot yet - that comes after emotions, empathy, RAG are built
