# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morgan is a self-hosted, distributed personal AI assistant with emotional intelligence and RAG capabilities. It's a Python monorepo with three main components that communicate via REST/WebSocket APIs.

**Key Principle**: Quality over speed (5-10s response time acceptable for thoughtful responses). Privacy first - all processing on local hardware.

## Build & Development Commands

### Installation (editable mode for development)
```bash
# Core RAG library
cd morgan-rag && pip install -e ".[server,dev]"

# FastAPI server
cd morgan-server && pip install -e ".[dev]"

# Terminal client
cd morgan-cli && pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests in a component
cd morgan-rag && pytest tests/
cd morgan-server && pytest tests/
cd morgan-cli && pytest tests/

# Run a single test file
pytest tests/test_emotional_intelligence.py

# Run a specific test
pytest tests/test_emotional_intelligence.py::test_emotion_detection -v

# With coverage
pytest --cov=morgan tests/
```

### Code Formatting & Linting
```bash
# Format (Black, line-length: 100 for server/cli, 88 for morgan-rag)
black morgan_server/ morgan_cli/
cd morgan-rag && make format

# Lint
ruff check .
ruff check --fix .

# Type checking
mypy morgan_server

# Pre-commit (runs all checks)
pre-commit run --all-files
```

### Running the Server
```bash
# Start server
cd morgan-server && python -m morgan_server --host 0.0.0.0 --port 8080

# Start with dependencies (Docker)
cd docker && docker-compose up -d
```

## Architecture

### Component Relationships
```
morgan-cli (Terminal UI) ──HTTP/WS──► morgan-server (FastAPI) ──► morgan-rag (Core Intelligence)
                                              │                           │
                                              └───► Qdrant (Vector DB)    ├── services/llm/
                                              └───► Redis (Cache)         ├── services/embeddings/
                                              └───► Ollama (LLM)          ├── services/reranking/
                                                                          ├── intelligence/ (emotions)
                                                                          ├── memory/
                                                                          └── search/
```

### Service Layer Pattern (morgan-rag)
Services are singletons accessed via factory functions:
```python
from morgan.services import get_llm_service, get_embedding_service, get_reranking_service

llm = get_llm_service()  # Singleton instance
response = await llm.agenerate("prompt")
```

Each service has multi-level fallback (e.g., reranking: remote → CrossEncoder → embedding similarity → BM25).

### Key Entry Points
- **morgan-server**: `morgan_server/__main__.py` → FastAPI app in `morgan_server/app.py`
- **morgan-rag**: `morgan/services/` for service layer, `morgan/core/` for assistant logic
- **API routes**: `morgan-server/morgan_server/api/routes/` (chat.py, memory.py, knowledge.py, profile.py)

### Data Flow
1. Client sends message to `/api/chat`
2. Server embeds query via `EmbeddingService` → Ollama
3. `SearchPipeline` finds context from Qdrant (vector search + reranking)
4. `MemoryProcessor` retrieves conversation history
5. `IntelligenceEngine` analyzes emotion
6. `LLMService` generates response with context
7. Response stored in Qdrant, returned to client

### Exception Hierarchy
All exceptions inherit from `MorganError` in `morgan/exceptions.py`:
- `LLMServiceError`, `EmbeddingServiceError`, `RerankingServiceError`
- `ConfigurationError`, `ValidationError`
- `InfrastructureError` → `ConnectionError`, `TimeoutError`

## Configuration

Environment variables prefixed with `MORGAN_`:
```bash
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b
MORGAN_QDRANT_URL=http://localhost:6333
MORGAN_REDIS_URL=redis://localhost:6379
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
```

Server config can also use YAML (`morgan-server/config.example.yaml`).

## External Services

- **Ollama** (localhost:11434): LLM serving with OpenAI-compatible API
- **Qdrant** (localhost:6333): Vector database for memories and search
- **Redis** (localhost:6379): Caching and session storage

## New Modules (morgan-rag/morgan/)

Twelve modules added to extend Morgan's core capabilities. All wired into the server lifespan, REST API, and CLI.

### Module Overview

| Module | Path | Purpose |
|--------|------|---------|
| **tools** | `morgan/tools/` | Pluggable tool execution framework (BaseTool, ToolExecutor, 5 built-in tools, permissions) |
| **workspace** | `morgan/workspace/` | SOUL.md/USER.md/MEMORY.md on-disk workspace with session-scoped memory gating |
| **compaction** | `morgan/compaction/` | Auto context-window compaction with token counting (tiktoken fallback to len//4) |
| **memory_consolidation** | `morgan/memory_consolidation/` | Daily log manager, hybrid BM25+vector search, LLM-driven MEMORY.md consolidation |
| **channels** | `morgan/channels/` | Multi-channel gateway with Telegram/Discord adapters and route resolver |
| **agents** | `morgan/agents/` | Agent/subagent spawning from code or markdown frontmatter definitions |
| **skills** | `morgan/skills/` | Skill/plugin system: markdown templates with YAML frontmatter and variable substitution |
| **scheduling** | `morgan/scheduling/` | CronService (APScheduler backend, graceful fallback) + HeartbeatManager (jittered intervals) |
| **task_manager** | `morgan/task_manager/` | Background task lifecycle (create, track, update, delete) |
| **hook_system** | `morgan/hook_system/` | Event-driven hook system with sync/async handlers and short-circuit abort |
| **app_state** | `morgan/app_state/` | Thread-safe observable state store with subscriber notifications |
| **security** | `morgan/security/` | MemoryGate, ChannelAllowlist, SessionPermissionMode |

### Feature Flags (environment variables)

```bash
MORGAN_ENABLE_TOOLS=true          # Tool execution framework
MORGAN_ENABLE_WORKSPACE=true      # SOUL.md workspace
MORGAN_ENABLE_COMPACTION=true     # Auto context compaction
MORGAN_ENABLE_CHANNELS=false      # Multi-channel (needs token config)
MORGAN_ENABLE_SCHEDULING=false    # Cron + heartbeat
MORGAN_ENABLE_AGENTS=true         # Agent spawning
MORGAN_WORKSPACE_PATH=            # Override workspace directory
MORGAN_TELEGRAM_TOKEN=            # Telegram bot token
MORGAN_DISCORD_TOKEN=             # Discord bot token
```

### New API Endpoints

```
GET  /api/tools                  - List tools with schemas
POST /api/tools/{name}           - Execute a tool
GET  /api/skills                 - List skills
POST /api/skills/{name}          - Run a skill
GET  /api/tasks                  - List background tasks
GET  /api/tasks/{id}             - Get task status
GET  /api/channels               - List active channels
GET  /api/workspace              - Get workspace status
POST /api/workspace/consolidate  - Trigger memory consolidation
```

### New CLI Commands

```bash
morgan tools                     # List available tools
morgan skills                    # List available skills
morgan tasks                     # List background tasks
morgan workspace                 # Show workspace status
morgan channels                  # List channels
morgan schedule list             # List cron jobs
morgan schedule add "*/5 * * * *" "check emails"  # Add cron job
```

### Optional Dependencies

```bash
pip install morgan-rag[channels]    # python-telegram-bot, discord.py
pip install morgan-rag[scheduling]  # apscheduler
pip install morgan-rag[tokens]      # tiktoken
```

### How Modules Connect

1. **Server lifespan** (`morgan-server/app.py`): Initializes workspace, AppStateStore, HookManager, TaskManager, CronService, HeartbeatManager, and ChannelGateway based on feature flags.
2. **API routes** (`morgan-server/api/routes/tools_api.py`): Exposes all modules via REST.
3. **CLI** (`morgan-cli/cli.py`): New Click commands for tools, skills, tasks, workspace, channels, schedule.
4. **Settings** (`morgan-rag/config/settings.py`): Feature flags control which modules are initialized.

### Running New Module Tests

```bash
cd morgan-rag && python -m pytest tests/test_tools.py tests/test_tool_executor.py tests/test_workspace.py tests/test_compaction.py tests/test_memory_consolidation.py tests/test_channels.py tests/test_agents.py tests/test_skills.py tests/test_scheduling.py tests/test_task_manager.py tests/test_hooks.py tests/test_app_state.py tests/test_security.py -v
```

## Code Style Notes

- **Line length**: 100 for morgan-server/morgan-cli, 88 for morgan-rag
- **Async**: Prefer async methods (`agenerate`, `aencode`, `arerank`)
- **Pre-commit excludes morgan-rag** from some hooks (has its own formatting rules via Makefile)
- **Python 3.11+** required
