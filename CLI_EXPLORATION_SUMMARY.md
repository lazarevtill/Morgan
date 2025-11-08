# Morgan v2-0.0.1 Branch - CLI Features & Design Summary

## Overview

The v2-0.0.1 branch implements a **human-first, CLI-only interface** for Morgan RAG with sophisticated distributed multi-host management capabilities. The design emphasizes simplicity, conversational interaction, and production-ready operations.

---

## 1. User-Facing CLI (`morgan-rag/morgan/cli/app.py`)

### Architecture
- **Framework**: Python `argparse` with Rich console output
- **Entry Point**: `morgan` command with subcommands
- **Philosophy**: KISS principle - simple commands that do exactly what humans expect
- **Output**: Rich formatted text with colors, progress bars, panels, and tables

### Core Commands

#### 1.1 Chat Command
```bash
morgan chat [--topic INITIAL_TOPIC]
```
- **Purpose**: Interactive multi-turn conversation
- **Features**:
  - Natural language input (multi-line with Ctrl+D or Enter)
  - Conversation history with emotional context
  - Source attribution for answers
  - Streaming response support for real-time feel
  - Exit commands: `quit`, `exit`, `bye`, `q`, Ctrl+C, Ctrl+D
  - Persistent conversation storage in `~/.morgan/conversations/{session_id}.json`

#### 1.2 Ask Command
```bash
morgan ask "<question>" [--sources] [--stream]
```
- **Purpose**: One-shot query for quick questions
- **Features**:
  - Stateless operation (no conversation history)
  - Optional source references via `--sources` flag
  - Optional streaming response via `--stream` flag
  - Related topic suggestions
  - Returns exit code 0 on success, 1 on error

#### 1.3 Learn Command
```bash
morgan learn <path|--url URL> [--type auto|pdf|web|code|markdown|text] [--progress]
```
- **Purpose**: Teach Morgan from documents or websites
- **Features**:
  - Local file/directory ingestion
  - Web URL learning
  - Auto-detection of document type (default)
  - Explicit type specification for edge cases
  - Progress bar during learning
  - Returns: documents_processed, chunks_created, processing_time, knowledge_areas
  - Success/failure status with detailed messages

#### 1.4 Serve Command
```bash
morgan serve [--host 0.0.0.0] [--port 8080] [--api-only]
```
- **Purpose**: Start web interface and/or REST API
- **Features**:
  - Web interface at `http://host:port`
  - REST API at `http://host:port/api`
  - API documentation at `http://host:port/docs`
  - Optional `--api-only` for headless operation
  - Proper shutdown handling via Ctrl+C

#### 1.5 Health Command
```bash
morgan health [--detailed]
```
- **Purpose**: Check system health of all components
- **Features**:
  - Overall status: healthy | degraded | unhealthy
  - Component status for: Knowledge Base, Memory, Search Engine, LLM Service, Vector DB
  - Detailed metrics when `--detailed` flag used
  - Color-coded output (green=healthy, yellow=warning, red=unhealthy)
  - Exit with code 1 if unhealthy

#### 1.6 Memory Command
```bash
morgan memory [--stats|--search QUERY|--cleanup DAYS]
```
- **Purpose**: Manage conversation memory
- **Features**:
  - `--stats`: Show conversation statistics (total_conversations, total_turns, avg_rating, feedback_percentage, common_topics)
  - `--search QUERY`: Search conversation history (returns 10 most relevant)
  - `--cleanup DAYS`: Remove conversations older than N days
  - Semantic search for finding relevant past conversations
  - Feedback integration for learning

#### 1.7 Knowledge Command
```bash
morgan knowledge [--stats|--search QUERY|--clear]
```
- **Purpose**: Manage knowledge base
- **Features**:
  - `--stats`: Show knowledge statistics (total_documents, chunks, knowledge_areas, storage_size, last_updated)
  - `--search QUERY`: Search knowledge base (returns 10 most relevant chunks)
  - `--clear`: Clear all knowledge (requires confirmation)
  - Source attribution in search results

#### 1.8 Cache Command
```bash
morgan cache [--stats|--metrics|--efficiency|--clear|--cleanup DAYS]
```
- **Purpose**: Monitor Git hash cache performance
- **Features**:
  - `--stats`: Basic cache statistics (hit_rate, total_requests, cache_hits, cache_misses)
  - `--metrics`: Detailed metrics with collection statistics
  - `--efficiency`: Efficiency report with recommendations
  - `--clear`: Clear cache metrics (requires confirmation)
  - `--cleanup DAYS`: Remove cache entries older than N days
  - Performance tracking: hit_rate, miss_rate, total_requests, hash_calculations

#### 1.9 Migrate Command
```bash
morgan migrate [analyze|plan|execute|validate|rollback|list-backups|cleanup]
```
- **Purpose**: Migrate knowledge bases to hierarchical format
- **Subcommands**:
  - `analyze [collection]`: Analyze readiness for migration
  - `plan <source>`: Create migration plan
  - `execute <source>`: Execute migration with optional `--dry-run` and `--confirm`
  - `validate <source> <target> [--sample-size]`: Validate completed migration
  - `rollback <backup_path>`: Restore from backup
  - `list-backups`: Show available backups
  - `cleanup [--keep-days]`: Remove old backups
- **Features**:
  - Migration planning with estimated time and batch size
  - Dry-run support before executing
  - Automatic backups before migration
  - Validation with sample-based consistency checking
  - Full rollback capability
  - Progress tracking

#### 1.10 Init Command
```bash
morgan init [--force]
```
- **Purpose**: Initialize Morgan in current directory
- **Features**:
  - Setup configuration files
  - Create data directories
  - Optional `--force` to overwrite existing config

### Global Options
```bash
morgan [--config CONFIG_FILE] [--verbose|-v] [--debug] <command>
```
- `--config`: Path to configuration file
- `--verbose/-v`: Enable verbose output
- `--debug`: Enable debug mode

### Output Formatting
- **Colors**: Red (error), Yellow (warning), Green (success), Blue (info), Cyan (secondary info)
- **Panels**: Rich panels for important information
- **Tables**: Rich tables for structured data (hosts, services, results)
- **Progress Bars**: Visual progress during long operations
- **Emotional Context**: Colored output based on detected emotional tone

---

## 2. Distributed Management CLI (`morgan-rag/morgan/cli/distributed_cli.py`)

### Architecture
- **Framework**: Python `click` library
- **Purpose**: Multi-host deployment and management
- **Design**: Enterprise-grade operations for distributed systems
- **Output**: Structured tables and JSON for automation

### Core Commands

#### 2.1 Deploy Command
```bash
python -m morgan.cli.distributed_cli deploy [--branch v2-0.0.1] [--force] [--parallel]
```
- **Purpose**: Deploy Morgan to all configured hosts
- **Features**:
  - Git branch selection (default: v2-0.0.1)
  - Optional `--force` to discard local changes
  - Optional `--parallel` for faster parallel deployment
  - Results summary with success/failure and duration per host
  - Host-by-host status reporting

#### 2.2 Update Command
```bash
python -m morgan.cli.distributed_cli update [--branch v2-0.0.1] [--rolling|--parallel]
```
- **Purpose**: Update all hosts with latest code
- **Features**:
  - Git branch selection
  - Rolling update (default): Zero-downtime sequential updates
  - Parallel update: Faster but with downtime
  - Update order for rolling: Background → Reranking → Embeddings → LLM#2 → LLM#1 → Manager
  - Per-host status and timing

#### 2.3 Health Command
```bash
python -m morgan.cli.distributed_cli health
```
- **Purpose**: Comprehensive health check across all hosts
- **Features**:
  - Overall health summary: total_hosts, healthy_hosts, unhealthy_hosts
  - Per-host detailed status:
    - Host name and role (CPU/GPU)
    - Error messages if unhealthy
    - Active services list
    - GPU info (model, utilization, memory)
  - Status icons: ✓ (healthy), ✗ (unhealthy)

#### 2.4 Restart Command
```bash
python -m morgan.cli.distributed_cli restart <service> [--hosts HOST1 HOST2 ...]
```
- **Purpose**: Restart specific service on selected hosts
- **Supported Services**: ollama, morgan, qdrant, redis, reranking, prometheus, grafana
- **Features**:
  - Service name mapping to internal service types
  - Optional host filtering (defaults to all hosts with service)
  - Per-host success/failure reporting

#### 2.5 Sync-Config Command
```bash
python -m morgan.cli.distributed_cli sync-config [--config-file .env] [--source HOSTNAME]
```
- **Purpose**: Synchronize configuration across all hosts
- **Features**:
  - File path specification (default: `.env`)
  - Optional source host (otherwise uses local file)
  - Per-host sync status
  - Broadcast configuration updates

#### 2.6 Status Command
```bash
python -m morgan.cli.distributed_cli status
```
- **Purpose**: Display current deployment configuration and status
- **Features**:
  - SSH key path
  - Total host count
  - Per-host details:
    - Hostname and role
    - GPU type (if applicable)
    - List of services

#### 2.7 Config Command
```bash
python -m morgan.cli.distributed_cli config
```
- **Purpose**: Show configuration as JSON
- **Features**:
  - Machine-readable JSON output
  - Full configuration dump
  - Suitable for automation/scripting

### Global Options
```bash
distributed_cli [--ssh-key PATH]
```
- `--ssh-key`: Path to SSH private key for host authentication

### Service Types Supported
- OLLAMA: LLM serving
- MORGAN_CORE: Main orchestration
- QDRANT: Vector database
- REDIS: Caching layer
- RERANKING_API: Result reranking
- PROMETHEUS: Metrics collection
- GRAFANA: Metrics visualization

---

## 3. Design Philosophy & Principles

### 3.1 Human-First Design
- **Natural Language**: Commands use human-friendly syntax, not technical jargon
- **Conversational**: Chat is the primary interaction mode
- **Intuitive**: Subcommands follow logical grouping
- **KISS Principle**: Simple commands that do exactly what expected

### 3.2 User Experience
- **Immediate Feedback**: Progress bars and real-time status updates
- **Colored Output**: Visual differentiation of information types and severity
- **Clear Error Messages**: User-friendly error descriptions, not stack traces
- **Help Text**: Comprehensive help for all commands via `--help`

### 3.3 Operational Excellence
- **Health Monitoring**: Easy visibility into system state
- **Graceful Degradation**: Works even if some components unavailable
- **Error Recovery**: Automatic retry on transient failures
- **Logging**: All operations logged to `~/.morgan/cli.log`

### 3.4 Distributed Architecture Support
- **Multi-Host Aware**: Commands work across distributed systems
- **Zero-Downtime Updates**: Rolling update strategy
- **Service Discovery**: Automatic service endpoint discovery
- **Flexible Deployment**: Support for 1-7+ hosts with dynamic allocation

---

## 4. Feature Highlights

### 4.1 Conversational Learning
```bash
# Interactive learning experience
morgan chat
# → Natural conversation with Morgan
# → Morgan learns from feedback
# → Emotional intelligence in responses

# Or quick learning from documents
morgan learn ./docs --progress
# → Visual progress bar
# → Automatic document type detection
# → Summary of learned content
```

### 4.2 Git Hash Cache Intelligence (R1.3, R9.1)
- **Incremental Updates**: Tracks file hashes to avoid re-processing
- **Performance Metrics**: Hit rate, miss rate, cache efficiency
- **Automatic Cleanup**: Remove stale cache entries
- **Optimization Reports**: Recommendations for cache performance

### 4.3 Knowledge Base Migration (R10.4, R10.5)
- **Format Evolution**: Migrate from legacy to hierarchical embeddings
- **Safe Operations**: Analyze → Plan → Execute → Validate → Rollback
- **Batch Processing**: Configurable batch sizes for large migrations
- **Backup Protection**: Automatic backups before migration

### 4.4 Distributed Deployment Management
- **Enterprise-Grade**: Production-ready multi-host operations
- **Zero-Downtime**: Rolling update strategy
- **Configuration Sync**: Keep all hosts consistent
- **Health Visibility**: Comprehensive monitoring and reporting

### 4.5 Service Integration
- **Consul DNS**: Service discovery via Consul DNS (`service.service.consul`)
- **HTTP Client**: Async HTTPx client with retry logic
- **Load Balancing**: Round-robin and random strategies
- **Circuit Breaker**: Automatic failover on service errors

---

## 5. Technical Implementation Details

### 5.1 Command Structure (app.py)
```python
# Main parser with subcommands
parser = argparse.ArgumentParser(
    prog="morgan",
    description="Morgan - Your Human-First AI Assistant"
)

# Subparsers for each command
subparsers = parser.add_subparsers(dest="command")
chat_parser = subparsers.add_parser("chat")
ask_parser = subparsers.add_parser("ask")
# ... etc for all commands
```

### 5.2 Distributed CLI Structure (distributed_cli.py)
```python
# Click-based CLI for distributed operations
@click.group()
@click.option('--ssh-key')
@click.pass_context
def cli(ctx, ssh_key):
    """Morgan Distributed Deployment Manager"""
    ctx.obj['manager'] = get_distributed_manager(ssh_key_path=ssh_key)

@cli.command()
@click.option('--branch')
@click.option('--parallel')
def deploy(ctx, branch, parallel):
    """Deploy to all hosts"""
```

### 5.3 Output Formatting
- **Rich Console**: Colored text, panels, tables, progress bars
- **Structured Logging**: All operations logged with context
- **Machine-Readable**: JSON output for `--json` flags
- **Human-Readable**: Formatted tables and panels for humans

### 5.4 Error Handling
- **User-Friendly Messages**: Translate technical errors to human language
- **Proper Exit Codes**: 0=success, 1=error, 2=service unavailable
- **Logging**: All errors logged to `~/.morgan/cli.log`
- **Retry Logic**: Automatic retry on transient failures

---

## 6. Integration with Morgan Services

### 6.1 Core Service Communication
- **Endpoint Discovery**: Uses Consul DNS or HTTP API
- **Async HTTP**: HTTPx async client for API calls
- **Connection Pooling**: Efficient connection reuse
- **Timeout Handling**: Configurable per-service (default 30s)

### 6.2 Service Endpoints Used
- `GET /health`: System health check
- `GET /status`: Detailed metrics
- `POST /api/text`: Text processing
- `POST /api/audio`: Audio processing
- `POST /api/ingest`: Document ingestion
- `GET /api/memory`: Conversation memory
- `GET /api/knowledge`: Knowledge base query

### 6.3 Distributed Manager Interface
```python
manager = get_distributed_manager(ssh_key_path=path)

# Deployment operations
await manager.deploy_all(git_branch="v2-0.0.1", parallel=True)
await manager.update_all(git_branch="v2-0.0.1", rolling=True)

# Health and status
status = await manager.health_check_all()
await manager.restart_service(service=ServiceType.OLLAMA, hosts=["host1"])

# Configuration
await manager.sync_config(config_file=".env", source_host="host1")
```

---

## 7. Configuration

### 7.1 User Configuration
- **Location**: `~/.morgan/config.yaml`
- **Environment Overrides**: `MORGAN_*` environment variables
- **Per-Command Options**: CLI flags override config file

### 7.2 Distributed Configuration
- **Consul KV Storage**: `morgan/config/` namespace
- **SSH Configuration**: Host list with credentials
- **Service Topology**: Host roles and service assignments

### 7.3 Environment Variables
```bash
MORGAN_CONFIG_DIR=/path/to/config
MORGAN_LOG_LEVEL=INFO
MORGAN_VERBOSE=true
MORGAN_DEBUG=false
```

---

## 8. Security & Privacy

### 8.1 Local-Only Operation
- No external APIs for core functionality
- All processing on local hardware
- SSH key-based authentication for distributed hosts
- TLS/mTLS support for inter-service communication (optional)

### 8.2 Data Protection
- Conversation history stored locally in `~/.morgan/conversations/`
- Knowledge base stored in Qdrant vector database
- Configuration stored in Consul or local files
- Encryption at rest supported (TBD)

### 8.3 Access Control
- SSH key authentication for distributed management
- Local user-only access to CLI
- Future: RBAC for multi-user scenarios

---

## 9. Performance Characteristics

### 9.1 CLI Response Times
- **Chat/Ask**: <500ms for response (backend may take 1-10s)
- **Learn**: ~100 docs/minute ingestion rate
- **Health**: <2s for complete health check across all hosts
- **Migrate**: Depends on dataset size (configurable batch size)

### 9.2 Scalability
- **Host Count**: Supports 1-7+ hosts (tested with 6)
- **Concurrent Users**: Single CLI instance, but supports background services
- **Data Size**: Tested with 100K+ documents
- **Response Latency**: +10-50ms additional latency for distributed

### 9.3 Resource Usage
- **CLI Memory**: <100MB for CLI process
- **Network**: <1Mbps average for typical operations
- **Disk**: Minimal (CLI only stores logs and conversation history)

---

## 10. Planned Features (Not Yet Implemented)

### 10.1 Advanced Queries
```bash
morgan query --advanced --reasoning "Explain the reasoning process"
```

### 10.2 Scheduled Tasks
```bash
morgan schedule --daily "Check email" --time "09:00"
```

### 10.3 Multi-User Support
```bash
morgan auth add-user username
morgan auth set-role username admin
```

### 10.4 Extended Analytics
```bash
morgan analytics --period 30d --format csv --output report.csv
```

### 10.5 Backup & Restore
```bash
morgan backup create --target s3://bucket/morgan-backup
morgan backup restore --from s3://bucket/morgan-backup
```

---

## 11. Files & Locations

### 11.1 CLI Source Files
- `morgan-rag/morgan/cli/app.py` - User-facing CLI (600+ lines)
- `morgan-rag/morgan/cli/distributed_cli.py` - Distributed management CLI (400+ lines)
- `morgan-rag/morgan/cli/main.py` - Entry point
- `morgan-rag/morgan/cli/console.py` - Rich output utilities
- `morgan-rag/morgan/cli/service_client.py` - HTTP client for services

### 11.2 Command Modules
- `morgan-rag/morgan/cli/commands/chat.py`
- `morgan-rag/morgan/cli/commands/ask.py`
- `morgan-rag/morgan/cli/commands/learn.py`
- `morgan-rag/morgan/cli/commands/serve.py`
- `morgan-rag/morgan/cli/commands/health.py`
- `morgan-rag/morgan/cli/commands/memory.py`
- `morgan-rag/morgan/cli/commands/knowledge.py`
- `morgan-rag/morgan/cli/commands/cache.py`
- `morgan-rag/morgan/cli/commands/migrate.py`

### 11.3 Data Locations
- **Conversations**: `~/.morgan/conversations/{session_id}.json`
- **Cache**: `~/.morgan/cache/`
- **Logs**: `~/.morgan/cli.log`
- **Config**: `~/.morgan/config.yaml`

### 11.4 Distributed Locations
- **Consul**: Port 8500 (HTTP), 8600 (DNS)
- **Service Discovery**: Consul KV at `morgan/config/`
- **Configuration**: Consul KV at `morgan/config/`
- **Host Capabilities**: Consul KV at `morgan/hosts/{hostname}/`

---

## 12. Key Technologies

### 12.1 CLI Framework
- **argparse**: Main CLI parser (app.py)
- **click**: Distributed management CLI (distributed_cli.py)
- **Rich**: Beautiful console output
- **Prompt Toolkit**: Interactive input for chat

### 12.2 Service Integration
- **httpx**: Async HTTP client for API calls
- **python-consul**: Consul client for service discovery
- **dnspython**: DNS resolution for Consul SRV records
- **tenacity**: Retry decorator for resilience

### 12.3 Data Management
- **Qdrant**: Vector database for semantic search
- **PostgreSQL**: Optional persistent storage
- **Redis**: Optional caching layer
- **MinIO**: S3-compatible object storage

### 12.4 LLM Integration
- **Ollama**: OpenAI-compatible LLM server
- **sentence-transformers**: Local embeddings
- **CrossEncoder**: Local reranking

---

## 13. Summary Table

| Aspect | User-Facing CLI | Distributed CLI |
|--------|-----------------|-----------------|
| **Framework** | argparse | click |
| **Entry Point** | `morgan <cmd>` | `python -m morgan.cli.distributed_cli <cmd>` |
| **Purpose** | User interaction | Infrastructure management |
| **Commands** | 10+ (chat, ask, learn, etc.) | 7 (deploy, update, health, etc.) |
| **Output** | Rich (colored, panels, tables) | Click (structured, JSON-ready) |
| **Scope** | Single-host user | Multi-host deployment |
| **Auth** | Local user | SSH keys |
| **Features** | Chat, learning, knowledge mgmt | Deployment, health, config sync |

---

## 14. Integration with Project (v2-0.0.1 Branch)

### 14.1 Architecture Alignment
- **Multi-Host**: Supports flexible 1-7+ host deployments
- **Service Discovery**: Integrates with Consul for service registration
- **Distributed Management**: Handles rolling updates and zero-downtime deployments
- **PostgreSQL Code-Only**: Python/SQLAlchemy integration (no stored procedures)
- **MinIO Storage**: Support for S3-compatible file storage

### 14.2 Implementation Timeline
- **Phase 1**: Foundation (Consul, CLI framework) - 2 weeks
- **Phase 2**: Data layer (Vector DB, PostgreSQL) - 3 weeks
- **Phase 3**: Service deployment (Multi-host orchestration) - 2 weeks
- **Phase 4**: Advanced features (Reasoning, proactive assistance) - 2.5 weeks
- **Phase 5**: Testing & production - 1.5 weeks
- **Total**: ~11 weeks (450 hours estimated)

### 14.3 Current Status (as of Nov 2, 2025)
- ✅ CLI structure designed
- ✅ User-facing commands defined
- ✅ Distributed management interface designed
- 🔄 Implementation in progress
- ⏳ Integration testing not yet started

---

## 15. Key Files to Review

1. **`morgan-rag/morgan/cli/app.py`** (600+ lines)
   - User-facing CLI with 10+ commands
   - Parser configuration and command handlers
   - Rich console output formatting

2. **`morgan-rag/morgan/cli/distributed_cli.py`** (400+ lines)
   - Distributed multi-host management
   - Click-based command groups
   - Enterprise deployment operations

3. **`morgan-rag/README.md`**
   - Human-first philosophy and KISS principles
   - Quick start guide with Docker setup
   - Real examples of chat, learning, and querying

4. **`.kiro/specs/morgan-multi-host-mvp/tasks.md`**
   - Comprehensive task breakdown (450 hours)
   - 5-phase implementation plan
   - Detailed acceptance criteria for each task

5. **`claude.md`**
   - Architecture overview (6-host distributed system)
   - Key design principles
   - Integration roadmap

---

## Conclusion

Morgan v2-0.0.1 branch implements a sophisticated **CLI-only interface** that prioritizes human-first design while supporting enterprise-grade distributed multi-host deployments. The design balances simplicity for end users with powerful management capabilities for operators, all without sacrificing functionality or usability.

The system is built to be:
- **Simple**: KISS principle for user-facing commands
- **Distributed**: Multi-host deployments with service discovery
- **Reliable**: Zero-downtime updates and health monitoring
- **Intelligent**: RAG-based learning and conversational interaction
- **Modular**: Easy to extend with new commands and features

