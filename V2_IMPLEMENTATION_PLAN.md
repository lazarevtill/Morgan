# Morgan v2-0.0.1 Implementation Plan
## Best CLI/Bash Experience Roadmap

> **Created**: November 8, 2025
> **Target Branch**: v2-0.0.1
> **Current Branch**: main (v0.2.0)
> **Status**: Planning Phase

---

## Executive Summary

This document provides a **prioritized implementation roadmap** for migrating Morgan from v0.2.0 (Docker Compose microservices) to v2-0.0.1 (Cloud-native multi-host RAG system with dual CLI architecture).

The plan prioritizes **CLI/bash user experience** as the primary interface, focusing on rapid iteration and testing while building toward the full distributed architecture.

---

## Gap Analysis

### Current State (main branch - v0.2.0)
- ✅ Docker Compose microservices (Core, LLM, TTS, STT, VAD)
- ✅ FastAPI-based services
- ✅ Basic health checks
- ❌ **NO CLI interface** (only HTTP API)
- ❌ No RAG system
- ❌ No multi-host support
- ❌ No service discovery (Consul)
- ❌ No emotional intelligence
- ❌ No learning system

### Target State (v2-0.0.1 branch)
- ✅ **Dual CLI architecture** (user CLI + distributed CLI)
- ✅ Advanced RAG with hierarchical search
- ✅ Multi-host distributed deployment (MicroK8s/Consul)
- ✅ Emotion engine (11 modules)
- ✅ Empathy engine (5 modules)
- ✅ Learning system (6 modules)
- ✅ Service discovery (Consul)
- ✅ Advanced monitoring and health
- ✅ 100+ source files, comprehensive architecture

### Migration Scope
- **Lines of code**: ~35,000 deleted, ~113,000 added
- **Files changed**: 410 files
- **New directories**: morgan-rag/, .kiro/specs/, extensive documentation
- **Architecture**: Complete paradigm shift (monolith → microservices → cloud-native)

---

## Implementation Philosophy

### Core Principles
1. **CLI-First Development**: Build user-facing CLI early for rapid feedback
2. **Incremental Migration**: Don't break existing functionality
3. **Testing at Each Step**: Validate before moving forward
4. **Documentation Driven**: Update docs as you implement
5. **Local-First**: Test locally before distributed deployment

### Success Metrics
- ✅ Users can chat via CLI within **Phase 1**
- ✅ Basic RAG working via CLI within **Phase 2**
- ✅ Multi-host deployment working within **Phase 3**
- ✅ Full emotional intelligence within **Phase 4**
- ✅ Production-ready within **Phase 5**

---

## Implementation Phases

## **PHASE 1: Foundation & Basic CLI (Week 1-2)**
### Goal: Get users chatting via CLI as fast as possible

### Priority: CRITICAL 🔴
This phase enables basic user interaction and validates the CLI design.

### Tasks

#### 1.1 Project Structure Setup
- [ ] Create `morgan-rag/` directory structure
- [ ] Set up `morgan/cli/` directory
- [ ] Create `morgan/core/` for basic assistant logic
- [ ] Set up `requirements.txt` with Click, Rich, httpx
- [ ] Create `morgan/__main__.py` for CLI entry point
- [ ] Add `.env.example` with basic configuration

**Deliverable**: Clean directory structure, installable package

#### 1.2 Basic User-Facing CLI (morgan command)
**File**: `morgan-rag/morgan/cli/app.py`

- [ ] Implement Click-based argument parser
- [ ] Create `morgan chat` command (basic)
- [ ] Create `morgan ask <question>` command
- [ ] Create `morgan health` command
- [ ] Create `morgan init` command
- [ ] Add Rich console for colored output
- [ ] Add progress bars using Rich
- [ ] Add basic error handling

**Deliverable**: Working CLI with 4 basic commands

#### 1.3 Simple Backend Service
**File**: `morgan-rag/morgan/core/assistant.py`

- [ ] Create `MorganAssistant` class with basic chat
- [ ] Implement simple in-memory conversation history
- [ ] Add Ollama integration (or OpenAI API fallback)
- [ ] Create basic response handler
- [ ] Add logging setup

**Deliverable**: CLI can chat with LLM

#### 1.4 Configuration System
**File**: `morgan-rag/morgan/config/settings.py`

- [ ] Create `Settings` class using Pydantic
- [ ] Support `.env` file loading
- [ ] Add configuration validation
- [ ] Create default config templates
- [ ] Add `morgan config` command to view/edit settings

**Deliverable**: Flexible configuration system

#### 1.5 Testing & Validation
- [ ] Create `tests/test_cli_basic.py`
- [ ] Test `morgan chat` interaction
- [ ] Test `morgan ask` one-shot query
- [ ] Test error handling (no LLM, network errors)
- [ ] Create basic integration test

**Deliverable**: Validated CLI that users can test

### Phase 1 Success Criteria
✅ User can run `morgan chat` and have conversation
✅ User can run `morgan ask "question"` and get answer
✅ Health check shows service status
✅ Configuration works via .env file
✅ Tests pass

**Estimated Time**: 1-2 weeks
**Key Files**: 5-10 files (~1,000 lines)

---

## **PHASE 2: RAG System Integration (Week 3-5)**
### Goal: Enable document ingestion and intelligent retrieval

### Priority: HIGH 🟠
This phase adds the core value proposition: RAG-based question answering.

### Tasks

#### 2.1 Document Ingestion CLI
**File**: `morgan-rag/morgan/cli/app.py` (extend)

- [ ] Add `morgan learn <path>` command
- [ ] Support file types: PDF, MD, TXT, DOCX
- [ ] Add progress bar for document processing
- [ ] Show summary after ingestion
- [ ] Add `morgan knowledge --stats` command

**Deliverable**: CLI can ingest documents

#### 2.2 Vector Database Setup
**File**: `morgan-rag/morgan/storage/vector.py`

- [ ] Set up Qdrant (Docker or local)
- [ ] Create collection manager
- [ ] Implement vector storage operations
- [ ] Add connection pooling
- [ ] Create health check for Qdrant

**Deliverable**: Vector database ready

#### 2.3 Document Processing Pipeline
**File**: `morgan-rag/morgan/ingestion/enhanced_processor.py`

- [ ] Implement semantic chunking
- [ ] Add metadata extraction
- [ ] Create document parsers (PDF, MD, TXT, DOCX)
- [ ] Add error handling for corrupted files
- [ ] Implement batch processing

**Deliverable**: Documents → chunks with metadata

#### 2.4 Embedding Generation
**File**: `morgan-rag/morgan/services/embedding_service.py`

- [ ] Integrate Ollama embedding API (qwen3-embedding)
- [ ] Add batching for efficiency
- [ ] Implement caching for duplicate chunks
- [ ] Add retry logic for transient failures
- [ ] Create embedding performance metrics

**Deliverable**: Text → embeddings

#### 2.5 Search & Retrieval
**File**: `morgan-rag/morgan/core/search.py`

- [ ] Implement basic vector search
- [ ] Add reciprocal rank fusion (RRF)
- [ ] Create result deduplication
- [ ] Add source attribution
- [ ] Implement search result caching

**Deliverable**: Query → relevant documents

#### 2.6 RAG-Augmented Responses
**File**: `morgan-rag/morgan/core/assistant.py` (extend)

- [ ] Integrate search results into LLM prompts
- [ ] Add context window management
- [ ] Implement source citation in responses
- [ ] Add "no relevant results" handling
- [ ] Create RAG performance logging

**Deliverable**: Answers based on user's documents

#### 2.7 Knowledge Management Commands
- [ ] `morgan knowledge --search "query"` - Search knowledge base
- [ ] `morgan knowledge --stats` - Show KB statistics
- [ ] `morgan knowledge --clear` - Clear knowledge base
- [ ] `morgan knowledge --export <path>` - Export KB

**Deliverable**: Full knowledge base management

#### 2.8 Testing & Validation
- [ ] Create `tests/test_rag_pipeline.py`
- [ ] Test document ingestion
- [ ] Test embedding generation
- [ ] Test vector search accuracy
- [ ] Test RAG response quality
- [ ] Create integration test with sample docs

**Deliverable**: Validated RAG system

### Phase 2 Success Criteria
✅ User can run `morgan learn ./docs` to ingest documents
✅ User can ask questions and get answers from their docs
✅ Source attribution shows which documents were used
✅ Search performance is <500ms
✅ Tests validate accuracy

**Estimated Time**: 2-3 weeks
**Key Files**: 15-20 files (~3,000 lines)

---

## **PHASE 3: Multi-Host Foundation (Week 6-8)**
### Goal: Enable distributed deployment across multiple hosts

### Priority: MEDIUM 🟡
This phase enables scalability but is not critical for MVP.

### Tasks

#### 3.1 Service Discovery Setup
**File**: `morgan-rag/morgan/infrastructure/distributed_manager.py`

- [ ] Set up Consul (Docker or native)
- [ ] Implement service registration
- [ ] Create health check integration
- [ ] Add DNS-based service discovery
- [ ] Create fallback for local-only mode

**Deliverable**: Consul service registry

#### 3.2 Distributed CLI Commands
**File**: `morgan-rag/morgan/cli/distributed_cli.py`

- [ ] Create `morgan-admin` entry point (or `morgan admin`)
- [ ] Implement `deploy` command
- [ ] Implement `health` command (all hosts)
- [ ] Implement `status` command
- [ ] Implement `restart` command
- [ ] Add host configuration management

**Deliverable**: Multi-host management CLI

#### 3.3 Remote Service Communication
**File**: `morgan-rag/morgan/infrastructure/distributed_llm.py`

- [ ] Implement service client with Consul discovery
- [ ] Add load balancing across hosts
- [ ] Create connection pooling
- [ ] Add retry logic with exponential backoff
- [ ] Implement request routing

**Deliverable**: Services can call each other across hosts

#### 3.4 Host Detection & Auto-Configuration
- [ ] Detect GPU capabilities (CUDA, VRAM)
- [ ] Detect CPU capabilities (cores, RAM)
- [ ] Auto-assign services based on hardware
- [ ] Create host inventory system
- [ ] Add dynamic service placement

**Deliverable**: Automatic host detection

#### 3.5 Deployment Automation
**File**: `scripts/deploy_multi_host.sh`

- [ ] Create deployment script
- [ ] Add SSH-based remote execution
- [ ] Implement rolling updates
- [ ] Add health checks during deployment
- [ ] Create rollback mechanism

**Deliverable**: Automated deployment

#### 3.6 Testing & Validation
- [ ] Create `tests/test_distributed.py`
- [ ] Test service discovery
- [ ] Test multi-host communication
- [ ] Test deployment automation
- [ ] Test failover scenarios

**Deliverable**: Validated distributed system

### Phase 3 Success Criteria
✅ System can deploy across 2+ hosts
✅ Services discover each other via Consul
✅ Load balancing works for LLM/embedding services
✅ `morgan admin health` shows all hosts
✅ Rolling updates work without downtime

**Estimated Time**: 2-3 weeks
**Key Files**: 10-15 files (~2,500 lines)

---

## **PHASE 4: Emotional Intelligence (Week 9-11)**
### Goal: Add emotion detection and empathetic responses

### Priority: MEDIUM 🟡
This phase adds differentiation but is not critical for basic functionality.

### Tasks

#### 4.1 Emotion Detection System
**Files**: `morgan-rag/morgan/emotions/` (11 modules)

- [ ] Implement basic emotion classifier
- [ ] Add emotion intensity detection
- [ ] Create emotion context tracking
- [ ] Add emotion pattern recognition
- [ ] Implement emotional memory

**Deliverable**: Detects user emotions

#### 4.2 Empathy Engine
**Files**: `morgan-rag/morgan/empathy/` (5 modules)

- [ ] Create empathetic response generator
- [ ] Add tone adjustment
- [ ] Implement emotional validation
- [ ] Create support response templates
- [ ] Add empathy scoring

**Deliverable**: Emotionally appropriate responses

#### 4.3 CLI Integration
- [ ] Add emotional tone to `morgan chat` output
- [ ] Use colored output for different emotions
- [ ] Add `--show-emotions` flag
- [ ] Create emotion analytics command
- [ ] Add emotion history tracking

**Deliverable**: Emotions visible in CLI

#### 4.4 Testing & Validation
- [ ] Create `tests/test_emotions.py`
- [ ] Test emotion detection accuracy
- [ ] Test empathy response quality
- [ ] Validate emotional context tracking
- [ ] Create user feedback system

**Deliverable**: Validated emotion system

### Phase 4 Success Criteria
✅ System detects user emotions accurately
✅ Responses adapt based on emotional state
✅ CLI shows emotional context (optional)
✅ Emotional memory persists across sessions
✅ Tests validate accuracy >80%

**Estimated Time**: 2-3 weeks
**Key Files**: 20-25 files (~5,000 lines)

---

## **PHASE 5: Learning & Adaptation (Week 12-14)**
### Goal: Enable continuous learning from user feedback

### Priority: LOW 🟢
This phase adds long-term value but is not needed for MVP.

### Tasks

#### 5.1 Feedback Collection
**File**: `morgan-rag/morgan/learning/feedback.py`

- [ ] Add thumbs up/down after responses
- [ ] Collect implicit feedback (follow-up questions)
- [ ] Track conversation success metrics
- [ ] Create feedback storage
- [ ] Add feedback analysis

**Deliverable**: Collects user feedback

#### 5.2 Learning Engine
**Files**: `morgan-rag/morgan/learning/` (6 modules)

- [ ] Implement preference learning
- [ ] Add pattern recognition
- [ ] Create adaptation strategies
- [ ] Add learning rate control
- [ ] Implement learning metrics

**Deliverable**: Learns from feedback

#### 5.3 Preference Management
- [ ] Add `morgan preferences` command
- [ ] Show learned preferences
- [ ] Allow preference overrides
- [ ] Create preference export/import
- [ ] Add preference reset

**Deliverable**: User-controllable learning

#### 5.4 Testing & Validation
- [ ] Create `tests/test_learning.py`
- [ ] Test feedback collection
- [ ] Test adaptation over time
- [ ] Validate preference learning
- [ ] Create long-term learning tests

**Deliverable**: Validated learning system

### Phase 5 Success Criteria
✅ System learns from user feedback
✅ Preferences improve over time
✅ Users can view/edit learned preferences
✅ Learning doesn't degrade performance
✅ Tests validate improvement

**Estimated Time**: 2-3 weeks
**Key Files**: 15-20 files (~4,000 lines)

---

## **PHASE 6: Production Hardening (Week 15-16)**
### Goal: Make system production-ready

### Priority: CRITICAL for deployment 🔴

### Tasks

#### 6.1 Error Handling & Recovery
- [ ] Add comprehensive error handling
- [ ] Implement automatic retry logic
- [ ] Create graceful degradation
- [ ] Add circuit breakers
- [ ] Implement timeout management

#### 6.2 Monitoring & Observability
- [ ] Add Prometheus metrics
- [ ] Create health dashboards
- [ ] Implement structured logging
- [ ] Add request tracing
- [ ] Create alerting system

#### 6.3 Performance Optimization
- [ ] Profile bottlenecks
- [ ] Optimize database queries
- [ ] Add caching layers
- [ ] Implement connection pooling
- [ ] Optimize batch processing

#### 6.4 Security Hardening
- [ ] Add authentication (optional for MVP)
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Secure API endpoints
- [ ] Add audit logging

#### 6.5 Documentation & Guides
- [ ] Create user documentation
- [ ] Write deployment guide
- [ ] Add troubleshooting guide
- [ ] Create API documentation
- [ ] Write contribution guide

#### 6.6 Testing & Validation
- [ ] Create comprehensive test suite
- [ ] Add integration tests
- [ ] Create load tests
- [ ] Add chaos testing
- [ ] Validate deployment scenarios

### Phase 6 Success Criteria
✅ System handles errors gracefully
✅ Monitoring shows key metrics
✅ Performance meets targets (<2s responses)
✅ Security best practices implemented
✅ Documentation complete
✅ Tests cover >80% of code

**Estimated Time**: 1-2 weeks
**Key Files**: Documentation + test files

---

## CLI/Bash Best Practices for v2

### 1. User Experience Principles

#### Immediate Feedback
```bash
# Good: Show progress immediately
$ morgan learn ./docs
📚 Scanning directory... Found 47 documents
⠋ Processing... ━━━━━━━━━━━━━━━━━━━━━━ 100% [23.4s]
✅ Complete! 312 chunks created

# Bad: Silent processing
$ morgan learn ./docs
[waits 23 seconds]
Done.
```

#### Rich Visual Output
- Use **Rich** library for tables, progress bars, panels
- Color-code status: ✅ green, ❌ red, ⚠️  yellow
- Show hierarchical information with tree views
- Use emoji sparingly for visual cues

#### Error Messages
```bash
# Good: Actionable error
❌ Error: Cannot connect to Ollama service
   → Check if Ollama is running: ollama serve
   → Verify URL in config: http://localhost:11434
   → Run: morgan health --detailed

# Bad: Cryptic error
Error: Connection refused
```

### 2. Command Design Patterns

#### Command Structure
```
morgan <command> [options] [arguments]

Categories:
- Interaction: chat, ask
- Knowledge: learn, knowledge
- System: health, init, config
- Admin: deploy, restart, status (distributed CLI)
```

#### Flags & Options
```bash
# Use intuitive flags
--verbose / -v         # More output
--quiet / -q           # Less output
--output / -o <file>   # Output to file
--format json|table    # Output format
--help / -h            # Help text

# Boolean flags
--progress             # Show progress (default: true)
--no-progress          # Hide progress
```

### 3. Performance Optimization

#### Startup Time
- Lazy load heavy dependencies
- Cache configuration
- Use async I/O for network calls
- Minimize import overhead

#### Response Time
```
Target Response Times:
- CLI startup: <500ms
- Health check: <2s
- Chat response: <2s (streaming)
- Document ingest: Real-time progress
```

### 4. Configuration Management

#### Priority Order
1. Command-line flags (highest)
2. Environment variables
3. User config file (`~/.morgan/config.yaml`)
4. System config (`/etc/morgan/config.yaml`)
5. Defaults (lowest)

#### Environment Variables
```bash
# Standard naming: MORGAN_<SECTION>_<KEY>
export MORGAN_LLM_URL=http://localhost:11434
export MORGAN_LLM_MODEL=qwen2.5:32b
export MORGAN_LOG_LEVEL=DEBUG
```

### 5. Testing Strategy

#### Unit Tests
```python
# Test CLI parsing
def test_chat_command():
    result = runner.invoke(cli, ['chat', '--help'])
    assert result.exit_code == 0
    assert 'Start interactive chat' in result.output
```

#### Integration Tests
```python
# Test end-to-end flow
@pytest.mark.integration
async def test_chat_flow():
    # Start chat
    # Send message
    # Verify response
    # Exit gracefully
```

#### Manual Testing Checklist
- [ ] First-time user experience (no config)
- [ ] Error handling (service down, invalid input)
- [ ] Performance (large document ingestion)
- [ ] Multi-platform (Linux, macOS, Windows)

### 6. Documentation Standards

#### Command Help Text
```bash
$ morgan learn --help

Usage: morgan learn [OPTIONS] PATH

  Ingest documents for RAG-based question answering.

  Supports: PDF, MD, TXT, DOCX, HTML, and code files.

Options:
  --recursive / --no-recursive  Scan subdirectories [default: recursive]
  --progress / --no-progress    Show progress bar [default: progress]
  --batch-size INTEGER          Embedding batch size [default: 32]
  -h, --help                    Show this message and exit

Examples:
  morgan learn ./company-docs
  morgan learn ~/Downloads/report.pdf --no-recursive
```

#### README Quick Start
```bash
# Installation
pip install morgan-rag

# Initialize
morgan init

# Start chatting
morgan chat

# Learn from documents
morgan learn ./docs

# Ask questions
morgan ask "How do I deploy to production?"
```

---

## Critical Path for Best CLI Experience

### Week 1: Minimal Viable CLI
```bash
# User can chat
morgan chat
> Hello!
> What is Python?
> /exit

# User can ask questions
morgan ask "What is Docker?"
```

### Week 2: Add Configuration
```bash
# User can configure
morgan init                    # Creates ~/.morgan/config.yaml
morgan config set llm.model qwen2.5:7b
morgan config show
```

### Week 3: Add Document Learning
```bash
# User can ingest docs
morgan learn ./docs
morgan knowledge --stats
morgan ask "What's in my docs?"
```

### Week 4: Polish & Refine
```bash
# Add rich output
morgan health --detailed       # Shows table with all services
morgan chat --show-emotions    # Shows emotional context
morgan learn ./docs --progress # Shows real-time progress
```

### Week 5-6: Multi-Host Admin CLI
```bash
# Admin can manage cluster
morgan-admin deploy
morgan-admin health
morgan-admin restart --host gpu-1
```

---

## Risk Mitigation

### High-Risk Items
1. **Consul Service Discovery Complexity**
   - Mitigation: Start with simple HTTP discovery, add Consul later
   - Fallback: Local-only mode without service discovery

2. **Multi-Host Testing Requirements**
   - Mitigation: Use Docker Compose to simulate multi-host locally
   - Fallback: Thorough single-host testing first

3. **Ollama Integration Issues**
   - Mitigation: Support OpenAI API as fallback
   - Fallback: Cached responses for testing

4. **Vector Database Performance**
   - Mitigation: Start with small document sets
   - Fallback: In-memory vector store for development

### Medium-Risk Items
1. **Emotion Detection Accuracy**
   - Mitigation: Use pre-trained models, extensive testing
   - Fallback: Make emotion detection optional

2. **Learning System Complexity**
   - Mitigation: Start with simple preference tracking
   - Fallback: Make learning opt-in

---

## Success Metrics

### Phase 1 Metrics
- [ ] CLI starts in <500ms
- [ ] Chat responses in <2s
- [ ] Zero crashes in 100 interactions
- [ ] Help text covers all commands

### Phase 2 Metrics
- [ ] Document ingestion: 100+ docs/min
- [ ] Search latency: <500ms
- [ ] RAG accuracy: >80% relevant results
- [ ] Source attribution: 100% of responses

### Phase 3 Metrics
- [ ] Multi-host deployment: <5min
- [ ] Service discovery: <2s
- [ ] Zero-downtime updates: 100%
- [ ] Load balancing: Even distribution

### Phase 4-6 Metrics
- [ ] Emotion detection: >80% accuracy
- [ ] Learning improvement: 10%+ over time
- [ ] System uptime: >99%
- [ ] Test coverage: >80%

---

## Recommended First Steps (This Week)

### Monday-Tuesday: Setup
1. Create `morgan-rag/` directory structure
2. Set up virtual environment
3. Install Click, Rich, httpx, pydantic
4. Create basic `morgan/__main__.py`
5. Test CLI entry point works

### Wednesday-Thursday: Basic CLI
1. Implement `morgan chat` (simple loop)
2. Add Rich console for colored output
3. Create basic LLM integration (Ollama or OpenAI)
4. Test interactive chat

### Friday: Validation & Demo
1. Create README with examples
2. Test with fresh environment
3. Demo to stakeholders
4. Gather feedback
5. Plan next week

---

## Next Phase Planning

After completing Phase 1, reassess priorities based on:
- User feedback on CLI experience
- Performance bottlenecks discovered
- Deployment environment constraints
- Team capacity and skills

---

## Conclusion

This implementation plan prioritizes **CLI user experience** as the primary interface, enabling rapid iteration and user testing while building toward the full distributed architecture described in v2-0.0.1.

**Key Success Factor**: Deliver working CLI in Week 1, then iterate based on real usage.

**Remember**: "Perfect is the enemy of good." Ship early, gather feedback, improve continuously.

---

**Document Status**: Ready for Implementation
**Next Action**: Begin Phase 1.1 (Project Structure Setup)
**Owner**: Development Team
**Review Date**: After Phase 1 completion
