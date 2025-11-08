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
- ❌ **NO CLI** (only HTTP API)
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

> **Note**: This implementation plan uses the 8-phase breakdown from V2_FULL_SYSTEM_PLAN.md for consistency.
> The phases are ordered to prioritize CLI functionality while building toward the complete emotional AI system.

## **PHASE 1: Core Infrastructure (Week 1-2)**
### Goal: Get the foundation right - all services ready

### Priority: CRITICAL 🔴
This phase sets up all infrastructure needed for the full emotional AI system.

### Tasks

#### 1.1 Project Structure & Dependencies
**Directory Structure**:
```bash
morgan-rag/
├── morgan/
│   ├── cli/              # CLI interface
│   ├── core/             # Assistant, conversation, integration
│   ├── emotions/         # 11 emotion modules
│   ├── empathy/          # 5 empathy modules
│   ├── learning/         # 6 learning modules
│   ├── memory/           # Memory processing
│   ├── search/           # Multi-stage search + companion memory
│   ├── storage/          # Vector DB, profile, memory storage
│   ├── services/         # Embedding, LLM services
│   ├── config/           # Configuration
│   └── utils/            # Error handling, logging, helpers
└── tests/                # Comprehensive tests
```

- [ ] Create complete directory structure
- [ ] Set up requirements.txt with all dependencies
- [ ] Create .env.example with all configuration options
- [ ] Set up virtual environment
- [ ] Install dependencies

**Deliverable**: Clean project structure, all dependencies installed

#### 1.2 Configuration System
**File**: `morgan/config/settings.py`

Must support:
- LLM settings (Ollama URL, model)
- Embedding settings (model, batch size)
- Reranking settings (Jina model)
- Qdrant connection
- PostgreSQL connection (optional)
- Redis connection (optional)
- Emotion detection settings
- Learning rate settings
- Memory retention settings

**Deliverable**: Complete configuration system with validation

#### 1.3 Storage Layer Setup
- [ ] Set up Qdrant (Docker or cloud)
- [ ] Create collections for different embedding granularities (coarse, medium, fine)
- [ ] Set up PostgreSQL for structured data (optional, fallback to JSON)
- [ ] Create local storage for conversations/preferences
- [ ] Verify all storage connections

**Deliverable**: All storage services running and tested

#### 1.4 Services Integration
- [ ] Ollama integration for LLM (qwen2.5:32b or 7b)
- [ ] Ollama integration for embeddings (qwen3-embedding:latest)
- [ ] Jina reranker setup (transformers + torch)
- [ ] Service health checks
- [ ] Connection pooling and retry logic

**Deliverable**: All AI services ready and accessible

#### 1.5 Basic CLI Framework
**File**: `morgan/cli/app.py`

Commands:
- `morgan chat` - Interactive chat (basic for now)
- `morgan ask` - One-shot query
- `morgan learn` - Document ingestion (prepared)
- `morgan health` - All services status
- `morgan info` - System information

**Deliverable**: Infrastructure ready, CLI skeleton works, health checks pass

**Estimated Time**: 1-2 weeks
**Key Files**: 10-15 files (~2,000 lines)

---

## **PHASE 2: Emotion Detection System (Week 3-4)**
### Goal: Implement all 11 emotion modules

### Priority: HIGH 🟠
This phase implements the core differentiation: emotional intelligence.

### Tasks

#### 2.1 Core Emotion Modules
**Files**: `morgan/emotions/`

All 11 modules from v2-0.0.1:
1. **analyzer.py** - Core emotion analysis
2. **classifier.py** - Emotion categorization (joy, sadness, anger, fear, etc.)
3. **intensity.py** - Emotion strength measurement
4. **context.py** - Contextual emotion understanding
5. **patterns.py** - Emotion pattern recognition
6. **memory.py** - Emotional event storage
7. **triggers.py** - Trigger identification
8. **recovery.py** - Emotional recovery tracking
9. **tracker.py** - Long-term emotion tracking
10. **detector.py** - Real-time emotion detection

**Deliverable**: All 11 emotion modules implemented

#### 2.2 Emotion Models
**File**: `morgan/emotional/models.py`

Data structures for:
- EmotionalState
- EmotionVector
- EmotionalContext
- EmotionalMemory
- EmotionalPattern

**Deliverable**: Complete emotion data models

#### 2.3 Intelligence Engine
**File**: `morgan/emotional/intelligence_engine.py`

Orchestrates all emotion modules:
- Detects emotions in user input
- Tracks emotional context
- Stores emotional memories
- Identifies patterns and triggers
- Provides emotion insights to empathy engine

**Deliverable**: Integrated emotion intelligence engine

#### 2.4 CLI Integration
**Update**: `morgan/cli/app.py`

Add emotion visualization:
- `morgan chat --show-emotions` - Display detected emotions
- `morgan emotions` - Emotion analytics command
- Rich console output for emotional context

**Deliverable**: Emotions visible and tracked in CLI

#### 2.5 Testing
- [ ] Test each emotion module independently
- [ ] Test emotion detection accuracy
- [ ] Test emotional context tracking
- [ ] Validate emotional memory storage
- [ ] Integration tests with CLI

**Deliverable**: Full emotion detection working with >80% accuracy

**Estimated Time**: 1-2 weeks
**Key Files**: 15-20 files (~4,000 lines)

---

## **PHASE 3: Empathy Engine (Week 5)**
### Goal: Generate emotionally-appropriate responses

### Priority: HIGH 🟠
This phase completes the emotional AI pipeline with empathetic responses.

### Tasks

#### 3.1 Empathy Modules
**Files**: `morgan/empathy/`

All 5 modules:
1. **generator.py** - Empathetic response generation
2. **mirror.py** - Emotional mirroring
3. **support.py** - Support response templates
4. **tone.py** - Tone adjustment based on emotions
5. **validator.py** - Empathy validation

**Deliverable**: All 5 empathy modules implemented

#### 3.2 Integration with Emotion Engine
Connect empathy to emotion detection:
- Receive emotional state from detector
- Adjust response tone
- Select appropriate support strategies
- Validate empathy effectiveness

**Deliverable**: Empathy engine connected to emotion detection

#### 3.3 Response Handler
**File**: `morgan/core/response_handler.py`

Combines:
- LLM-generated content
- Emotional tone adjustment
- Empathy layer
- Personalization

**Deliverable**: Response handler with empathy integration

#### 3.4 CLI Output Enhancement
Show empathy in action with rich console output and emotional validation scores.

**Deliverable**: Empathetic responses working in CLI

**Estimated Time**: 1 week
**Key Files**: 8-10 files (~2,500 lines)

---

## **PHASE 4: Advanced RAG System (Week 6-8)**
### Goal: Implement hierarchical search with reranking

### Priority: HIGH 🟠
This phase adds intelligent document-based knowledge to responses.

### Tasks

#### 4.1 Document Processing
**File**: `morgan/ingestion/enhanced_processor.py`

- PDF, DOCX, MD, HTML, TXT parsing
- Semantic chunking (not just fixed-size)
- Metadata extraction
- Hierarchical embedding generation (coarse, medium, fine)

**Deliverable**: Complete document processing pipeline

#### 4.2 Embedding Service
**File**: `morgan/services/embedding_service.py`

- Ollama qwen3-embedding integration
- Batch processing for efficiency
- Three granularity levels (coarse, medium, fine)
- Caching for duplicate content

**Deliverable**: Embedding service with hierarchical support

#### 4.3 Multi-Stage Search
**File**: `morgan/search/multi_stage_search.py`

Implements the full search pipeline:
1. **Coarse Search**: Find relevant documents
2. **Medium Search**: Narrow to relevant sections
3. **Fine Search**: Extract precise chunks
4. **Reciprocal Rank Fusion**: Merge results
5. **Reranking**: Use Jina reranker for final ordering

**Deliverable**: Multi-stage search with reranking

#### 4.4 Reranking Service
**File**: `morgan/jina/reranking/service.py`

- Load Jina reranker model (jina-reranker-v2-base-multilingual)
- Cross-encoder scoring
- GPU acceleration if available
- Batch processing

**Deliverable**: Reranking service operational

#### 4.5 Companion Memory Search
**File**: `morgan/search/companion_memory_search.py`

Special search for relationship/emotional memories:
- Search by emotional context
- Search by time period
- Search by relationship milestone
- Combine with factual search

**Deliverable**: Companion memory search integrated

#### 4.6 Knowledge Management
**File**: `morgan/core/knowledge.py`

- Ingest documents via CLI
- Update existing knowledge
- Query knowledge base
- Track knowledge sources
- Manage collections

**Deliverable**: Complete knowledge management system

#### 4.7 CLI Integration
Add commands:
- `morgan learn ./docs` - Ingest documents
- `morgan knowledge --search "query"` - Search knowledge
- `morgan knowledge --stats` - Statistics
- `morgan ask` - Now uses RAG for answers

**Deliverable**: Full RAG pipeline working via CLI

**Estimated Time**: 2-3 weeks
**Key Files**: 15-20 files (~5,000 lines)

---

## **PHASE 5: Memory System (Week 9-10)**
### Goal: Multi-layer memory with emotional context

### Priority: MEDIUM 🟡
This phase adds persistent memory across all interaction types.

### Tasks

#### 5.1 Memory Types

**Conversation Memory** (`morgan/storage/memory.py`):
- Short-term conversation history
- Message-level storage
- Efficient retrieval
- Context window management

**Knowledge Memory** (`morgan/storage/vector.py`):
- Long-term factual knowledge
- Document embeddings
- Searchable via RAG

**Emotional Memory** (`morgan/emotions/memory.py`):
- Significant emotional events
- Emotional patterns over time
- Triggers and recovery tracking

**Relationship Memory** (`morgan/companion/storage.py`):
- User preferences
- Communication style
- Milestones and trust level
- Interaction patterns

**Deliverable**: All memory types implemented

#### 5.2 Memory Processor
**File**: `morgan/memory/memory_processor.py`

Consolidates memories:
- Identifies important conversations
- Extracts key facts
- Recognizes emotional events
- Updates relationship model
- Prunes old/irrelevant memories

**Deliverable**: Memory consolidation working

#### 5.3 Companion Relationship Manager
**File**: `morgan/companion/relationship_manager.py`

Tracks relationship dynamics:
- Trust building
- Communication preferences
- Emotional bonds
- Shared experiences
- Milestones

**Deliverable**: Relationship tracking functional

#### 5.4 CLI Integration
Add commands:
- `morgan memory --stats` - Memory statistics
- `morgan memory --search "topic"` - Search memories
- `morgan relationship` - Relationship insights

**Deliverable**: Complete memory system accessible via CLI

**Estimated Time**: 1-2 weeks
**Key Files**: 10-15 files (~3,000 lines)

---

## **PHASE 6: Learning & Adaptation (Week 11-12)**
### Goal: Continuous improvement based on feedback

### Priority: MEDIUM 🟡
This phase enables the system to improve over time.

### Tasks

#### 6.1 Feedback Collection
**File**: `morgan/learning/feedback.py`

Collect:
- Explicit feedback (thumbs up/down, ratings)
- Implicit feedback (follow-up questions, conversation length)
- Emotional feedback (user emotions during interaction)
- Outcome feedback (did it help?)

**Deliverable**: Feedback collection system

#### 6.2 Pattern Recognition
**File**: `morgan/learning/patterns.py`

Identify:
- Common questions
- Preferred response styles
- Successful interactions
- Failed interactions
- Emotional triggers

**Deliverable**: Pattern recognition working

#### 6.3 Adaptation Engine
**File**: `morgan/learning/adaptation.py`

Adapt:
- Response style
- Detail level
- Emotional tone
- Proactivity
- Formality

**Deliverable**: Adaptation engine functional

#### 6.4 Preference Learning
**File**: `morgan/learning/preferences.py`

Learn:
- Communication style preferences
- Topic interests
- Response length preferences
- Emotional support needs
- Learning style

**Deliverable**: Preference learning operational

#### 6.5 CLI Integration
Add commands:
- `morgan preferences --view` - Show learned preferences
- Feedback prompts after responses

**Deliverable**: Learning system working via CLI

**Estimated Time**: 1-2 weeks
**Key Files**: 10-15 files (~3,000 lines)

---

## **PHASE 7: System Integration (Week 13-14)**
### Goal: Everything works together seamlessly

### Priority: CRITICAL 🔴
This phase ensures all components work as a unified system.

### Tasks

#### 7.1 Core Assistant
**File**: `morgan/core/assistant.py`

Main orchestrator that:
1. Receives user input
2. Detects emotions
3. Searches memory/knowledge
4. Applies empathy
5. Generates response with LLM
6. Updates memories
7. Learns from interaction
8. Returns emotionally-aware, contextual response

**Deliverable**: Complete assistant orchestration

#### 7.2 System Integration
**File**: `morgan/core/system_integration.py`

Coordinates all subsystems:
- Emotional intelligence pipeline
- Empathy engine
- RAG system
- Memory system
- Learning system
- Storage layer

**Deliverable**: All systems integrated

#### 7.3 End-to-End Testing
- [ ] Test complete conversation flows
- [ ] Validate all components working together
- [ ] Performance testing (<3s response time)
- [ ] Stress testing (1000+ interactions)
- [ ] Error handling and graceful degradation

**Deliverable**: Validated end-to-end system

#### 7.4 CLI Polish
- Rich formatting
- Progress indicators
- Error handling
- Graceful degradation
- Help text
- Examples

**Deliverable**: Complete system working end-to-end with polished CLI

**Estimated Time**: 1-2 weeks
**Key Files**: Integration code + extensive testing

---

## **PHASE 8: Distributed Deployment (Week 15-16)**
### Goal: Multi-host support (optional for CLI MVP)

### Priority: LOW 🟢
This phase enables distributed deployment across multiple hosts.

### Tasks

#### 8.1 Service Discovery
- Consul setup
- Service registration
- Health checks
- DNS resolution

**Deliverable**: Consul service discovery working

#### 8.2 Distributed Services
- LLM service on GPU hosts
- Embedding service on GPU hosts
- Reranking service on GPU hosts
- Qdrant on high-RAM hosts
- Core assistant on CPU hosts

**Deliverable**: Services distributed across hosts

#### 8.3 Admin CLI
**File**: `morgan/cli/distributed_cli.py`

Commands:
- `morgan-admin deploy` - Deploy to cluster
- `morgan-admin health` - Check all hosts
- `morgan-admin restart --service llm` - Restart services

**Deliverable**: Multi-host deployment working

#### 8.4 Testing & Validation
- [ ] Multi-host deployment tests
- [ ] Service discovery tests
- [ ] Load balancing validation
- [ ] Failover testing

**Deliverable**: Multi-host deployment validated

**Estimated Time**: 1-2 weeks
**Key Files**: 10-15 files (~2,500 lines)

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
```text
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
```text
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
