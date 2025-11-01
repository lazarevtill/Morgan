# Morgan RAG - Human-First AI Assistant

## Overview

Morgan RAG is a **human-first** AI assistant built with KISS principles (Keep It Simple, Stupid). Unlike complex RAG systems that focus on technical features, Morgan is designed to be genuinely helpful, easy to use, and simple to maintain.

Built on proven patterns from InspecTor but redesigned for human interaction, Morgan provides intelligent conversation, learning from documents, and continuous improvement through feedback.

## Philosophy: Human-First, Not Code-First

🤝 **Conversational**: Talk to Morgan naturally, like you would with a knowledgeable colleague  
🧠 **Learning**: Morgan learns from your documents and remembers your conversations  
🎯 **Helpful**: Focused on actually helping humans, not just being technically impressive  
🔧 **Simple**: Easy to set up, use, and maintain - no PhD in AI required  
⚡ **Fast**: Quick responses without sacrificing quality  
🔒 **Private**: Your data stays on your servers, period  
🐳 **Ready**: Works out of the box with Docker 

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│              "How do I deploy a Docker container?"              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY PROCESSING                              │
│  ├─ Intent Classification                                       │
│  ├─ Query Enhancement                                           │
│  └─ Domain Routing                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE RETRIEVAL                            │
│  ├─ Hierarchical Search (coarse→medium→fine)                   │
│  ├─ Multi-Domain Fusion                                        │
│  ├─ Conversation Memory Search                                 │
│  └─ Context Assembly                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE GENERATION                          │
│  ├─ LLM Generation (OpenAI Compatible)                         │
│  ├─ Source Attribution                                         │
│  └─ Response Enhancement                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LEARNING & MEMORY                             │
│  ├─ Conversation Storage                                       │
│  ├─ Feedback Learning                                          │
│  └─ Knowledge Graph Updates                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                  INTELLIGENT RESPONSE
```

## Quick Start (5 Minutes to Your First Chat)

### 1. Get Morgan Running

```bash
# Clone and setup
git clone <your-repo> morgan-rag
cd morgan-rag

# Copy configuration template
cp .env.example .env

# Edit .env file - just set your LLM endpoint
nano .env  # Set LLM_BASE_URL and LLM_API_KEY

# Start with Docker (easiest)
docker-compose up -d

# Or install locally
pip install -r requirements.txt
python -m morgan init
```

### 2. Teach Morgan Something

```bash
# Learn from your documents
morgan learn ./docs

# Learn from a website
morgan learn --url https://docs.docker.com

# Learn from code
morgan learn ./my-project --type code
```

### 3. Start Chatting

```bash
# Interactive chat (recommended)
morgan chat

# Quick question
morgan ask "How do I deploy Docker containers?"

# Web interface
morgan serve
# Then open http://localhost:8080
```

That's it! Morgan is now ready to help you with anything in your documents.

## Configuration

### Environment Variables (.env)

```bash
# LLM Configuration
LLM_BASE_URL=https://gpt.lazarev.cloud/ollama/v1
LLM_API_KEY=your-api-key
LLM_MODEL=llama3.1:8b

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional-api-key

# Embedding Service
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=100

# System Settings
MORGAN_DATA_DIR=./data
MORGAN_LOG_LEVEL=INFO
MORGAN_MAX_CONTEXT=8192
```

## Core Components

### 1. Document Ingestion (`morgan/ingestion/`)
- **Universal Parsers**: PDF, DOCX, MD, HTML, TXT, Code files
- **Intelligent Chunking**: Semantic-aware text splitting
- **Metadata Extraction**: Automatic metadata detection
- **Batch Processing**: Efficient bulk ingestion

### 2. Vector Database (`morgan/vector_db/`)
- **Hierarchical Embeddings**: Coarse/medium/fine scale search
- **Collection Management**: Domain-specific collections
- **Caching System**: Git hash tracking for incremental updates
- **Performance Optimization**: Batch operations and indexing

### 3. Search Engine (`morgan/search/`)
- **Multi-Stage Retrieval**: 6 search strategies with fusion
- **Query Enhancement**: Automatic query expansion
- **Relevance Scoring**: Advanced ranking algorithms
- **Context Assembly**: Intelligent context building

### 4. Memory System (`morgan/memory/`)
- **Conversation History**: Persistent chat storage
- **Semantic Memory**: Vector-based conversation search
- **Learning Feedback**: User feedback integration
- **Knowledge Graph**: Entity and relationship tracking

### 5. LLM Integration (`morgan/llm/`)
- **OpenAI Compatible**: Works with any OpenAI-compatible API
- **Streaming Support**: Real-time response streaming
- **Context Management**: Intelligent context window management
- **Response Enhancement**: Source attribution and formatting

## Real Examples (What Morgan Actually Does)

### Natural Conversation

```bash
$ morgan chat

🤖 Morgan: Hello! I'm your AI assistant. How can I help you today?

👤 You: I need to deploy a web app with Docker but I'm new to this

🤖 Morgan: I'd be happy to help you deploy a web app with Docker! Let me break this down into simple steps:

1. **Create a Dockerfile** - This tells Docker how to build your app
2. **Build the image** - Package your app into a container
3. **Run the container** - Start your app

What type of web app are you working with? (Node.js, Python, PHP, etc.)

👤 You: It's a Python Flask app

🤖 Morgan: Perfect! Here's a simple Dockerfile for a Flask app:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Then run these commands:
```bash
docker build -t my-flask-app .
docker run -p 5000:5000 my-flask-app
```

Would you like me to explain any of these steps in more detail?

[Sources: docker-flask-tutorial.md, dockerfile-best-practices.md]
```

### Quick Questions

```bash
$ morgan ask "What's the difference between Docker and VM?"

🤖 Morgan: Great question! Here are the key differences:

**Docker (Containers):**
- Shares the host OS kernel
- Lightweight (MBs)
- Starts in seconds
- Better resource efficiency

**Virtual Machines:**
- Each VM has its own OS
- Heavier (GBs)
- Takes minutes to start
- More isolated but resource-intensive

Think of containers like apartments in a building (shared infrastructure) vs VMs like separate houses (complete isolation).

[Sources: docker-vs-vm-comparison.md]
```

### Learning from Your Docs

```bash
$ morgan learn ./company-docs

📚 Teaching Morgan from: ./company-docs
⠋ Processing documents... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✅ Learning Complete!
📚 Documents processed: 47
🧩 Knowledge chunks: 312
⏱️  Processing time: 23.4s
🎯 Knowledge areas: DevOps, API Guidelines, Security Policies, Deployment Procedures

$ morgan ask "What's our API versioning policy?"

🤖 Morgan: According to your company's API guidelines, here's the versioning policy:

1. **Semantic Versioning**: Use MAJOR.MINOR.PATCH format
2. **URL Versioning**: Include version in URL path (/api/v1/)
3. **Backward Compatibility**: Maintain for at least 2 major versions
4. **Deprecation Notice**: 6 months minimum before removing endpoints

[Source: api-guidelines.md, section 3.2]
```

### Python API (Simple Integration)

```python
from morgan import create_assistant

# Create Morgan
morgan = create_assistant()

# Teach Morgan about your project
morgan.learn_from_documents("./docs")

# Ask questions
response = morgan.ask("How do I set up authentication?")
print(response.answer)

# Start a conversation
conv_id = morgan.start_conversation("API Development")
response = morgan.ask("What's our auth flow?", conversation_id=conv_id)

# Provide feedback to help Morgan learn
morgan.provide_feedback(conv_id, rating=5, comment="Very helpful!")
```

## Performance Characteristics

### Ingestion Performance
- **Documents**: ~100 docs/minute
- **Code Files**: ~500 files/minute  
- **Web Pages**: ~50 pages/minute
- **Batch Processing**: 10x faster than individual

### Search Performance
- **Query Response**: <200ms (cached)
- **First Search**: <500ms (cold)
- **Hierarchical Filtering**: 90% reduction in candidates
- **Memory Usage**: ~2GB for 100K documents

### LLM Integration
- **Context Assembly**: <100ms
- **Response Generation**: Depends on LLM (typically 1-5s)
- **Streaming**: Real-time token delivery
- **Concurrent Users**: 50+ (with proper LLM backend)

## Advanced Features

### 1. Domain-Specific Collections

```bash
# Create specialized collections
morgan collection create "python-docs" --domain programming
morgan collection create "company-policies" --domain business
morgan collection create "research-papers" --domain academic

# Query specific domains
morgan ask "How to handle exceptions?" --domain programming
```

### 2. Learning from Feedback

```bash
# Provide feedback on responses
morgan feedback --conversation-id abc123 --rating 5 --comment "Very helpful"

# View learning analytics
morgan analytics --period 30d
```

### 3. Knowledge Graph

```bash
# View entity relationships
morgan graph --entity "Docker" --depth 2

# Export knowledge graph
morgan graph export --format graphml
```

### 4. Custom Embeddings

```python
# Use custom embedding models
from morgan.embeddings import CustomEmbedding

embedding = CustomEmbedding(
    model_path="./my-embedding-model",
    device="cuda"
)

morgan = MorganRAG(embedding_service=embedding)
```

## Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  morgan:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LLM_BASE_URL=https://gpt.lazarev.cloud/ollama/v1
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./data:/app/data
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morgan-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: morgan-rag
  template:
    metadata:
      labels:
        app: morgan-rag
    spec:
      containers:
      - name: morgan
        image: morgan-rag:latest
        ports:
        - containerPort: 8080
        env:
        - name: LLM_BASE_URL
          value: "https://gpt.lazarev.cloud/ollama/v1"
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
```

## Monitoring & Analytics

### Health Checks

```bash
# System health
morgan health

# Component status
morgan status --detailed

# Performance metrics
morgan metrics --component search --period 1h
```

### Logging

```bash
# View logs
morgan logs --level INFO --follow

# Search logs
morgan logs --query "error" --since 1h

# Export logs
morgan logs --export --format json --output logs.json
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](issues)
- 💬 Discussions: [GitHub Discussions](discussions)
- 📧 Email: support@morgan-rag.com

---

**Morgan RAG** - Intelligent Knowledge for Intelligent Assistants