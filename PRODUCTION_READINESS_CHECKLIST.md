# Production Readiness Checklist

This document provides a comprehensive checklist for validating that Morgan server is production-ready.

## Quick Validation

Run the automated validation script:

```bash
cd morgan-server
python validate_production_readiness.py
```

This script will automatically check all items in this checklist.

## Manual Validation Steps

### 1. Test Suite ✓

**Objective**: Ensure all tests pass

**Steps**:
```bash
cd morgan-server
python -m pytest tests/ -v
```

**Expected Result**:
- All tests pass (1235+ tests)
- No failures or errors
- Coverage is adequate

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 2. Docker Compose Stack ✓

**Objective**: Verify Docker Compose configuration works

**Steps**:
```bash
# Validate syntax
docker-compose -f docker/docker-compose.yml config

# Start the stack
docker-compose -f docker/docker-compose.yml up -d

# Check services are running
docker-compose -f docker/docker-compose.yml ps

# Check logs
docker-compose -f docker/docker-compose.yml logs morgan-server
docker-compose -f docker/docker-compose.yml logs qdrant

# Stop the stack
docker-compose -f docker/docker-compose.yml down
```

**Expected Result**:
- Configuration is valid
- All services start successfully
- Services are healthy
- No error logs

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 3. Real LLM (Ollama) ✓

**Objective**: Test connectivity with real Ollama instance

**Prerequisites**:
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull a model
ollama pull gemma3
# or
ollama pull llama2
```

**Steps**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

**Expected Result**:
- Ollama responds to API calls
- At least one model is available
- Generation works correctly

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 4. Real Vector Database (Qdrant) ✓

**Objective**: Test connectivity with real Qdrant instance

**Prerequisites**:
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Steps**:
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# List collections
curl http://localhost:6333/collections

# Create a test collection
curl -X PUT http://localhost:6333/collections/test \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

# Delete test collection
curl -X DELETE http://localhost:6333/collections/test
```

**Expected Result**:
- Qdrant responds to health checks
- Collections can be listed
- Collections can be created and deleted

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 5. Monitoring and Metrics ✓

**Objective**: Verify monitoring endpoints work

**Steps**:
```bash
# Start Morgan server
python -m morgan_server

# Check health endpoint
curl http://localhost:8080/health

# Check detailed status
curl http://localhost:8080/api/status

# Check Prometheus metrics
curl http://localhost:8080/metrics
```

**Expected Result**:
- Health endpoint returns 200 with status "healthy"
- Status endpoint shows all components as "up"
- Metrics endpoint returns Prometheus-formatted data
- Metrics include:
  - Request counts
  - Response times
  - Error rates
  - Active sessions
  - Uptime

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 6. Graceful Shutdown ✓

**Objective**: Verify server shuts down gracefully

**Steps**:
```bash
# Start server
python -m morgan_server

# In another terminal, get the PID
ps aux | grep morgan_server

# Send SIGTERM
kill -TERM <pid>

# Check logs for graceful shutdown messages
```

**Expected Result**:
- Server receives SIGTERM signal
- Server logs "Shutting down gracefully..."
- All connections are closed
- Pending data is persisted
- Server exits cleanly

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### 7. Documentation Completeness ✓

**Objective**: Ensure all documentation is present and complete

**Required Documents**:

- [ ] `morgan-server/README.md` - Server documentation
  - Installation instructions
  - Configuration guide
  - API documentation
  - Deployment guide
  
- [ ] `morgan-cli/README.md` - Client documentation
  - Installation instructions
  - Usage examples
  - Configuration guide
  
- [ ] `docker/docker-compose.yml` - Docker Compose configuration
  - All services defined
  - Environment variables documented
  - Volumes configured
  - Networks configured
  
- [ ] `MIGRATION.md` - Migration guide
  - Migration steps from old system
  - Breaking changes documented
  - Compatibility notes
  
- [ ] `DOCUMENTATION.md` - Main documentation
  - Architecture overview
  - Component descriptions
  - Development guide
  
- [ ] `DEPRECATION_NOTICE.md` - Deprecation notice
  - Old system marked as deprecated
  - Timeline for removal
  - Migration instructions

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

## End-to-End Integration Tests

### Test 1: Complete Chat Flow

**Steps**:
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be ready
sleep 30

# Send a chat message
curl -X POST http://localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Hello, Morgan! Tell me about yourself.",
    "user_id": "test_user"
  }'
```

**Expected Result**:
- Server responds with a chat response
- Response includes answer, conversation_id, confidence
- Emotional tone is detected
- Response is personalized

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### Test 2: Document Learning Flow

**Steps**:
```bash
# Create a test document
echo "Morgan is an AI assistant with empathic and knowledge engines." > test_doc.txt

# Learn from document
curl -X POST http://localhost:8080/api/knowledge/learn \
  -F "file=@test_doc.txt" \
  -F "doc_type=text"

# Search knowledge base
curl -X GET "http://localhost:8080/api/knowledge/search?query=empathic&limit=5"

# Get knowledge stats
curl http://localhost:8080/api/knowledge/stats
```

**Expected Result**:
- Document is processed successfully
- Search returns relevant results
- Stats show document count increased

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

### Test 3: Memory and Profile Flow

**Steps**:
```bash
# Get user profile
curl http://localhost:8080/api/profile/test_user

# Update preferences
curl -X PUT http://localhost:8080/api/profile/test_user \
  -H 'Content-Type: application/json' \
  -d '{
    "preferred_name": "Alex",
    "communication_style": "friendly",
    "response_length": "detailed"
  }'

# Get memory stats
curl http://localhost:8080/api/memory/stats?user_id=test_user

# Search memory
curl -X GET "http://localhost:8080/api/memory/search?query=hello&user_id=test_user"
```

**Expected Result**:
- Profile is created/retrieved
- Preferences are updated
- Memory stats are returned
- Memory search works

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

## Performance Validation

### Load Testing

**Objective**: Verify server handles concurrent requests

**Steps**:
```bash
# Install Apache Bench (if not installed)
# apt-get install apache2-utils  # Ubuntu/Debian
# brew install apache2-utils      # macOS

# Run load test
ab -n 1000 -c 10 -p chat_request.json -T application/json \
  http://localhost:8080/api/chat
```

**Expected Result**:
- 95th percentile response time < 5 seconds
- No errors or timeouts
- Server remains stable

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

## Security Validation

### Configuration Security

**Checklist**:
- [ ] No hardcoded API keys in code
- [ ] Environment variables used for sensitive data
- [ ] Default passwords changed
- [ ] CORS configured appropriately
- [ ] Rate limiting considered (if needed)

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

## Deployment Validation

### Docker Deployment

**Steps**:
```bash
# Build server image
docker build -t morgan-server:latest -f docker/Dockerfile.server .

# Run server container
docker run -d \
  -p 8080:8080 \
  -e MORGAN_LLM_ENDPOINT=http://host.docker.internal:11434 \
  -e MORGAN_VECTOR_DB_URL=http://host.docker.internal:6333 \
  --name morgan-server \
  morgan-server:latest

# Check logs
docker logs morgan-server

# Test health
curl http://localhost:8080/health

# Stop container
docker stop morgan-server
docker rm morgan-server
```

**Expected Result**:
- Image builds successfully
- Container starts without errors
- Health check passes
- Server is accessible

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete

---

## Final Checklist

Before marking this task as complete, ensure:

- [x] All tests pass (1235+ tests)
- [ ] Docker Compose stack works
- [ ] Real LLM (Ollama) connectivity verified
- [ ] Real vector database (Qdrant) connectivity verified
- [ ] Monitoring and metrics endpoints work
- [ ] Graceful shutdown works
- [ ] All documentation is complete
- [ ] End-to-end integration tests pass
- [ ] Performance is acceptable
- [ ] Security considerations addressed
- [ ] Docker deployment works

---

## Notes

Add any additional notes or observations here:

```
[Your notes here]
```

---

## Sign-off

**Validated by**: _______________
**Date**: _______________
**Signature**: _______________

---

## Automated Validation

To run the automated validation script:

```bash
cd morgan-server
python validate_production_readiness.py
```

This will check:
1. ✓ All tests pass
2. ✓ Docker Compose syntax is valid
3. ✓ Qdrant is accessible and healthy
4. ✓ Ollama is accessible with models
5. ✓ Morgan server is healthy
6. ✓ Metrics endpoint works
7. ✓ Graceful shutdown is implemented
8. ✓ Documentation is complete

The script will output a summary and exit with code 0 if all checks pass, or 1 if any fail.
