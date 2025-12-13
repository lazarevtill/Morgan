# Production Readiness Validation Guide

This guide explains how to validate that Morgan server is production-ready.

## Quick Validation (Recommended)

For a fast check without running the full test suite:

```bash
cd morgan-server
python quick_validate.py
```

This checks:
- ✅ Documentation files exist
- ✅ Services are accessible (Qdrant, Ollama, Morgan Server)
- ✅ Test structure is in place

**Time**: ~5 seconds

## Full Validation

For comprehensive validation including all tests:

```bash
cd morgan-server
python validate_production_readiness.py
```

This checks everything in quick validation PLUS:
- ✅ All 1235+ tests pass
- ✅ Docker Compose syntax validation
- ✅ Detailed service health checks
- ✅ Metrics endpoint validation

**Time**: ~5-10 minutes (depending on test execution)

## Prerequisites

Before running validation, you may need to start services:

### 1. Qdrant (Vector Database)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use Docker Compose:
```bash
docker-compose -f ../docker/docker-compose.yml up qdrant -d
```

### 2. Ollama (LLM)

Install from https://ollama.ai/download, then:

```bash
ollama serve
ollama pull gemma3  # or another model
```

### 3. Morgan Server

```bash
cd morgan-server
python -m morgan_server
```

## Validation Checklist

### Documentation ✅

- [x] `morgan-server/README.md` - Server documentation
- [x] `morgan-cli/README.md` - Client documentation  
- [x] `docker/docker-compose.yml` - Docker configuration
- [x] `MIGRATION.md` - Migration guide
- [x] `DOCUMENTATION.md` - Main documentation

### Services

- [ ] Qdrant running at http://localhost:6333
- [ ] Ollama running at http://localhost:11434
- [ ] Morgan Server running at http://localhost:8080

### Tests

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All property-based tests pass
- [ ] Performance tests pass

### Docker

- [ ] docker-compose.yml syntax is valid
- [ ] Docker images build successfully
- [ ] Containers start without errors
- [ ] Health checks pass

## Troubleshooting

### "Cannot connect to Qdrant"

**Solution**: Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "Cannot connect to Ollama"

**Solution**: Install and start Ollama:
1. Download from https://ollama.ai/download
2. Run `ollama serve`
3. Pull a model: `ollama pull gemma3`

### "Cannot connect to Morgan Server"

**Solution**: Start the server:
```bash
cd morgan-server
python -m morgan_server
```

### "Tests are taking too long"

**Solution**: Use quick validation instead:
```bash
python quick_validate.py
```

### "Character encoding errors on Windows"

The scripts handle this automatically by falling back to ASCII symbols.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Production Readiness

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          cd morgan-server
          pip install -e ".[dev]"
      
      - name: Run quick validation
        run: |
          cd morgan-server
          python quick_validate.py
      
      - name: Run full validation
        run: |
          cd morgan-server
          python validate_production_readiness.py
```

## Manual Testing

For manual end-to-end testing, see [PRODUCTION_READINESS_CHECKLIST.md](../PRODUCTION_READINESS_CHECKLIST.md).

## Exit Codes

Both validation scripts use standard exit codes:

- `0` - All checks passed
- `1` - Some checks failed or error occurred

This makes them suitable for CI/CD pipelines.

## What Gets Validated

### Quick Validation

1. **Documentation** - All required docs exist
2. **Services** - Qdrant, Ollama, Morgan Server accessible
3. **Test Structure** - Test files present

### Full Validation

Everything in quick validation PLUS:

4. **Test Suite** - All 1235+ tests pass
5. **Docker Compose** - Configuration syntax valid
6. **Service Health** - Detailed health checks
7. **Metrics** - Prometheus metrics endpoint works
8. **Graceful Shutdown** - Implementation verified

## Recommendations

1. **Development**: Use `quick_validate.py` for fast feedback
2. **Pre-commit**: Run `quick_validate.py` before committing
3. **CI/CD**: Run `validate_production_readiness.py` in pipeline
4. **Pre-deployment**: Run full validation + manual checklist
5. **Production**: Set up continuous monitoring

## Support

For issues or questions:
1. Check [PRODUCTION_READINESS_CHECKLIST.md](../PRODUCTION_READINESS_CHECKLIST.md)
2. Review [TASK_47_PRODUCTION_READINESS.md](../TASK_47_PRODUCTION_READINESS.md)
3. Check server logs: `docker-compose logs morgan-server`
