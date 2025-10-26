# Morgan Assistant - Final Fixes Log

## Date: 2025-10-25

### Issues Found During Service Startup

After completing the major refactoring (VAD service removal, Docker image optimization), we encountered startup issues that required additional fixes:

---

## Fix #1: Syntax Error in Logging Module

**File**: `shared/utils/logging.py:124`

**Error**:
```
SyntaxError: invalid decimal literal
self.logger.info(f"{self.operation} completed in {duration".3f"}s")
```

**Root Cause**: Invalid f-string syntax - incorrect quote placement inside the f-string.

**Fix**:
```python
# Before (INCORRECT):
self.logger.info(f"{self.operation} completed in {duration".3f"}s")

# After (CORRECT):
self.logger.info(f"{self.operation} completed in {duration:.3f}s")
```

**Impact**: All services (Core, LLM, TTS, STT) failed to start due to this syntax error in the shared logging module.

---

## Fix #2: Missing Dependencies in Service Dockerfiles

**Files**:
- `services/llm/Dockerfile`
- `services/tts/Dockerfile`
- `services/stt/Dockerfile`

**Error**:
```
ModuleNotFoundError: No module named 'yaml'
```

**Root Cause**: Service Dockerfiles were only installing service-specific dependencies (openai, TTS, faster-whisper) but not the base FastAPI/shared dependencies that are required by the shared modules (pyyaml, pydantic, fastapi, etc.).

**Fix**: Updated all service Dockerfiles to include base dependencies:

### LLM Service
```dockerfile
# Before:
RUN uv pip install openai --system

# After:
RUN uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv openai --system
```

### TTS Service
```dockerfile
# Before:
RUN uv pip install TTS soundfile librosa pydub torch torchaudio torchvision --system

# After:
RUN uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv TTS soundfile librosa pydub torch torchaudio torchvision --system
```

### STT Service
```dockerfile
# Before:
RUN uv pip install faster-whisper soundfile librosa pydub torch torchaudio silero-vad --system

# After:
RUN uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv faster-whisper soundfile librosa pydub torch torchaudio silero-vad --system
```

**Added Dependencies**:
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic` - Data validation
- `aiohttp` - Async HTTP client
- `pyyaml` - YAML config parsing
- `python-dotenv` - Environment variable loading

**Impact**: Services can now import shared modules (`shared.config.base`, `shared.utils.logging`, etc.) which depend on these base packages.

---

## Verification Steps

1. Fixed syntax error in `shared/utils/logging.py`
2. Updated all service Dockerfiles with complete dependency lists
3. Rebuilt all services:
   - Core: ✓ Completed
   - LLM: ✓ Completed
   - TTS: 🔄 Building (large CUDA dependencies)
   - STT: 🔄 Building (large CUDA dependencies)

4. Next: Start services with `docker-compose up -d` after builds complete

---

## Root Cause Analysis

The initial Dockerfiles assumed services would be isolated and only needed their ML-specific dependencies. However, the shared module architecture requires all services to have access to:
- Config system (pyyaml)
- Logging utilities (structlog implied by usage)
- HTTP communication (aiohttp)
- Data validation (pydantic)
- FastAPI framework (for API servers)

**Lesson**: When using shared modules across microservices, all services need the base dependency stack, not just service-specific packages.

---

## Files Modified

1. `shared/utils/logging.py` - Fixed f-string syntax
2. `services/llm/Dockerfile` - Added base dependencies
3. `services/tts/Dockerfile` - Added base dependencies
4. `services/stt/Dockerfile` - Added base dependencies

---

## Expected Outcome

After rebuilds complete:
- All services should start without module import errors
- Health checks should pass
- Core service should successfully communicate with all microservices
- Audio processing pipeline should work: Audio → STT (with VAD) → LLM → TTS → Core
