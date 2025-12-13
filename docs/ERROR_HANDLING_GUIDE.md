# Morgan Error Handling Guide

> **⚠️ DEPRECATED:** This document is for the old Morgan system. For the new client-server architecture, see:
> - [Server Documentation](../morgan-server/README.md)
> - [Deployment Guide](../morgan-server/docs/DEPLOYMENT.md)
> - [Migration Guide](../MIGRATION.md)

## Overview

This document describes the production-grade error handling system for Morgan AI Assistant (deprecated).

## Table of Contents

1. [Architecture](#architecture)
2. [Exception Hierarchy](#exception-hierarchy)
3. [Error Decorators](#error-decorators)
4. [Error Handling](#error-handling)
5. [Best Practices](#best-practices)
6. [Migration Guide](#migration-guide)
7. [Examples](#examples)

## Architecture

The error handling system consists of three main components:

### 1. Exception Hierarchy (`shared/utils/exceptions.py`)

- **Custom exception classes** with rich context
- **Error classification** (transient vs permanent)
- **Correlation IDs** for distributed tracing
- **User-friendly messages** with recovery suggestions
- **Structured error responses**

### 2. Error Decorators (`shared/utils/error_decorators.py`)

- **Retry logic** with exponential backoff
- **Circuit breaker** pattern for fault tolerance
- **Timeout handling**
- **Error aggregation** for batch operations
- **Fallback strategies**

### 3. Error Handling (`shared/utils/error_handling.py`)

- **Structured error responses**
- **Error context management**
- **Correlation ID tracking**
- **HTTP status code mapping**
- **Structured logging**

## Exception Hierarchy

### Base Exception

```python
from shared.utils.exceptions import MorganException

# All custom exceptions inherit from MorganException
raise MorganException(
    message="Something went wrong",
    category=ErrorCategory.INTERNAL_ERROR,
    severity=ErrorSeverity.ERROR,
    is_transient=True,
    is_retryable=True,
    correlation_id="custom-id",  # Optional
    user_message="Please try again",
    recovery_suggestions=["Check your input", "Try again later"]
)
```

### Service Exceptions

```python
from shared.utils.exceptions import (
    ServiceUnavailableError,
    ServiceTimeoutError,
    ServiceDegradedError
)

# Service unavailable (transient, retryable)
raise ServiceUnavailableError(
    service_name="llm",
    message="LLM service is temporarily unavailable"
)

# Service timeout (transient, retryable)
raise ServiceTimeoutError(
    service_name="tts",
    timeout_seconds=30.0
)
```

### Model Exceptions

```python
from shared.utils.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
    ModelOutOfMemoryError
)

# Model not found (permanent, not retryable)
raise ModelNotFoundError(model_name="llama3.2")

# Model load failure (permanent, not retryable)
raise ModelLoadError(
    model_name="whisper-large",
    reason="Insufficient GPU memory"
)

# Inference failure (transient, retryable)
raise ModelInferenceError(
    model_name="csm-1b",
    reason="CUDA out of memory"
)

# GPU out of memory (transient, retryable)
raise ModelOutOfMemoryError(
    model_name="whisper",
    required_memory_mb=8192
)
```

### Audio Exceptions

```python
from shared.utils.exceptions import (
    AudioFormatError,
    AudioProcessingError,
    AudioEncodingError,
    AudioDecodingError
)

# Invalid format (permanent, not retryable)
raise AudioFormatError(
    message="Unsupported audio format",
    expected_format="WAV or MP3"
)

# Processing failure (transient, retryable)
raise AudioProcessingError(
    message="Failed to process audio",
    operation="noise_reduction"
)
```

### Configuration Exceptions

```python
from shared.utils.exceptions import (
    ConfigMissingError,
    ConfigInvalidError
)

# Missing configuration (permanent, not retryable)
raise ConfigMissingError(config_key="OPENAI_API_KEY")

# Invalid configuration (permanent, not retryable)
raise ConfigInvalidError(
    config_key="temperature",
    invalid_value=2.5,
    reason="Must be between 0 and 2"
)
```

### Validation Exceptions

```python
from shared.utils.exceptions import (
    ValidationError,
    InvalidInputError,
    MissingRequiredFieldError
)

# General validation error
raise ValidationError(
    message="Invalid input data",
    field_name="email"
)

# Invalid input
raise InvalidInputError(
    message="Email format is invalid",
    input_name="email"
)

# Missing required field
raise MissingRequiredFieldError(field_name="prompt")
```

### Resource Exceptions

```python
from shared.utils.exceptions import (
    ResourceExhaustedError,
    QuotaExceededError,
    GPUOutOfMemoryError
)

# Resource exhausted (transient, retryable)
raise ResourceExhaustedError(resource_name="GPU memory")

# Quota exceeded (permanent, not retryable)
raise QuotaExceededError(
    quota_name="API calls",
    limit=1000,
    current=1001
)

# GPU OOM (transient, retryable)
raise GPUOutOfMemoryError(required_memory_mb=16384)
```

## Error Decorators

### Retry Decorator

Automatically retry operations on transient failures:

```python
from shared.utils.error_decorators import retry_on_transient_error

@retry_on_transient_error(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0
)
async def fetch_from_external_api():
    # This will retry up to 3 times with exponential backoff
    # on transient errors
    return await external_api.get_data()
```

### Circuit Breaker

Prevent cascading failures:

```python
from shared.utils.error_decorators import with_circuit_breaker

@with_circuit_breaker(
    name="llm_service",
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2
)
async def call_llm_service(prompt: str):
    # Circuit opens after 5 failures
    # Stays open for 60 seconds
    # Needs 2 successes to close again
    return await llm_service.generate(prompt)
```

### Timeout

Add timeout protection:

```python
from shared.utils.error_decorators import with_timeout

@with_timeout(timeout_seconds=30.0)
async def slow_operation():
    # Raises ServiceTimeoutError after 30 seconds
    await asyncio.sleep(60)
```

### Error Handling with Logging

Handle and log errors gracefully:

```python
from shared.utils.error_decorators import handle_errors

@handle_errors(
    logger_instance=my_logger,
    suppress_errors=True,
    return_value_on_error=None
)
async def optional_operation():
    # Errors are logged but suppressed
    # Returns None on error
    return await risky_operation()
```

### Fallback Strategy

Provide fallback on error:

```python
from shared.utils.error_decorators import with_fallback

async def fallback_response():
    return "Service temporarily unavailable"

@with_fallback(fallback_response)
async def get_llm_response(prompt: str):
    # Falls back to default message on error
    return await llm_service.generate(prompt)
```

### Batch Error Aggregation

Handle errors in batch operations:

```python
from shared.utils.error_decorators import aggregate_errors

@aggregate_errors(operation_name="batch_transcription")
async def transcribe_batch(audio_files):
    results = []
    errors = []

    for audio in audio_files:
        try:
            result = await transcribe(audio)
            results.append(result)
        except Exception as e:
            errors.append((audio, e))

    return results, errors
```

### Combining Decorators

Stack decorators for comprehensive protection:

```python
@retry_on_transient_error(max_attempts=3)
@with_circuit_breaker(name="external_api")
@with_timeout(timeout_seconds=30.0)
@handle_errors(logger_instance=logger)
async def robust_api_call():
    return await external_api.call()
```

## Error Handling

### Using ErrorHandler

```python
from shared.utils.error_handling import ErrorHandler, create_error_context

# Create error handler
error_handler = ErrorHandler(
    logger_instance=logger,
    service_name="llm",
    include_stack_trace=False  # Set to True in development
)

# Create error context
context = create_error_context(
    user_id="user123",
    request_id="req456"
)

# Handle exception
try:
    result = await risky_operation()
except Exception as e:
    error_response = error_handler.handle_error(
        exception=e,
        context=context
    )
    # error_response is a structured dict ready to return to client
    return error_response
```

### Structured Error Responses

```python
from shared.utils.error_handling import ErrorResponse

# From exception
try:
    do_something()
except Exception as e:
    response = ErrorResponse.from_exception(
        exception=e,
        include_stack_trace=False,
        correlation_id="custom-correlation-id"
    )
    # Returns:
    # {
    #     "error": {
    #         "message": "...",
    #         "user_message": "...",
    #         "category": "...",
    #         "severity": "...",
    #         "correlation_id": "...",
    #         "is_transient": true/false,
    #         "is_retryable": true/false,
    #         "http_status": 500,
    #         "recovery_suggestions": [...]
    #     }
    # }

# Success response
response = ErrorResponse.success(
    data={"result": "success"},
    metadata={"processing_time": 1.23},
    correlation_id="custom-id"
)
```

## Best Practices

### 1. Always Use Specific Exceptions

**BAD:**
```python
try:
    result = await model.generate()
except Exception as e:
    logger.error(f"Error: {e}")
    raise
```

**GOOD:**
```python
from shared.utils.exceptions import ModelInferenceError, ModelOutOfMemoryError

try:
    result = await model.generate()
except torch.cuda.OutOfMemoryError as e:
    raise ModelOutOfMemoryError(
        model_name=model_name,
        required_memory_mb=estimate_memory()
    )
except RuntimeError as e:
    raise ModelInferenceError(
        model_name=model_name,
        reason=str(e)
    )
```

### 2. Never Use Bare Except

**BAD:**
```python
try:
    cleanup_temp_files()
except:
    pass
```

**GOOD:**
```python
try:
    cleanup_temp_files()
except (OSError, PermissionError) as e:
    logger.debug(f"Cleanup failed: {e}")
```

### 3. Provide Context and Recovery Suggestions

**BAD:**
```python
raise ValueError("Invalid input")
```

**GOOD:**
```python
raise InvalidInputError(
    message=f"Text length {len(text)} exceeds maximum of {max_length}",
    input_name="text",
    context={"length": len(text), "max_length": max_length},
    recovery_suggestions=[
        f"Reduce text length to {max_length} characters or less",
        "Split text into multiple requests"
    ]
)
```

### 4. Use Correlation IDs

```python
# Generate at request entry point
correlation_id = str(uuid.uuid4())

# Pass through call chain
try:
    result = await service.process(data, correlation_id=correlation_id)
except MorganException as e:
    # Correlation ID is automatically included
    logger.error(f"Processing failed: {e}", extra={"correlation_id": e.correlation_id})
    raise
```

### 5. Classify Errors Correctly

```python
# Transient errors (retry these)
raise ServiceUnavailableError(...)  # is_transient=True, is_retryable=True

# Permanent errors (don't retry)
raise ValidationError(...)  # is_transient=False, is_retryable=False
```

### 6. Use Decorators for Cross-Cutting Concerns

```python
# Instead of manual retry logic
@retry_on_transient_error(max_attempts=3)
async def fetch_data():
    return await api.get()

# Instead of manual circuit breaker
@with_circuit_breaker(name="external_service")
async def call_external():
    return await external.call()
```

### 7. Log with Structured Data

```python
from shared.utils.error_handling import ErrorLogger, ErrorContext

logger = ErrorLogger(service_name="stt")
context = ErrorContext(
    correlation_id="abc123",
    user_id="user456",
    operation="transcribe"
)

try:
    result = await transcribe()
except Exception as e:
    logger.log_exception(
        exception=e,
        context=context,
        additional_data={"audio_length": audio_length}
    )
    raise
```

## Migration Guide

### Step 1: Replace Old Exceptions

**Before:**
```python
from shared.utils.errors import ModelError, ErrorCode

raise ModelError(
    "Model load failed",
    ErrorCode.MODEL_LOAD_ERROR
)
```

**After:**
```python
from shared.utils.exceptions import ModelLoadError

raise ModelLoadError(
    model_name="whisper",
    reason="Insufficient GPU memory"
)
```

### Step 2: Update Exception Handlers

**Before:**
```python
from shared.utils.errors import ErrorHandler

error_handler = ErrorHandler(logger)

try:
    result = await process()
except Exception as e:
    error_handler.handle_error(e)
    raise
```

**After:**
```python
from shared.utils.error_handling import ErrorHandler, create_error_context

error_handler = ErrorHandler(logger_instance=logger, service_name="my_service")
context = create_error_context(request_id="req123")

try:
    result = await process()
except Exception as e:
    error_response = error_handler.handle_error(e, context)
    # Return error_response to client or re-raise
    raise
```

### Step 3: Add Decorators

**Before:**
```python
async def fetch_data():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await api.get()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

**After:**
```python
from shared.utils.error_decorators import retry_on_transient_error

@retry_on_transient_error(max_attempts=3, base_delay=1.0)
async def fetch_data():
    return await api.get()
```

### Step 4: Fix Bare Except Clauses

**Before:**
```python
try:
    cleanup()
except:
    pass
```

**After:**
```python
try:
    cleanup()
except (OSError, PermissionError) as e:
    logger.debug(f"Cleanup failed: {e}")
```

## Examples

### Complete Service Example

```python
from shared.utils.exceptions import (
    ModelInferenceError,
    ServiceTimeoutError,
    AudioProcessingError
)
from shared.utils.error_decorators import (
    retry_on_transient_error,
    with_circuit_breaker,
    with_timeout
)
from shared.utils.error_handling import (
    ErrorHandler,
    create_error_context,
    ErrorResponse
)
import logging

logger = logging.getLogger(__name__)
error_handler = ErrorHandler(logger_instance=logger, service_name="tts")


class TTSService:
    @retry_on_transient_error(max_attempts=3)
    @with_circuit_breaker(name="tts_model", failure_threshold=5)
    @with_timeout(timeout_seconds=30.0)
    async def generate_speech(self, text: str, correlation_id: str = None):
        """Generate speech from text with comprehensive error handling"""
        context = create_error_context(
            correlation_id=correlation_id,
            operation="generate_speech",
            text_length=len(text)
        )

        try:
            # Validate input
            if not text or len(text) > 5000:
                from shared.utils.exceptions import InvalidInputError
                raise InvalidInputError(
                    message=f"Text length must be 1-5000 characters, got {len(text)}",
                    input_name="text",
                    recovery_suggestions=[
                        "Provide text between 1-5000 characters",
                        "Split longer text into chunks"
                    ]
                )

            # Generate speech
            audio_data = await self.model.generate(text)

            # Return success response
            return ErrorResponse.success(
                data={"audio_data": audio_data},
                metadata={"text_length": len(text)},
                correlation_id=correlation_id
            )

        except torch.cuda.OutOfMemoryError as e:
            from shared.utils.exceptions import GPUOutOfMemoryError
            raise GPUOutOfMemoryError(
                required_memory_mb=8192,
                correlation_id=correlation_id
            )

        except Exception as e:
            # Log and return structured error
            error_response = error_handler.handle_error(e, context)
            return error_response


# API endpoint using the service
from fastapi import FastAPI, HTTPException

app = FastAPI()
tts_service = TTSService()


@app.post("/generate")
async def generate_speech_endpoint(text: str):
    try:
        result = await tts_service.generate_speech(text)

        # Check if result is an error
        if "error" in result:
            # Map to HTTP status code
            http_status = result["error"].get("http_status", 500)
            raise HTTPException(status_code=http_status, detail=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        # Unexpected error
        error_response = ErrorResponse.from_exception(e)
        http_status = error_response["error"].get("http_status", 500)
        raise HTTPException(status_code=http_status, detail=error_response)
```

### Batch Processing Example

```python
from shared.utils.error_decorators import aggregate_errors
from shared.utils.exceptions import AudioProcessingError


@aggregate_errors(operation_name="batch_transcription")
async def transcribe_batch(audio_files: List[bytes]):
    results = []
    errors = []

    for i, audio in enumerate(audio_files):
        try:
            result = await transcribe_single(audio)
            results.append({
                "index": i,
                "text": result.text,
                "confidence": result.confidence
            })
        except Exception as e:
            errors.append({
                "index": i,
                "error": str(e),
                "is_retryable": getattr(e, "is_retryable", False)
            })

    return results, errors


# Usage
results, errors = await transcribe_batch(audio_files)

if errors:
    logger.warning(f"Batch processing had {len(errors)} errors")
    # Retry failed items
    retryable = [err for err in errors if err["is_retryable"]]
    if retryable:
        logger.info(f"Retrying {len(retryable)} failed items")
```

## Summary

The new error handling system provides:

1. **Rich exception hierarchy** with context and metadata
2. **Error classification** (transient vs permanent, retryable vs not)
3. **Correlation IDs** for distributed tracing
4. **User-friendly messages** with recovery suggestions
5. **Retry logic** with exponential backoff
6. **Circuit breaker** for fault tolerance
7. **Structured error responses** ready for clients
8. **Comprehensive logging** with context
9. **Type safety** with specific exception types
10. **Production-ready** patterns and best practices

By following this guide, you'll have robust, maintainable error handling across all Morgan services.
