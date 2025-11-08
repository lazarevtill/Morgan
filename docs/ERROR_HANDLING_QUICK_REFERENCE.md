# Error Handling Quick Reference

Quick reference card for using Morgan's production-grade error handling system.

## Import Statements

```python
# Exceptions
from shared.utils.exceptions import (
    # Service errors
    ServiceUnavailableError,
    ServiceTimeoutError,

    # Model errors
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
    ModelOutOfMemoryError,

    # Audio errors
    AudioFormatError,
    AudioProcessingError,

    # Validation errors
    ValidationError,
    InvalidInputError,
    MissingRequiredFieldError,

    # Configuration errors
    ConfigMissingError,
    ConfigInvalidError,

    # Resource errors
    ResourceExhaustedError,
    QuotaExceededError,
    GPUOutOfMemoryError,

    # Network errors
    NetworkConnectionError,
    NetworkTimeoutError,

    # Database errors
    DatabaseConnectionError,
    DatabaseQueryError
)

# Decorators
from shared.utils.error_decorators import (
    retry_on_transient_error,
    with_circuit_breaker,
    with_timeout,
    handle_errors,
    aggregate_errors,
    with_fallback
)

# Error handling utilities
from shared.utils.error_handling import (
    ErrorHandler,
    ErrorResponse,
    ErrorContext,
    create_error_context
)
```

## Common Patterns

### 1. Service Unavailable

```python
raise ServiceUnavailableError(
    service_name="llm",
    message="LLM service is temporarily unavailable"
)
```

### 2. Model Not Found

```python
raise ModelNotFoundError(model_name="whisper-large-v3")
```

### 3. Model Inference Failed

```python
raise ModelInferenceError(
    model_name="csm-1b",
    reason="CUDA out of memory"
)
```

### 4. Invalid Input

```python
raise InvalidInputError(
    message=f"Text length {len(text)} exceeds maximum {max_len}",
    input_name="text"
)
```

### 5. Missing Configuration

```python
raise ConfigMissingError(config_key="OPENAI_API_KEY")
```

## Decorator Usage

### Retry on Transient Errors

```python
@retry_on_transient_error(max_attempts=3, base_delay=1.0)
async def fetch_data():
    return await api.get()
```

### Circuit Breaker

```python
@with_circuit_breaker(name="llm_service", failure_threshold=5)
async def call_llm(prompt: str):
    return await llm.generate(prompt)
```

### Timeout

```python
@with_timeout(timeout_seconds=30.0)
async def slow_operation():
    return await long_running_task()
```

### Combine Multiple Decorators

```python
@retry_on_transient_error(max_attempts=3)
@with_circuit_breaker(name="external_api")
@with_timeout(timeout_seconds=30.0)
async def robust_api_call():
    return await external_api.call()
```

### Fallback

```python
async def fallback_response():
    return "Service temporarily unavailable"

@with_fallback(fallback_response)
async def get_response():
    return await primary_service()
```

## Error Handling

### Basic Handler

```python
error_handler = ErrorHandler(
    logger_instance=logger,
    service_name="tts"
)

try:
    result = await risky_operation()
except Exception as e:
    context = create_error_context(
        user_id="user123",
        operation="generate_speech"
    )
    error_response = error_handler.handle_error(e, context)
    return error_response
```

### API Endpoint Pattern

```python
from fastapi import FastAPI, HTTPException

@app.post("/api/endpoint")
async def endpoint(request: Request):
    correlation_id = str(uuid.uuid4())

    try:
        result = await service.process(request, correlation_id=correlation_id)

        return ErrorResponse.success(
            data=result,
            correlation_id=correlation_id
        )

    except MorganException as e:
        error_response = ErrorResponse.from_exception(e)
        http_status = error_response["error"]["http_status"]
        raise HTTPException(status_code=http_status, detail=error_response)

    except Exception as e:
        error_response = ErrorResponse.from_exception(e, correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=error_response)
```

## Batch Operations

```python
@aggregate_errors(operation_name="batch_process")
async def process_batch(items):
    results = []
    errors = []

    for item in items:
        try:
            result = await process(item)
            results.append(result)
        except Exception as e:
            errors.append({"item": item, "error": str(e)})

    return results, errors

# Usage
results, errors = await process_batch(items)
if errors:
    logger.warning(f"Processing had {len(errors)} errors")
```

## Exception Properties

Every `MorganException` has:

```python
exception.message           # Technical message
exception.user_message      # User-friendly message
exception.category          # ErrorCategory enum
exception.severity          # ErrorSeverity enum
exception.is_transient      # bool - temporary error?
exception.is_retryable      # bool - safe to retry?
exception.correlation_id    # str - for tracing
exception.context           # dict - structured metadata
exception.recovery_suggestions  # list - actionable advice
exception.cause             # Exception - original error
exception.timestamp         # datetime - when error occurred
```

## Response Format

### Error Response

```json
{
  "error": {
    "message": "Technical error message",
    "user_message": "User-friendly message",
    "category": "error_category",
    "severity": "error",
    "correlation_id": "uuid",
    "is_transient": true,
    "is_retryable": true,
    "http_status": 503,
    "timestamp": "2025-11-08T12:34:56.789Z",
    "context": { },
    "recovery_suggestions": ["suggestion 1", "suggestion 2"]
  }
}
```

### Success Response

```json
{
  "success": true,
  "data": { },
  "metadata": { },
  "correlation_id": "uuid",
  "timestamp": "2025-11-08T12:34:56.789Z"
}
```

## Best Practices Checklist

- ✅ Use specific exception types, not generic Exception
- ✅ Never use bare `except:` clauses
- ✅ Always provide error context
- ✅ Include correlation IDs for tracing
- ✅ Classify errors correctly (transient vs permanent)
- ✅ Add user-friendly messages
- ✅ Provide recovery suggestions
- ✅ Use decorators for cross-cutting concerns
- ✅ Log with structured data
- ✅ Handle batch operations properly

## Anti-Patterns to Avoid

### ❌ Bare Except
```python
try:
    do_something()
except:
    pass
```

### ❌ Generic Exception
```python
try:
    result = await model.generate()
except Exception as e:
    logger.error(f"Error: {e}")
```

### ❌ Swallowing Errors
```python
try:
    important_operation()
except Exception:
    return None  # Error lost!
```

### ❌ No Context
```python
raise ValueError("Invalid input")
```

## Migration Examples

### Before → After

#### Generic Exception
```python
# Before
try:
    result = await model.generate()
except Exception as e:
    logger.error(f"Error: {e}")
    raise

# After
from shared.utils.exceptions import ModelInferenceError

try:
    result = await model.generate()
except RuntimeError as e:
    raise ModelInferenceError(
        model_name="my_model",
        reason=str(e),
        correlation_id=correlation_id
    )
```

#### Manual Retry
```python
# Before
max_retries = 3
for attempt in range(max_retries):
    try:
        return await api.get()
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        await asyncio.sleep(2 ** attempt)

# After
@retry_on_transient_error(max_attempts=3, base_delay=1.0)
async def fetch_data():
    return await api.get()
```

## HTTP Status Codes

| Exception Type | HTTP Status |
|---------------|-------------|
| `ValidationException` | 400 Bad Request |
| `ConfigurationException` | 500 Internal Server Error |
| `ResourceException` (Quota) | 429 Too Many Requests |
| `ResourceException` (Other) | 507 Insufficient Storage |
| `ServiceException` (Timeout) | 504 Gateway Timeout |
| `ServiceException` (Other) | 503 Service Unavailable |
| `NetworkException` | 503 Service Unavailable |
| `ModelException` | 503 Service Unavailable |
| `AudioException` | 422 Unprocessable Entity |
| `DatabaseException` | 503 Service Unavailable |
| `ExternalIntegrationException` | 502 Bad Gateway |
| Default | 500 Internal Server Error |

## Logging Example

```python
from shared.utils.error_handling import ErrorLogger, ErrorContext

logger = ErrorLogger(service_name="tts")
context = ErrorContext(
    correlation_id="abc123",
    user_id="user456",
    operation="generate_speech"
)

try:
    result = await generate()
except Exception as e:
    logger.log_exception(
        exception=e,
        context=context,
        additional_data={"text_length": len(text)}
    )
    raise
```

## Testing Exceptions

```python
import pytest
from shared.utils.exceptions import ModelInferenceError

def test_model_error():
    with pytest.raises(ModelInferenceError) as exc_info:
        raise ModelInferenceError(
            model_name="test",
            reason="test error"
        )

    error = exc_info.value
    assert error.model_name == "test"
    assert error.is_transient == True
    assert error.is_retryable == True
    assert error.correlation_id is not None
```

## Documentation Links

- Full Guide: `docs/ERROR_HANDLING_GUIDE.md`
- Implementation Summary: `ERROR_HANDLING_IMPROVEMENTS.md`
- Exception Reference: `shared/utils/exceptions.py`
- Decorator Reference: `shared/utils/error_decorators.py`
- Handler Reference: `shared/utils/error_handling.py`

---

**Remember:** Good error handling improves reliability, debuggability, and user experience!
