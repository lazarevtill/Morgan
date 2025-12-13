# Logging System Documentation

## Overview

The Morgan Server logging system provides comprehensive structured logging with support for:
- JSON and text output formats
- Log level filtering
- Context fields (request_id, user_id, conversation_id)
- Log rotation
- Multiple output handlers (console, file)

**Validates: Requirements 10.2, 10.4, 10.5**

## Features

### Structured Logging (JSON Format)

All log entries in JSON format include:
- `timestamp`: ISO 8601 timestamp
- `level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logger`: Logger name
- `message`: Log message
- Additional context fields from `extra` parameter

Example JSON log entry:
```json
{
  "timestamp": "2025-12-08T10:30:00Z",
  "level": "INFO",
  "logger": "morgan.api.chat",
  "message": "Chat request processed",
  "request_id": "req_123",
  "user_id": "user_456",
  "conversation_id": "conv_789",
  "response_time_ms": 1234,
  "status_code": 200
}
```

### Log Level Filtering

The system supports standard Python log levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Potentially harmful situations
- **ERROR**: Error events that might still allow the application to continue
- **CRITICAL**: Severe errors that cause the application to abort

Only messages at or above the configured level are logged.

### Context Fields

The logging system supports adding context fields to log records:
- `request_id`: Unique identifier for each request
- `user_id`: User identifier
- `conversation_id`: Conversation identifier
- Custom fields via `extra` parameter

### Log Rotation

File-based logging supports automatic rotation:
- Rotate when file size exceeds configured maximum (default: 100 MB)
- Keep configurable number of backup files (default: 30)
- Automatic compression of old logs

## Usage

### Basic Configuration

```python
from morgan_server.logging_config import configure_logging

# Configure with JSON format
configure_logging(
    log_level="INFO",
    log_format="json",
    log_file="./logs/morgan.log",
    enable_rotation=True,
    max_bytes=100 * 1024 * 1024,  # 100 MB
    backup_count=30,
    enable_console=True
)
```

### Getting a Logger

```python
from morgan_server.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

### Logging with Context

```python
from morgan_server.logging_config import get_logger, setup_request_logging

logger = get_logger(__name__)

# Create context
context = setup_request_logging(
    request_id="req_123",
    user_id="user_456",
    conversation_id="conv_789"
)

# Log with context
logger.info("Processing request", extra=context)
```

### Using LogContext Manager

```python
from morgan_server.logging_config import get_logger, LogContext

logger = get_logger(__name__)

# All logs within context will include these fields
with LogContext(request_id="req_123", user_id="user_456"):
    logger.info("Starting processing")
    logger.info("Processing complete")
```

### Logging Exceptions

```python
from morgan_server.logging_config import get_logger

logger = get_logger(__name__)

try:
    # Some operation
    raise ValueError("Something went wrong")
except ValueError:
    logger.error("An error occurred", exc_info=True)
```

## Configuration Options

### Environment Variables

- `MORGAN_LOG_LEVEL`: Log level (default: INFO)
- `MORGAN_LOG_FORMAT`: Log format - "json" or "text" (default: json)
- `MORGAN_LOG_FILE`: Path to log file (optional)

### Programmatic Configuration

```python
configure_logging(
    log_level="INFO",           # Log level
    log_format="json",          # "json" or "text"
    log_file="./logs/app.log",  # Optional log file path
    enable_rotation=True,       # Enable log rotation
    max_bytes=100*1024*1024,    # Max file size before rotation
    backup_count=30,            # Number of backup files to keep
    enable_console=True         # Log to console
)
```

## Integration with Middleware

The logging system is integrated with the middleware layer:

### Request Logging

All HTTP requests are automatically logged with:
- Request method and path
- Query parameters
- User context (user_id, conversation_id)
- Response status code
- Response time

### Error Logging

All errors are automatically logged with:
- Error type and message
- Stack trace
- Request context
- User context

## Testing

The logging system includes comprehensive tests:

### Property-Based Tests

- **Property 25: Log level filtering** - Verifies that only messages at or above the configured level are logged
- **Property 26: Structured logging format** - Verifies that JSON output is valid and contains all required fields

### Integration Tests

- File logging with rotation
- Console and file logging
- Context field handling
- Exception logging
- Multiple loggers
- Text format logging

## Best Practices

1. **Use appropriate log levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: General informational messages
   - WARNING: Potentially harmful situations
   - ERROR: Error events
   - CRITICAL: Severe errors

2. **Include context fields**:
   ```python
   logger.info("Processing request", extra={
       "request_id": request_id,
       "user_id": user_id,
       "conversation_id": conversation_id
   })
   ```

3. **Log exceptions with stack traces**:
   ```python
   try:
       # operation
   except Exception:
       logger.error("Operation failed", exc_info=True)
   ```

4. **Use structured logging (JSON) in production**:
   - Easier to parse and analyze
   - Better for log aggregation tools
   - Consistent format

5. **Use text format for development**:
   - More human-readable
   - Easier to debug locally

## Performance Considerations

- JSON formatting has minimal overhead
- Log rotation is handled efficiently
- File I/O is buffered
- Context fields are added without copying the entire record

## Troubleshooting

### Logs not appearing

Check that:
1. Log level is set correctly
2. Logger name starts with "morgan"
3. Handlers are configured

### File permission errors

Ensure:
1. Log directory exists and is writable
2. Process has write permissions
3. File is not locked by another process

### JSON parsing errors

Verify:
1. All context field values are JSON-serializable
2. No circular references in context objects
3. Special characters are properly escaped

## Future Enhancements

Potential improvements:
- Async logging for better performance
- Log aggregation integration (ELK, Splunk)
- Distributed tracing support
- Custom log formatters
- Log sampling for high-volume scenarios
