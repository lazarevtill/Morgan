# Session Management System

## Overview

The session management system provides robust session tracking, isolation, and cleanup for concurrent clients connecting to the Morgan server. It ensures that multiple users can interact with the server simultaneously without data leakage or interference.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

## Features

- **Session Creation and Tracking**: Create and manage client sessions with unique identifiers
- **Session Isolation**: Ensure complete data isolation between sessions
- **Automatic Cleanup**: Expired sessions are automatically cleaned up
- **Request Tracking**: Track active requests within each session
- **Connection Pooling Support**: Designed to work with connection pooling
- **Concurrent Request Handling**: Handle multiple concurrent requests efficiently
- **Graceful Shutdown**: Clean up all sessions on server shutdown

## Architecture

### Session Model

Each session contains:
- `session_id`: Unique session identifier (UUID)
- `user_id`: Optional user identifier
- `conversation_id`: Optional conversation identifier
- `created_at`: Session creation timestamp
- `last_activity`: Last activity timestamp
- `metadata`: Custom session metadata
- `connection_count`: Number of active connections
- `active_requests`: Set of active request IDs

### SessionManager

The `SessionManager` class provides:
- Session lifecycle management (create, get, delete)
- Automatic cleanup of expired sessions
- Request tracking with context managers
- Metrics and monitoring
- Thread-safe operations with async locks

## Usage

### Basic Session Creation

```python
from morgan_server.session import initialize_session_manager, get_session_manager

# Initialize session manager
manager = initialize_session_manager(
    session_timeout_minutes=60,
    cleanup_interval_seconds=300,
    max_concurrent_requests=100
)

# Start the manager
await manager.start()

# Create a session
session = await manager.create_session(
    user_id="user-123",
    conversation_id="conv-456",
    metadata={"client_type": "web"}
)

print(f"Session ID: {session.session_id}")
```

### Retrieving Sessions

```python
# Get session by ID
session = await manager.get_session(session_id)

# Get session by user ID
session = await manager.get_session_by_user(user_id)
```

### Tracking Requests

Use the context manager to track requests within a session:

```python
async with manager.track_request(session_id) as session:
    # Process request
    # Session automatically tracks this as an active request
    result = await process_chat_message(session, message)
    
# Request is automatically removed when context exits
```

### Session Cleanup

```python
# Manual cleanup
await manager.delete_session(session_id)

# Automatic cleanup happens based on:
# - Session timeout (inactive sessions)
# - Server shutdown (all sessions)
```

### Integration with FastAPI

```python
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.on_event("startup")
async def startup():
    manager = initialize_session_manager()
    await manager.start()
    app.state.session_manager = manager

@app.on_event("shutdown")
async def shutdown():
    await app.state.session_manager.stop()

@app.post("/api/chat")
async def chat(request: Request, session_id: str, message: str):
    manager = request.app.state.session_manager
    
    # Track request within session
    async with manager.track_request(session_id) as session:
        # Process message
        response = await process_message(session, message)
        return {"response": response}
```

## Configuration

### Session Manager Parameters

- `session_timeout_minutes` (default: 60): How long before inactive sessions expire
- `cleanup_interval_seconds` (default: 300): How often to run cleanup task
- `max_concurrent_requests` (default: 100): Maximum concurrent requests allowed

### Environment Variables

Configure via `ServerConfig`:

```python
MORGAN_SESSION_TIMEOUT=60  # Session timeout in minutes
MORGAN_MAX_CONCURRENT=100  # Max concurrent requests
```

## Session Lifecycle

### 1. Creation

```
User connects → Create session → Store in manager → Return session ID
```

### 2. Active Use

```
Request arrives → Track in session → Process → Remove from tracking
```

### 3. Expiration

```
Inactive > timeout → Cleanup task detects → Remove session
```

### 4. Manual Cleanup

```
User disconnects → Delete session → Clean up resources
```

## Isolation Guarantees

The session management system provides strong isolation guarantees:

1. **Data Isolation**: Session data never crosses session boundaries
2. **Request Isolation**: Requests are tracked per-session
3. **Cleanup Isolation**: Cleaning up one session doesn't affect others
4. **Concurrent Safety**: Thread-safe operations with async locks

## Monitoring and Metrics

### Get Session Metrics

```python
metrics = manager.get_metrics()

# Returns:
{
    "active_sessions": 10,
    "total_sessions_created": 100,
    "total_sessions_expired": 50,
    "total_sessions_cleaned": 60,
    "session_timeout_minutes": 60,
    "max_concurrent_requests": 100
}
```

### Integration with Health System

```python
from morgan_server.health import get_health_system

health_system = get_health_system()

# Track session creation
session = await manager.create_session(user_id="user-123")
health_system.increment_active_sessions()

# Track session deletion
await manager.delete_session(session_id)
health_system.decrement_active_sessions()
```

## Best Practices

### 1. Always Use Context Manager for Requests

```python
# Good
async with manager.track_request(session_id) as session:
    await process_request(session)

# Bad - request not tracked
session = await manager.get_session(session_id)
await process_request(session)
```

### 2. Handle Session Not Found

```python
session = await manager.get_session(session_id)
if not session:
    raise HTTPException(status_code=404, detail="Session not found")
```

### 3. Clean Up on Disconnect

```python
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        # Handle messages
        pass
    finally:
        # Clean up session on disconnect
        await manager.delete_session(session_id)
```

### 4. Set Appropriate Timeouts

```python
# For interactive applications
manager = initialize_session_manager(session_timeout_minutes=30)

# For long-running tasks
manager = initialize_session_manager(session_timeout_minutes=120)
```

## Error Handling

### Session Not Found

```python
try:
    async with manager.track_request(session_id) as session:
        pass
except ValueError as e:
    # Session not found
    raise HTTPException(status_code=404, detail=str(e))
```

### Concurrent Request Limit

The session manager doesn't enforce the max_concurrent_requests limit directly. This should be handled at the application level:

```python
if manager.get_active_session_count() >= max_concurrent_requests:
    raise HTTPException(
        status_code=503,
        detail="Server at capacity, please try again later"
    )
```

## Testing

### Unit Tests

See `tests/test_session_management.py` for comprehensive unit tests covering:
- Session creation and retrieval
- Session isolation
- Request tracking
- Cleanup behavior
- Concurrent operations

### Property-Based Tests

See `tests/test_session_properties.py` for property-based tests:
- **Property 12**: Concurrent request handling
- **Property 14**: Session cleanup isolation

Run tests:

```bash
# Unit tests
pytest tests/test_session_management.py -v

# Property-based tests
pytest tests/test_session_properties.py -v
```

## Examples

See `examples/session_management_example.py` for complete working examples:
1. Basic session usage
2. Request tracking
3. Concurrent request handling
4. Session cleanup
5. FastAPI integration

Run examples:

```bash
python examples/session_management_example.py
```

## Performance Considerations

### Memory Usage

- Each session stores minimal data (< 1KB typically)
- Active requests are tracked as string IDs
- Cleanup task runs periodically to free memory

### Concurrency

- Async locks protect shared state
- Operations are non-blocking
- Designed for high concurrency

### Scalability

For single-server deployment:
- Handles 100+ concurrent sessions easily
- Cleanup task is lightweight
- No external dependencies

For multi-server deployment:
- Consider using Redis or similar for shared session storage
- Current implementation is in-memory only

## Troubleshooting

### Sessions Not Being Cleaned Up

Check:
1. Is cleanup task running? (manager should be started)
2. Are sessions expired? (check last_activity timestamp)
3. Do sessions have active requests? (they won't be cleaned up)

### Session Not Found Errors

Check:
1. Was session created successfully?
2. Has session expired?
3. Was session manually deleted?

### High Memory Usage

Check:
1. Are sessions being cleaned up properly?
2. Is session timeout too long?
3. Are there memory leaks in session metadata?

## Future Enhancements

Potential improvements:
1. Redis-backed session storage for multi-server deployments
2. Session persistence across server restarts
3. Session migration between servers
4. Advanced metrics and monitoring
5. Rate limiting per session
6. Session priority levels

## References

- Requirements: 6.1, 6.2, 6.3, 6.5
- Design Document: Section on Session Management
- API Models: `morgan_server/api/models.py`
- Health System: `morgan_server/health.py`
