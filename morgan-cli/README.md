# Morgan CLI

Terminal client for Morgan AI Assistant - a lightweight TUI client that communicates with the Morgan server via HTTP/WebSocket APIs.

## Features

- **HTTP Client**: REST API calls for all Morgan server endpoints
- **WebSocket Client**: Real-time chat with streaming responses
- **Connection Management**: Automatic retry logic and connection status tracking
- **Error Handling**: Graceful error handling with informative messages
- **Configuration**: Flexible configuration via environment variables or code

## Installation

```bash
pip install morgan-cli
```

## Quick Start

### Using HTTP Client

```python
import asyncio
from morgan_cli.client import MorganClient, ClientConfig

async def main():
    config = ClientConfig(
        server_url="http://localhost:8080",
        user_id="my_user"
    )
    
    async with MorganClient(config) as client:
        # Send a chat message
        response = await client.http.chat("Hello, Morgan!")
        print(response["answer"])
        
        # Get user profile
        profile = await client.http.get_profile()
        print(f"Trust level: {profile['trust_level']}")
        
        # Check server health
        health = await client.http.health_check()
        print(f"Server status: {health['status']}")

asyncio.run(main())
```

### Using WebSocket Client

```python
import asyncio
from morgan_cli.client import MorganClient, ClientConfig

async def main():
    config = ClientConfig(
        server_url="http://localhost:8080",
        user_id="my_user"
    )
    
    client = MorganClient(config)
    
    # Connect WebSocket
    await client.ws.connect()
    
    # Send a message
    await client.ws.send_message("Tell me a story")
    
    # Receive responses
    async for message in client.ws.receive_messages():
        print(message["answer"])
        break  # Exit after first response
    
    await client.close()

asyncio.run(main())
```

## Configuration

The client can be configured using `ClientConfig`:

```python
from morgan_cli.client import ClientConfig

config = ClientConfig(
    server_url="http://localhost:8080",  # Morgan server URL
    api_key=None,                         # Optional API key
    user_id="my_user",                    # Optional user identifier
    timeout_seconds=60,                   # Request timeout
    retry_attempts=3,                     # Number of retry attempts
    retry_delay_seconds=2                 # Delay between retries
)
```

### Environment Variables

You can also configure the client using environment variables:

- `MORGAN_SERVER_URL`: Server URL (default: http://localhost:8080)
- `MORGAN_API_KEY`: API key (optional)
- `MORGAN_USER_ID`: User identifier (optional)
- `MORGAN_TIMEOUT`: Request timeout in seconds (default: 60)
- `MORGAN_RETRY_ATTEMPTS`: Number of retry attempts (default: 3)
- `MORGAN_RETRY_DELAY`: Delay between retries in seconds (default: 2)

## API Methods

### Chat

```python
response = await client.http.chat(
    message="Hello!",
    user_id="optional_user_id",
    conversation_id="optional_conv_id",
    metadata={"key": "value"}
)
```

### Memory

```python
# Get memory statistics
stats = await client.http.get_memory_stats()

# Search conversation history
results = await client.http.search_memory("search query", limit=10)

# Clean up old conversations
result = await client.http.cleanup_memory()
```

### Knowledge

```python
# Add documents to knowledge base
result = await client.http.learn(
    source="/path/to/document.pdf",
    doc_type="pdf"
)

# Search knowledge base
results = await client.http.search_knowledge("query", limit=10)

# Get knowledge statistics
stats = await client.http.get_knowledge_stats()
```

### Profile

```python
# Get user profile
profile = await client.http.get_profile()

# Update preferences
updated = await client.http.update_profile(
    communication_style="friendly",
    response_length="detailed",
    topics_of_interest=["AI", "Python"],
    preferred_name="Alex"
)

# Get timeline
timeline = await client.http.get_timeline()
```

### Health & Status

```python
# Quick health check
health = await client.http.health_check()

# Detailed status
status = await client.http.get_status()
```

## Connection Status

Track connection status with callbacks:

```python
from morgan_cli.client import ConnectionStatus

def on_status_change(status: ConnectionStatus):
    print(f"Connection status: {status.value}")

client.http.add_status_callback(on_status_change)
client.ws.add_status_callback(on_status_change)
```

## Error Handling

The client provides specific exceptions for different error scenarios:

```python
from morgan_cli.client import (
    ConnectionError,
    RequestError,
    TimeoutError,
    ValidationError
)

try:
    response = await client.http.chat("Hello!")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except RequestError as e:
    print(f"Request failed: {e} (status: {e.status_code})")
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=morgan_cli
```

### Property-Based Tests

The client includes comprehensive property-based tests using Hypothesis:

```bash
pytest tests/test_client_properties.py -v
```

## License

MIT.

## Features

- **Rich Terminal UI**: Beautiful markdown rendering with Rich
- **Interactive Chat**: Real-time chat via WebSocket
- **Command History**: Navigate previous commands with arrow keys
- **Multiple Commands**: Chat, ask, learn, memory, knowledge, health
- **Flexible Configuration**: Connect to any Morgan server instance

## Installation

```bash
cd morgan-cli
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file or set environment variables:

```bash
MORGAN_SERVER_URL=http://localhost:8080
MORGAN_USER_ID=my-user-id
```

Or pass via command line:

```bash
morgan --server http://localhost:8080 chat
```

## Usage

### Interactive Chat

```bash
morgan chat
```

### Single Question

```bash
morgan ask "What is the weather today?"
```

### Learn from Documents

```bash
morgan learn --file document.pdf
morgan learn --url https://example.com/article
```

### Memory Management

```bash
morgan memory stats
morgan memory search "keyword"
morgan memory cleanup
```

### Knowledge Base

```bash
morgan knowledge search "query"
morgan knowledge stats
```

### Health Check

```bash
morgan health
```

## Commands

- `chat` - Start interactive chat session
- `ask` - Ask a single question
- `learn` - Add documents to knowledge base
- `memory` - Manage conversation memory
- `knowledge` - Search and manage knowledge base
- `health` - Check server health

## License

MIT
