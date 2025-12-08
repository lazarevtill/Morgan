# Morgan CLI

Terminal client for Morgan AI Assistant.

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
