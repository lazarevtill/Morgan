# Morgan Documentation Index

Complete documentation for the Morgan AI Assistant client-server architecture.

> **⚠️ Important:** The old monolithic system in `morgan-rag/` and `cli.py.old` is **DEPRECATED**. See [DEPRECATION_NOTICE.md](./DEPRECATION_NOTICE.md) for details and [MIGRATION.md](./MIGRATION.md) for migration instructions.

## Quick Start

- **[Server Quick Start](./morgan-server/README.md#quick-start)** - Get the server running in 5 minutes
- **[Client Quick Start](./morgan-cli/README.md#quick-start)** - Start chatting with Morgan
- **[Docker Quick Start](./docker/README.md#quick-start)** - Deploy with Docker Compose

## Core Documentation

### Server

- **[Server README](./morgan-server/README.md)** - Overview, installation, and quick start
- **[Configuration Guide](./morgan-server/docs/CONFIGURATION.md)** - Complete configuration reference
- **[Embedding Configuration](./morgan-server/docs/EMBEDDING_CONFIGURATION.md)** - Embedding provider setup
- **[Deployment Guide](./morgan-server/docs/DEPLOYMENT.md)** - Docker and bare metal deployment
- **[API Documentation](./morgan-server/docs/API.md)** - REST and WebSocket API reference

### Client

- **[Client README](./morgan-cli/README.md)** - Overview, installation, and usage
- **[Client API Reference](./morgan-cli/README.md#api-methods)** - Python client library

### Docker

- **[Docker README](./docker/README.md)** - Docker deployment guide

### Migration

- **[Migration Guide](./MIGRATION.md)** - Migrating from old Morgan system

## Documentation by Topic

### Getting Started

1. **[Installation](./morgan-server/README.md#installation)** - Install server and client
2. **[Configuration](./morgan-server/docs/CONFIGURATION.md)** - Configure for your environment
3. **[First Deployment](./docker/README.md#quick-start)** - Deploy with Docker Compose
4. **[First Chat](./morgan-cli/README.md#quick-start)** - Start chatting with Morgan

### Configuration

- **[Server Settings](./morgan-server/docs/CONFIGURATION.md#server-settings)** - Host, port, workers
- **[LLM Configuration](./morgan-server/docs/CONFIGURATION.md#llm-settings)** - Ollama, OpenAI-compatible
- **[Vector Database](./morgan-server/docs/CONFIGURATION.md#vector-database-settings)** - Qdrant setup
- **[Embedding Providers](./morgan-server/docs/EMBEDDING_CONFIGURATION.md)** - Local, Ollama, OpenAI
- **[Cache Settings](./morgan-server/docs/CONFIGURATION.md#cache-settings)** - Cache configuration
- **[Logging](./morgan-server/docs/CONFIGURATION.md#logging-settings)** - Log levels and formats
- **[Performance](./morgan-server/docs/CONFIGURATION.md#performance-settings)** - Tuning options

### Deployment

- **[Docker Compose](./docker/README.md#basic-deployment)** - Full stack deployment
- **[Docker Standalone](./morgan-server/docs/DEPLOYMENT.md#option-2-docker-server-only)** - Server-only container
- **[Bare Metal Linux](./morgan-server/docs/DEPLOYMENT.md#bare-metal-deployment)** - Direct installation
- **[Systemd Service](./morgan-server/docs/DEPLOYMENT.md#step-5-create-systemd-service-linux)** - Linux service
- **[Production Setup](./morgan-server/docs/DEPLOYMENT.md#production-considerations)** - Security, HA, monitoring

### API Usage

- **[Chat API](./morgan-server/docs/API.md#chat-endpoints)** - Send messages and get responses
- **[Memory API](./morgan-server/docs/API.md#memory-endpoints)** - Conversation history
- **[Knowledge API](./morgan-server/docs/API.md#knowledge-endpoints)** - Document ingestion and search
- **[Profile API](./morgan-server/docs/API.md#profile-endpoints)** - User preferences
- **[System API](./morgan-server/docs/API.md#system-endpoints)** - Health checks and metrics
- **[WebSocket API](./morgan-server/docs/API.md#websocket-api)** - Real-time chat

### Client Usage

- **[HTTP Client](./morgan-cli/README.md#using-http-client)** - REST API calls
- **[WebSocket Client](./morgan-cli/README.md#using-websocket-client)** - Real-time chat
- **[Configuration](./morgan-cli/README.md#configuration)** - Client configuration
- **[Error Handling](./morgan-cli/README.md#error-handling)** - Handle errors gracefully

### Operations

- **[Health Monitoring](./morgan-server/docs/DEPLOYMENT.md#health-monitoring)** - Check system health
- **[Log Management](./morgan-server/docs/DEPLOYMENT.md#log-management)** - Logs and rotation
- **[Backup and Recovery](./morgan-server/docs/DEPLOYMENT.md#backup-and-recovery)** - Data backup
- **[Updates](./morgan-server/docs/DEPLOYMENT.md#updates-and-maintenance)** - Update server
- **[Troubleshooting](./morgan-server/docs/DEPLOYMENT.md#troubleshooting)** - Common issues

### Advanced Topics

- **[Security](./morgan-server/docs/DEPLOYMENT.md#security)** - Security best practices
- **[Performance Tuning](./morgan-server/docs/DEPLOYMENT.md#performance-optimization)** - Optimize performance
- **[High Availability](./morgan-server/docs/DEPLOYMENT.md#high-availability)** - HA setup
- **[Monitoring](./morgan-server/docs/DEPLOYMENT.md#monitoring-and-maintenance)** - Metrics and alerting

## Architecture

### Overview

Morgan uses a clean client-server architecture:

```
┌─────────────┐
│   Clients   │
│  (TUI, Web) │
└──────┬──────┘
       │ HTTP/WebSocket
       │
┌──────▼──────────────────────┐
│     Morgan Server           │
│  ┌────────────────────────┐ │
│  │   API Gateway          │ │
│  │   (FastAPI)            │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │  Empathic Engine       │ │
│  │  - Emotional Intel     │ │
│  │  - Personality         │ │
│  │  - Relationships       │ │
│  └────────────────────────┘ │
│  ┌────────────────────────┐ │
│  │  Knowledge Engine      │ │
│  │  - RAG System          │ │
│  │  - Vector Search       │ │
│  │  - Doc Processing      │ │
│  └────────────────────────┘ │
│  ┌────────────────────────┐ │
│  │  Personalization       │ │
│  │  - User Profiles       │ │
│  │  - Preferences         │ │
│  │  - Memory              │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
       │
       │
┌──────▼──────────────────────┐
│   External Services         │
│  - Ollama (LLM)             │
│  - Qdrant (Vector DB)       │
└─────────────────────────────┘
```

### Components

- **[Server Architecture](./morgan-server/README.md#architecture)** - Server components
- **[API Layer](./morgan-server/docs/API.md)** - REST and WebSocket APIs
- **[Empathic Engine](./morgan-server/README.md#features)** - Emotional intelligence
- **[Knowledge Engine](./morgan-server/README.md#features)** - RAG and search
- **[Personalization](./morgan-server/README.md#features)** - User profiles and preferences

## Examples

### Configuration Examples

- **[Development Setup](./morgan-server/docs/CONFIGURATION.md#development-local-everything)** - Local development
- **[Self-Hosted Production](./morgan-server/docs/CONFIGURATION.md#production-self-hosted)** - Self-hosted deployment
- **[Cloud Production](./morgan-server/docs/CONFIGURATION.md#production-cloud-services)** - Cloud services
- **[Docker Compose](./morgan-server/docs/CONFIGURATION.md#docker-compose)** - Docker setup

### API Examples

- **[Chat Example](./morgan-server/docs/API.md#post-apichat)** - Send a message
- **[Knowledge Example](./morgan-server/docs/API.md#post-apiknowledgelearn)** - Add documents
- **[WebSocket Example](./morgan-server/docs/API.md#ws-wsuser_id)** - Real-time chat
- **[Python Client Example](./morgan-cli/README.md#using-http-client)** - Use Python client

### Deployment Examples

- **[Docker Compose](./docker/README.md#basic-deployment)** - Full stack
- **[Bare Metal](./morgan-server/docs/DEPLOYMENT.md#bare-metal-deployment)** - Direct installation
- **[Systemd Service](./morgan-server/docs/DEPLOYMENT.md#step-5-create-systemd-service-linux)** - Linux service

## Troubleshooting

### Common Issues

- **[Server Won't Start](./morgan-server/docs/DEPLOYMENT.md#server-wont-start)** - Startup issues
- **[Connection Issues](./morgan-server/docs/DEPLOYMENT.md#connection-issues)** - Can't connect to services
- **[Performance Issues](./morgan-server/docs/DEPLOYMENT.md#performance-issues)** - Slow responses
- **[Data Issues](./morgan-server/docs/DEPLOYMENT.md#data-issues)** - Lost data or search problems

### Configuration Issues

- **[Configuration Not Loading](./morgan-server/docs/CONFIGURATION.md#configuration-not-loading)** - Config file issues
- **[Environment Variables](./morgan-server/docs/CONFIGURATION.md#environment-variables-not-working)** - Env var problems
- **[Validation Errors](./morgan-server/docs/CONFIGURATION.md#validation-errors)** - Invalid configuration

### Docker Issues

- **[Service Won't Start](./docker/README.md#service-wont-start)** - Docker startup issues
- **[Can't Connect to Ollama](./docker/README.md#cant-connect-to-ollama)** - Ollama connection
- **[Qdrant Issues](./docker/README.md#qdrant-connection-issues)** - Vector DB problems

## Support

### Getting Help

- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share ideas
- **Documentation** - Search this documentation
- **Logs** - Check server logs for errors

### Contributing

- **Contributing Guide** - How to contribute (coming soon)
- **Development Setup** - Set up development environment
- **Testing** - Run tests and add new ones
- **Code Style** - Follow code style guidelines

## Deprecated Documentation

The following documentation is for the old system and is deprecated:

- **[Old CI/CD Guide](./docs/CI_CD.md)** - For old system (deprecated)
- **[Old Error Handling Guide](./docs/ERROR_HANDLING_GUIDE.md)** - For old system (deprecated)
- **[Old Error Handling Reference](./docs/ERROR_HANDLING_QUICK_REFERENCE.md)** - For old system (deprecated)
- **[Old Morgan RAG README](./morgan-rag/README.md)** - Old system overview (deprecated)

For current documentation, see the sections above.

See **[DEPRECATION_NOTICE.md](./DEPRECATION_NOTICE.md)** for complete deprecation information.

## Version

This documentation is for Morgan v0.1.0 (new client-server architecture).

Last updated: December 8, 2025
