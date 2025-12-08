# Morgan Server Documentation

Welcome to the Morgan Server documentation. This directory contains comprehensive guides for configuring, deploying, and using Morgan Server.

## Documentation Index

### Getting Started

- **[Main README](../README.md)** - Quick start and overview
- **[Configuration Guide](CONFIGURATION.md)** - Complete configuration reference
- **[Embedding Configuration](EMBEDDING_CONFIGURATION.md)** - Embedding provider setup

### Configuration Guides

- **[Configuration Guide](CONFIGURATION.md)**
  - Configuration sources and precedence
  - Environment variables reference
  - Configuration file formats (YAML, JSON, .env)
  - Validation rules
  - Example configurations
  - Troubleshooting

- **[Embedding Configuration](EMBEDDING_CONFIGURATION.md)**
  - Local embeddings (sentence-transformers)
  - Remote Ollama embeddings
  - OpenAI-compatible embeddings
  - Model recommendations
  - Performance considerations
  - Migration guide
  - Troubleshooting

### Deployment (Coming Soon)

- **Deployment Guide** - Docker, Kubernetes, bare metal
- **Security Guide** - Best practices for production
- **Monitoring Guide** - Metrics, logging, alerting

### API Reference (Coming Soon)

- **API Documentation** - REST and WebSocket endpoints
- **Client Guide** - Using the TUI client
- **Integration Guide** - Building custom clients

## Quick Links

### Configuration

- [Server Settings](CONFIGURATION.md#server-settings)
- [LLM Configuration](CONFIGURATION.md#llm-settings)
- [Vector Database](CONFIGURATION.md#vector-database-settings)
- [Embedding Providers](EMBEDDING_CONFIGURATION.md#embedding-providers)
- [Cache Settings](CONFIGURATION.md#cache-settings)
- [Logging](CONFIGURATION.md#logging-settings)

### Embedding Providers

- [Local Embeddings](EMBEDDING_CONFIGURATION.md#1-local-embeddings-default)
- [Remote Ollama](EMBEDDING_CONFIGURATION.md#2-remote-ollama-embeddings)
- [OpenAI-Compatible](EMBEDDING_CONFIGURATION.md#3-openai-compatible-embeddings)

### Examples

- [Development Setup](CONFIGURATION.md#development-local-everything)
- [Self-Hosted Production](CONFIGURATION.md#production-self-hosted)
- [Cloud Production](CONFIGURATION.md#production-cloud-services)
- [Docker Compose](CONFIGURATION.md#docker-compose)

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Contributing**: See CONTRIBUTING.md (coming soon)

## Version

This documentation is for Morgan Server v0.1.0.
