# Changelog

All notable changes to Morgan Server will be documented in this file.

## [0.1.0] - 2025-12-08

### Added

#### Remote Embedding Support
- Added support for remote embedding providers (Ollama, OpenAI-compatible)
- New configuration fields:
  - `embedding_provider`: Choose between `local`, `ollama`, or `openai-compatible`
  - `embedding_endpoint`: Remote embedding service endpoint
  - `embedding_api_key`: API key for remote embedding services
- Validation for remote embedding configuration
- Property-based tests for embedding configuration validation

#### Configuration System
- Comprehensive configuration validation
- Support for YAML, JSON, and .env configuration files
- Environment variable support with `MORGAN_` prefix
- Configuration precedence: env vars > config files > defaults
- Clear error messages for invalid configuration
- Property-based tests for configuration validation (100+ test cases per property)

#### Documentation
- Complete configuration guide (`docs/CONFIGURATION.md`)
- Comprehensive embedding configuration guide (`docs/EMBEDDING_CONFIGURATION.md`)
- Documentation index (`docs/README.md`)
- Updated main README with quick start guide
- Example configuration file (`config.example.yaml`)

#### Testing
- 46 configuration tests (unit + property-based)
- 13 property-based tests for invalid configuration rejection
- 3 property-based tests for configuration format support
- 3 property-based tests for embedding configuration
- All tests use Hypothesis with 100 iterations minimum

### Configuration Options

#### Server Settings
- `host`: Server host address (default: `0.0.0.0`)
- `port`: Server port (default: `8080`)
- `workers`: Number of worker processes (default: `4`)

#### LLM Settings
- `llm_provider`: LLM provider type (default: `ollama`)
- `llm_endpoint`: LLM service endpoint (default: `http://localhost:11434`)
- `llm_model`: LLM model name (default: `llama2`)
- `llm_api_key`: API key for LLM service (optional)

#### Vector Database Settings
- `vector_db_url`: Qdrant URL (default: `http://localhost:6333`)
- `vector_db_api_key`: Vector database API key (optional)

#### Embedding Settings
- `embedding_provider`: Embedding provider (default: `local`)
- `embedding_model`: Embedding model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `embedding_device`: Device for local embeddings (default: `cpu`)
- `embedding_endpoint`: Remote embedding endpoint (required for remote providers)
- `embedding_api_key`: API key for remote embeddings (optional)

#### Cache Settings
- `cache_dir`: Cache directory path (default: `./data/cache`)
- `cache_size_mb`: Maximum cache size in MB (default: `1000`)

#### Logging Settings
- `log_level`: Log level (default: `INFO`)
- `log_format`: Log format (default: `json`)

#### Performance Settings
- `max_concurrent_requests`: Maximum concurrent requests (default: `100`)
- `request_timeout_seconds`: Request timeout (default: `60`)
- `session_timeout_minutes`: Session timeout (default: `60`)

### Validation Rules

- Port must be between 1 and 65535
- LLM provider must be `ollama` or `openai-compatible`
- Embedding provider must be `local`, `ollama`, or `openai-compatible`
- Embedding device must be `cpu`, `cuda`, or `mps`
- Log level must be `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`
- Log format must be `json` or `text`
- URLs must start with `http://` or `https://`
- Remote embedding providers require `embedding_endpoint`
- All numeric values must be positive

### Breaking Changes

None - this is the initial release.

### Migration Guide

Not applicable - this is the initial release.

## [Unreleased]

### Planned Features

- API implementation (chat, memory, knowledge, profile endpoints)
- Empathic Engine (emotional intelligence, personality, relationships)
- Knowledge Engine (RAG, document processing, search)
- Personalization Layer (profiles, preferences, memory)
- LLM client implementations (Ollama, OpenAI-compatible)
- Health checks and monitoring
- Docker deployment
- Kubernetes deployment
- API documentation
- Client implementation

## Version History

- **0.1.0** (2025-12-08) - Initial release with configuration system and remote embedding support
