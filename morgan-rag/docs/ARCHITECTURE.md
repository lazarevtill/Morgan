# Morgan RAG - Clean Modular Architecture

## Overview

Morgan RAG follows KISS (Keep It Simple, Stupid) principles with a clean modular architecture where each module has a single, focused responsibility.

## Module Structure

```
morgan/
├── models/          # Unified model management (local + remote)
│   ├── manager.py   # Central model coordination
│   ├── local.py     # Local model integration (Ollama, Transformers)
│   ├── lazarev.py   # gpt.lazarev.cloud endpoint integration
│   ├── cache.py     # Model caching and optimization
│   └── selector.py  # Model selection logic
│
├── storage/         # Unified data persistence
│   ├── vector.py    # Vector database operations
│   ├── memory.py    # Conversation and emotional memory storage
│   ├── profile.py   # User profiles and preferences storage
│   ├── cache.py     # Performance caching and optimization
│   └── backup.py    # Data backup and recovery operations
│
├── config/          # Centralized configuration management
│   └── settings.py  # Core settings management and validation
│
├── utils/           # Shared utilities following DRY principles
│   ├── logger.py    # Centralized logging configuration
│   ├── error_handling.py # Error handling and recovery
│   ├── validators.py # Input validation utilities
│   ├── cache.py     # Caching utilities
│   ├── model_helpers.py # Model utility functions
│   └── storage_helpers.py # Storage utility functions
│
├── core/            # Core orchestration and assistant logic
├── emotional/       # Emotional intelligence engine
├── companion/       # Relationship management
├── memory/          # Conversation memory processing
├── search/          # Multi-stage search engine
├── vectorization/   # Hierarchical embeddings
├── jina/            # Jina AI integration
├── background/      # Background processing
├── ingestion/       # Document processing
├── caching/         # Existing caching infrastructure
├── vector_db/       # Vector database client
├── monitoring/      # System monitoring
└── interfaces/      # User interfaces
```

## Design Principles

### KISS Architecture Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Clean Interfaces**: Simple, well-defined APIs between modules
3. **Dependency Injection**: Modules depend on abstractions, not implementations
4. **Configuration-Driven**: Behavior controlled through configuration, not code changes
5. **Local-First**: All processing happens locally, no external API dependencies (except gpt.lazarev.cloud)

### Module Guidelines

- **Composition over Inheritance**: Use simple base classes with composition
- **Minimal Data Structures**: Avoid over-engineering with too many specialized classes
- **Clear Relationships**: Simple, obvious data relationships
- **Standard Types**: Use built-in Python types where possible
- **Privacy-Focused**: All data structures support local-only processing

## Requirements Addressed

This clean module structure addresses the following requirements:

- **23.1**: Complete local operation with modular architecture
- **23.2**: Local model management through models/ module
- **23.3**: Unified storage through storage/ module
- **23.4**: Centralized configuration through config/ module
- **23.5**: Shared utilities through utils/ module following DRY principles

## Module Responsibilities

### models/
- **Single Responsibility**: Manage all AI models (local and remote)
- **Key Features**: Model loading, caching, selection, and optimization
- **Dependencies**: config/, utils/

### storage/
- **Single Responsibility**: Provide unified data persistence
- **Key Features**: Vector storage, memory storage, caching, backup
- **Dependencies**: config/, utils/

### config/
- **Single Responsibility**: Centralized configuration management
- **Key Features**: Settings validation, environment configuration
- **Dependencies**: utils/

### utils/
- **Single Responsibility**: Shared utilities following DRY principles
- **Key Features**: Logging, error handling, validation, helpers
- **Dependencies**: None (base utilities)

## Integration Points

The modules integrate through well-defined interfaces:

1. **Configuration Flow**: config/ → models/, storage/, core/
2. **Utility Usage**: utils/ → all modules (logging, error handling)
3. **Model Access**: core/ → models/ → storage/ (for caching)
4. **Data Flow**: ingestion/ → storage/ → search/ → core/

## Testing Strategy

Each module is independently testable:

- Unit tests for individual module functionality
- Integration tests for module interactions
- End-to-end tests for complete workflows
- Performance tests for optimization validation

## Maintenance Guidelines

1. **Keep modules focused**: Resist the temptation to add unrelated functionality
2. **Maintain clean interfaces**: Changes should not break dependent modules
3. **Document dependencies**: Clear documentation of module relationships
4. **Regular cleanup**: Remove unused code and deprecated functionality
5. **Performance monitoring**: Track module performance and optimize as needed