# Implementation Plan: Client-Server Separation

## Overview

This implementation plan outlines the tasks for creating a complete client-server separation for Morgan, with a focus on building a personal assistant with empathic and knowledge engines. All components will be built from scratch.

## Tasks

- [x] 1. Set up new project structure and dependencies
  - Create `morgan-server/` directory with package structure
  - Create `morgan-cli/` directory with package structure
  - Create `pyproject.toml` for server with dependencies (FastAPI, Qdrant client, sentence-transformers, pydantic, uvicorn)
  - Create `pyproject.toml` for client with dependencies (rich, aiohttp, click, websockets)
  - Create Docker directory with Dockerfile templates
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 2. Implement server configuration system







  - Create `morgan_server/config.py` with Pydantic models for configuration
  - Implement environment variable loading with precedence rules
  - Implement configuration file loading (YAML, JSON, .env)
  - Add validation for required fields (LLM endpoint, vector DB URL)
  - Add default values for non-critical settings
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2.1 Write property test for configuration precedence



  - **Property 2: Configuration precedence consistency**
  - **Validates: Requirements 3.3**
  - Use Hypothesis library with minimum 100 iterations

- [x] 2.2 Write property test for invalid configuration rejection








  - **Property 3: Invalid configuration rejection**
  - **Validates: Requirements 1.4, 3.4, 3.5**
  - Use Hypothesis library with minimum 100 iterations

- [x] 2.3 Write property test for configuration format support








  - **Property 9: Configuration format support**
  - **Validates: Requirements 3.2**
  - Use Hypothesis library with minimum 100 iterations

- [x] 3. Implement LLM client layer





  - Create `morgan_server/llm/__init__.py` with base LLM client interface
  - Implement `morgan_server/llm/ollama.py` for Ollama integration
  - Implement `morgan_server/llm/openai_compatible.py` for OpenAI-compatible endpoints
  - Add error handling and retry logic
  - Add streaming support for responses
  - _Requirements: 1.1_

- [x] 3.1 Write unit tests for LLM clients



  - Test Ollama client connection and requests
  - Test OpenAI-compatible client connection and requests
  - Test error handling and retries
  - _Requirements: 1.1_


- [x] 4. Implement vector database client









  - Create `morgan_server/knowledge/vectordb.py` with Qdrant client
  - Implement collection creation and management
  - Implement vector insertion and search
  - Add connection pooling and error handling
  - _Requirements: 1.1_

- [x] 4.1 Write unit tests for vector database client





  - Test collection operations
  - Test vector insertion and search
  - Test error handling
  - _Requirements: 1.1_

- [x] 5. Implement Empathic Engine - Emotional Intelligence





  - Create `morgan_server/empathic/emotional.py`
  - Implement emotional tone detection from user messages
  - Implement emotional tone adjustment for responses
  - Implement emotional pattern tracking over time
  - Add support for celebrating positive moments and providing support
  - _Requirements: 1.1_

- [x] 5.1 Write unit tests for emotional intelligence


  - Test tone detection
  - Test tone adjustment
  - Test pattern tracking
  - _Requirements: 1.1_


- [x] 6. Implement Empathic Engine - Personality System



  - Create `morgan_server/empathic/personality.py`
  - Implement base personality traits configuration
  - Implement consistent personality across conversations
  - Implement adaptive behavior based on relationship depth
  - Add natural conversational style logic
  - _Requirements: 1.1_


- [x] 6.1 Write unit tests for personality system




  - Test personality trait application
  - Test consistency across conversations
  - Test adaptive behavior
  - _Requirements: 1.1_


- [x] 7. Implement Empathic Engine - Roleplay System




  - Create `morgan_server/empathic/roleplay.py`
  - Implement base roleplay configuration (personality, tone, style)
  - Add context-aware response logic
  - Add emotional intelligence integration
  - Add relationship-aware behavior
  - _Requirements: 1.1_

- [x] 7.1 Write unit tests for roleplay system







  - Test roleplay configuration loading
  - Test context-aware responses
  - Test emotional integration
  - _Requirements: 1.1_


- [x] 8. Implement Empathic Engine - Relationship Management




  - Create `morgan_server/empathic/relationships.py`
  - Implement interaction history tracking
  - Implement trust level calculation
  - Implement milestone recognition and celebration
  - Add relationship depth metrics
  - _Requirements: 1.1_



- [x] 8.1 Write unit tests for relationship management









  - Test interaction tracking
  - Test trust level calculation


  - Test milestone recognition

  - _Requirements: 1.1_

- [x] 9. Implement Knowledge Engine - Document Processing


  - Create `morgan_server/knowledge/ingestion.py`
  - Implement document loaders (PDF, markdown, text, web pages)


  - Implement intelligent chunking with overlap
  - Implement metadata extraction
  - Add support for incremental updates
  - _Requirements: 1.1_

- [x] 9.1 Write unit tests for document processing





  - Test document loading for different formats
  - Test chunking logic
  - Test metadata extraction
  - _Requirements: 1.1_


- [x] 10. Implement Knowledge Engine - RAG System



  - Create `morgan_server/knowledge/rag.py`
  - Implement semantic search using vector embeddings
  - Implement context-aware document retrieval
  - Implement multi-stage ranking (vector similarity + reranking)
  - Add source attribution and confidence scoring
  - _Requirements: 1.1_

- [x] 10.1 Write unit tests for RAG system






  - Test semantic search
  - Test context retrieval
  - Test ranking logic
  - Test confidence scoring
  - _Requirements: 1.1_


- [x] 11. Implement Knowledge Engine - Search System








  - Create `morgan_server/knowledge/search.py`
  - Implement vector similarity search
  - Implement hybrid search (vector + keyword)
  - Implement result reranking
  - Add relevance filtering
  - _Requirements: 1.1_





- [x] 11.1 Write unit tests for search system




  - Test vector search
  - Test hybrid search



  - Test reranking
  - _Requirements: 1.1_

- [x] 12. Implement Personalization Layer - User Profile




  - Create `morgan_server/personalization/profile.py`
  - Implement user profile model (name, preferences, metrics)

  - Implement profile persistence
  - Add trust and engagement metrics
  - _Requirements: 1.1_


- [x] 12.1 Write unit tests for user profile





  - Test profile creation and updates
  - Test persistence
  - Test metrics calculation
  - _Requirements: 1.1_


- [x] 13. Implement Personalization Layer - Preferences




  - Create `morgan_server/personalization/preferences.py`
  - Implement communication style preferences
  - Implement response length preferences
  - Implement topic interest tracking
  - Add preference learning from interactions



  - _Requirements: 1.1_



- [-] 13.1 Write unit tests for preferences



  - Test preference storage and retrieval
  - Test preference learning
  - Test preference application
  - _Requirements: 1.1_

- [x] 14. Implement Personalization Layer - Memory System




  - Create `morgan_server/personalization/memory.py`
  - Implement conversation history storage
  - Implement long-term memory across sessions
  - Implement context retrieval from memory
  - Add memory summarization
  - _Requirements: 1.1_

- [x] 14.1 Write unit tests for memory system







  - Test conversation storage
  - Test memory retrieval
  - Test summarization
  - _Requirements: 1.1_

- [x] 15. Implement core assistant orchestration




  - Create `morgan_server/assistant.py`
  - Integrate Empathic Engine, Knowledge Engine, and Personalization Layer
  - Implement main chat flow (receive message → process → generate response)
  - Add context management and conversation flow
  - Implement response generation with all engines
  - _Requirements: 1.1_


- [x] 15.1 Write integration tests for assistant


  - Test full chat flow
  - Test engine integration
  - Test context management
  - _Requirements: 1.1_


- [x] 16. Implement API models










  - Create `morgan_server/api/models.py`
  - Define Pydantic models for all API requests (ChatRequest, LearnRequest, etc.)
  - Define Pydantic models for all API responses (ChatResponse, ProfileResponse, etc.)
  - Add validation rules

  - _Requirements: 7.2_




- [x] 16.1 Write property test for API consistency




  - **Property 15: API consistency**

  - **Validates: Requirements 7.2**
  - Use Hypothesis library with minimum 100 iterations





- [x] 16.2 Write property test for error response structure





  - **Property 16: Error response structure**
  - **Validates: Requirements 7.3**
  - Use Hypothesis library with minimum 100 iterations



- [x] 17. Implement API routes - Chat






  - Create `morgan_server/api/routes/chat.py`
  - Implement POST `/api/chat` endpoint
  - Implement WebSocket `/ws/{user_id}` endpoint
  - Add request validation and error handling
  - Add response formatting

  - _Requirements: 1.3, 7.1, 7.2, 7.3_

- [x] 17.1 Write unit tests for chat routes




  - Test POST endpoint
  - Test WebSocket endpoint
  - Test error handling
  - _Requirements: 1.3, 7.1_



- [x] 18. Implement API routes - Memory






  - Create `morgan_server/api/routes/memory.py`
  - Implement GET `/api/memory/stats` endpoint
  - Implement GET `/api/memory/search` endpoint
  - Implement DELETE `/api/memory/cleanup` endpoint
  - _Requirements: 7.1_



- [x] 18.1 Write unit tests for memory routes



  - Test stats endpoint
  - Test search endpoint
  - Test cleanup endpoint


  - _Requirements: 7.1_

- [x] 19. Implement API routes - Knowledge






  - Create `morgan_server/api/routes/knowledge.py`
  - Implement POST `/api/knowledge/learn` endpoint
  - Implement GET `/api/knowledge/search` endpoint


  - Implement GET `/api/knowledge/stats` endpoint
  - _Requirements: 7.1_




- [x] 19.1 Write unit tests for knowledge routes





  - Test learn endpoint
  - Test search endpoint


  - Test stats endpoint
  - _Requirements: 7.1_




- [x] 20. Implement API routes - Profile

  - Create `morgan_server/api/routes/profile.py`

  - Implement GET `/api/profile/{user_id}` endpoint


  - Implement PUT `/api/profile/{user_id}` endpoint
  - Implement GET `/api/timeline/{user_id}` endpoint
  - _Requirements: 7.1_
  
- [x] 20.1 Write unit tests for profile routes







  - Test profile retrieval
  - Test profile updates
  - Test timeline retrieval

  - _Requirements: 7.1_




- [x] 21. Implement health check system






  - Create `morgan_server/health.py`
  - Implement GET `/health` endpoint (simple health check)
  - Implement GET `/api/status` endpoint (detailed status)
  - Add component health checks (vector DB, LLM)

  - Add response time tracking

  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 21.1 Write property test for health check responsiveness









  - **Property 10: Health check responsiveness**
  - **Validates: Requirements 4.1, 4.2, 4.3**

  - Use Hypothesis library with minimum 100 iterations




- [x] 22. Implement metrics endpoint




  - Add GET `/metrics` endpoint for Prometheus
  - Implement request counting

  - Implement response time tracking
  - Implement error rate tracking
  - _Requirements: 4.5, 10.3_



- [x] 22.1 Write property test for metrics accuracy



  - **Property 11: Metrics accuracy**

  - **Validates: Requirements 4.5**

  - Use Hypothesis library with minimum 100 iterations


- [x] 23. Implement middleware





  - Create `morgan_server/middleware.py`
  - Implement logging middleware (request/response logging)
  - Implement CORS middleware


  - Implement error handling middleware
  - Implement request validation middleware
  - _Requirements: 10.1_




- [x] 23.1 Write property test for request logging completeness







  - **Property 23: Request logging completeness**
  - **Validates: Requirements 10.1**
  - Use Hypothesis library with minimum 100 iterations

- [x] 23.2 Write property test for error logging detail




  - **Property 24: Error logging detail**
  - **Validates: Requirements 10.2**

  - Use Hypothesis library with minimum 100 iterations


- [x] 24. Implement logging system






  - Configure structured logging (JSON format)
  - Implement log level filtering

  - Add context fields (request_id, user_id, conversation_id)
  - Configure log rotation
  - _Requirements: 10.2, 10.4, 10.5_




- [x] 24.1 Write property test for log level filtering







  - **Property 25: Log level filtering**
  - **Validates: Requirements 10.4**
  - Use Hypothesis library with minimum 100 iterations




- [x] 24.2 Write property test for structured logging format



  - **Property 26: Structured logging format**
  - **Validates: Requirements 10.5**
  - Use Hypothesis library with minimum 100 iterations






- [x] 25. Implement FastAPI application factory



  - Create `morgan_server/app.py`
  - Implement `create_app()` factory function
  - Register all routes
  - Add middleware

  - Configure CORS
  - Add startup and shutdown events
  - _Requirements: 1.1, 1.2, 1.3, 1.5_



- [x] 25.1 Write property test for server initialization independence







  - **Property 1: Server initialization independence**
  - **Validates: Requirements 1.1**
  - Use Hypothesis library with minimum 100 iterations




- [x] 25.2 Write property test for graceful shutdown preservation







  - **Property 4: Graceful shutdown preservation**
  - **Validates: Requirements 1.5**
  - Use Hypothesis library with minimum 100 iterations


- [x] 26. Implement session management







  - Add session tracking for concurrent clients
  - Implement session isolation
  - Add session cleanup on disconnect
  - Implement connection pooling
  - _Requirements: 6.1, 6.2, 6.3, 6.5_





- [x] 26.1 Write property test for concurrent request handling









  - **Property 12: Concurrent request handling**
  - **Validates: Requirements 6.1, 6.2**
  - Use Hypothesis library with minimum 100 iterations


- [x] 26.2 Write property test for session cleanup isolation








  - **Property 14: Session cleanup isolation**
  - **Validates: Requirements 6.5**
  - Use Hypothesis library with minimum 100 iterations








- [x] 27. Checkpoint - Ensure all server tests pass



  - Ensure all tests pass, ask the user if questions arise.





- [x] 28. Create TUI client - HTTP/WebSocket client




  - Create `morgan_cli/client.py`
  - Implement HTTP client for REST API calls
  - Implement WebSocket client for real-time chat
  - Add error handling and retry logic
  - Add connection status tracking
  - _Requirements: 2.2, 2.3, 2.4, 2.5_





- [x] 28.1 Write property test for client API-only communication







  - **Property 5: Client API-only communication**
  - **Validates: Requirements 2.1, 2.3**


  - Use Hypothesis library with minimum 100 iterations




- [x] 28.2 Write property test for client configuration flexibility





  - **Property 6: Client configuration flexibility**
  - **Validates: Requirements 2.2**
  - Use Hypothesis library with minimum 100 iterations




- [x] 28.3 Write property test for client error handling






  - **Property 7: Client error handling**
  - **Validates: Requirements 2.4**
  - Use Hypothesis library with minimum 100 iterations



- [x] 28.4 Write property test for client cleanup isolation








  - **Property 8: Client cleanup isolation**
  - **Validates: Requirements 2.5**
  - Use Hypothesis library with minimum 100 iterations








- [x] 29. Create TUI client - Rich UI components




  - Create `morgan_cli/ui.py`
  - Implement markdown rendering using Rich
  - Implement typing indicators and progress feedback
  - Implement scrolling and pagination for long responses
  - Add error message display with user-friendly formatting


  - _Requirements: 9.1, 9.2, 9.4, 9.5_




- [x] 29.1 Write property test for markdown rendering





  - **Property 20: Markdown rendering**





  - **Validates: Requirements 9.1**

  - Use Hypothesis library with minimum 100 iterations


- [x] 29.2 Write property test for error message clarity








  - **Property 22: Error message clarity**
  - **Validates: Requirements 9.5**


  - Use Hypothesis library with minimum 100 iterations

- [x] 30. Create TUI client - Click CLI



  - Create `morgan_cli/cli.py`

  - Implement `chat` command for interactive chat


  - Implement `ask` command for single questions
  - Implement `learn` command for document ingestion
  - Implement `memory` command for memory management
  - Implement `knowledge` command for knowledge base management
  - Implement `health` command for server health check


  - Add command history and auto-completion
  - _Requirements: 2.1, 2.2, 9.3_


- [x] 30.1 Write property test for command history (✓ passed)


  - **Property 21: Command history**
  - **Validates: Requirements 9.3**


  - Use Hypothesis library with minimum 100 iterations

- [x] 31. Create TUI client - Configuration





  - Create `morgan_cli/config.py`
  - Implement configuration loading from environment variables
  - Implement configuration loading from command-line arguments


  - Add default values
  - _Requirements: 2.2_



- [x] 31.1 Write unit tests for client configuration





  - Test environment variable loading
  - Test command-line argument parsing
  - Test default values
  - _Requirements: 2.2_







- [x] 32. Checkpoint - Ensure all client tests pass




  - Ensure all tests pass, ask the user if questions arise.




- [x] 33. Create Docker configuration for server













  - Create `docker/Dockerfile.server`
  - Configure production-ready image with only server dependencies
  - Add non-root user

  - Add health check
  - Configure SIGTERM handling
- - Make sure that all using actual docker config (tests too)
  - _Requirements: 8.1, 8.2, 8.3, 8.4_


- [x] 33.1 Write property test for container configuration











  - **Property 18: Container configuration**
  - **Validates: Requirements 8.2**
  - Use Hypothesis library with minimum 100 iterations







- [x] 33.2 Write property test for container signal handling











  - **Property 19: Container signal handling**
  - **Validates: Requirements 8.4**

  - Use Hypothesis library with minimum 100 iterations
  - use docker for actual tests

- [x] 34. Create Docker Compose configuration






  - Create `docker/docker-compose.yml`
  - Add morgan-server service
  - Dont add ollama service - that will be remote
  - Add qdrant service
  - Add prometheus service (optional)
  - Configure volumes and networks

  - _Requirements: 8.5_



- [x] 35. Create server entry point





  - Create `morgan_server/__main__.py`
  - Add CLI for starting server

  - Add configuration loading

  - Add graceful shutdown handling
  - _Requirements: 1.2, 1.5_



- [x] 36. Create client entry point




  - Create `morgan_cli/__main__.py`
  - Add CLI entry point
  - Configure as console script in pyproject.toml
  - _Requirements: 11.4_




- [x] 37. Write integration tests




  - Test full client-server communication
  - Test chat flow end-to-end
  - Test document learning flow
  - Test memory and knowledge retrieval
  - Test error scenarios
  - _Requirements: All_






- [ ] 38. Write performance tests


  - Test concurrent client connections
  - Measure response times under load
  - Verify 95th percentile under 5 seconds
  - Test resource cleanup

  - _Requirements: 6.4_


- [ ] 38.1 Write property test for performance under load


  - **Property 13: Performance under load**



  - **Validates: Requirements 6.4**
  - Use Hypothesis library with minimum 100 iterations



- [x] 39. Create documentation








  - Write README for server package
  - Write README for client package

  - Write deployment guide (Docker, bare metal)

  - Write configuration guide
  - Write API documentation
  - Create migration guide from old system
  - _Requirements: All_


- [x] 40. Final checkpoint - Complete system validation





  - Ensure all tests pass, ask the user if questions arise.
  - Verify all requirements are met
  - Test full deployment with Docker Compose
  - Verify documentation is complete


## Final Migration and Consolidation

- [x] 41. Migrate useful components from old morgan-rag system







  - Review and migrate reranking functionality from `morgan/infrastructure/local_reranking.py`
  - Review and migrate advanced embedding features from `morgan/jina/embeddings/`
  - Review and migrate emotional intelligence components from `morgan/emotional/` and `morgan/emotions/`
  - Review and migrate relationship management from `morgan/relationships/`
  - Review and migrate communication preferences from `morgan/communication/`
  - Review and migrate learning/adaptation features from `morgan/learning/`
  - Review and migrate caching strategies from `morgan/caching/intelligent_cache.py`
  - Review and migrate monitoring/metrics from `morgan/monitoring/`
  - _Requirements: All_

- [x] 41.1 Write integration tests for migrated components


  - Test reranking integration with search
  - Test emotional intelligence integration
  - Test relationship management integration
  - Test communication preferences
  - _Requirements: All_

- [x] 42. Implement missing User Profile functionality





  - Complete user profile model implementation
  - Add profile persistence to vector database
  - Integrate profile with personalization layer
  - Add profile API endpoints
  - _Requirements: 1.1_



- [x] 42.1 Write unit tests for user profile




  - Test profile creation and updates
  - Test persistence
  - Test metrics calculation
  - _Requirements: 1.1_

- [x] 43. Test with actual conversation data




  - Import existing conversation history from old system
  - Test chat flow with real conversation data
  - Verify emotional intelligence works with real conversations
  - Verify relationship tracking with real data
  - Test memory retrieval with actual queries
  - _Requirements: All_


- [x] 43.1 Write integration tests with real data

  - Test end-to-end chat with imported conversations
  - Test memory search with real queries
  - Test knowledge retrieval with actual documents
  - _Requirements: All_

- [x] 44. Performance optimization and testing




  - Profile server performance under load
  - Optimize slow endpoints
  - Implement caching where beneficial
  - Test concurrent client connections
  - Measure response times at different load levels
  - Verify 95th percentile stays under 5 seconds
  - _Requirements: 6.4_

- [x] 44.1 Write property test for performance under load


  - **Property 13: Performance under load**
  - **Validates: Requirements 6.4**
  - Use Hypothesis library with minimum 100 iterations

- [x] 45. Clean up and deprecate old system




  - Mark old `morgan/` directory as deprecated
  - Update root README.md to point to new system
  - Create MIGRATION.md guide for users
  - Archive old code (don't delete yet, keep for reference)
  - Update all documentation to reference new system
  - _Requirements: All_





- [ ] 46. Final validation and documentation
  - Run full test suite (unit, integration, property-based)
  - Verify all requirements are met
  - Test Docker deployment end-to-end
  - Update all API documentation
  - Create deployment guides




  - Create user migration guide
  - _Requirements: All_


- [ ] 47. Final checkpoint - Production readiness validation
  - Ensure all tests pass
  - Verify Docker Compose stack works
  - Test with real LLM (Ollama)
  - Test with real vector database (Qdrant)
  - Verify monitoring and metrics work
  - Confirm graceful shutdown works
  - Validate all documentation is complete
