# Archived Tests

This directory contains tests that were written for an older architecture of morgan-server where it had its own internal implementations of domain modules.

## Why These Tests Were Archived

Morgan has evolved from a monolithic architecture to a clean client-server architecture where:
- **morgan-server** is a thin API wrapper that delegates to morgan-rag
- **morgan-rag** contains all domain logic (knowledge, empathic, personalization modules)

These archived tests expected morgan-server to have modules like:
- `morgan_server.knowledge.vectordb`
- `morgan_server.knowledge.rag`
- `morgan_server.knowledge.search`
- `morgan_server.empathic.relationships`
- `morgan_server.empathic.personality`
- `morgan_server.empathic.emotional`
- `morgan_server.personalization.memory`
- `morgan_server.personalization.profile`
- `morgan_server.personalization.preferences`

These modules don't exist in the current architecture because morgan-server delegates all domain logic to morgan-rag.

## Current Test Strategy

The current test suite focuses on:
1. **API Endpoint Testing**: Testing the REST/WebSocket APIs using FastAPI TestClient
2. **Integration Testing**: Testing the integration between morgan-server and morgan-rag
3. **Configuration Testing**: Testing server configuration and setup

## Archived Files

### Domain Logic Tests (Should be in morgan-rag)
- `test_vectordb.py` - Vector database client tests
- `test_search_system.py` - Search system tests
- `test_rag_system.py` - RAG system tests
- `test_relationship_management.py` - Relationship management tests
- `test_personality_system.py` - Personality system tests
- `test_emotional_intelligence.py` - Emotional intelligence tests
- `test_roleplay_system.py` - Roleplay system tests
- `test_user_profile.py` - User profile tests
- `test_preferences.py` - Preferences tests
- `test_memory_system.py` - Memory system tests
- `test_document_processing.py` - Document processing tests

### Integration Tests (Outdated)
- `test_assistant_integration.py` - Old integration tests
- `integration/` - Old integration test suite

## Migration Notes

If you need to test domain logic:
1. **Write tests in morgan-rag**: Domain logic tests belong in the morgan-rag test suite
2. **Test via API**: If testing from morgan-server, use the API endpoints
3. **Integration tests**: Create new integration tests that test the API → morgan-rag flow

## Date Archived

December 26, 2025

## Refactoring Context

This cleanup was part of Phase 2 of the Morgan refactoring project, which focused on aligning the server test suite with the current architecture.
