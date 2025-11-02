"""
Morgan RAG v2.0 - Production-Grade AI Assistant
================================================

Clean Architecture implementation with:
- Domain-Driven Design (DDD)
- SOLID Principles
- Dependency Injection
- Hexagonal Architecture (Ports & Adapters)

Architecture Layers:
    domain/         - Pure business logic (no dependencies)
    application/    - Use cases and orchestration
    infrastructure/ - Technical implementations
    interfaces/     - External access points (CLI, API, Web)
    shared/         - Cross-cutting concerns
    di/             - Dependency injection

Usage:
    from morgan_v2 import create_assistant

    async def main():
        assistant = await create_assistant()
        response = await assistant.process_query("How do I deploy Docker?")
        print(response.answer)
"""

__version__ = "2.0.0"
__author__ = "Morgan RAG Team"

from morgan_v2.di.container import Container
from morgan_v2.application.dto.query_response import QueryResponse


async def create_assistant():
    """
    Create and initialize Morgan assistant with all dependencies.

    Returns:
        Container: Dependency injection container with all services

    Example:
        >>> container = await create_assistant()
        >>> use_case = container.process_query_use_case()
        >>> response = await use_case.execute(QueryRequest(query="Hello"))
    """
    container = Container()
    container.init_resources()
    await container.wire(modules=[__name__])
    return container


__all__ = [
    "create_assistant",
    "Container",
    "QueryResponse",
]
