"""
Unified LLM Service for Morgan AI Assistant.

This module provides a single, consolidated LLM service that supports:
- Single endpoint mode (direct OpenAI-compatible API)
- Distributed mode (load-balanced across multiple hosts)
- Automatic failover and health monitoring
- Both sync and async interfaces
- Streaming support

Usage:
    from morgan.services.llm import get_llm_service, LLMResponse

    # Get singleton service
    service = get_llm_service()

    # Generate response (sync)
    response = service.generate("What is Python?")
    print(response.content)

    # Generate response (async)
    response = await service.agenerate("Explain Docker")
    print(response.content)

    # Use fast model for simple queries
    response = service.generate("Hi!", use_fast_model=True)
"""

from morgan.services.llm.models import LLMResponse, LLMMode
from morgan.services.llm.service import (
    LLMService,
    get_llm_service,
    reset_llm_service,
)

__all__ = [
    "LLMService",
    "LLMResponse",
    "LLMMode",
    "get_llm_service",
    "reset_llm_service",
]
