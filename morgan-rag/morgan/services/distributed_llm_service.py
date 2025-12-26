"""
Distributed LLM Service for Morgan.

Integrates the distributed LLM infrastructure with Morgan's service layer,
providing a unified interface for LLM operations across multiple hosts.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from morgan.config import get_settings
from morgan.infrastructure.distributed_llm import DistributedLLMClient
from morgan.infrastructure.distributed_gpu_manager import (
    DistributedGPUManager,
    HostRole,
    get_distributed_gpu_manager,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM response wrapper."""

    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str
    endpoint_used: Optional[str] = None
    latency_ms: Optional[float] = None


class DistributedLLMService:
    """
    Distributed LLM service integrating multi-host infrastructure.

    Provides:
    - Automatic endpoint discovery from distributed GPU manager
    - Load-balanced LLM requests across multiple hosts
    - Failover and health monitoring
    - Both sync and async interfaces
    - Streaming support

    Example:
        >>> service = DistributedLLMService()
        >>>
        >>> # Synchronous
        >>> response = service.generate("What is Python?")
        >>> print(response.content)
        >>>
        >>> # Asynchronous
        >>> response = await service.agenerate("Explain Docker")
        >>> print(response.content)
        >>>
        >>> # Streaming
        >>> async for chunk in service.astream("Write a poem"):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        model: Optional[str] = None,
        fast_model: Optional[str] = None,
        strategy: str = "round_robin",
        auto_discover: bool = True,
    ):
        """
        Initialize distributed LLM service.

        Args:
            endpoints: List of LLM endpoint URLs (auto-discovered if None)
            model: Main LLM model name
            fast_model: Fast LLM model name for simple queries
            strategy: Load balancing strategy
            auto_discover: Auto-discover endpoints from GPU manager
        """
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._initialized = False

        # Model configuration
        self.main_model = model or getattr(
            self.settings, "llm_model", "qwen2.5:32b-instruct-q4_K_M"
        )
        self.fast_model = fast_model or getattr(
            self.settings, "llm_fast_model", "qwen2.5:7b-instruct-q5_K_M"
        )
        self.strategy = strategy

        # Store endpoints for initialization
        self._endpoints = endpoints
        self._auto_discover = auto_discover

        # Clients (lazy initialization)
        self._main_client: Optional[DistributedLLMClient] = None
        self._fast_client: Optional[DistributedLLMClient] = None
        self._gpu_manager: Optional[DistributedGPUManager] = None

        logger.info(
            "DistributedLLMService created: " "main_model=%s, fast_model=%s",
            self.main_model,
            self.fast_model,
        )

    async def _ensure_initialized(self):
        """Ensure service is initialized with endpoints."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            endpoints = self._endpoints

            # Auto-discover endpoints if needed
            if endpoints is None and self._auto_discover:
                try:
                    self._gpu_manager = get_distributed_gpu_manager()
                    endpoints = await self._gpu_manager.get_endpoints(HostRole.MAIN_LLM)
                    logger.info("Auto-discovered %d LLM endpoints", len(endpoints))
                except Exception as e:
                    logger.warning("Failed to auto-discover endpoints: %s", e)

            # Fallback to settings
            if not endpoints:
                base_url = getattr(
                    self.settings, "llm_base_url", "http://localhost:11434/v1"
                )
                endpoints = [base_url]
                logger.info("Using fallback endpoint: %s", base_url)

            # Initialize main LLM client
            self._main_client = DistributedLLMClient(
                endpoints=endpoints,
                model=self.main_model,
                strategy=self.strategy,
            )

            # Initialize fast LLM client (same endpoints, different model)
            self._fast_client = DistributedLLMClient(
                endpoints=endpoints,
                model=self.fast_model,
                strategy=self.strategy,
            )

            self._initialized = True
            logger.info("DistributedLLMService initialized")

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Generate response asynchronously.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Returns:
            LLM response
        """
        await self._ensure_initialized()

        client = self._fast_client if use_fast_model else self._main_client
        model = self.fast_model if use_fast_model else self.main_model

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Format as single prompt (DistributedLLMClient uses simple prompts)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            import time

            start = time.time()

            content = await client.generate(
                prompt=full_prompt,
                temperature=temperature or self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                stream=False,
            )

            latency = (time.time() - start) * 1000

            return LLMResponse(
                content=content,
                usage={},  # Ollama doesn't always return usage
                model=model,
                finish_reason="stop",
                latency_ms=latency,
            )

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    async def astream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream response asynchronously.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Yields:
            Response chunks
        """
        await self._ensure_initialized()

        client = self._fast_client if use_fast_model else self._main_client

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            stream = await client.generate(
                prompt=full_prompt,
                temperature=temperature or self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content

        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            raise

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Chat with message history asynchronously.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_fast_model: Use fast model for simple queries

        Returns:
            LLM response
        """
        # Extract system prompt and user messages
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)

        # Format conversation
        prompt_parts = []
        for msg in user_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)

        return await self.agenerate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            use_fast_model=use_fast_model,
        )

    # Synchronous wrappers

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """Synchronous generate wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.agenerate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                use_fast_model=use_fast_model,
            )
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """Synchronous chat wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.achat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                use_fast_model=use_fast_model,
            )
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> Iterator[str]:
        """Synchronous streaming generator."""

        async def _async_gen():
            async for chunk in self.astream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                use_fast_model=use_fast_model,
            ):
                yield chunk

        loop = asyncio.new_event_loop()
        try:
            gen = _async_gen()
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all LLM endpoints."""
        await self._ensure_initialized()

        health = await self._main_client.health_check()
        stats = self._main_client.get_stats()

        return {
            "status": "healthy" if any(health.values()) else "unhealthy",
            "endpoints": health,
            "stats": stats,
        }

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            asyncio.get_event_loop().run_until_complete(self._ensure_initialized())
            health = asyncio.get_event_loop().run_until_complete(
                self._main_client.health_check()
            )
            return any(health.values())
        except Exception as e:
            logger.error("LLM availability check failed: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        if self._main_client:
            return self._main_client.get_stats()
        return {}


# Singleton instance
_service_instance: Optional[DistributedLLMService] = None
_service_lock = threading.Lock()


def get_distributed_llm_service(
    endpoints: Optional[List[str]] = None,
    **kwargs,
) -> DistributedLLMService:
    """
    Get singleton distributed LLM service instance.

    Args:
        endpoints: Optional endpoint URLs
        **kwargs: Additional service configuration

    Returns:
        Shared DistributedLLMService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DistributedLLMService(
                    endpoints=endpoints,
                    **kwargs,
                )

    return _service_instance
