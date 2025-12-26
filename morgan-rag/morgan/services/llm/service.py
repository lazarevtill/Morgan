"""
Unified LLM Service for Morgan AI Assistant.

Consolidates llm_service.py and distributed_llm_service.py into a single
implementation that supports both single and distributed modes.
"""

import asyncio
import threading
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from openai import OpenAI

from morgan.config import get_settings
from morgan.services.llm.models import LLMMode, LLMResponse
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    Unified LLM service supporting single and distributed modes.

    This service consolidates the functionality from both llm_service.py
    and distributed_llm_service.py into a single implementation.

    Features:
    - Single mode: Direct OpenAI-compatible API calls
    - Distributed mode: Load-balanced calls across multiple hosts
    - Automatic failover and health monitoring
    - Both sync and async interfaces
    - Fast model support for simple queries
    - Streaming support

    Example:
        >>> # Single mode (default)
        >>> service = LLMService()
        >>> response = service.generate("What is Python?")

        >>> # Distributed mode
        >>> service = LLMService(
        ...     mode="distributed",
        ...     endpoints=["http://host1:11434/v1", "http://host2:11434/v1"]
        ... )
        >>> response = service.generate("What is Python?")

        >>> # Async usage
        >>> response = await service.agenerate("Explain Docker")

        >>> # Fast model for simple queries
        >>> response = service.generate("Hi!", use_fast_model=True)
    """

    def __init__(
        self,
        mode: str = "single",
        endpoints: Optional[List[str]] = None,
        model: Optional[str] = None,
        fast_model: Optional[str] = None,
        strategy: str = "round_robin",
        auto_discover: bool = True,
    ):
        """
        Initialize LLM service.

        Args:
            mode: "single" or "distributed"
            endpoints: List of endpoint URLs for distributed mode
            model: Main LLM model name (uses settings default if not specified)
            fast_model: Fast LLM model name for simple queries
            strategy: Load balancing strategy for distributed mode
            auto_discover: Auto-discover endpoints from distributed config
        """
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._initialized = False

        # Determine mode
        self.mode = LLMMode(mode)

        # Model configuration
        self.main_model = model or getattr(
            self.settings, "llm_model", "qwen2.5:32b-instruct-q4_K_M"
        )
        self.fast_model = fast_model or getattr(
            self.settings, "llm_fast_model", "qwen2.5:7b-instruct-q5_K_M"
        )
        self.strategy = strategy

        # Store for lazy initialization
        self._endpoints = endpoints
        self._auto_discover = auto_discover

        # Clients (lazy initialization)
        self._client: Optional[OpenAI] = None
        self._distributed_client = None

        logger.info(
            "LLMService created: mode=%s, main_model=%s, fast_model=%s",
            self.mode.value,
            self.main_model,
            self.fast_model,
        )

    def _ensure_initialized(self):
        """Ensure service is initialized (sync version)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if self.mode == LLMMode.DISTRIBUTED:
                self._init_distributed()
            else:
                self._init_single()

            self._initialized = True

    def _init_single(self):
        """Initialize single-endpoint mode."""
        base_url = getattr(self.settings, "llm_base_url", "http://localhost:11434/v1")
        api_key = getattr(self.settings, "llm_api_key", "ollama")

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "ollama",
        )
        self._distributed_client = None

        logger.info("LLM single mode initialized: %s", base_url)

    def _init_distributed(self):
        """Initialize distributed mode."""
        try:
            from morgan.infrastructure.distributed_llm import DistributedLLMClient

            endpoints = self._endpoints

            # Auto-discover endpoints if needed
            if endpoints is None and self._auto_discover:
                try:
                    from morgan.config.distributed_config import get_distributed_config

                    config = get_distributed_config()
                    llm_hosts = config.get_hosts_by_role("llm")
                    if llm_hosts:
                        endpoints = [
                            f"http://{h.address}:{h.port}{h.api_path}"
                            for h in llm_hosts
                        ]
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

            # Initialize distributed client
            self._distributed_client = DistributedLLMClient(
                endpoints=endpoints,
                model=self.main_model,
                strategy=self.strategy,
            )

            # Start health monitoring
            self._distributed_client.start_health_monitoring()

            self._client = None  # Not used in distributed mode

            logger.info(
                "LLM distributed mode initialized with %d endpoints",
                len(endpoints),
            )

        except ImportError as e:
            logger.warning(
                "Distributed mode unavailable: %s, falling back to single", e
            )
            self.mode = LLMMode.SINGLE
            self._init_single()

    # =========================================================================
    # Synchronous Methods
    # =========================================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Generate response from LLM (synchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED:
            return self._generate_distributed(
                prompt, max_tokens, temperature, system_prompt, use_fast_model
            )
        else:
            return self._generate_single(
                prompt, max_tokens, temperature, system_prompt, use_fast_model
            )

    def _generate_single(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        system_prompt: Optional[str],
        use_fast_model: bool,
    ) -> LLMResponse:
        """Generate using single endpoint."""
        model = self.fast_model if use_fast_model else self.main_model
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start_time = time.time()

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump() if response.usage else {},
                latency_ms=latency_ms,
                endpoint_used=self.settings.llm_base_url,
            )

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    def _generate_distributed(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        system_prompt: Optional[str],
        use_fast_model: bool,
    ) -> LLMResponse:
        """Generate using distributed client."""
        model = self.fast_model if use_fast_model else self.main_model

        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            start_time = time.time()

            # Run async in sync context
            loop = asyncio.new_event_loop()
            try:
                content = loop.run_until_complete(
                    self._distributed_client.generate(
                        prompt=full_prompt,
                        temperature=temperature or self.settings.llm_temperature,
                        max_tokens=max_tokens or self.settings.llm_max_tokens,
                    )
                )
            finally:
                loop.close()

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=model,
                finish_reason="stop",
                usage={},  # Distributed client doesn't return usage
                latency_ms=latency_ms,
                endpoint_used="distributed",
            )

        except Exception as e:
            logger.error("Distributed LLM generation failed: %s", e)
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Chat with LLM using message history (synchronous).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_fast_model: Use fast model for simple queries

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED:
            # Convert messages to single prompt for distributed client
            system_prompt = None
            prompt_parts = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                else:
                    prompt_parts.append(f"{role.upper()}: {content}")

            return self.generate(
                prompt="\n".join(prompt_parts),
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                use_fast_model=use_fast_model,
            )

        # Single mode - direct chat
        model = self.fast_model if use_fast_model else self.main_model

        try:
            start_time = time.time()

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump() if response.usage else {},
                latency_ms=latency_ms,
                endpoint_used=self.settings.llm_base_url,
            )

        except Exception as e:
            logger.error("LLM chat failed: %s", e)
            raise

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> Iterator[str]:
        """
        Stream response from LLM (synchronous generator).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Yields:
            Response text chunks
        """
        self._ensure_initialized()

        model = self.fast_model if use_fast_model else self.main_model
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.mode == LLMMode.SINGLE and self._client:
                stream = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                    temperature=temperature or self.settings.llm_temperature,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Distributed mode - fall back to non-streaming
                response = self.generate(
                    prompt, max_tokens, temperature, system_prompt, use_fast_model
                )
                yield response.content

        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            raise

    # =========================================================================
    # Asynchronous Methods
    # =========================================================================

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Generate response from LLM (asynchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED and self._distributed_client:
            model = self.fast_model if use_fast_model else self.main_model

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            start_time = time.time()

            content = await self._distributed_client.generate(
                prompt=full_prompt,
                temperature=temperature or self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=model,
                finish_reason="stop",
                usage={},
                latency_ms=latency_ms,
                endpoint_used="distributed",
            )
        else:
            # Fall back to sync in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generate(
                    prompt, max_tokens, temperature, system_prompt, use_fast_model
                ),
            )

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        """
        Chat with LLM using message history (asynchronous).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_fast_model: Use fast model for simple queries

        Returns:
            LLMResponse with generated content
        """
        # Extract system prompt and format messages
        system_prompt = None
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return await self.agenerate(
            prompt="\n".join(prompt_parts),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            use_fast_model=use_fast_model,
        )

    async def astream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream response from LLM (asynchronous generator).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            use_fast_model: Use fast model for simple queries

        Yields:
            Response text chunks
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED and self._distributed_client:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            try:
                stream = await self._distributed_client.generate(
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
                logger.error("LLM async streaming failed: %s", e)
                raise
        else:
            # Fall back to non-streaming
            response = await self.agenerate(
                prompt, max_tokens, temperature, system_prompt, use_fast_model
            )
            yield response.content

    # =========================================================================
    # Health & Status Methods
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if LLM service is available.

        Returns:
            True if available
        """
        try:
            self._ensure_initialized()

            if self.mode == LLMMode.DISTRIBUTED and self._distributed_client:
                stats = self._distributed_client.get_stats()
                return stats.get("healthy_endpoints", 0) > 0

            # Single mode - test with minimal request
            self._client.chat.completions.create(
                model=self.main_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            logger.error("LLM availability check failed: %s", e)
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of LLM service (async).

        Returns:
            Dict with health status
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED and self._distributed_client:
            health = await self._distributed_client.health_check()
            stats = self._distributed_client.get_stats()

            return {
                "status": "healthy" if any(health.values()) else "unhealthy",
                "mode": self.mode.value,
                "endpoints": health,
                "stats": stats,
            }

        # Single mode
        available = self.is_available()
        return {
            "status": "healthy" if available else "unhealthy",
            "mode": self.mode.value,
            "endpoint": getattr(self.settings, "llm_base_url", "unknown"),
            "model": self.main_model,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dict with service stats
        """
        self._ensure_initialized()

        if self.mode == LLMMode.DISTRIBUTED and self._distributed_client:
            stats = self._distributed_client.get_stats()
            stats["mode"] = self.mode.value
            stats["fast_model"] = self.fast_model
            return stats

        return {
            "mode": self.mode.value,
            "endpoint": getattr(self.settings, "llm_base_url", "unknown"),
            "main_model": self.main_model,
            "fast_model": self.fast_model,
            "available": self.is_available(),
        }

    def shutdown(self):
        """Shutdown service and cleanup resources."""
        if self._distributed_client:
            self._distributed_client.stop_health_monitoring()
        logger.info("LLM service shutdown complete")


# =============================================================================
# Singleton Management
# =============================================================================

_llm_service_instance: Optional[LLMService] = None
_llm_service_lock = threading.Lock()


def get_llm_service(
    mode: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
    force_new: bool = False,
    **kwargs,
) -> LLMService:
    """
    Get singleton LLM service instance.

    Args:
        mode: "single" or "distributed" (uses env var if not specified)
        endpoints: Endpoint URLs for distributed mode
        force_new: Force create new instance
        **kwargs: Additional service configuration

    Returns:
        Shared LLMService instance
    """
    global _llm_service_instance

    if _llm_service_instance is None or force_new:
        with _llm_service_lock:
            if _llm_service_instance is None or force_new:
                # Determine mode from settings if not specified
                settings = get_settings()
                actual_mode = mode or getattr(settings, "llm_mode", "single")

                _llm_service_instance = LLMService(
                    mode=actual_mode,
                    endpoints=endpoints,
                    **kwargs,
                )

    return _llm_service_instance


def reset_llm_service():
    """Reset singleton instance (useful for testing)."""
    global _llm_service_instance

    with _llm_service_lock:
        if _llm_service_instance:
            _llm_service_instance.shutdown()
        _llm_service_instance = None
