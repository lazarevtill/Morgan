"""
LLM Service for Morgan RAG.

Supports:
- Single endpoint (standard)
- Distributed endpoints with load balancing (optional)
- OpenAI-compatible API

Uses configuration from settings to determine mode.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from openai import AsyncOpenAI, OpenAI

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM response wrapper."""

    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str


class LLMService:
    """
    LLM service with support for single and distributed endpoints.

    Features:
    - Single endpoint mode (default)
    - Distributed mode with load balancing
    - Sync and async support
    - Streaming support
    """

    def __init__(self):
        """Initialize LLM service."""
        self.settings = get_settings()
        self._distributed_client = None
        self._sync_client = None
        self._async_client = None

        # Check if distributed mode is enabled
        self.distributed_enabled = self.settings.llm_distributed_enabled

        if self.distributed_enabled:
            self._init_distributed()
        else:
            self._init_single()

        logger.info(
            f"LLM service initialized: "
            f"model={self.settings.llm_model}, "
            f"distributed={self.distributed_enabled}"
        )

    def _init_single(self):
        """Initialize single endpoint client."""
        self._sync_client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "ollama",
            timeout=self.settings.llm_timeout,
        )

        self._async_client = AsyncOpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "ollama",
            timeout=self.settings.llm_timeout,
        )

        logger.info(f"Single endpoint: {self.settings.llm_base_url}")

    def _init_distributed(self):
        """Initialize distributed client with load balancing."""
        from morgan.infrastructure import DistributedLLMClient

        endpoints = self.settings.get_llm_endpoints()

        self._distributed_client = DistributedLLMClient(
            endpoints=endpoints,
            model=self.settings.llm_model,
            strategy=self.settings.llm_load_balancing_strategy,
            api_key=self.settings.llm_api_key or "ollama",
            timeout=self.settings.llm_timeout,
            health_check_interval=self.settings.llm_health_check_interval,
        )

        # Start health monitoring
        self._distributed_client.start_health_monitoring()

        logger.info(
            f"Distributed mode: {len(endpoints)} endpoints, "
            f"strategy={self.settings.llm_load_balancing_strategy}"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response from LLM (synchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response from LLM (asynchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self.chat_async(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Chat with LLM using message history (synchronous).

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        if self.distributed_enabled:
            # Use async client via event loop
            return asyncio.get_event_loop().run_until_complete(
                self.chat_async(messages, max_tokens, temperature)
            )

        try:
            response = self._sync_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else self.settings.llm_temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "stop",
            )

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Chat with LLM using message history (asynchronous).

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        if self.distributed_enabled and self._distributed_client:
            # Use distributed client
            try:
                # Distributed client expects a single prompt, so we format messages
                formatted_prompt = self._format_messages_for_prompt(messages)

                content = await self._distributed_client.generate(
                    prompt=formatted_prompt,
                    temperature=temperature
                    if temperature is not None
                    else self.settings.llm_temperature,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                )

                return LLMResponse(
                    content=content,
                    usage={},
                    model=self.settings.llm_model,
                    finish_reason="stop",
                )

            except Exception as e:
                logger.error(f"Distributed LLM chat failed: {e}")
                raise

        try:
            response = await self._async_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else self.settings.llm_temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "stop",
            )

        except Exception as e:
            logger.error(f"LLM async chat failed: {e}")
            raise

    def _format_messages_for_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages list into a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n\n".join(parts)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream response from LLM (synchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Yields:
            Response chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            stream = self._sync_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else self.settings.llm_temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise

    async def stream_generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response from LLM (asynchronous).

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Yields:
            Response chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self._async_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature
                if temperature is not None
                else self.settings.llm_temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM async streaming failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if LLM service is available.

        Returns:
            True if available
        """
        try:
            if self.distributed_enabled and self._distributed_client:
                # Check distributed endpoints
                stats = self._distributed_client.get_stats()
                return stats["healthy_endpoints"] > 0

            # Single endpoint check
            self._sync_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            logger.error(f"LLM availability check failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on LLM service.

        Returns:
            Health status dictionary
        """
        if self.distributed_enabled and self._distributed_client:
            health = await self._distributed_client.health_check()
            stats = self._distributed_client.get_stats()
            return {
                "mode": "distributed",
                "healthy": stats["healthy_endpoints"] > 0,
                "endpoints": stats,
                "endpoint_health": health,
            }

        try:
            response = await self._async_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return {
                "mode": "single",
                "healthy": True,
                "endpoint": self.settings.llm_base_url,
                "model": self.settings.llm_model,
            }

        except Exception as e:
            return {
                "mode": "single",
                "healthy": False,
                "endpoint": self.settings.llm_base_url,
                "error": str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get LLM service statistics.

        Returns:
            Statistics dictionary
        """
        if self.distributed_enabled and self._distributed_client:
            return self._distributed_client.get_stats()

        return {
            "mode": "single",
            "endpoint": self.settings.llm_base_url,
            "model": self.settings.llm_model,
        }

    def shutdown(self):
        """Shutdown LLM service and cleanup resources."""
        if self._distributed_client:
            self._distributed_client.stop_health_monitoring()
            logger.info("Distributed LLM client shutdown")


# Singleton instance
_llm_service_instance = None
_llm_service_lock = threading.Lock()


def get_llm_service() -> LLMService:
    """
    Get singleton LLM service instance.

    Returns:
        Shared LLMService instance
    """
    global _llm_service_instance

    if _llm_service_instance is None:
        with _llm_service_lock:
            if _llm_service_instance is None:
                _llm_service_instance = LLMService()

    return _llm_service_instance


def reset_llm_service():
    """Reset the LLM service singleton (for testing)."""
    global _llm_service_instance

    with _llm_service_lock:
        if _llm_service_instance:
            _llm_service_instance.shutdown()
        _llm_service_instance = None
