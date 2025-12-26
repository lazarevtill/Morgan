"""
LLM Service for Morgan RAG.

Unified LLM interface supporting:
- Single endpoint (OpenAI-compatible)
- Distributed multi-host setup with load balancing
- Automatic failover and health monitoring
- Performance tracking and caching
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class LLMMode(str, Enum):
    """LLM operation mode."""

    SINGLE = "single"  # Single endpoint
    DISTRIBUTED = "distributed"  # Multi-host with load balancing


@dataclass
class LLMResponse:
    """LLM response wrapper."""

    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str
    endpoint: Optional[str] = None
    response_time_ms: Optional[float] = None


class LLMService:
    """
    Unified LLM service supporting single and distributed modes.

    Single mode: Direct OpenAI-compatible API calls
    Distributed mode: Load-balanced calls across multiple hosts

    Example:
        # Single mode (default)
        >>> service = LLMService()
        >>> response = service.generate("What is Python?")

        # Distributed mode
        >>> service = LLMService(
        ...     mode="distributed",
        ...     endpoints=["http://host1:11434/v1", "http://host2:11434/v1"]
        ... )
        >>> response = service.generate("What is Python?")
    """

    def __init__(
        self,
        mode: str = "single",
        endpoints: Optional[List[str]] = None,
        load_balancing_strategy: str = "round_robin",
    ):
        """
        Initialize LLM service.

        Args:
            mode: "single" or "distributed"
            endpoints: List of endpoint URLs for distributed mode
            load_balancing_strategy: Strategy for distributed mode
        """
        self.settings = get_settings()
        self.mode = LLMMode(mode)
        self.distributed_client = None

        # Initialize based on mode
        if self.mode == LLMMode.DISTRIBUTED:
            self._init_distributed(endpoints, load_balancing_strategy)
        else:
            self._init_single()

        logger.info("LLM service initialized in %s mode", self.mode.value)

    def _init_single(self):
        """Initialize single-endpoint mode."""
        self.client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "dummy-key",
        )
        self.distributed_client = None
        logger.info("LLM single mode: %s", self.settings.llm_base_url)

    def _init_distributed(self, endpoints: Optional[List[str]], strategy: str):
        """Initialize distributed mode."""
        try:
            from morgan.infrastructure.distributed_llm import DistributedLLMClient

            # Use provided endpoints or try to get from config
            if not endpoints:
                from morgan.config.distributed_config import get_distributed_config

                config = get_distributed_config()

                # Get LLM hosts
                llm_hosts = config.get_hosts_by_role("llm")
                if llm_hosts:
                    endpoints = [
                        f"http://{h.address}:{h.port}{h.api_path}" for h in llm_hosts
                    ]
                else:
                    # Fallback to single mode
                    logger.warning(
                        "No distributed endpoints configured, "
                        "falling back to single mode"
                    )
                    self.mode = LLMMode.SINGLE
                    self._init_single()
                    return

            self.distributed_client = DistributedLLMClient(
                endpoints=endpoints,
                model=self.settings.llm_model,
                strategy=strategy,
            )
            self.client = None  # Not used in distributed mode

            # Start health monitoring
            self.distributed_client.start_health_monitoring()

            logger.info("LLM distributed mode with %d endpoints", len(endpoints))

        except ImportError as e:
            logger.warning(
                "Distributed mode unavailable: %s, falling back to single", e
            )
            self.mode = LLMMode.SINGLE
            self._init_single()

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        if self.mode == LLMMode.DISTRIBUTED:
            return self._generate_distributed(
                prompt, max_tokens, temperature, system_prompt
            )
        else:
            return self._generate_single(prompt, max_tokens, temperature, system_prompt)

    def _generate_single(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using single endpoint."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                endpoint=self.settings.llm_base_url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    def _generate_distributed(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using distributed client."""
        try:
            start_time = time.time()

            # Build full prompt with system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Run async in sync context
            loop = asyncio.new_event_loop()
            try:
                content = loop.run_until_complete(
                    self.distributed_client.generate(
                        prompt=full_prompt,
                        temperature=temperature or self.settings.llm_temperature,
                        max_tokens=max_tokens or self.settings.llm_max_tokens,
                    )
                )
            finally:
                loop.close()

            elapsed_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                usage={},  # Distributed client doesn't return usage
                model=self.settings.llm_model,
                finish_reason="stop",
                endpoint="distributed",
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Distributed LLM generation failed: %s", e)
            raise

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Async generate response from LLM.

        Useful for distributed mode where async is native.
        """
        if self.mode == LLMMode.DISTRIBUTED and self.distributed_client:
            start_time = time.time()

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            content = await self.distributed_client.generate(
                prompt=full_prompt,
                temperature=temperature or self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                usage={},
                model=self.settings.llm_model,
                finish_reason="stop",
                endpoint="distributed",
                response_time_ms=elapsed_ms,
            )
        else:
            # Fall back to sync in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, max_tokens, temperature, system_prompt),
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Chat with LLM using message history.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        if self.mode == LLMMode.DISTRIBUTED:
            # Convert messages to single prompt for distributed client
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.upper()}: {content}")

            return self.generate(
                prompt="\n".join(prompt_parts),
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # Single mode - direct chat
        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                endpoint=self.settings.llm_base_url,
                response_time_ms=elapsed_ms,
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
    ) -> Iterator[str]:
        """
        Stream response from LLM.

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
            stream = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            raise

    def is_available(self) -> bool:
        """
        Check if LLM service is available.

        Returns:
            True if available
        """
        if self.mode == LLMMode.DISTRIBUTED and self.distributed_client:
            # Check if any endpoint is healthy
            stats = self.distributed_client.get_stats()
            return stats.get("healthy_endpoints", 0) > 0

        try:
            self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            logger.error("LLM availability check failed: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dict with service stats
        """
        if self.mode == LLMMode.DISTRIBUTED and self.distributed_client:
            return self.distributed_client.get_stats()

        return {
            "mode": self.mode.value,
            "endpoint": self.settings.llm_base_url,
            "model": self.settings.llm_model,
            "available": self.is_available(),
        }


# Singleton instance
_llm_service_instance = None
_llm_service_lock = threading.Lock()


def get_llm_service(
    mode: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
    force_new: bool = False,
) -> LLMService:
    """
    Get singleton LLM service instance.

    Args:
        mode: "single" or "distributed" (uses env var if not specified)
        endpoints: Endpoint URLs for distributed mode
        force_new: Force create new instance

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
                )

    return _llm_service_instance
