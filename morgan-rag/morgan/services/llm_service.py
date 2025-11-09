"""
LLM Service for Morgan RAG.

Simple OpenAI-compatible LLM interface.
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

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
    Simple LLM service using OpenAI-compatible API.
    """

    def __init__(self):
        """Initialize LLM service."""
        self.settings = get_settings()

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "dummy-key",
        )

        logger.info(f"LLM service initialized with model: {self.settings.llm_model}")

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
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

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
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=temperature or self.settings.llm_temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
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
            logger.error(f"LLM streaming failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if LLM service is available.

        Returns:
            True if available
        """
        try:
            self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            logger.error(f"LLM availability check failed: {e}")
            return False


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
