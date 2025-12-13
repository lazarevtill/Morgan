"""Ollama LLM client implementation."""

import asyncio
import json
from typing import List, Optional, AsyncIterator, Dict, Any
import aiohttp
from aiohttp import ClientTimeout, ClientError

from . import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
)


class OllamaClient(LLMClient):
    """
    Client for Ollama LLM service.

    Ollama provides a local API for running LLMs like Llama, Mistral, etc.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Ollama client.

        Args:
            endpoint: Ollama API endpoint (e.g., http://localhost:11434)
            model: Model name (e.g., "gemma3", "llama2")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format messages for Ollama API."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Any:
        """
        Make HTTP request with retry logic.

        Args:
            endpoint: API endpoint path
            payload: Request payload
            stream: Whether to stream the response

        Returns:
            Response data

        Raises:
            LLMConnectionError: If connection fails after retries
            LLMTimeoutError: If request times out
            LLMRateLimitError: If rate limit exceeded
        """
        session = await self._get_session()
        url = f"{self.endpoint}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload) as response:
                    # Check for rate limiting
                    if response.status == 429:
                        raise LLMRateLimitError(
                            f"Rate limit exceeded for {self.model}"
                        )

                    # Check for other errors
                    if response.status >= 400:
                        error_text = await response.text()
                        raise LLMError(
                            f"Ollama API error (status {response.status}): {error_text}"
                        )

                    if stream:
                        return response
                    else:
                        return await response.json()

            except asyncio.TimeoutError as e:
                if attempt == self.max_retries - 1:
                    raise LLMTimeoutError(
                        f"Request to {self.model} timed out after {self.timeout}s"
                    ) from e
                await asyncio.sleep(self.retry_delay * (attempt + 1))

            except ClientError as e:
                if attempt == self.max_retries - 1:
                    raise LLMConnectionError(
                        f"Failed to connect to Ollama at {self.endpoint}: {e}"
                    ) from e
                await asyncio.sleep(self.retry_delay * (attempt + 1))

            except (LLMRateLimitError, LLMError):
                # Don't retry on rate limit or API errors
                raise

        raise LLMConnectionError(
            f"Failed to connect to Ollama after {self.max_retries} attempts"
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Ollama.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama-specific parameters

        Returns:
            LLMResponse with generated content
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)

        response_data = await self._make_request("/api/chat", payload)

        # Extract response content
        content = response_data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self.model,
            finish_reason=response_data.get("done_reason"),
            usage={
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": (
                    response_data.get("prompt_eval_count", 0) +
                    response_data.get("eval_count", 0)
                ),
            },
            metadata={
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_duration": response_data.get("eval_duration"),
            }
        )

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from Ollama.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama-specific parameters

        Yields:
            Chunks of generated text
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)

        session = await self._get_session()
        url = f"{self.endpoint}/api/chat"

        try:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise LLMError(
                        f"Ollama API error (status {response.status}): {error_text}"
                    )

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Streaming request to {self.model} timed out"
            ) from e
        except ClientError as e:
            raise LLMConnectionError(
                f"Connection error during streaming: {e}"
            ) from e

    async def health_check(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.endpoint}/api/tags"

            async with session.get(url) as response:
                return response.status == 200

        except Exception:
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
