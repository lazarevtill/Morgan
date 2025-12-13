"""OpenAI-compatible LLM client implementation."""

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


class OpenAICompatibleClient(LLMClient):
    """
    Client for OpenAI-compatible LLM endpoints.

    Works with any service that implements the OpenAI Chat Completions API,
    including LM Studio, LocalAI, vLLM, and others.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            endpoint: API endpoint (e.g., http://localhost:1234/v1)
            model: Model name
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.timeout)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format messages for OpenAI API."""
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
                            f"API error (status {response.status}): {error_text}"
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
                        f"Failed to connect to endpoint {self.endpoint}: {e}"
                    ) from e
                await asyncio.sleep(self.retry_delay * (attempt + 1))

            except (LLMRateLimitError, LLMError):
                # Don't retry on rate limit or API errors
                raise

        raise LLMConnectionError(
            f"Failed to connect after {self.max_retries} attempts"
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using OpenAI-compatible API.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API-specific parameters

        Returns:
            LLMResponse with generated content
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "stream": False,
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add any additional parameters
        payload.update(kwargs)

        response_data = await self._make_request("/chat/completions", payload)

        # Extract response content
        choice = response_data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=response_data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            usage=response_data.get("usage", {}),
            metadata={
                "id": response_data.get("id"),
                "created": response_data.get("created"),
                "object": response_data.get("object"),
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
        Generate a streaming response using OpenAI-compatible API.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API-specific parameters

        Yields:
            Chunks of generated text
        """
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add any additional parameters
        payload.update(kwargs)

        session = await self._get_session()
        url = f"{self.endpoint}/chat/completions"

        try:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise LLMError(
                        f"API error (status {response.status}): {error_text}"
                    )

                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()

                        # Skip empty lines
                        if not line_str:
                            continue

                        # OpenAI streaming format: "data: {json}\n\n"
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix

                            # Check for end of stream
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
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
        Check if the API endpoint is available.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.endpoint}/models"

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
