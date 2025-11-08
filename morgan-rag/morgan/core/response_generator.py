"""
Response generation with LLM integration.

Features:
- Prompt construction from context + RAG + emotion
- LLM API calls with retry logic
- Streaming support
- Response post-processing
- Response validation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from morgan.core.types import (
    AssistantResponse,
    ConversationContext,
    MessageRole,
    SearchSource,
)
from morgan.emotions.types import EmotionResult
from morgan.core.search import SearchResult

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Error during response generation."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id
        self.recoverable = recoverable


class ValidationError(Exception):
    """Error validating response."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id


class ResponseGenerator:
    """
    Generates assistant responses using LLM.

    Features:
    - Emotion-aware prompting
    - RAG-enhanced responses with citations
    - Retry logic with exponential backoff
    - Streaming support
    - Response validation
    - Performance tracking
    """

    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2:latest",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        max_retries: int = 3,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize response generator.

        Args:
            llm_base_url: Base URL for LLM API (Ollama)
            llm_model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
            max_retries: Maximum retry attempts
            correlation_id: Correlation ID for tracing
        """
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.correlation_id = correlation_id

        # HTTP client
        self._client = httpx.AsyncClient(
            base_url=llm_base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

        # Metrics
        self._metrics = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_tokens_generated": 0,
        }

        logger.info(
            "Response generator initialized",
            extra={
                "llm_model": llm_model,
                "base_url": llm_base_url,
                "correlation_id": correlation_id,
            },
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self._client.aclose()
        logger.info("Response generator cleaned up")

    async def generate(
        self,
        context: ConversationContext,
        user_message: str,
        rag_results: Optional[List[SearchResult]] = None,
        detected_emotion: Optional[EmotionResult] = None,
    ) -> AssistantResponse:
        """
        Generate assistant response.

        Args:
            context: Conversation context
            user_message: Current user message
            rag_results: Optional RAG search results
            detected_emotion: Optional detected emotion

        Returns:
            Assistant response

        Raises:
            GenerationError: If generation fails
        """
        start_time = time.time()
        response_id = self._generate_id()

        try:
            # Build prompt
            prompt = await self._build_prompt(
                context,
                user_message,
                rag_results,
                detected_emotion,
            )

            # Generate response with retries
            response_text = await self._generate_with_retry(prompt)

            # Post-process response
            processed_text = await self._post_process_response(
                response_text,
                rag_results,
            )

            # Validate response
            await self._validate_response(processed_text)

            # Convert RAG results to sources
            sources = self._convert_rag_to_sources(rag_results) if rag_results else []

            # Create response
            duration_ms = (time.time() - start_time) * 1000

            response = AssistantResponse(
                content=processed_text,
                response_id=response_id,
                timestamp=datetime.now(),
                sources=sources,
                emotion=detected_emotion,
                confidence=1.0,
                generation_time_ms=duration_ms,
                metadata={
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    "rag_sources": len(sources),
                },
            )

            self._metrics["total_generations"] += 1
            self._metrics["successful_generations"] += 1

            logger.info(
                "Response generated successfully",
                extra={
                    "response_id": response_id,
                    "duration_ms": round(duration_ms, 2),
                    "sources": len(sources),
                },
            )

            return response

        except Exception as e:
            self._metrics["total_generations"] += 1
            self._metrics["failed_generations"] += 1

            logger.error(
                "Response generation failed",
                extra={
                    "error": str(e),
                    "correlation_id": self.correlation_id,
                },
            )

            raise GenerationError(
                f"Failed to generate response: {e}",
                correlation_id=self.correlation_id,
            ) from e

    async def generate_stream(
        self,
        context: ConversationContext,
        user_message: str,
        rag_results: Optional[List[SearchResult]] = None,
        detected_emotion: Optional[EmotionResult] = None,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.

        Args:
            context: Conversation context
            user_message: Current user message
            rag_results: Optional RAG search results
            detected_emotion: Optional detected emotion

        Yields:
            Response chunks

        Raises:
            GenerationError: If generation fails
        """
        try:
            # Build prompt
            prompt = await self._build_prompt(
                context,
                user_message,
                rag_results,
                detected_emotion,
            )

            # Stream response
            async for chunk in self._generate_stream_with_retry(prompt):
                yield chunk

            self._metrics["total_generations"] += 1
            self._metrics["successful_generations"] += 1

        except Exception as e:
            self._metrics["total_generations"] += 1
            self._metrics["failed_generations"] += 1

            logger.error(
                "Streaming generation failed",
                extra={"error": str(e)},
            )

            raise GenerationError(
                f"Failed to stream response: {e}",
                correlation_id=self.correlation_id,
            ) from e

    async def _build_prompt(
        self,
        context: ConversationContext,
        user_message: str,
        rag_results: Optional[List[SearchResult]],
        detected_emotion: Optional[EmotionResult],
    ) -> str:
        """
        Build prompt from context, RAG, and emotion.

        Args:
            context: Conversation context
            user_message: User message
            rag_results: RAG results
            detected_emotion: Detected emotion

        Returns:
            Constructed prompt
        """
        prompt_parts: List[str] = []

        # System prompt
        system_prompt = self._build_system_prompt(
            context,
            detected_emotion,
        )
        prompt_parts.append(system_prompt)

        # RAG context
        if rag_results:
            rag_context = self._build_rag_context(rag_results)
            prompt_parts.append(rag_context)

        # Conversation history
        for message in context.messages[-10:]:  # Last 10 messages
            role = message.role.value
            content = message.content
            prompt_parts.append(f"{role}: {content}")

        # Current user message
        prompt_parts.append(f"user: {user_message}")
        prompt_parts.append("assistant:")

        return "\n\n".join(prompt_parts)

    def _build_system_prompt(
        self,
        context: ConversationContext,
        detected_emotion: Optional[EmotionResult],
    ) -> str:
        """
        Build system prompt with emotion awareness.

        Args:
            context: Conversation context
            detected_emotion: Detected emotion

        Returns:
            System prompt
        """
        parts = [
            "You are Morgan, a helpful and empathetic AI assistant.",
            "You provide thoughtful, accurate responses based on the conversation context.",
        ]

        # Add emotional awareness
        if detected_emotion and detected_emotion.primary_emotion:
            emotion_type = detected_emotion.primary_emotion.emotion_type.value
            intensity = detected_emotion.primary_emotion.intensity

            parts.append(
                f"The user seems to be feeling {emotion_type} "
                f"(intensity: {intensity:.2f}). "
                "Please respond with appropriate empathy and support."
            )

        # Add user preferences if available
        if context.user_profile and context.user_profile.preferences:
            parts.append(
                "Consider the user's preferences and communication style "
                "when crafting your response."
            )

        return "\n".join(parts)

    def _build_rag_context(
        self,
        rag_results: List[SearchResult],
    ) -> str:
        """
        Build RAG context section.

        Args:
            rag_results: RAG search results

        Returns:
            RAG context string
        """
        if not rag_results:
            return ""

        parts = ["Relevant information from knowledge base:\n"]

        for idx, result in enumerate(rag_results[:5], 1):  # Top 5
            parts.append(
                f"[{idx}] {result.content}"
                f" (source: {result.metadata.get('source', 'unknown')})"
            )

        parts.append(
            "\nUse this information to enhance your response when relevant. "
            "Cite sources when appropriate."
        )

        return "\n".join(parts)

    async def _generate_with_retry(
        self,
        prompt: str,
    ) -> str:
        """
        Generate response with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            Generated response

        Raises:
            GenerationError: If all retries fail
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
        ):
            with attempt:
                # Call Ollama API
                response = await self._client.post(
                    "/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    },
                )

                response.raise_for_status()
                data = response.json()

                return data.get("response", "")

        raise GenerationError(
            "Failed to generate response after retries",
            correlation_id=self.correlation_id,
        )

    async def _generate_stream_with_retry(
        self,
        prompt: str,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response with retry logic.

        Args:
            prompt: Input prompt

        Yields:
            Response chunks
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
        ):
            with attempt:
                # Call Ollama streaming API
                async with self._client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    },
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                chunk = data.get("response", "")
                                if chunk:
                                    yield chunk
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Failed to decode streaming chunk",
                                    extra={"line": line[:100]},
                                )

    async def _post_process_response(
        self,
        response: str,
        rag_results: Optional[List[SearchResult]],
    ) -> str:
        """
        Post-process generated response.

        Args:
            response: Raw response
            rag_results: RAG results for citation

        Returns:
            Processed response
        """
        # Clean up response
        processed = response.strip()

        # Add citations if RAG was used
        if rag_results and "[" in processed:
            # Citations are already in the format [1], [2], etc.
            # Could enhance citation formatting here
            pass

        return processed

    async def _validate_response(
        self,
        response: str,
    ) -> None:
        """
        Validate generated response.

        Args:
            response: Response to validate

        Raises:
            ValidationError: If validation fails
        """
        if not response or not response.strip():
            raise ValidationError(
                "Generated response is empty",
                correlation_id=self.correlation_id,
            )

        # Check minimum length
        if len(response.strip()) < 10:
            raise ValidationError(
                "Generated response too short",
                correlation_id=self.correlation_id,
            )

        # Could add more validation:
        # - Content safety
        # - Factual consistency
        # - Citation validation
        # etc.

    def _convert_rag_to_sources(
        self,
        rag_results: List[SearchResult],
    ) -> List[SearchSource]:
        """
        Convert RAG results to SearchSource objects.

        Args:
            rag_results: RAG search results

        Returns:
            List of SearchSource objects
        """
        sources: List[SearchSource] = []

        for result in rag_results:
            source = SearchSource(
                content=result.content,
                source=result.metadata.get("source", "unknown"),
                score=result.score,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                metadata=result.metadata,
            )
            sources.append(source)

        return sources

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import hashlib
        return hashlib.sha256(
            f"{time.time()}{id(self)}".encode()
        ).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "config": {
                "model": self.llm_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "metrics": self._metrics.copy(),
        }
