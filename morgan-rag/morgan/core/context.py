"""
Context management for conversation handling.

Manages:
- Context window tracking (token limits)
- Context relevance scoring
- Context pruning strategies
- Context compression
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from morgan.core.types import (
    ConversationContext,
    ContextPruningStrategy,
    EmotionalState,
    Message,
    MessageRole,
    UserProfile,
)

logger = logging.getLogger(__name__)


class ContextError(Exception):
    """Base exception for context operations."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id


class ContextOverflowError(ContextError):
    """Context exceeds token limits."""

    pass


class ContextManager:
    """
    Manages conversation context with intelligent pruning and compression.

    Features:
    - Token counting and limits
    - Importance-based message scoring
    - Multiple pruning strategies
    - Context compression via summarization
    - Fast operation (< 50ms target)
    """

    def __init__(
        self,
        max_context_tokens: int = 8000,
        target_context_tokens: int = 6000,
        system_prompt_tokens: int = 500,
        default_pruning_strategy: ContextPruningStrategy = ContextPruningStrategy.HYBRID,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize context manager.

        Args:
            max_context_tokens: Hard limit for context size
            target_context_tokens: Target size for pruning
            system_prompt_tokens: Reserve tokens for system prompt
            default_pruning_strategy: Default strategy for pruning
            correlation_id: Correlation ID for tracing
        """
        self.max_context_tokens = max_context_tokens
        self.target_context_tokens = target_context_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.default_pruning_strategy = default_pruning_strategy
        self.correlation_id = correlation_id

        # Metrics
        self._metrics = {
            "contexts_built": 0,
            "contexts_pruned": 0,
            "tokens_pruned": 0,
        }

        logger.info(
            "Context manager initialized",
            extra={
                "max_tokens": max_context_tokens,
                "target_tokens": target_context_tokens,
                "correlation_id": correlation_id,
            },
        )

    async def build_context(
        self,
        messages: List[Message],
        user_id: str,
        session_id: str,
        user_profile: Optional[UserProfile] = None,
        emotional_state: Optional[EmotionalState] = None,
        max_tokens: Optional[int] = None,
    ) -> ConversationContext:
        """
        Build conversation context from messages.

        Args:
            messages: List of conversation messages
            user_id: User identifier
            session_id: Session identifier
            user_profile: Optional user profile
            emotional_state: Optional emotional state
            max_tokens: Optional max tokens override

        Returns:
            Conversation context

        Raises:
            ContextOverflowError: If context is too large even after pruning
        """
        start_time = time.time()

        try:
            # Calculate token limits
            max_tokens = max_tokens or self.max_context_tokens
            available_tokens = max_tokens - self.system_prompt_tokens

            # Calculate current token count
            total_tokens = await self._count_tokens(messages)

            # Prune if necessary
            if total_tokens > available_tokens:
                logger.debug(
                    "Context exceeds limit, pruning",
                    extra={
                        "current_tokens": total_tokens,
                        "available_tokens": available_tokens,
                    },
                )

                messages = await self.prune_context(
                    messages,
                    target_tokens=self.target_context_tokens,
                )

                total_tokens = await self._count_tokens(messages)

                # Check if still too large
                if total_tokens > available_tokens:
                    raise ContextOverflowError(
                        f"Context too large even after pruning: {total_tokens} > {available_tokens}",
                        correlation_id=self.correlation_id,
                    )

                self._metrics["contexts_pruned"] += 1

            # Build context
            context = ConversationContext(
                messages=messages,
                user_id=user_id,
                session_id=session_id,
                user_profile=user_profile,
                emotional_state=emotional_state,
                total_tokens=total_tokens,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )

            self._metrics["contexts_built"] += 1

            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Context built successfully",
                extra={
                    "messages": len(messages),
                    "tokens": total_tokens,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return context

        except ContextOverflowError:
            raise
        except Exception as e:
            logger.error(
                "Failed to build context",
                extra={"error": str(e)},
            )
            raise ContextError(
                f"Failed to build context: {e}",
                correlation_id=self.correlation_id,
            ) from e

    async def prune_context(
        self,
        messages: List[Message],
        target_tokens: int,
        strategy: Optional[ContextPruningStrategy] = None,
    ) -> List[Message]:
        """
        Prune messages to fit within token limit.

        Args:
            messages: Messages to prune
            target_tokens: Target token count
            strategy: Pruning strategy to use

        Returns:
            Pruned messages
        """
        strategy = strategy or self.default_pruning_strategy

        if strategy == ContextPruningStrategy.SLIDING_WINDOW:
            return await self._prune_sliding_window(messages, target_tokens)
        elif strategy == ContextPruningStrategy.IMPORTANCE_BASED:
            return await self._prune_importance_based(messages, target_tokens)
        elif strategy == ContextPruningStrategy.RECENCY_WEIGHTED:
            return await self._prune_recency_weighted(messages, target_tokens)
        else:  # HYBRID
            return await self._prune_hybrid(messages, target_tokens)

    async def _prune_sliding_window(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> List[Message]:
        """
        Prune using sliding window (keep most recent).

        Args:
            messages: Messages to prune
            target_tokens: Target token count

        Returns:
            Pruned messages
        """
        pruned_messages: List[Message] = []
        current_tokens = 0

        # Take messages from the end (most recent)
        for message in reversed(messages):
            msg_tokens = await self._estimate_message_tokens(message)

            if current_tokens + msg_tokens <= target_tokens:
                pruned_messages.insert(0, message)
                current_tokens += msg_tokens
            else:
                break

        tokens_removed = await self._count_tokens(messages) - current_tokens
        self._metrics["tokens_pruned"] += tokens_removed

        logger.debug(
            "Sliding window pruning completed",
            extra={
                "original": len(messages),
                "pruned": len(pruned_messages),
                "tokens_removed": tokens_removed,
            },
        )

        return pruned_messages

    async def _prune_importance_based(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> List[Message]:
        """
        Prune based on message importance scores.

        Args:
            messages: Messages to prune
            target_tokens: Target token count

        Returns:
            Pruned messages
        """
        # Score and sort messages by importance
        scored_messages = [
            (msg, msg.importance_score) for msg in messages
        ]
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Select top messages that fit
        pruned_messages: List[Message] = []
        current_tokens = 0

        for message, _ in scored_messages:
            msg_tokens = await self._estimate_message_tokens(message)

            if current_tokens + msg_tokens <= target_tokens:
                pruned_messages.append(message)
                current_tokens += msg_tokens
            else:
                break

        # Re-sort by timestamp to maintain conversation order
        pruned_messages.sort(key=lambda m: m.timestamp)

        tokens_removed = await self._count_tokens(messages) - current_tokens
        self._metrics["tokens_pruned"] += tokens_removed

        logger.debug(
            "Importance-based pruning completed",
            extra={
                "original": len(messages),
                "pruned": len(pruned_messages),
                "tokens_removed": tokens_removed,
            },
        )

        return pruned_messages

    async def _prune_recency_weighted(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> List[Message]:
        """
        Prune with recency weighting (recent = higher score).

        Args:
            messages: Messages to prune
            target_tokens: Target token count

        Returns:
            Pruned messages
        """
        if not messages:
            return []

        # Calculate recency scores
        now = datetime.now()
        scored_messages: List[Tuple[Message, float]] = []

        for idx, message in enumerate(messages):
            # Recency factor (more recent = higher)
            recency_factor = (idx + 1) / len(messages)

            # Time decay
            time_diff = (now - message.timestamp).total_seconds()
            time_decay = 1.0 / (1.0 + time_diff / 3600.0)  # Decay over hours

            # Combined score
            score = (
                message.importance_score * 0.4
                + recency_factor * 0.4
                + time_decay * 0.2
            )

            scored_messages.append((message, score))

        # Sort by score
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Select top messages
        pruned_messages: List[Message] = []
        current_tokens = 0

        for message, _ in scored_messages:
            msg_tokens = await self._estimate_message_tokens(message)

            if current_tokens + msg_tokens <= target_tokens:
                pruned_messages.append(message)
                current_tokens += msg_tokens
            else:
                break

        # Re-sort by timestamp
        pruned_messages.sort(key=lambda m: m.timestamp)

        tokens_removed = await self._count_tokens(messages) - current_tokens
        self._metrics["tokens_pruned"] += tokens_removed

        logger.debug(
            "Recency-weighted pruning completed",
            extra={
                "original": len(messages),
                "pruned": len(pruned_messages),
                "tokens_removed": tokens_removed,
            },
        )

        return pruned_messages

    async def _prune_hybrid(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> List[Message]:
        """
        Hybrid pruning: keep recent + important messages.

        Strategy:
        1. Always keep last N messages (recency)
        2. Fill remaining space with important messages

        Args:
            messages: Messages to prune
            target_tokens: Target token count

        Returns:
            Pruned messages
        """
        if not messages:
            return []

        # Reserve 50% of tokens for recent messages
        recent_tokens = target_tokens // 2
        important_tokens = target_tokens - recent_tokens

        # Get recent messages (sliding window)
        recent_messages: List[Message] = []
        current_tokens = 0

        for message in reversed(messages):
            msg_tokens = await self._estimate_message_tokens(message)

            if current_tokens + msg_tokens <= recent_tokens:
                recent_messages.insert(0, message)
                current_tokens += msg_tokens
            else:
                break

        # Get important messages from older ones
        older_messages = messages[:-len(recent_messages)] if recent_messages else messages

        # Score older messages by importance
        scored_older = [
            (msg, msg.importance_score) for msg in older_messages
        ]
        scored_older.sort(key=lambda x: x[1], reverse=True)

        # Add important older messages
        important_messages: List[Message] = []
        important_current_tokens = 0

        for message, _ in scored_older:
            msg_tokens = await self._estimate_message_tokens(message)

            if important_current_tokens + msg_tokens <= important_tokens:
                important_messages.append(message)
                important_current_tokens += msg_tokens
            else:
                break

        # Combine and sort by timestamp
        pruned_messages = important_messages + recent_messages
        pruned_messages.sort(key=lambda m: m.timestamp)

        tokens_removed = await self._count_tokens(messages) - (current_tokens + important_current_tokens)
        self._metrics["tokens_pruned"] += tokens_removed

        logger.debug(
            "Hybrid pruning completed",
            extra={
                "original": len(messages),
                "pruned": len(pruned_messages),
                "recent": len(recent_messages),
                "important": len(important_messages),
                "tokens_removed": tokens_removed,
            },
        )

        return pruned_messages

    async def compress_context(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """
        Compress context by summarizing older messages.

        Note: This is a placeholder. In production, you'd want to:
        1. Use an LLM to summarize older messages
        2. Replace multiple messages with a summary message
        3. Preserve recent messages for context

        Args:
            messages: Messages to compress

        Returns:
            Compressed messages
        """
        # For now, just return messages as-is
        # In production, implement actual summarization
        logger.debug("Context compression not yet implemented")
        return messages

    async def _count_tokens(
        self,
        messages: List[Message],
    ) -> int:
        """
        Count total tokens in messages.

        Args:
            messages: Messages to count

        Returns:
            Total token count
        """
        total = 0
        for message in messages:
            if message.tokens is not None:
                total += message.tokens
            else:
                total += await self._estimate_message_tokens(message)
        return total

    async def _estimate_message_tokens(
        self,
        message: Message,
    ) -> int:
        """
        Estimate tokens for a message.

        Simple estimation: ~4 characters per token.
        In production, use tiktoken or similar.

        Args:
            message: Message to estimate

        Returns:
            Estimated token count
        """
        if message.tokens is not None:
            return message.tokens

        # Simple estimation
        char_count = len(message.content)
        token_count = max(1, char_count // 4)

        return token_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "config": {
                "max_context_tokens": self.max_context_tokens,
                "target_context_tokens": self.target_context_tokens,
                "system_prompt_tokens": self.system_prompt_tokens,
                "default_strategy": self.default_pruning_strategy.value,
            },
            "metrics": self._metrics.copy(),
        }
