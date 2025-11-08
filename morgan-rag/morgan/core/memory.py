"""
Multi-layer memory system for the assistant.

Provides:
- Short-term memory (current conversation, in-memory)
- Working memory (active processing, Redis/in-memory)
- Long-term memory (historical, persistent storage)
- Memory consolidation (background task)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.core.types import (
    ConversationContext,
    EmotionalState,
    MemoryEntry,
    MemoryType,
    Message,
    MessageRole,
    UserProfile,
)
from morgan.learning.exceptions import LearningStorageError

logger = logging.getLogger(__name__)


class MemoryError(Exception):
    """Base exception for memory operations."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id
        self.recoverable = recoverable


class MemoryRetrievalError(MemoryError):
    """Error during memory retrieval."""

    pass


class MemoryStorageError(MemoryError):
    """Error during memory storage."""

    pass


class MemorySystem:
    """
    Multi-layer memory system.

    Layers:
    1. Short-term: Current conversation (fast, in-memory)
    2. Working: Processing buffer (Redis/in-memory)
    3. Long-term: Historical conversations (persistent)
    4. Consolidated: Important patterns (background processed)

    Features:
    - Fast retrieval (< 100ms target)
    - Automatic cleanup of expired memories
    - Memory consolidation
    - Importance-based storage
    - Session management
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_short_term_messages: int = 100,
        max_working_memory_size: int = 1000,
        cleanup_interval_seconds: int = 300,
        consolidation_interval_hours: int = 24,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize memory system.

        Args:
            storage_path: Path for persistent storage
            max_short_term_messages: Max messages per session in short-term
            max_working_memory_size: Max entries in working memory
            cleanup_interval_seconds: How often to cleanup expired memories
            consolidation_interval_hours: How often to consolidate memories
            correlation_id: Correlation ID for tracing
        """
        self.storage_path = storage_path or Path.home() / ".morgan" / "memory"
        self.max_short_term_messages = max_short_term_messages
        self.max_working_memory_size = max_working_memory_size
        self.cleanup_interval = cleanup_interval_seconds
        self.consolidation_interval = consolidation_interval_hours
        self.correlation_id = correlation_id or self._generate_id()

        # In-memory stores
        self._short_term: Dict[str, List[Message]] = defaultdict(list)  # session_id -> messages
        self._working: Dict[str, MemoryEntry] = {}  # entry_id -> entry
        self._consolidated: Dict[str, List[MemoryEntry]] = defaultdict(list)  # user_id -> entries

        # User profiles cache
        self._user_profiles: Dict[str, UserProfile] = {}
        self._emotional_states: Dict[str, EmotionalState] = {}

        # Synchronization
        self._lock = asyncio.Lock()
        self._session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics = {
            "total_stores": 0,
            "total_retrievals": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Memory system initialized",
            extra={
                "storage_path": str(self.storage_path),
                "correlation_id": self.correlation_id,
            },
        )

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.sha256(
            f"{time.time()}{id(self)}".encode()
        ).hexdigest()[:16]

    async def initialize(self) -> None:
        """Initialize background tasks."""
        logger.info("Starting memory system background tasks")

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Start consolidation task
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        logger.info("Stopping memory system")

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

        # Persist remaining data
        await self._persist_all()

        logger.info("Memory system stopped")

    async def store_message(
        self,
        session_id: str,
        message: Message,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Store a message in short-term memory.

        Args:
            session_id: Session identifier
            message: Message to store
            user_id: Optional user identifier

        Raises:
            MemoryStorageError: If storage fails
        """
        try:
            async with self._session_locks[session_id]:
                # Add to short-term memory
                self._short_term[session_id].append(message)

                # Trim if too large
                if len(self._short_term[session_id]) > self.max_short_term_messages:
                    # Keep most recent messages
                    self._short_term[session_id] = self._short_term[session_id][
                        -self.max_short_term_messages :
                    ]

                self._metrics["total_stores"] += 1

                logger.debug(
                    "Message stored in short-term memory",
                    extra={
                        "session_id": session_id,
                        "role": message.role.value,
                        "message_id": message.message_id,
                    },
                )

        except Exception as e:
            logger.error(
                "Failed to store message",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise MemoryStorageError(
                f"Failed to store message: {e}",
                correlation_id=self.correlation_id,
            ) from e

    async def retrieve_context(
        self,
        session_id: str,
        n_messages: Optional[int] = None,
    ) -> List[Message]:
        """
        Retrieve recent conversation context.

        Args:
            session_id: Session identifier
            n_messages: Number of recent messages to retrieve

        Returns:
            List of recent messages

        Raises:
            MemoryRetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            async with self._session_locks[session_id]:
                messages = self._short_term.get(session_id, [])

                if n_messages is not None:
                    messages = messages[-n_messages:]

                self._metrics["total_retrievals"] += 1

                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    "Context retrieved",
                    extra={
                        "session_id": session_id,
                        "messages": len(messages),
                        "duration_ms": round(duration_ms, 2),
                    },
                )

                return list(messages)  # Return copy

        except Exception as e:
            logger.error(
                "Failed to retrieve context",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise MemoryRetrievalError(
                f"Failed to retrieve context: {e}",
                correlation_id=self.correlation_id,
            ) from e

    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[Message]:
        """
        Search memories by semantic similarity.

        Note: This is a simplified implementation using keyword matching.
        In production, you'd want to use embeddings for semantic search.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
            memory_types: Optional filter by memory types

        Returns:
            List of relevant messages
        """
        start_time = time.time()

        try:
            # Collect all relevant memories
            all_messages: List[Message] = []

            # Search short-term (all sessions for this user)
            for session_id, messages in self._short_term.items():
                # Simple session filtering - in production use proper user mapping
                all_messages.extend(messages)

            # Search working memory
            for entry in self._working.values():
                if entry.user_id == user_id:
                    if memory_types is None or entry.memory_type in memory_types:
                        all_messages.append(entry.message)

            # Search consolidated
            for entry in self._consolidated.get(user_id, []):
                if memory_types is None or entry.memory_type in memory_types:
                    all_messages.append(entry.message)

            # Simple keyword-based scoring (replace with embedding search in production)
            query_lower = query.lower()
            scored_messages: List[Tuple[Message, float]] = []

            for msg in all_messages:
                content_lower = msg.content.lower()
                # Simple scoring: count keyword matches
                score = sum(
                    word in content_lower for word in query_lower.split()
                )
                if score > 0:
                    scored_messages.append((msg, score))

            # Sort by score
            scored_messages.sort(key=lambda x: x[1], reverse=True)

            # Return top results
            results = [msg for msg, _ in scored_messages[:limit]]

            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Memory search completed",
                extra={
                    "user_id": user_id,
                    "query_length": len(query),
                    "results": len(results),
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Memory search failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            return []

    async def get_user_profile(
        self,
        user_id: str,
    ) -> Optional[UserProfile]:
        """
        Get user profile from cache or storage.

        Args:
            user_id: User identifier

        Returns:
            User profile if found
        """
        # Check cache first
        if user_id in self._user_profiles:
            self._metrics["cache_hits"] += 1
            return self._user_profiles[user_id]

        self._metrics["cache_misses"] += 1

        # Try to load from storage
        profile_path = self.storage_path / "profiles" / f"{user_id}.json"

        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)

                # Reconstruct profile (simplified - you'd want proper deserialization)
                profile = UserProfile(
                    user_id=data["user_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_active=datetime.fromisoformat(data["last_active"]),
                    conversation_count=data.get("conversation_count", 0),
                    total_messages=data.get("total_messages", 0),
                    metadata=data.get("metadata", {}),
                )

                # Cache it
                self._user_profiles[user_id] = profile

                return profile

            except Exception as e:
                logger.error(
                    "Failed to load user profile",
                    extra={"user_id": user_id, "error": str(e)},
                )

        return None

    async def update_user_profile(
        self,
        profile: UserProfile,
    ) -> None:
        """
        Update user profile in cache and storage.

        Args:
            profile: User profile to update
        """
        # Update cache
        self._user_profiles[profile.user_id] = profile

        # Persist to storage
        profile_path = self.storage_path / "profiles" / f"{profile.user_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "user_id": profile.user_id,
                "created_at": profile.created_at.isoformat(),
                "last_active": profile.last_active.isoformat(),
                "conversation_count": profile.conversation_count,
                "total_messages": profile.total_messages,
                "metadata": profile.metadata,
            }

            with open(profile_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                "User profile updated",
                extra={"user_id": profile.user_id},
            )

        except Exception as e:
            logger.error(
                "Failed to persist user profile",
                extra={"user_id": profile.user_id, "error": str(e)},
            )

    async def get_emotional_state(
        self,
        user_id: str,
    ) -> Optional[EmotionalState]:
        """
        Get current emotional state for user.

        Args:
            user_id: User identifier

        Returns:
            Emotional state if available
        """
        return self._emotional_states.get(user_id)

    async def update_emotional_state(
        self,
        state: EmotionalState,
    ) -> None:
        """
        Update emotional state for user.

        Args:
            state: Emotional state to update
        """
        self._emotional_states[state.user_id] = state

        logger.debug(
            "Emotional state updated",
            extra={"user_id": state.user_id},
        )

    async def consolidate_memories(
        self,
        user_id: str,
    ) -> int:
        """
        Consolidate memories for a user.

        This identifies important patterns and moves them to consolidated storage.

        Args:
            user_id: User identifier

        Returns:
            Number of memories consolidated
        """
        logger.info("Starting memory consolidation", extra={"user_id": user_id})

        consolidated_count = 0

        try:
            # Collect all memories for user
            all_entries: List[MemoryEntry] = []

            # From working memory
            for entry in self._working.values():
                if entry.user_id == user_id:
                    all_entries.append(entry)

            # Sort by importance
            all_entries.sort(key=lambda e: e.importance_score, reverse=True)

            # Move top important ones to consolidated
            consolidation_threshold = 0.7
            for entry in all_entries:
                if entry.importance_score >= consolidation_threshold:
                    # Create consolidated entry
                    consolidated_entry = MemoryEntry(
                        entry_id=entry.entry_id,
                        user_id=entry.user_id,
                        session_id=entry.session_id,
                        memory_type=MemoryType.CONSOLIDATED,
                        message=entry.message,
                        created_at=entry.created_at,
                        importance_score=entry.importance_score,
                        access_count=entry.access_count,
                        last_accessed=entry.last_accessed,
                        metadata=entry.metadata,
                    )

                    self._consolidated[user_id].append(consolidated_entry)
                    consolidated_count += 1

            logger.info(
                "Memory consolidation completed",
                extra={
                    "user_id": user_id,
                    "consolidated": consolidated_count,
                },
            )

            return consolidated_count

        except Exception as e:
            logger.error(
                "Memory consolidation failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            return 0

    async def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Clear a session from short-term memory.

        Args:
            session_id: Session to clear
        """
        async with self._session_locks[session_id]:
            if session_id in self._short_term:
                del self._short_term[session_id]

            logger.debug(
                "Session cleared",
                extra={"session_id": session_id},
            )

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired memories."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", extra={"error": str(e)})

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from working memory."""
        now = datetime.now()
        expired_ids: List[str] = []

        async with self._lock:
            for entry_id, entry in self._working.items():
                if entry.expires_at and entry.expires_at < now:
                    expired_ids.append(entry_id)

            for entry_id in expired_ids:
                del self._working[entry_id]

        if expired_ids:
            logger.debug(
                "Expired memories cleaned up",
                extra={"count": len(expired_ids)},
            )

    async def _consolidation_loop(self) -> None:
        """Background task for memory consolidation."""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval * 3600)
                await self._consolidate_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consolidation loop error", extra={"error": str(e)})

    async def _consolidate_all(self) -> None:
        """Consolidate memories for all users."""
        # Get all unique user IDs
        user_ids: Set[str] = set()

        for entry in self._working.values():
            user_ids.add(entry.user_id)

        for user_id in self._consolidated.keys():
            user_ids.add(user_id)

        # Consolidate for each user
        for user_id in user_ids:
            await self.consolidate_memories(user_id)

    async def _persist_all(self) -> None:
        """Persist all in-memory data to storage."""
        logger.info("Persisting all memory data")

        try:
            # Persist user profiles
            for profile in self._user_profiles.values():
                await self.update_user_profile(profile)

            # Could also persist other data structures here
            # For now, we just persist profiles

        except Exception as e:
            logger.error("Failed to persist data", extra={"error": str(e)})

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "short_term_sessions": len(self._short_term),
            "short_term_messages": sum(
                len(msgs) for msgs in self._short_term.values()
            ),
            "working_memory_entries": len(self._working),
            "consolidated_users": len(self._consolidated),
            "consolidated_entries": sum(
                len(entries) for entries in self._consolidated.values()
            ),
            "user_profiles_cached": len(self._user_profiles),
            "emotional_states_cached": len(self._emotional_states),
            "metrics": self._metrics.copy(),
        }
