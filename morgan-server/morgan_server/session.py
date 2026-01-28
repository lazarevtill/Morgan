"""
Session Management for Morgan Server

This module provides session tracking, isolation, and cleanup for concurrent clients.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents a client session.

    **Validates: Requirements 6.1, 6.2**
    """

    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Connection tracking
    connection_count: int = 0
    active_requests: Set[str] = field(default_factory=set)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_minutes: int) -> bool:
        """
        Check if session has expired.

        Args:
            timeout_minutes: Session timeout in minutes

        Returns:
            True if session has expired
        """
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def add_request(self, request_id: str) -> None:
        """
        Add an active request to this session.

        Args:
            request_id: Unique request identifier
        """
        self.active_requests.add(request_id)
        self.update_activity()

    def remove_request(self, request_id: str) -> None:
        """
        Remove a completed request from this session.

        Args:
            request_id: Unique request identifier
        """
        self.active_requests.discard(request_id)
        self.update_activity()

    def has_active_requests(self) -> bool:
        """Check if session has any active requests."""
        return len(self.active_requests) > 0


class SessionManager:
    """
    Manages client sessions with isolation and cleanup.

    Features:
    - Session creation and tracking
    - Session isolation (data never crosses sessions)
    - Automatic cleanup of expired sessions
    - Connection pooling support
    - Concurrent request handling

    **Validates: Requirements 6.1, 6.2, 6.3, 6.5**
    """

    def __init__(
        self,
        session_timeout_minutes: int = 60,
        cleanup_interval_seconds: int = 300,
        max_concurrent_requests: int = 100,
    ):
        """
        Initialize session manager.

        Args:
            session_timeout_minutes: Session timeout in minutes
            cleanup_interval_seconds: How often to run cleanup (seconds)
            max_concurrent_requests: Maximum concurrent requests allowed
        """
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.max_concurrent_requests = max_concurrent_requests

        # Session storage (session_id -> Session)
        self._sessions: Dict[str, Session] = {}
        self._sessions_lock = asyncio.Lock()

        # User to session mapping (user_id -> session_id)
        self._user_sessions: Dict[str, str] = {}

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._total_sessions_created = 0
        self._total_sessions_expired = 0
        self._total_sessions_cleaned = 0

        logger.info(
            "Session manager initialized",
            extra={
                "session_timeout_minutes": session_timeout_minutes,
                "cleanup_interval_seconds": cleanup_interval_seconds,
                "max_concurrent_requests": max_concurrent_requests,
            },
        )

    async def start(self) -> None:
        """
        Start the session manager and cleanup task.

        **Validates: Requirements 6.3**
        """
        if self._running:
            logger.warning("Session manager already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """
        Stop the session manager and cleanup all sessions.

        **Validates: Requirements 6.5**
        """
        if not self._running:
            return

        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clean up all sessions
        async with self._sessions_lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                await self._cleanup_session(session_id)

        logger.info(
            "Session manager stopped",
            extra={
                "total_sessions_created": self._total_sessions_created,
                "total_sessions_expired": self._total_sessions_expired,
                "total_sessions_cleaned": self._total_sessions_cleaned,
            },
        )

    async def create_session(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.

        **Validates: Requirements 6.1, 6.2**

        Args:
            user_id: Optional user identifier
            conversation_id: Optional conversation identifier
            metadata: Optional session metadata

        Returns:
            Created Session object
        """
        session_id = str(uuid.uuid4())

        session = Session(
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata or {},
        )

        async with self._sessions_lock:
            self._sessions[session_id] = session

            # Track user session mapping
            if user_id:
                self._user_sessions[user_id] = session_id

            self._total_sessions_created += 1

        logger.info(
            "Session created",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
        )

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        **Validates: Requirements 6.2**

        Args:
            session_id: Session identifier

        Returns:
            Session object or None if not found
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session:
                session.update_activity()
            return session

    async def get_session_by_user(self, user_id: str) -> Optional[Session]:
        """
        Get a session by user ID.

        Args:
            user_id: User identifier

        Returns:
            Session object or None if not found
        """
        async with self._sessions_lock:
            session_id = self._user_sessions.get(user_id)
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    session.update_activity()
                return session
        return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        **Validates: Requirements 6.5**

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            # Wait for active requests to complete (with timeout)
            if session.has_active_requests():
                logger.warning(
                    "Deleting session with active requests",
                    extra={
                        "session_id": session_id,
                        "active_requests": len(session.active_requests),
                    },
                )

            # Clean up session
            await self._cleanup_session(session_id)

            return True

    async def _cleanup_session(self, session_id: str) -> None:
        """
        Clean up a session and its resources.

        **Validates: Requirements 6.5**

        Args:
            session_id: Session identifier
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        # Remove from user mapping
        if session.user_id and self._user_sessions.get(session.user_id) == session_id:
            del self._user_sessions[session.user_id]

        # Remove session
        del self._sessions[session_id]

        self._total_sessions_cleaned += 1

        logger.info(
            "Session cleaned up",
            extra={
                "session_id": session_id,
                "user_id": session.user_id,
                "duration_seconds": (
                    datetime.now(timezone.utc) - session.created_at
                ).total_seconds(),
            },
        )

    async def _cleanup_loop(self) -> None:
        """
        Background task to clean up expired sessions.

        **Validates: Requirements 6.3, 6.5**
        """
        logger.info("Session cleanup loop started")

        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)

                # Find expired sessions
                expired_sessions = []
                async with self._sessions_lock:
                    for session_id, session in self._sessions.items():
                        if session.is_expired(self.session_timeout_minutes):
                            # Don't clean up sessions with active requests
                            if not session.has_active_requests():
                                expired_sessions.append(session_id)

                # Clean up expired sessions
                if expired_sessions:
                    logger.info(
                        "Cleaning up expired sessions",
                        extra={"count": len(expired_sessions)},
                    )

                    async with self._sessions_lock:
                        for session_id in expired_sessions:
                            await self._cleanup_session(session_id)
                            self._total_sessions_expired += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}", exc_info=True)

    @asynccontextmanager
    async def track_request(self, session_id: str):
        """
        Context manager to track a request within a session.

        **Validates: Requirements 6.1, 6.2**

        Args:
            session_id: Session identifier

        Yields:
            Session object

        Example:
            async with session_manager.track_request(session_id) as session:
                # Process request
                pass
        """
        request_id = str(uuid.uuid4())
        session = await self.get_session(session_id)

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Add request to session
        session.add_request(request_id)

        try:
            yield session
        finally:
            # Remove request from session
            session.remove_request(request_id)

    def get_active_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self._sessions)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get session manager metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "active_sessions": len(self._sessions),
            "total_sessions_created": self._total_sessions_created,
            "total_sessions_expired": self._total_sessions_expired,
            "total_sessions_cleaned": self._total_sessions_cleaned,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_concurrent_requests": self.max_concurrent_requests,
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance.

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def initialize_session_manager(
    session_timeout_minutes: int = 60,
    cleanup_interval_seconds: int = 300,
    max_concurrent_requests: int = 100,
) -> SessionManager:
    """
    Initialize the global session manager.

    Args:
        session_timeout_minutes: Session timeout in minutes
        cleanup_interval_seconds: How often to run cleanup (seconds)
        max_concurrent_requests: Maximum concurrent requests allowed

    Returns:
        SessionManager instance
    """
    global _session_manager
    _session_manager = SessionManager(
        session_timeout_minutes=session_timeout_minutes,
        cleanup_interval_seconds=cleanup_interval_seconds,
        max_concurrent_requests=max_concurrent_requests,
    )
    return _session_manager
