"""
Unit tests for session management system.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from morgan_server.session import (
    Session,
    SessionManager,
    initialize_session_manager,
    get_session_manager,
)


class TestSession:
    """Test Session class."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = Session(
            session_id="test-123",
            user_id="user-456",
            conversation_id="conv-789"
        )
        
        assert session.session_id == "test-123"
        assert session.user_id == "user-456"
        assert session.conversation_id == "conv-789"
        assert session.connection_count == 0
        assert len(session.active_requests) == 0
    
    def test_session_update_activity(self):
        """Test updating session activity."""
        session = Session(session_id="test-123")
        original_time = session.last_activity
        
        # Wait a bit
        import time
        time.sleep(0.01)
        
        session.update_activity()
        assert session.last_activity > original_time
    
    def test_session_expiration(self):
        """Test session expiration check."""
        session = Session(session_id="test-123")
        
        # Not expired immediately
        assert not session.is_expired(timeout_minutes=60)
        
        # Set last activity to past
        session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=61)
        assert session.is_expired(timeout_minutes=60)
    
    def test_session_request_tracking(self):
        """Test tracking active requests."""
        session = Session(session_id="test-123")
        
        # Add requests
        session.add_request("req-1")
        session.add_request("req-2")
        
        assert len(session.active_requests) == 2
        assert session.has_active_requests()
        
        # Remove request
        session.remove_request("req-1")
        assert len(session.active_requests) == 1
        
        # Remove non-existent request (should not error)
        session.remove_request("req-999")
        assert len(session.active_requests) == 1
        
        # Remove last request
        session.remove_request("req-2")
        assert not session.has_active_requests()


class TestSessionManager:
    """Test SessionManager class."""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a session manager for testing."""
        manager = SessionManager(
            session_timeout_minutes=1,  # Short timeout for testing
            cleanup_interval_seconds=1,  # Fast cleanup for testing
            max_concurrent_requests=10
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a session."""
        session = await session_manager.create_session(
            user_id="user-123",
            conversation_id="conv-456",
            metadata={"key": "value"}
        )
        
        assert session.session_id is not None
        assert session.user_id == "user-123"
        assert session.conversation_id == "conv-456"
        assert session.metadata["key"] == "value"
        
        # Verify session is stored
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
    
    @pytest.mark.asyncio
    async def test_get_session_by_user(self, session_manager):
        """Test getting session by user ID."""
        session = await session_manager.create_session(user_id="user-123")
        
        retrieved = await session_manager.get_session_by_user("user-123")
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        assert retrieved.user_id == "user-123"
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, session_manager):
        """Test that sessions are isolated from each other."""
        # Create two sessions
        session1 = await session_manager.create_session(
            user_id="user-1",
            metadata={"data": "session1"}
        )
        session2 = await session_manager.create_session(
            user_id="user-2",
            metadata={"data": "session2"}
        )
        
        # Verify they are different
        assert session1.session_id != session2.session_id
        assert session1.metadata["data"] == "session1"
        assert session2.metadata["data"] == "session2"
        
        # Verify retrieval is isolated
        retrieved1 = await session_manager.get_session(session1.session_id)
        retrieved2 = await session_manager.get_session(session2.session_id)
        
        assert retrieved1.session_id == session1.session_id
        assert retrieved2.session_id == session2.session_id
        assert retrieved1.metadata["data"] == "session1"
        assert retrieved2.metadata["data"] == "session2"
    
    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager):
        """Test deleting a session."""
        session = await session_manager.create_session(user_id="user-123")
        
        # Verify session exists
        assert await session_manager.get_session(session.session_id) is not None
        
        # Delete session
        result = await session_manager.delete_session(session.session_id)
        assert result is True
        
        # Verify session is gone
        assert await session_manager.get_session(session.session_id) is None
        
        # Try to delete again
        result = await session_manager.delete_session(session.session_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_expiration(self, session_manager):
        """Test that expired sessions are cleaned up."""
        # Create a session
        session = await session_manager.create_session(user_id="user-123")
        
        # Manually expire it
        session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        # Wait for cleanup to run
        await asyncio.sleep(2)
        
        # Session should be cleaned up
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_session_not_cleaned_with_active_requests(self, session_manager):
        """Test that sessions with active requests are not cleaned up."""
        # Create a session
        session = await session_manager.create_session(user_id="user-123")
        
        # Add an active request
        session.add_request("req-1")
        
        # Manually expire it
        session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        # Wait for cleanup to run
        await asyncio.sleep(2)
        
        # Session should still exist because it has active requests
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is not None
    
    @pytest.mark.asyncio
    async def test_track_request_context_manager(self, session_manager):
        """Test tracking requests with context manager."""
        session = await session_manager.create_session(user_id="user-123")
        
        # Track a request
        async with session_manager.track_request(session.session_id) as tracked_session:
            assert tracked_session.session_id == session.session_id
            assert tracked_session.has_active_requests()
        
        # After context, request should be removed
        retrieved = await session_manager.get_session(session.session_id)
        assert not retrieved.has_active_requests()
    
    @pytest.mark.asyncio
    async def test_track_request_with_invalid_session(self, session_manager):
        """Test tracking request with invalid session ID."""
        with pytest.raises(ValueError, match="Session not found"):
            async with session_manager.track_request("invalid-session-id"):
                pass
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, session_manager):
        """Test handling multiple concurrent sessions."""
        # Create multiple sessions concurrently
        tasks = []
        for i in range(10):
            tasks.append(
                session_manager.create_session(
                    user_id=f"user-{i}",
                    metadata={"index": i}
                )
            )
        
        sessions = await asyncio.gather(*tasks)
        
        # Verify all sessions were created
        assert len(sessions) == 10
        assert len(set(s.session_id for s in sessions)) == 10  # All unique
        
        # Verify each session can be retrieved
        for session in sessions:
            retrieved = await session_manager.get_session(session.session_id)
            assert retrieved is not None
            assert retrieved.session_id == session.session_id
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, session_manager):
        """Test getting session manager metrics."""
        # Create some sessions
        await session_manager.create_session(user_id="user-1")
        await session_manager.create_session(user_id="user-2")
        
        metrics = session_manager.get_metrics()
        
        assert metrics["active_sessions"] == 2
        assert metrics["total_sessions_created"] == 2
        assert metrics["session_timeout_minutes"] == 1
        assert metrics["max_concurrent_requests"] == 10
    
    @pytest.mark.asyncio
    async def test_session_manager_stop_cleans_all_sessions(self):
        """Test that stopping session manager cleans up all sessions."""
        manager = SessionManager(
            session_timeout_minutes=60,
            cleanup_interval_seconds=300
        )
        await manager.start()
        
        # Create sessions
        await manager.create_session(user_id="user-1")
        await manager.create_session(user_id="user-2")
        await manager.create_session(user_id="user-3")
        
        assert manager.get_active_session_count() == 3
        
        # Stop manager
        await manager.stop()
        
        # All sessions should be cleaned up
        assert manager.get_active_session_count() == 0


class TestSessionManagerGlobal:
    """Test global session manager functions."""
    
    def test_initialize_and_get_session_manager(self):
        """Test initializing and getting global session manager."""
        manager = initialize_session_manager(
            session_timeout_minutes=30,
            cleanup_interval_seconds=60,
            max_concurrent_requests=50
        )
        
        assert manager is not None
        assert manager.session_timeout_minutes == 30
        assert manager.max_concurrent_requests == 50
        
        # Get the same instance
        same_manager = get_session_manager()
        assert same_manager is manager
