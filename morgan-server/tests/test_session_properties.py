"""
Property-based tests for session management.

**Feature: client-server-separation**
**Validates: Requirements 6.1, 6.2, 6.5**
"""

import pytest
import asyncio
from hypothesis import given, settings, strategies as st
from datetime import datetime, timezone, timedelta

from morgan_server.session import (
    Session,
    SessionManager,
)


# ============================================================================
# Property 12: Concurrent request handling
# **Validates: Requirements 6.1, 6.2**
# ============================================================================

@settings(max_examples=100)
@given(
    num_sessions=st.integers(min_value=1, max_value=20),
    requests_per_session=st.integers(min_value=1, max_value=10),
)
@pytest.mark.asyncio
async def test_concurrent_request_handling(num_sessions, requests_per_session):
    """
    **Feature: client-server-separation, Property 12: Concurrent request handling**
    
    Property: For any set of concurrent client connections, the server should
    handle all requests without blocking, maintaining session isolation such that
    data from one session never appears in another session's responses.
    
    This test verifies:
    1. Multiple sessions can be created concurrently
    2. Multiple requests can be processed concurrently within each session
    3. Session data remains isolated (no cross-contamination)
    4. All requests complete successfully without blocking
    
    **Validates: Requirements 6.1, 6.2**
    """
    # Create session manager
    manager = SessionManager(
        session_timeout_minutes=60,
        cleanup_interval_seconds=300,
        max_concurrent_requests=1000
    )
    await manager.start()
    
    try:
        # Create multiple sessions concurrently
        session_creation_tasks = []
        for i in range(num_sessions):
            session_creation_tasks.append(
                manager.create_session(
                    user_id=f"user-{i}",
                    metadata={"session_index": i, "data": f"session-{i}-data"}
                )
            )
        
        sessions = await asyncio.gather(*session_creation_tasks)
        
        # Verify all sessions were created
        assert len(sessions) == num_sessions, (
            f"Expected {num_sessions} sessions, got {len(sessions)}"
        )
        
        # Verify all session IDs are unique
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == num_sessions, (
            "Session IDs are not unique - sessions are not properly isolated"
        )
        
        # Process concurrent requests within each session
        all_request_tasks = []
        
        for session in sessions:
            for req_idx in range(requests_per_session):
                # Create a task that simulates processing a request
                async def process_request(sess, req_id):
                    async with manager.track_request(sess.session_id) as tracked_session:
                        # Verify we got the correct session
                        assert tracked_session.session_id == sess.session_id, (
                            f"Session isolation violated: expected {sess.session_id}, "
                            f"got {tracked_session.session_id}"
                        )
                        
                        # Verify session metadata is correct (no cross-contamination)
                        assert tracked_session.metadata["session_index"] == sess.metadata["session_index"], (
                            f"Session data contamination: expected index {sess.metadata['session_index']}, "
                            f"got {tracked_session.metadata['session_index']}"
                        )
                        
                        assert tracked_session.metadata["data"] == sess.metadata["data"], (
                            f"Session data contamination: expected {sess.metadata['data']}, "
                            f"got {tracked_session.metadata['data']}"
                        )
                        
                        # Simulate some work
                        await asyncio.sleep(0.001)
                        
                        return {
                            "session_id": sess.session_id,
                            "request_id": req_id,
                            "user_id": sess.user_id,
                        }
                
                all_request_tasks.append(process_request(session, f"req-{req_idx}"))
        
        # Execute all requests concurrently
        results = await asyncio.gather(*all_request_tasks)
        
        # Verify all requests completed
        expected_total_requests = num_sessions * requests_per_session
        assert len(results) == expected_total_requests, (
            f"Expected {expected_total_requests} results, got {len(results)}"
        )
        
        # Verify each result has correct session data (no cross-contamination)
        for result in results:
            session_id = result["session_id"]
            
            # Find the original session
            original_session = next(s for s in sessions if s.session_id == session_id)
            
            # Verify user_id matches
            assert result["user_id"] == original_session.user_id, (
                f"User ID mismatch: expected {original_session.user_id}, "
                f"got {result['user_id']}"
            )
        
        # Verify all sessions still exist and have correct data
        for session in sessions:
            retrieved = await manager.get_session(session.session_id)
            assert retrieved is not None, (
                f"Session {session.session_id} was lost during concurrent processing"
            )
            assert retrieved.metadata["session_index"] == session.metadata["session_index"], (
                f"Session metadata was corrupted during concurrent processing"
            )
            
            # Verify no active requests remain
            assert not retrieved.has_active_requests(), (
                f"Session {session.session_id} still has active requests after completion"
            )
    
    finally:
        await manager.stop()


# ============================================================================
# Property 14: Session cleanup isolation
# **Validates: Requirements 6.5**
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    num_sessions=st.integers(min_value=2, max_value=20),
    sessions_to_cleanup=st.integers(min_value=1, max_value=15),
)
@pytest.mark.asyncio
async def test_session_cleanup_isolation(num_sessions, sessions_to_cleanup):
    """
    **Feature: client-server-separation, Property 14: Session cleanup isolation**
    
    Property: For any client that disconnects, the server should clean up that
    client's session resources without affecting other active sessions or their data.
    
    This test verifies:
    1. Cleaning up one session does not affect other sessions
    2. Session data remains intact for non-cleaned sessions
    3. Cleaned sessions are properly removed
    4. Active sessions continue to function after cleanup
    
    **Validates: Requirements 6.5**
    """
    # Ensure we don't try to cleanup more sessions than we create
    sessions_to_cleanup = min(sessions_to_cleanup, num_sessions - 1)
    
    # Create session manager
    manager = SessionManager(
        session_timeout_minutes=60,
        cleanup_interval_seconds=300,
        max_concurrent_requests=100
    )
    await manager.start()
    
    try:
        # Create multiple sessions
        sessions = []
        for i in range(num_sessions):
            session = await manager.create_session(
                user_id=f"user-{i}",
                metadata={
                    "session_index": i,
                    "data": f"important-data-{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            sessions.append(session)
        
        # Verify all sessions exist
        assert manager.get_active_session_count() == num_sessions
        
        # Select sessions to cleanup (first N sessions)
        sessions_to_delete = sessions[:sessions_to_cleanup]
        sessions_to_keep = sessions[sessions_to_cleanup:]
        
        # Store original data for sessions we're keeping
        original_data = {}
        for session in sessions_to_keep:
            original_data[session.session_id] = {
                "user_id": session.user_id,
                "metadata": session.metadata.copy(),
                "conversation_id": session.conversation_id,
            }
        
        # Clean up selected sessions
        for session in sessions_to_delete:
            result = await manager.delete_session(session.session_id)
            assert result is True, (
                f"Failed to delete session {session.session_id}"
            )
        
        # Verify cleaned sessions are gone
        for session in sessions_to_delete:
            retrieved = await manager.get_session(session.session_id)
            assert retrieved is None, (
                f"Session {session.session_id} should be deleted but still exists"
            )
        
        # Verify remaining sessions still exist and have correct data
        for session in sessions_to_keep:
            retrieved = await manager.get_session(session.session_id)
            
            assert retrieved is not None, (
                f"Session {session.session_id} was incorrectly deleted during cleanup"
            )
            
            # Verify session data is intact
            original = original_data[session.session_id]
            
            assert retrieved.user_id == original["user_id"], (
                f"User ID changed after cleanup: expected {original['user_id']}, "
                f"got {retrieved.user_id}"
            )
            
            assert retrieved.metadata["session_index"] == original["metadata"]["session_index"], (
                f"Session metadata corrupted after cleanup"
            )
            
            assert retrieved.metadata["data"] == original["metadata"]["data"], (
                f"Session data corrupted after cleanup: expected {original['metadata']['data']}, "
                f"got {retrieved.metadata['data']}"
            )
            
            assert retrieved.conversation_id == original["conversation_id"], (
                f"Conversation ID changed after cleanup"
            )
        
        # Verify session count is correct
        expected_count = num_sessions - sessions_to_cleanup
        actual_count = manager.get_active_session_count()
        assert actual_count == expected_count, (
            f"Session count mismatch after cleanup: expected {expected_count}, "
            f"got {actual_count}"
        )
        
        # Verify remaining sessions can still be used for requests
        for session in sessions_to_keep:
            async with manager.track_request(session.session_id) as tracked_session:
                # Verify session is functional
                assert tracked_session.session_id == session.session_id
                assert tracked_session.has_active_requests()
                
                # Verify data is still correct
                assert tracked_session.metadata["data"] == original_data[session.session_id]["metadata"]["data"]
        
        # Verify no active requests remain
        for session in sessions_to_keep:
            retrieved = await manager.get_session(session.session_id)
            assert not retrieved.has_active_requests(), (
                f"Session {session.session_id} has lingering active requests"
            )
    
    finally:
        await manager.stop()


# ============================================================================
# Additional Property: Session expiration does not affect active sessions
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    num_active_sessions=st.integers(min_value=1, max_value=10),
    num_expired_sessions=st.integers(min_value=1, max_value=10),
)
@pytest.mark.asyncio
async def test_expired_session_cleanup_isolation(num_active_sessions, num_expired_sessions):
    """
    Property: When expired sessions are cleaned up, active sessions should not be affected.
    
    This test verifies:
    1. Expired sessions are cleaned up
    2. Active sessions remain untouched
    3. Active session data is not corrupted
    """
    # Create session manager with short timeout
    manager = SessionManager(
        session_timeout_minutes=1,
        cleanup_interval_seconds=1,
        max_concurrent_requests=100
    )
    await manager.start()
    
    try:
        # Create active sessions
        active_sessions = []
        for i in range(num_active_sessions):
            session = await manager.create_session(
                user_id=f"active-user-{i}",
                metadata={"type": "active", "index": i}
            )
            active_sessions.append(session)
        
        # Create sessions that will expire
        expired_sessions = []
        for i in range(num_expired_sessions):
            session = await manager.create_session(
                user_id=f"expired-user-{i}",
                metadata={"type": "expired", "index": i}
            )
            # Manually expire them
            session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=2)
            expired_sessions.append(session)
        
        # Wait for cleanup to run
        await asyncio.sleep(2)
        
        # Verify expired sessions are cleaned up
        for session in expired_sessions:
            retrieved = await manager.get_session(session.session_id)
            assert retrieved is None, (
                f"Expired session {session.session_id} was not cleaned up"
            )
        
        # Verify active sessions still exist with correct data
        for session in active_sessions:
            retrieved = await manager.get_session(session.session_id)
            assert retrieved is not None, (
                f"Active session {session.session_id} was incorrectly cleaned up"
            )
            assert retrieved.metadata["type"] == "active", (
                f"Active session metadata corrupted"
            )
            assert retrieved.metadata["index"] == session.metadata["index"], (
                f"Active session data corrupted"
            )
    
    finally:
        await manager.stop()
