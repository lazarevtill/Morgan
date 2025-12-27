"""
Example: Using Session Management in API Routes

This example demonstrates how to use the session management system
in FastAPI routes to track client sessions and ensure isolation.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**
"""

import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from morgan_server.session import initialize_session_manager, get_session_manager
from morgan_server.health import initialize_health_system, get_health_system


# ============================================================================
# Example 1: Basic Session Creation and Tracking
# ============================================================================


async def example_basic_session_usage():
    """
    Example showing basic session creation and retrieval.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Session Usage")
    print("=" * 70)

    # Initialize session manager
    manager = initialize_session_manager(
        session_timeout_minutes=60,
        cleanup_interval_seconds=300,
        max_concurrent_requests=100,
    )
    await manager.start()

    try:
        # Create a session for a user
        session = await manager.create_session(
            user_id="user-123",
            conversation_id="conv-456",
            metadata={"client_type": "web", "version": "1.0"},
        )

        print(f"Created session: {session.session_id}")
        print(f"  User ID: {session.user_id}")
        print(f"  Conversation ID: {session.conversation_id}")
        print(f"  Metadata: {session.metadata}")

        # Retrieve the session
        retrieved = await manager.get_session(session.session_id)
        print(f"\nRetrieved session: {retrieved.session_id}")
        print(f"  Active requests: {len(retrieved.active_requests)}")

        # Get session by user ID
        user_session = await manager.get_session_by_user("user-123")
        print(f"\nFound session for user-123: {user_session.session_id}")

    finally:
        await manager.stop()


# ============================================================================
# Example 2: Request Tracking with Context Manager
# ============================================================================


async def example_request_tracking():
    """
    Example showing how to track requests within a session.
    """
    print("\n" + "=" * 70)
    print("Example 2: Request Tracking")
    print("=" * 70)

    manager = initialize_session_manager()
    await manager.start()

    try:
        # Create a session
        session = await manager.create_session(user_id="user-456")
        print(f"Created session: {session.session_id}")

        # Track a request using context manager
        print("\nProcessing request...")
        async with manager.track_request(session.session_id) as tracked_session:
            print(f"  Session ID: {tracked_session.session_id}")
            print(f"  Active requests: {len(tracked_session.active_requests)}")
            print(f"  Has active requests: {tracked_session.has_active_requests()}")

            # Simulate processing
            await asyncio.sleep(0.1)
            print("  Request processing complete")

        # After context, request is removed
        retrieved = await manager.get_session(session.session_id)
        print(f"\nAfter request:")
        print(f"  Active requests: {len(retrieved.active_requests)}")
        print(f"  Has active requests: {retrieved.has_active_requests()}")

    finally:
        await manager.stop()


# ============================================================================
# Example 3: Concurrent Request Handling
# ============================================================================


async def example_concurrent_requests():
    """
    Example showing concurrent request handling across multiple sessions.
    """
    print("\n" + "=" * 70)
    print("Example 3: Concurrent Request Handling")
    print("=" * 70)

    manager = initialize_session_manager()
    await manager.start()

    try:
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = await manager.create_session(
                user_id=f"user-{i}", metadata={"index": i}
            )
            sessions.append(session)
            print(f"Created session {i}: {session.session_id}")

        # Process concurrent requests
        async def process_request(session, request_num):
            async with manager.track_request(session.session_id) as tracked:
                print(
                    f"  Processing request {request_num} for session {tracked.metadata['index']}"
                )
                await asyncio.sleep(0.1)
                return f"Result for session {tracked.metadata['index']}, request {request_num}"

        # Create tasks for concurrent requests
        tasks = []
        for session in sessions:
            for req_num in range(2):
                tasks.append(process_request(session, req_num))

        print(f"\nProcessing {len(tasks)} concurrent requests...")
        results = await asyncio.gather(*tasks)

        print(f"\nCompleted {len(results)} requests")
        for result in results:
            print(f"  - {result}")

        # Verify all sessions are clean
        print("\nVerifying session state:")
        for session in sessions:
            retrieved = await manager.get_session(session.session_id)
            print(
                f"  Session {retrieved.metadata['index']}: "
                f"{len(retrieved.active_requests)} active requests"
            )

    finally:
        await manager.stop()


# ============================================================================
# Example 4: Session Cleanup
# ============================================================================


async def example_session_cleanup():
    """
    Example showing session cleanup and isolation.
    """
    print("\n" + "=" * 70)
    print("Example 4: Session Cleanup")
    print("=" * 70)

    manager = initialize_session_manager()
    await manager.start()

    try:
        # Create multiple sessions
        session1 = await manager.create_session(
            user_id="user-1", metadata={"name": "Session 1"}
        )
        session2 = await manager.create_session(
            user_id="user-2", metadata={"name": "Session 2"}
        )
        session3 = await manager.create_session(
            user_id="user-3", metadata={"name": "Session 3"}
        )

        print(f"Created 3 sessions")
        print(f"Active sessions: {manager.get_active_session_count()}")

        # Delete one session
        print(f"\nDeleting session 2...")
        await manager.delete_session(session2.session_id)

        print(f"Active sessions: {manager.get_active_session_count()}")

        # Verify session 2 is gone
        retrieved2 = await manager.get_session(session2.session_id)
        print(f"Session 2 exists: {retrieved2 is not None}")

        # Verify other sessions still exist
        retrieved1 = await manager.get_session(session1.session_id)
        retrieved3 = await manager.get_session(session3.session_id)

        print(f"Session 1 exists: {retrieved1 is not None}")
        print(f"Session 3 exists: {retrieved3 is not None}")

        # Verify data is intact
        print(f"\nSession 1 metadata: {retrieved1.metadata}")
        print(f"Session 3 metadata: {retrieved3.metadata}")

    finally:
        await manager.stop()


# ============================================================================
# Example 5: Integration with FastAPI Routes
# ============================================================================


def create_example_app():
    """
    Create an example FastAPI app with session management.
    """
    app = FastAPI(title="Session Management Example")

    # Store session manager in app state
    @app.on_event("startup")
    async def startup():
        manager = initialize_session_manager()
        await manager.start()
        app.state.session_manager = manager

        health_system = initialize_health_system()
        app.state.health_system = health_system

        print("Session manager started")

    @app.on_event("shutdown")
    async def shutdown():
        if hasattr(app.state, "session_manager"):
            await app.state.session_manager.stop()
            print("Session manager stopped")

    @app.post("/api/sessions")
    async def create_session(request: Request, user_id: str):
        """Create a new session."""
        manager = request.app.state.session_manager
        health_system = request.app.state.health_system

        session = await manager.create_session(user_id=user_id)

        # Track session in health system
        health_system.increment_active_sessions()

        return JSONResponse(
            content={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
            }
        )

    @app.get("/api/sessions/{session_id}")
    async def get_session(request: Request, session_id: str):
        """Get session information."""
        manager = request.app.state.session_manager

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return JSONResponse(
            content={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "active_requests": len(session.active_requests),
            }
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(request: Request, session_id: str):
        """Delete a session."""
        manager = request.app.state.session_manager
        health_system = request.app.state.health_system

        result = await manager.delete_session(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update health system
        health_system.decrement_active_sessions()

        return JSONResponse(content={"status": "deleted"})

    @app.post("/api/chat")
    async def chat(request: Request, session_id: str, message: str):
        """
        Process a chat message within a session.

        This demonstrates using the session manager to track requests.
        """
        manager = request.app.state.session_manager

        # Track this request within the session
        async with manager.track_request(session_id) as session:
            # Simulate processing
            await asyncio.sleep(0.1)

            return JSONResponse(
                content={
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "message": message,
                    "response": f"Echo: {message}",
                }
            )

    @app.get("/api/metrics")
    async def get_metrics(request: Request):
        """Get session metrics."""
        manager = request.app.state.session_manager
        metrics = manager.get_metrics()

        return JSONResponse(content=metrics)

    return app


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Session Management Examples")
    print("=" * 70)

    await example_basic_session_usage()
    await example_request_tracking()
    await example_concurrent_requests()
    await example_session_cleanup()

    print("\n" + "=" * 70)
    print("FastAPI Integration Example")
    print("=" * 70)
    print("\nTo run the FastAPI example:")
    print("  1. Save this file")
    print("  2. Run: uvicorn session_management_example:create_example_app --factory")
    print("  3. Visit: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
