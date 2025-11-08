#!/usr/bin/env python3
"""
Morgan Web API Demo - Demonstrates web interface functionality.

This script shows how to start and use the Morgan Web API.
"""

import asyncio
from pathlib import Path

from morgan.interfaces import create_app


async def main():
    """Run web API demo."""
    print("=" * 70)
    print("Morgan Web API Demo")
    print("=" * 70)
    print()

    # Create FastAPI app
    print("Creating FastAPI application...")
    app = create_app(
        storage_path=Path.home() / ".morgan" / "api_demo",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:latest",
        vector_db_url="http://localhost:6333",
        enable_emotion_detection=True,
        enable_learning=True,
        enable_rag=True,
        cors_origins=["*"],
    )

    print("   ✓ Application created")
    print()

    print("Available endpoints:")
    print("   GET  /               - Root endpoint")
    print("   GET  /health         - Health check")
    print("   POST /chat           - Synchronous chat")
    print("   POST /chat/stream    - Streaming chat")
    print("   POST /feedback       - Submit feedback")
    print("   GET  /learning/stats - Learning statistics")
    print("   WS   /ws             - WebSocket connection")
    print()

    print("To start the server, run:")
    print()
    print("   # Development mode")
    print("   python -m morgan.interfaces.web_interface")
    print()
    print("   # or with uvicorn")
    print("   uvicorn morgan.interfaces.web_interface:app --reload")
    print()
    print("   # Production mode")
    print("   uvicorn morgan.interfaces.web_interface:app \\")
    print("     --host 0.0.0.0 --port 8000 --workers 4")
    print()

    print("Example API requests:")
    print()

    print("1. Health check:")
    print("   curl http://localhost:8000/health")
    print()

    print("2. Chat request:")
    print("   curl -X POST http://localhost:8000/chat \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print('       "message": "What is AI?",')
    print('       "user_id": "user123",')
    print('       "include_sources": true,')
    print('       "include_emotion": true')
    print("     }'")
    print()

    print("3. Feedback:")
    print("   curl -X POST http://localhost:8000/feedback \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print('       "response_id": "resp123",')
    print('       "user_id": "user123",')
    print('       "session_id": "session456",')
    print('       "rating": 0.9,')
    print('       "comment": "Excellent response!"')
    print("     }'")
    print()

    print("4. Learning stats:")
    print("   curl 'http://localhost:8000/learning/stats?user_id=user123'")
    print()

    print("=" * 70)
    print("Documentation: http://localhost:8000/docs (when running)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
