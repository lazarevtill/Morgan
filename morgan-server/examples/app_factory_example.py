"""
Example: Using the Morgan Server Application Factory

This example demonstrates how to create and run the Morgan server
using the application factory pattern.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from morgan_server.app import create_app, create_app_from_env, create_app_from_file
from morgan_server.config import ServerConfig


def example_basic():
    """Create app with default configuration."""
    print("Example 1: Basic app creation with defaults")
    app = create_app()
    print(f"  App created: {app.title}")
    print(f"  Version: {app.version}")
    print(f"  Config host: {app.state.config.host}")
    print(f"  Config port: {app.state.config.port}")
    print()


def example_with_config():
    """Create app with custom configuration."""
    print("Example 2: App creation with custom config")
    
    config = ServerConfig(
        host="127.0.0.1",
        port=9000,
        llm_provider="ollama",
        llm_endpoint="http://localhost:11434",
        llm_model="gemma3",
        vector_db_url="http://localhost:6333",
        log_level="DEBUG",
    )
    
    app = create_app(config=config)
    print(f"  App created: {app.title}")
    print(f"  Config host: {app.state.config.host}")
    print(f"  Config port: {app.state.config.port}")
    print(f"  Log level: {app.state.config.log_level}")
    print()


def example_with_overrides():
    """Create app with configuration overrides."""
    print("Example 3: App creation with overrides")
    
    app = create_app(
        port=8888,
        log_level="WARNING",
        llm_model="llama2",
    )
    
    print(f"  App created: {app.title}")
    print(f"  Config port: {app.state.config.port}")
    print(f"  Log level: {app.state.config.log_level}")
    print(f"  LLM model: {app.state.config.llm_model}")
    print()


def example_from_env():
    """Create app from environment variables."""
    print("Example 4: App creation from environment")
    print("  Set environment variables:")
    print("    MORGAN_HOST=0.0.0.0")
    print("    MORGAN_PORT=8080")
    print("    MORGAN_LLM_ENDPOINT=http://localhost:11434")
    print("    MORGAN_VECTOR_DB_URL=http://localhost:6333")
    print()
    
    # In practice, you would set these in your environment
    # app = create_app_from_env()


def example_from_file():
    """Create app from configuration file."""
    print("Example 5: App creation from config file")
    print("  Create a config.yaml file:")
    print("    host: 0.0.0.0")
    print("    port: 8080")
    print("    llm_endpoint: http://localhost:11434")
    print("    vector_db_url: http://localhost:6333")
    print()
    
    # In practice, you would have a config file
    # app = create_app_from_file("config.yaml")


def example_run_server():
    """Example of running the server."""
    print("Example 6: Running the server")
    print()
    print("  Option 1: Using uvicorn directly")
    print("    uvicorn morgan_server.app:create_app --factory --host 0.0.0.0 --port 8080")
    print()
    print("  Option 2: Using Python")
    print("    app = create_app()")
    print("    uvicorn.run(app, host='0.0.0.0', port=8080)")
    print()
    print("  Option 3: Using the app factory in code")
    
    # Create app
    app = create_app(
        host="127.0.0.1",
        port=8080,
    )
    
    print(f"    App created: {app.title}")
    print("    To run: uvicorn.run(app, host='127.0.0.1', port=8080)")
    print()


def example_access_routes():
    """Example of accessing routes."""
    print("Example 7: Available routes")
    
    app = create_app()
    
    print("  Root endpoint:")
    print("    GET / - Server information")
    print()
    print("  Health endpoints:")
    print("    GET /health - Simple health check")
    print("    GET /api/status - Detailed status")
    print()
    print("  Chat endpoints:")
    print("    POST /api/chat - Send message")
    print("    WS /ws/{user_id} - WebSocket chat")
    print()
    print("  Memory endpoints:")
    print("    GET /api/memory/stats - Memory statistics")
    print("    GET /api/memory/search - Search conversations")
    print()
    print("  Knowledge endpoints:")
    print("    POST /api/knowledge/learn - Add documents")
    print("    GET /api/knowledge/search - Search knowledge")
    print()
    print("  Documentation:")
    print("    GET /docs - OpenAPI documentation")
    print("    GET /redoc - ReDoc documentation")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Morgan Server Application Factory Examples")
    print("=" * 70)
    print()
    
    example_basic()
    example_with_config()
    example_with_overrides()
    example_from_env()
    example_from_file()
    example_run_server()
    example_access_routes()
    
    print("=" * 70)
    print("For more information, see the documentation at /docs")
    print("=" * 70)
