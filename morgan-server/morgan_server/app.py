"""
FastAPI Application Factory for Morgan Server

This module provides the application factory for creating and configuring
the Morgan server FastAPI application.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from morgan_server.config import ServerConfig, get_config, ConfigurationError
from morgan_server.middleware import setup_middleware
from morgan_server.health import initialize_health_system, get_health_system
from morgan_server.logging_config import configure_logging
from morgan_server.session import initialize_session_manager, get_session_manager
from morgan_server.api.routes import (
    chat_router,
    memory_router,
    knowledge_router,
    health_router,
    profile_router,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifecycle Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle (startup and shutdown).

    This context manager handles:
    - Component initialization on startup
    - Graceful shutdown on termination

    **Validates: Requirements 1.1, 1.5**

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    # ========================================================================
    # STARTUP
    # ========================================================================
    logger.info("Starting Morgan Server...")

    try:
        # Get configuration from app state
        config: ServerConfig = app.state.config
        health_system = get_health_system()

        # Set health system in routes so they can access it
        from morgan_server.api.routes.health import set_health_system

        set_health_system(health_system)

        # Initialize and start session manager
        session_manager = get_session_manager()
        await session_manager.start()
        app.state.session_manager = session_manager

        logger.info(
            "Server configuration loaded",
            extra={
                "host": config.host,
                "port": config.port,
                "llm_provider": config.llm_provider,
                "llm_endpoint": config.llm_endpoint,
                "vector_db_url": config.vector_db_url,
                "embedding_provider": config.embedding_provider,
                "session_timeout_minutes": config.session_timeout_minutes,
            },
        )

        # Initialize core components
        # Note: Actual component initialization will be done here
        # For now, we're setting up the structure

        # Initialize assistant using Core
        logger.info("Initializing Morgan assistant (Core)...")
        from morgan_server.assistant import MorganAssistant
        from morgan_server.api.routes.chat import set_assistant

        # We can pass config_path if config object has it, otherwise default
        config_path = getattr(config, "config_file", None)

        assistant = MorganAssistant(config_path=config_path)
        app.state.assistant = assistant

        # Set assistant in chat routes
        set_assistant(assistant)

        # Register components with health system
        health_system.register_component("assistant", assistant)
        logger.info("Morgan assistant initialized successfully")

        logger.info(
            "Morgan Server started successfully",
            extra={
                "version": health_system.version,
                "components_initialized": len(health_system.component_checkers),
            },
        )

        # Application is now running
        yield

    except Exception as e:
        logger.error(f"Failed to start Morgan Server: {e}", exc_info=True)
        raise

    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    finally:
        logger.info("Shutting down Morgan Server...")

        try:
            # Close all component connections gracefully
            shutdown_tasks = []

            # Shutdown session manager
            if hasattr(app.state, "session_manager"):
                logger.info("Shutting down session manager...")
                shutdown_tasks.append(app.state.session_manager.stop())

            # Shutdown assistant
            if hasattr(app.state, "assistant"):
                logger.info("Shutting down assistant...")
                await app.state.assistant.shutdown()

            # Wait for all shutdowns to complete
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)

            logger.info("Morgan Server shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# ============================================================================
# Application Factory
# ============================================================================


def create_app(
    config: Optional[ServerConfig] = None,
    config_file: Optional[str] = None,
    **config_overrides,
) -> FastAPI:
    """
    Create and configure the Morgan Server FastAPI application.

    This is the main application factory that:
    1. Loads configuration from multiple sources
    2. Configures logging
    3. Initializes health check system
    4. Sets up middleware
    5. Registers API routes
    6. Configures lifecycle management

    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**

    Args:
        config: Optional pre-configured ServerConfig instance
        config_file: Optional path to configuration file
        **config_overrides: Additional configuration overrides

    Returns:
        Configured FastAPI application instance

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> app = create_app()
        >>> # Or with custom config
        >>> app = create_app(config_file="config.yaml", port=9000)
    """
    # ========================================================================
    # Step 1: Load Configuration
    # ========================================================================
    if config is None:
        try:
            from pathlib import Path

            config_path = Path(config_file) if config_file else None
            config = get_config(
                config_file=config_path, validate=True, **config_overrides
            )
        except ConfigurationError as e:
            # Log error and re-raise
            print(f"Configuration error: {e}")
            raise

    # ========================================================================
    # Step 2: Configure Logging
    # ========================================================================
    configure_logging(log_level=config.log_level, log_format=config.log_format)

    logger.info(
        "Configuring Morgan Server",
        extra={
            "version": "0.1.0",
            "log_level": config.log_level,
            "log_format": config.log_format,
        },
    )

    # ========================================================================
    # Step 3: Initialize Health Check System
    # ========================================================================
    health_system = initialize_health_system(version="0.1.0")

    # ========================================================================
    # Step 3.5: Initialize Session Manager
    # ========================================================================
    session_manager = initialize_session_manager(
        session_timeout_minutes=config.session_timeout_minutes,
        cleanup_interval_seconds=300,  # 5 minutes
        max_concurrent_requests=config.max_concurrent_requests,
    )

    # ========================================================================
    # Step 4: Create FastAPI Application
    # ========================================================================
    app = FastAPI(
        title="Morgan Server",
        description="Personal AI Assistant with Empathic and Knowledge Engines",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store configuration and health system in app state
    app.state.config = config
    app.state.health_system = health_system

    # ========================================================================
    # Step 5: Setup Middleware
    # ========================================================================
    setup_middleware(
        app=app,
        enable_logging=True,
        enable_error_handling=True,
        enable_validation=True,
        enable_cors=True,
        max_request_size=10 * 1024 * 1024,  # 10 MB
        cors_origins=["*"],  # Allow all origins for single-user deployment
    )

    logger.info("Middleware configured")

    # ========================================================================
    # Step 6: Register API Routes
    # ========================================================================

    # Health check routes (no prefix)
    app.include_router(health_router, tags=["Health"])

    # API routes (routers already have /api prefix)
    app.include_router(chat_router, tags=["Chat"])
    app.include_router(memory_router, tags=["Memory"])
    app.include_router(knowledge_router, tags=["Knowledge"])

    # Profile router
    app.include_router(profile_router, tags=["Profile"])

    logger.info("API routes registered")

    # ========================================================================
    # Step 7: Add Root Endpoint
    # ========================================================================

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with server information."""
        return JSONResponse(
            content={
                "name": "Morgan Server",
                "version": "0.1.0",
                "description": "Personal AI Assistant API",
                "docs": "/docs",
                "health": "/health",
                "status": "/api/status",
            }
        )

    logger.info("Morgan Server application created successfully")

    return app


# ============================================================================
# Convenience Functions
# ============================================================================


def create_app_from_env() -> FastAPI:
    """
    Create application using only environment variables.

    This is useful for containerized deployments where all configuration
    comes from environment variables.

    Returns:
        Configured FastAPI application
    """
    return create_app()


def create_app_from_file(config_file: str) -> FastAPI:
    """
    Create application from a configuration file.

    Args:
        config_file: Path to configuration file (YAML, JSON, or .env)

    Returns:
        Configured FastAPI application
    """
    return create_app(config_file=config_file)
