"""
FastAPI Application Factory for Morgan Server

This module provides the application factory for creating and configuring
the Morgan server FastAPI application.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
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
    features_router,
    tools_router,
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
        from morgan_server.api.routes.features import set_assistant as set_features_assistant

        # We can pass config_path if config object has it, otherwise default
        config_path = getattr(config, "config_file", None)

        assistant = MorganAssistant(config_path=config_path)
        app.state.assistant = assistant

        # Set assistant in chat routes and feature routes
        set_assistant(assistant)
        set_features_assistant(assistant)

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

        # ================================================================
        # Initialize new modules (Task 13: Integration Wiring)
        # ================================================================
        try:
            from morgan.config.settings import get_settings as get_rag_settings

            rag_settings = get_rag_settings()

            # --- Workspace ---
            if rag_settings.morgan_enable_workspace:
                try:
                    from morgan.workspace import WorkspaceManager, get_workspace_path
                    ws_path = rag_settings.morgan_workspace_path or str(
                        get_workspace_path()
                    )
                    workspace_mgr = WorkspaceManager(workspace_dir=Path(ws_path))
                    workspace_mgr.bootstrap()
                    app.state.workspace_manager = workspace_mgr
                    logger.info("Workspace bootstrapped at %s", ws_path)
                except Exception as exc:
                    logger.warning("Workspace init failed (non-fatal): %s", exc)

            # --- AppStateStore ---
            try:
                from morgan.app_state import AppStateStore
                app.state.app_state_store = AppStateStore()
                logger.info("AppStateStore initialized")
            except Exception as exc:
                logger.warning("AppStateStore init failed (non-fatal): %s", exc)

            # --- HookManager ---
            try:
                from morgan.hook_system import HookManager
                app.state.hook_manager = HookManager()
                logger.info("HookManager initialized")

                # Share the hook manager with the assistant/orchestrator pipeline.
                if hasattr(app.state, "assistant") and hasattr(app.state.assistant, "core"):
                    app.state.assistant.core.hook_manager = app.state.hook_manager
                    if hasattr(app.state.assistant.core, "orchestrator"):
                        app.state.assistant.core.orchestrator._hook_manager = app.state.hook_manager
                    if hasattr(app.state.assistant.core, "compactor") and app.state.assistant.core.compactor:
                        app.state.assistant.core.compactor._hook_manager = app.state.hook_manager
            except Exception as exc:
                logger.warning("HookManager init failed (non-fatal): %s", exc)

            # --- TaskManager ---
            try:
                from morgan.task_manager import TaskManager
                app.state.task_manager = TaskManager()
                logger.info("TaskManager initialized")
            except Exception as exc:
                logger.warning("TaskManager init failed (non-fatal): %s", exc)

            # --- Scheduling (CronService + HeartbeatManager) ---
            if rag_settings.morgan_enable_scheduling:
                try:
                    from morgan.memory_consolidation import MemoryConsolidator
                    from morgan.scheduling import CronJob, CronService, HeartbeatManager
                    import os as _os
                    _ws = _os.environ.get("MORGAN_WORKSPACE_PATH") or str(Path.home() / ".morgan")
                    cron_persistence = Path(_ws) / "cron_jobs.json"
                    cron_persistence.parent.mkdir(parents=True, exist_ok=True)
                    cron_service = CronService(persistence_path=str(cron_persistence))
                    cron_service.load()

                    async def _run_memory_consolidation() -> None:
                        if not hasattr(app.state, "workspace_manager"):
                            return
                        workspace_dir = app.state.workspace_manager._dir
                        consolidator = MemoryConsolidator(workspace_dir=workspace_dir)
                        try:
                            consolidator.consolidate(days_to_review=7)
                            logger.info("Scheduled memory consolidation completed")
                        except Exception as consolidate_exc:
                            logger.warning(
                                "Scheduled memory consolidation failed: %s",
                                consolidate_exc,
                            )

                    def _cron_job_handler(job: CronJob) -> None:
                        if job.metadata.get("job_type") == "memory_consolidation":
                            asyncio.create_task(_run_memory_consolidation())

                    cron_service.set_job_handler(_cron_job_handler)
                    # Only add default job if not already loaded from persistence
                    if cron_service.get_job("memory-consolidation-daily") is None:
                        cron_service.add_job(
                            CronJob(
                                job_id="memory-consolidation-daily",
                                schedule="0 2 * * *",
                                prompt="Consolidate recent memory logs",
                                channel="system",
                                metadata={"job_type": "memory_consolidation"},
                            )
                        )
                    cron_service.save()
                    await cron_service.start()
                    app.state.cron_service = cron_service

                    heartbeat_mgr = HeartbeatManager()
                    await heartbeat_mgr.start()
                    app.state.heartbeat_manager = heartbeat_mgr
                    logger.info("Scheduling (Cron + Heartbeat) initialized")
                except Exception as exc:
                    logger.warning("Scheduling init failed (non-fatal): %s", exc)

            # --- ChannelGateway ---
            if rag_settings.morgan_enable_channels:
                try:
                    from morgan.channels import ChannelGateway
                    gateway = ChannelGateway(default_agent_id="main")

                    # Register Telegram channel if token is configured
                    if rag_settings.morgan_telegram_token:
                        try:
                            from morgan.channels.telegram_channel import TelegramChannel
                            telegram_ch = TelegramChannel(
                                token=rag_settings.morgan_telegram_token,
                                require_mention_in_groups=False,
                            )
                            gateway.register_channel(telegram_ch)
                            logger.info("Telegram channel registered with gateway")
                        except Exception as exc:
                            logger.warning("Telegram channel init failed (non-fatal): %s", exc)

                    # Register Synology Chat channel if token + URL are configured
                    if rag_settings.morgan_synology_token and rag_settings.morgan_synology_incoming_url:
                        try:
                            from morgan.channels.synology_channel import SynologyChannel
                            synology_ch = SynologyChannel(
                                token=rag_settings.morgan_synology_token,
                                incoming_url=rag_settings.morgan_synology_incoming_url,
                                webhook_path=rag_settings.morgan_synology_webhook_path,
                                webhook_port=rag_settings.morgan_synology_webhook_port,
                                bot_name=rag_settings.morgan_synology_bot_name,
                                rate_limit_per_minute=rag_settings.morgan_synology_rate_limit,
                            )
                            gateway.register_channel(synology_ch)
                            logger.info("Synology Chat channel registered with gateway")
                        except Exception as exc:
                            logger.warning("Synology Chat channel init failed (non-fatal): %s", exc)

                    # Wire gateway to assistant so channel messages get responses.
                    # The session_key string encodes session type (dm/group)
                    # which _infer_session_type in the server assistant parses
                    # to control memory gating (MEMORY.md only for dm, not group).
                    async def _channel_agent_handler(agent_id, session_key, message):
                        """Route channel messages through the Morgan assistant.

                        Memory behaviour:
                        - DM sessions (session_key contains \":dm:\") → full
                          memory access (MEMORY.md, daily logs, workspace).
                        - Group sessions (\":group:\") → memory gated out by
                          WorkspaceManager.load_session_context().
                        """
                        try:
                            logger.info(
                                "Channel handler: processing message from %s via %s: %s",
                                message.peer_id,
                                message.channel,
                                message.content[:100],
                            )
                            # Pass channel metadata so tools know the chat context
                            # (chat_id, message_id, thread_id for Telegram actions)
                            channel_meta = dict(message.metadata)
                            channel_meta["channel"] = message.channel
                            channel_meta["group_id"] = message.group_id
                            channel_meta["peer_id"] = message.peer_id

                            # Build approval callback for dangerous tool execution.
                            # If the message came from a Telegram channel, use its
                            # inline-button approval mechanism; otherwise skip.
                            approval_cb = None
                            if message.channel == "telegram":
                                tg_channel = gateway.channels.get("telegram")
                                if tg_channel and hasattr(tg_channel, "request_approval"):
                                    _chat_id = int(
                                        str(message.group_id).split(":")[0]
                                        if message.group_id
                                        else message.peer_id
                                    )
                                    _thread_id = message.metadata.get("message_thread_id")

                                    async def _approval_callback(
                                        tool_name: str,
                                        description: str,
                                        tool_input: dict,
                                        _cid: int = _chat_id,
                                        _tid=_thread_id,
                                        _ch=tg_channel,
                                    ) -> bool:
                                        return await _ch.request_approval(
                                            chat_id=_cid,
                                            tool_name=tool_name,
                                            description=description,
                                            tool_input=tool_input,
                                            thread_id=_tid,
                                        )

                                    approval_cb = _approval_callback

                            result = await assistant.chat(
                                message=message.content,
                                user_id=message.peer_id,
                                conversation_id=str(session_key),
                                channel_metadata=channel_meta,
                                approval_callback=approval_cb,
                            )
                            answer = result.answer if hasattr(result, "answer") else str(result)
                            logger.info("Channel handler: got answer (%d chars)", len(answer) if answer else 0)
                            if not answer or not answer.strip():
                                return "I processed your message but couldn't generate a response. Please try again."
                            return answer
                        except Exception as chat_exc:
                            logger.error("Channel chat failed: %s", chat_exc, exc_info=True)
                            return f"Sorry, I encountered an error: {str(chat_exc)[:200]}"

                    gateway.set_agent_handler(_channel_agent_handler)

                    await gateway.start()
                    app.state.channel_gateway = gateway
                    logger.info("ChannelGateway initialized with agent handler")
                except Exception as exc:
                    logger.warning("ChannelGateway init failed (non-fatal): %s", exc)

        except Exception as exc:
            logger.warning("New module initialization failed (non-fatal): %s", exc)

        if hasattr(app.state, "hook_manager"):
            try:
                from morgan.hook_system import HookType

                await app.state.hook_manager.trigger(
                    HookType.SESSION_START, {"scope": "server"}
                )
            except Exception as exc:
                logger.warning("SESSION_START hook failed (non-fatal): %s", exc)

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
            if hasattr(app.state, "hook_manager"):
                try:
                    from morgan.hook_system import HookType

                    await app.state.hook_manager.trigger(
                        HookType.SESSION_END, {"scope": "server"}
                    )
                except Exception as exc:
                    logger.warning("SESSION_END hook failed (non-fatal): %s", exc)

            # Close all component connections gracefully
            shutdown_tasks = []

            # Shutdown session manager
            if hasattr(app.state, "session_manager"):
                logger.info("Shutting down session manager...")
                shutdown_tasks.append(app.state.session_manager.stop())

            # Shutdown new module services
            if hasattr(app.state, "channel_gateway"):
                logger.info("Shutting down channel gateway...")
                shutdown_tasks.append(app.state.channel_gateway.stop())
            if hasattr(app.state, "heartbeat_manager"):
                logger.info("Shutting down heartbeat manager...")
                shutdown_tasks.append(app.state.heartbeat_manager.stop())
            if hasattr(app.state, "cron_service"):
                logger.info("Shutting down cron service...")
                try:
                    app.state.cron_service.save()
                except Exception as exc:
                    logger.warning("Cron service save on shutdown failed: %s", exc)
                shutdown_tasks.append(app.state.cron_service.stop())

            # Wait for async shutdowns first
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)

            # Shutdown task manager (sync, clear all tracked tasks)
            if hasattr(app.state, "task_manager"):
                logger.info("Shutting down task manager...")
                try:
                    tm = app.state.task_manager
                    if hasattr(tm, "shutdown"):
                        result = tm.shutdown()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception as exc:
                    logger.warning("TaskManager shutdown error (non-fatal): %s", exc)

            # Clear hook manager (unregister all handlers)
            if hasattr(app.state, "hook_manager"):
                logger.info("Shutting down hook manager...")
                # Nothing async to do, but clear references
                app.state.hook_manager = None

            # Clear app state store
            if hasattr(app.state, "app_state_store"):
                logger.info("Shutting down app state store...")
                app.state.app_state_store = None

            # Clear workspace manager reference
            if hasattr(app.state, "workspace_manager"):
                logger.info("Shutting down workspace manager...")
                app.state.workspace_manager = None

            # Shutdown assistant last (depends on other modules)
            if hasattr(app.state, "assistant"):
                logger.info("Shutting down assistant...")
                await app.state.assistant.shutdown()

            # Reset tools_api module caches
            try:
                from morgan_server.api.routes import tools_api
                tools_api._cached_tool_executor = None
                tools_api._cached_skill_registry = None
            except Exception:
                pass

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
    initialize_session_manager(
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

    # Feature module routes (suggestions, wellness, habits, quality)
    app.include_router(features_router, tags=["Features"])

    # New module routes (tools, skills, tasks, channels, workspace)
    app.include_router(tools_router, tags=["Modules"])

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
