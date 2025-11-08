"""
Morgan AI Assistant - Core Service
Modern orchestration service with async/await support
"""

import asyncio
import signal
import sys
import os
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from shared.config.base import ServiceConfig
from shared.models.base import (
    Message,
    ConversationContext,
    Response,
    Command,
    ProcessingResult,
)
from shared.utils.logging import setup_logging, Timer
from shared.utils.exceptions import MorganException, ErrorCategory
from shared.utils.http_client import service_registry

# Import core components (relative imports)
from api.server import APIServer
from conversation.manager import ConversationManager
from handlers.registry import HandlerRegistry
from integrations.manager import IntegrationManager
from services.streaming_orchestrator import StreamingOrchestrator
from memory.manager import MemoryManager
from tools.mcp_manager import MCPToolsManager


class CoreConfig(BaseModel):
    """Core service configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    https_port: int = 8443
    llm_service_url: str = "http://llm-service:8001"
    tts_service_url: str = "http://tts-service:8002"
    stt_service_url: str = "http://stt-service:8003"
    conversation_timeout: int = 1800
    max_history: int = 50
    log_level: str = "INFO"
    enable_prometheus: bool = True
    enable_memory: bool = True
    enable_tools: bool = True
    use_https: bool = False
    ssl_cert_path: str = "/app/cert.pem"

    # Streaming optimization
    streaming_enabled: bool = True
    stream_chunk_size: int = 8192
    stream_timeout: int = 60
    real_time_processing: bool = True
    ssl_key_path: str = "/app/cert.pfx"
    external_llm_api_base: str = "https://gpt.lazarev.cloud/ollama/v1"
    external_llm_api_key: str = ""
    embedding_model: str = "qwen3-embedding:latest"


class MorganCore:
    """Modern Morgan Core Service"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("core")
        self.logger = setup_logging(
            "morgan_core", self.config.get("log_level", "INFO"), "logs/core.log"
        )

        # Load configuration with proper defaults
        config_data = self.config.all()
        self.core_config = CoreConfig(**config_data)

        # Service components
        self.conversation_manager = None
        self.handler_registry = None
        self.integration_manager = None
        self.streaming_orchestrator = (
            None  # Unified streaming and non-streaming orchestrator
        )
        self.api_server = None
        self.memory_manager = None
        self.tools_manager = None

        # Runtime state
        self.running = False
        self.start_time = time.time()
        self.request_count = 0

        self.logger.info("Morgan Core Service initialized")

    async def start(self):
        """Start the Morgan Core Service"""
        self.running = True
        self.logger.info("Morgan Core Service starting...")

        try:
            # Initialize service components
            await self._initialize_components()

            # Start API server
            self.api_server = APIServer(
                self,
                host=self.core_config.host,
                port=self.core_config.port,
                https_port=self.core_config.https_port,
                use_https=self.core_config.use_https,
                ssl_cert_path=self.core_config.ssl_cert_path,
                ssl_key_path=self.core_config.ssl_key_path,
            )
            await self.api_server.start()

            # Start background tasks
            asyncio.create_task(self._periodic_cleanup())
            asyncio.create_task(self._health_monitoring())

            self.logger.info("Morgan Core Service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start Morgan Core Service: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop the Morgan Core Service"""
        self.logger.info("Morgan Core Service stopping...")
        self.running = False

        try:
            # Stop API server
            if self.api_server:
                await self.api_server.stop()

            # Stop streaming orchestrator
            if self.streaming_orchestrator:
                await self.streaming_orchestrator.stop()

            # Stop conversation manager and disconnect from databases
            if self.conversation_manager:
                if self.conversation_manager.db:
                    await self.conversation_manager.db.disconnect()
                if self.conversation_manager.redis:
                    await self.conversation_manager.redis.disconnect()

            # Stop database managers
            if self.memory_manager:
                await self.memory_manager.stop()
            if self.tools_manager:
                await self.tools_manager.stop()

            # Disconnect from service registry
            await service_registry.disconnect_all()

            self.logger.info("Morgan Core Service stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _initialize_components(self):
        """Initialize core components"""
        try:
            # Register services
            await self._register_services()

            # Initialize database clients (PostgreSQL + Redis)
            db_client, redis_client = await self._initialize_database_clients()

            # Initialize database managers (Memory + Tools)
            await self._initialize_databases()

            # Initialize conversation manager with database clients
            self.conversation_manager = ConversationManager(
                config=self.config,
                db_client=db_client,
                redis_client=redis_client,
                max_history=self.core_config.max_history,
                timeout=self.core_config.conversation_timeout,
            )

            # Initialize handler registry
            self.handler_registry = HandlerRegistry(self)

            # Initialize integration manager
            self.integration_manager = IntegrationManager(self.config)

            # Initialize streaming orchestrator (unified for both streaming and non-streaming)
            self.streaming_orchestrator = StreamingOrchestrator(
                config=self.config,
                conversation_manager=self.conversation_manager,
                handler_registry=self.handler_registry,
                integration_manager=self.integration_manager,
                memory_manager=self.memory_manager,
                tools_manager=self.tools_manager,
                redis_client=redis_client,
                db_client=db_client,
            )

            await self.streaming_orchestrator.start()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def _initialize_database_clients(self):
        """Initialize PostgreSQL and Redis clients with proper validation"""
        from shared.utils.database import DatabaseClient
        from shared.utils.redis_client import RedisClient

        db_client = None
        redis_client = None

        # Check if database configuration is provided
        # Check both MORGAN_POSTGRES_HOST (from config) and standard POSTGRES_HOST (from env)
        postgres_host = self.config.get("postgres_host") or os.getenv("POSTGRES_HOST")
        postgres_port = self.config.get("postgres_port") or os.getenv("POSTGRES_PORT")
        postgres_db = self.config.get("postgres_db") or os.getenv("POSTGRES_DB")
        postgres_user = self.config.get("postgres_user") or os.getenv("POSTGRES_USER")
        postgres_password = self.config.get("postgres_password") or os.getenv(
            "POSTGRES_PASSWORD"
        )

        redis_host = self.config.get("redis_host") or os.getenv(
            "REDIS_HOST", "localhost"
        )
        redis_port = self.config.get("redis_port") or os.getenv("REDIS_PORT")
        redis_password = self.config.get("redis_password") or os.getenv(
            "REDIS_PASSWORD"
        )

        # Initialize PostgreSQL if configured
        if postgres_host:
            try:
                self.logger.info(f"Connecting to PostgreSQL at {postgres_host}...")
                db_client = DatabaseClient(
                    host=postgres_host,
                    port=int(postgres_port) if postgres_port else None,
                    database=postgres_db,
                    user=postgres_user,
                    password=postgres_password,
                )
                await db_client.connect()

                # Validate connection by executing a simple query
                if db_client.pool:
                    async with db_client.acquire() as conn:
                        result = await conn.fetchval("SELECT 1")
                        if result == 1:
                            self.logger.info(
                                "PostgreSQL connection validated successfully"
                            )
                        else:
                            raise Exception("PostgreSQL connection validation failed")
                else:
                    raise Exception("PostgreSQL connection pool not initialized")

            except Exception as e:
                self.logger.error(f"Failed to connect to PostgreSQL: {e}")
                self.logger.warning(
                    "Database persistence will be disabled. Running with in-memory fallback."
                )
                db_client = None
        else:
            self.logger.info(
                "PostgreSQL not configured. Using in-memory conversation storage."
            )

        # Initialize Redis if configured
        try:
            self.logger.info(f"Connecting to Redis at {redis_host}...")
            redis_client = RedisClient(
                host=redis_host,
                port=int(redis_port) if redis_port else None,
                password=redis_password,
            )
            await redis_client.connect()

            # Validate connection by executing PING
            if redis_client.client:
                ping_result = await redis_client.client.ping()
                if ping_result:
                    self.logger.info("Redis connection validated successfully")
                else:
                    raise Exception("Redis PING validation failed")
            else:
                raise Exception("Redis client not initialized")

        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.logger.warning(
                "Redis caching will be disabled. Running with in-memory fallback."
            )
            redis_client = None

        return db_client, redis_client

    async def _initialize_databases(self):
        """Initialize database managers (Memory + Tools)"""
        # Check if memory and tools are enabled
        enable_memory = self.config.get("enable_memory", True)
        enable_tools = self.config.get("enable_tools", True)

        if not enable_memory and not enable_tools:
            self.logger.info(
                "Memory and tools disabled - skipping database initialization"
            )
            return

        try:
            # Initialize Memory Manager if enabled
            if enable_memory:
                qdrant_host = self.config.get("qdrant_host", "qdrant")
                qdrant_port = self.config.get("qdrant_port", 6333)
                embedding_dim = self.config.get("embedding_dimension", 384)

                # Construct PostgreSQL DSN
                postgres_host = self.config.get("postgres_host", "postgres")
                postgres_port = self.config.get("postgres_port", 5432)
                postgres_db = self.config.get("postgres_db", "morgan")
                postgres_user = self.config.get("postgres_user", "morgan")
                postgres_password = self.config.get(
                    "postgres_password", "morgan_secure_password"
                )

                postgres_dsn = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

                # Use existing DatabaseClient from conversation manager
                self.memory_manager = MemoryManager(
                    postgres_dsn=postgres_dsn,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port,
                    embedding_dimension=embedding_dim,
                )
                await self.memory_manager.start()
                self.logger.info("Memory Manager initialized")

            # Initialize MCP Tools Manager if enabled
            if enable_tools:
                self.tools_manager = MCPToolsManager(postgres_dsn=postgres_dsn)
                await self.tools_manager.start()
                self.logger.info("MCP Tools Manager initialized")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize database managers: {e}", exc_info=True
            )
            raise

    async def _register_services(self):
        """Register external services"""
        try:
            # Register LLM service
            service_registry.register_service(
                "llm", self.core_config.llm_service_url, timeout=30.0, max_retries=3
            )

            # Register TTS service
            service_registry.register_service(
                "tts", self.core_config.tts_service_url, timeout=30.0, max_retries=3
            )

            # Register STT service
            service_registry.register_service(
                "stt", self.core_config.stt_service_url, timeout=30.0, max_retries=3
            )

            self.logger.info("Services registered successfully")

        except Exception as e:
            self.logger.error(f"Failed to register services: {e}")
            raise

    async def process_text_request(
        self,
        text: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Process text input request"""
        with Timer(self.logger, f"Text processing for user {user_id}"):
            try:
                self.request_count += 1

                # Get or create conversation context (async)
                context = await self.conversation_manager.get_context(user_id)

                # Add user message (async)
                user_message = Message(role="user", content=text, metadata=metadata)
                await self.conversation_manager.add_message(user_id, user_message)

                # Process through streaming orchestrator
                response = await self.streaming_orchestrator.process_request(
                    context, metadata
                )

                # Add assistant response to context (async)
                assistant_message = Message(
                    role="assistant",
                    content=response.text,
                    metadata={
                        "actions": len(response.actions) if response.actions else 0
                    },
                )
                await self.conversation_manager.add_message(user_id, assistant_message)

                return response

            except Exception as e:
                self.logger.error(f"Error processing text request: {e}")
                return Response(
                    text="I'm sorry, I'm having trouble processing your request. Please try again.",
                    metadata={"error": True, "error_message": str(e)},
                )

    async def process_audio_request(
        self,
        audio_data: bytes,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Process audio input request"""
        with Timer(self.logger, f"Audio processing for user {user_id}"):
            try:
                self.request_count += 1

                # Get or create conversation context (async)
                context = await self.conversation_manager.get_context(user_id)

                # Process through streaming orchestrator with audio
                response = await self.streaming_orchestrator.process_audio_request(
                    audio_data, context, metadata
                )

                # Add assistant response to context (async)
                assistant_message = Message(
                    role="assistant",
                    content=response.text,
                    metadata={"audio_processed": True},
                )
                await self.conversation_manager.add_message(user_id, assistant_message)

                return response

            except Exception as e:
                self.logger.error(f"Error processing audio request: {e}")
                return Response(
                    text="I'm sorry, I couldn't process the audio. Please try again.",
                    metadata={"error": True, "error_message": str(e)},
                )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get service health
            service_health = await service_registry.health_check_all()

            # Get component status
            orchestrator_status = await self.streaming_orchestrator.health_check()
            conversation_status = self.conversation_manager.get_status()

            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            uptime_str = self._format_uptime(uptime_seconds)

            # Get version from pyproject.toml if available
            version = "0.2.0"  # Default version

            return {
                "version": version,
                "status": "healthy" if all(service_health.values()) else "degraded",
                "uptime": uptime_str,
                "uptime_seconds": uptime_seconds,
                "request_count": self.request_count,
                "services": service_health,
                "orchestrator": orchestrator_status,
                "conversations": conversation_status,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "version": "0.2.0",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            }

    async def _periodic_cleanup(self):
        """Periodic cleanup tasks"""
        while self.running:
            try:
                # Clean up expired conversations
                self.conversation_manager.cleanup_expired()

                # Health check services
                await service_registry.health_check_all()

                await asyncio.sleep(60)  # Run every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

    async def _health_monitoring(self):
        """Health monitoring and metrics"""
        while self.running:
            try:
                # Log current status periodically
                if self.request_count > 0 and self.request_count % 100 == 0:
                    status = await self.get_system_status()
                    self.logger.info(
                        f"System status: {status['status']}, requests: {status['request_count']}"
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(30)

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)


async def main():
    """Main entry point"""
    # Setup configuration
    config = ServiceConfig("core")
    config_data = config.all()

    # Setup logging
    logger = setup_logging("morgan_core_main", config.get("log_level", "INFO"))

    logger.info("Starting Morgan Core Service...")

    try:
        core_config = CoreConfig(**config_data)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error(f"Available config keys: {list(config_data.keys())}")
        logger.error(f"Expected CoreConfig fields: {CoreConfig.model_fields.keys()}")
        raise

    # Create and start core service
    morgan_core = MorganCore(config)

    # Setup signal handling
    def signal_handler():
        logger.info("Shutting down Morgan Core Service...")
        asyncio.create_task(morgan_core.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: signal_handler())

    try:
        await morgan_core.start()

        # Keep the service running
        while morgan_core.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        await morgan_core.stop()
    except Exception as e:
        logger.error(f"Morgan Core Service failed: {e}")
        await morgan_core.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
