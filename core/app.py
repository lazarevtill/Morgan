"""
Morgan AI Assistant - Core Service
Modern orchestration service with async/await support
"""
import asyncio
import signal
import sys
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
from shared.models.base import Message, ConversationContext, Response, Command, ProcessingResult
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, ErrorCode
from shared.utils.http_client import service_registry

# Import core components (relative imports)
from api.server import APIServer
from conversation.manager import ConversationManager
from handlers.registry import HandlerRegistry
from integrations.manager import IntegrationManager
from services.orchestrator import ServiceOrchestrator
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
    ssl_key_path: str = "/app/cert.pfx"
    external_llm_api_base: str = "https://gpt.lazarev.cloud/ollama/v1"
    external_llm_api_key: str = ""
    embedding_model: str = "qwen3-embedding:latest"


class MorganCore:
    """Modern Morgan Core Service"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("core")
        self.error_handler = ErrorHandler()
        self.logger = setup_logging(
            "morgan_core",
            self.config.get("log_level", "INFO"),
            "logs/core.log"
        )

        # Load configuration with proper defaults
        config_data = self.config.all()
        self.core_config = CoreConfig(**config_data)

        # Service components
        self.conversation_manager = None
        self.handler_registry = None
        self.integration_manager = None
        self.service_orchestrator = None
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
                ssl_key_path=self.core_config.ssl_key_path
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

            # Stop service orchestrator
            if self.service_orchestrator:
                await self.service_orchestrator.stop()

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

            # Initialize database managers
            await self._initialize_databases()

            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                config=self.config,
                max_history=self.core_config.max_history,
                timeout=self.core_config.conversation_timeout
            )
            await self.conversation_manager.initialize()

            # Initialize handler registry
            self.handler_registry = HandlerRegistry(self)

            # Initialize integration manager
            self.integration_manager = IntegrationManager(self.config)

            # Initialize service orchestrator
            self.service_orchestrator = ServiceOrchestrator(
                config=self.config,
                conversation_manager=self.conversation_manager,
                handler_registry=self.handler_registry,
                integration_manager=self.integration_manager,
                memory_manager=self.memory_manager,
                tools_manager=self.tools_manager
            )

            await self.service_orchestrator.start()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def _initialize_databases(self):
        """Initialize database connections"""
        # Check if memory and tools are enabled
        enable_memory = self.config.get("enable_memory", True)
        enable_tools = self.config.get("enable_tools", True)

        if not enable_memory and not enable_tools:
            self.logger.info("Memory and tools disabled - skipping database initialization")
            return

        try:
            # Get database configuration from YAML/env
            postgres_host = self.config.get("postgres_host", "postgres")
            postgres_port = self.config.get("postgres_port", 5432)
            postgres_db = self.config.get("postgres_db", "morgan")
            postgres_user = self.config.get("postgres_user", "morgan")
            postgres_password = self.config.get("postgres_password", "morgan_secure_password")

            # Build PostgreSQL DSN
            postgres_dsn = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

            # Initialize Memory Manager if enabled
            if enable_memory:
                qdrant_host = self.config.get("qdrant_host", "qdrant")
                qdrant_port = self.config.get("qdrant_port", 6333)
                embedding_dim = self.config.get("embedding_dimension", 384)

                self.memory_manager = MemoryManager(
                    postgres_dsn=postgres_dsn,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port,
                    embedding_dimension=embedding_dim
                )
                await self.memory_manager.start()
                self.logger.info("Memory Manager initialized")

            # Initialize MCP Tools Manager if enabled
            if enable_tools:
                self.tools_manager = MCPToolsManager(postgres_dsn=postgres_dsn)
                await self.tools_manager.start()
                self.logger.info("MCP Tools Manager initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize databases: {e}", exc_info=True)
            raise

    async def _register_services(self):
        """Register external services"""
        try:
            # Register LLM service
            service_registry.register_service(
                "llm",
                self.core_config.llm_service_url,
                timeout=30.0,
                max_retries=3
            )

            # Register TTS service
            service_registry.register_service(
                "tts",
                self.core_config.tts_service_url,
                timeout=30.0,
                max_retries=3
            )

            # Register STT service
            service_registry.register_service(
                "stt",
                self.core_config.stt_service_url,
                timeout=30.0,
                max_retries=3
            )

            self.logger.info("Services registered successfully")

        except Exception as e:
            self.logger.error(f"Failed to register services: {e}")
            raise

    async def process_text_request(self, text: str, user_id: str = "default",
                                 metadata: Optional[Dict[str, Any]] = None) -> Response:
        """Process text input request"""
        with Timer(self.logger, f"Text processing for user {user_id}"):
            try:
                self.request_count += 1

                # Get or create conversation context
                context = self.conversation_manager.get_context(user_id)

                # Add user message
                user_message = Message(
                    role="user",
                    content=text,
                    metadata=metadata
                )
                context.add_message(user_message)

                # Process through orchestrator
                response = await self.service_orchestrator.process_request(context, metadata)

                # Add assistant response to context
                assistant_message = Message(
                    role="assistant",
                    content=response.text,
                    metadata={"actions": len(response.actions) if response.actions else 0}
                )
                context.add_message(assistant_message)

                return response

            except Exception as e:
                self.logger.error(f"Error processing text request: {e}")
                error_response = self.error_handler.create_error_response(
                    "I'm sorry, I'm having trouble processing your request. Please try again.",
                    ErrorCode.INTERNAL_ERROR
                )
                return Response(
                    text=error_response["error"]["message"],
                    metadata={"error": True}
                )

    async def process_audio_request(self, audio_data: bytes, user_id: str = "default",
                                  metadata: Optional[Dict[str, Any]] = None) -> Response:
        """Process audio input request"""
        with Timer(self.logger, f"Audio processing for user {user_id}"):
            try:
                self.request_count += 1

                # Get or create conversation context
                context = self.conversation_manager.get_context(user_id)

                # Process through orchestrator with audio
                response = await self.service_orchestrator.process_audio_request(
                    audio_data, context, metadata
                )

                # Add assistant response to context
                assistant_message = Message(
                    role="assistant",
                    content=response.text,
                    metadata={"audio_processed": True}
                )
                context.add_message(assistant_message)

                return response

            except Exception as e:
                self.logger.error(f"Error processing audio request: {e}")
                error_response = self.error_handler.create_error_response(
                    "I'm sorry, I couldn't process the audio. Please try again.",
                    ErrorCode.AUDIO_PROCESSING_ERROR
                )
                return Response(
                    text=error_response["error"]["message"],
                    metadata={"error": True}
                )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get service health
            service_health = await service_registry.health_check_all()

            # Get component status
            orchestrator_status = await self.service_orchestrator.health_check()
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
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "version": "0.2.0",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
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
                    self.logger.info(f"System status: {status['status']}, requests: {status['request_count']}")

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
    logger = setup_logging(
        "morgan_core_main",
        config.get("log_level", "INFO")
    )

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
