"""
FastAPI server for Morgan Core Service
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from shared.config.base import ServiceConfig
from shared.models.base import Response, ProcessingResult
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode


class TextRequest(BaseModel):
    """Text input request"""
    text: str = Field(..., description="Input text")
    user_id: Optional[str] = Field("default", description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AudioRequest(BaseModel):
    """Audio input request"""
    user_id: Optional[str] = Field("default", description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ConversationResetRequest(BaseModel):
    """Conversation reset request"""
    user_id: Optional[str] = Field("default", description="User identifier")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime: str
    uptime_seconds: float
    request_count: int
    services: Dict[str, bool]
    orchestrator: Dict[str, Any]
    conversations: Dict[str, Any]


class StatusResponse(BaseModel):
    """System status response"""
    version: str
    status: str
    uptime: str
    uptime_seconds: float
    request_count: int
    services: Dict[str, bool]
    orchestrator: Dict[str, Any]
    conversations: Dict[str, Any]
    timestamp: float


class APIServer:
    """FastAPI server for Morgan Core"""

    def __init__(self, core_service, host: str = "0.0.0.0", port: int = 8000):
        self.core_service = core_service
        self.host = host
        self.port = port
        self.logger = logging.getLogger("api_server")
        self.app = None
        self.server = None

    async def start(self):
        """Start the API server"""
        # Create FastAPI app
        self.app = FastAPI(
            title="Morgan AI Assistant",
            description="Modern AI assistant with voice and text capabilities",
            version="0.2.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

        # Create server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )
        self.server = uvicorn.Server(config)

        self.logger.info(f"Starting API server on {self.host}:{self.port}")
        await self.server.serve()

    async def stop(self):
        """Stop the API server"""
        self.logger.info("Stopping API server...")
        if self.server:
            self.server.should_exit = True
        self.logger.info("API server stopped")

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint"""
            try:
                status = await self.core_service.get_system_status()
                return HealthResponse(**status)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

        @self.app.get("/status", response_model=StatusResponse)
        async def system_status() -> StatusResponse:
            """Detailed system status"""
            try:
                status = await self.core_service.get_system_status()
                return StatusResponse(**status)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Status check failed: {e}")

        @self.app.post("/api/text")
        async def process_text(request: TextRequest) -> Dict[str, Any]:
            """Process text input"""
            try:
                response = await self.core_service.process_text_request(
                    request.text,
                    request.user_id,
                    request.metadata
                )

                return {
                    "text": response.text,
                    "audio": response.audio_data.hex() if response.audio_data else None,
                    "actions": [action.dict() for action in response.actions] if response.actions else [],
                    "metadata": response.metadata,
                    "confidence": response.confidence
                }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Text processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Text processing failed: {e}")

        @self.app.post("/api/audio")
        async def process_audio(
            file: UploadFile = File(...),
            user_id: str = Form("default"),
            metadata: Optional[str] = Form(None)
        ) -> Dict[str, Any]:
            """Process audio input"""
            try:
                # Read audio file
                audio_data = await file.read()

                # Parse metadata if provided
                parsed_metadata = None
                if metadata:
                    import json
                    parsed_metadata = json.loads(metadata)

                response = await self.core_service.process_audio_request(
                    audio_data,
                    user_id,
                    parsed_metadata
                )

                return {
                    "text": response.text,
                    "transcribed_text": parsed_metadata.get("transcribed_text") if parsed_metadata else None,
                    "audio": response.audio_data.hex() if response.audio_data else None,
                    "actions": [action.dict() for action in response.actions] if response.actions else [],
                    "metadata": response.metadata,
                    "confidence": response.confidence
                }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Audio processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")

        @self.app.post("/api/conversation/reset")
        async def reset_conversation(request: ConversationResetRequest) -> Dict[str, Any]:
            """Reset conversation for a user"""
            try:
                # Reset conversation in manager
                self.core_service.conversation_manager.reset_context(request.user_id)

                return {
                    "success": True,
                    "message": f"Conversation reset for user: {request.user_id}",
                    "user_id": request.user_id
                }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Conversation reset error: {e}")
                raise HTTPException(status_code=500, detail=f"Conversation reset failed: {e}")

        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Morgan AI Assistant",
                "version": "0.2.0",
                "status": "running",
                "docs": "/docs",
                "health": "/health"
            }

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all requests"""
            start_time = asyncio.get_event_loop().time()

            try:
                response = await call_next(request)
                process_time = asyncio.get_event_loop().time() - start_time

                # Log request details
                self.logger.info(
                    f"{request.method} {request.url.path} - {response.status_code} - {process_time".3f"}s"
                )

                # Add processing time header
                response.headers["X-Process-Time"] = str(process_time)

                return response

            except Exception as e:
                process_time = asyncio.get_event_loop().time() - start_time
                self.logger.error(f"Request error: {request.method} {request.url.path} - {e} - {process_time".3f"}s")
                raise
