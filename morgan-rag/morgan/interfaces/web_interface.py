"""
Morgan Web Interface - FastAPI Application.

Provides REST API endpoints for:
- Chat (synchronous and streaming)
- Document ingestion
- Health checks
- Feedback submission
- Metrics and statistics
- User preferences
- Session management

Full async/await with FastAPI.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install fastapi uvicorn"
    )

from morgan.core.assistant import AssistantError, MorganAssistant
from morgan.core.types import AssistantResponse as CoreAssistantResponse
from morgan.emotions.types import EmotionResult
from morgan.jina.reranking.service import RerankingService
from morgan.learning.types import FeedbackSignal, FeedbackType, LearningMetrics
from morgan.services.embedding_service import EmbeddingService
from morgan.vector_db.client import QdrantClient

logger = logging.getLogger(__name__)


# ==================== Request/Response Models ====================


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    include_sources: bool = Field(True, description="Include RAG sources")
    include_emotion: bool = Field(True, description="Include emotion analysis")
    include_metrics: bool = Field(False, description="Include performance metrics")


class EmotionResponse(BaseModel):
    """Emotion response model."""

    primary_emotion: str
    intensity: float
    confidence: float
    emotions: Dict[str, float]


class SourceResponse(BaseModel):
    """RAG source response model."""

    content: str
    source: str
    score: float
    document_id: str


class MetricsResponse(BaseModel):
    """Performance metrics response model."""

    total_duration_ms: float
    emotion_detection_ms: float
    memory_retrieval_ms: float
    rag_search_ms: float
    response_generation_ms: float
    learning_update_ms: float


class ChatResponse(BaseModel):
    """Chat response model."""

    response_id: str
    content: str
    timestamp: datetime
    emotion: Optional[EmotionResponse] = None
    sources: List[SourceResponse] = Field(default_factory=list)
    metrics: Optional[MetricsResponse] = None
    confidence: float = 1.0


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    components: Dict[str, Dict[str, any]]


class FeedbackRequest(BaseModel):
    """Feedback request model."""

    response_id: str
    user_id: str
    session_id: str
    rating: float = Field(..., ge=0.0, le=1.0, description="Rating from 0.0 to 1.0")
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    """Feedback response model."""

    feedback_id: str
    status: str
    message: str


class LearningStatsResponse(BaseModel):
    """Learning statistics response model."""

    patterns_detected: int
    feedback_processed: int
    preferences_learned: int
    adaptations_made: int
    consolidations_performed: int
    avg_confidence: float


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime


# ==================== Application ====================


class MorganWebApp:
    """Morgan Web Application."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2:latest",
        vector_db_url: str = "http://localhost:6333",
        vector_db_collection: str = "morgan_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_emotion_detection: bool = True,
        enable_learning: bool = True,
        enable_rag: bool = True,
        cors_origins: Optional[List[str]] = None,
    ):
        """
        Initialize Morgan Web Application.

        Args:
            storage_path: Path for persistent storage
            llm_base_url: LLM API base URL
            llm_model: LLM model name
            vector_db_url: Vector database URL
            vector_db_collection: Vector database collection name
            embedding_model: Embedding model name
            enable_emotion_detection: Enable emotion detection
            enable_learning: Enable learning system
            enable_rag: Enable RAG search
            cors_origins: Allowed CORS origins
        """
        self.storage_path = storage_path or Path.home() / ".morgan"
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.vector_db_url = vector_db_url
        self.vector_db_collection = vector_db_collection
        self.embedding_model = embedding_model
        self.enable_emotion_detection = enable_emotion_detection
        self.enable_learning = enable_learning
        self.enable_rag = enable_rag
        self.cors_origins = cors_origins or ["*"]

        # Assistant instance (initialized on startup)
        self.assistant: Optional[MorganAssistant] = None

        # Create FastAPI app
        self.app = self._create_app()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        # Startup
        logger.info("Starting Morgan Web Interface")
        await self._startup()
        yield
        # Shutdown
        logger.info("Shutting down Morgan Web Interface")
        await self._shutdown()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Morgan AI Assistant API",
            description="Intelligent, emotionally-aware assistant with learning capabilities",
            version="2.0.0",
            lifespan=self._lifespan,
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes(app)

        # Add exception handlers
        self._register_exception_handlers(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register all API routes."""

        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "name": "Morgan AI Assistant",
                "version": "2.0.0",
                "status": "operational",
            }

        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return await self._health_check()

        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """
            Process chat message.

            Processes a user message and returns a complete response with
            optional emotion, sources, and metrics.
            """
            return await self._process_chat(request)

        @app.post("/chat/stream")
        async def chat_stream(request: ChatRequest):
            """
            Stream chat response.

            Streams the response token-by-token using Server-Sent Events (SSE).
            """
            return StreamingResponse(
                self._stream_chat(request),
                media_type="text/event-stream",
            )

        @app.post("/feedback", response_model=FeedbackResponse)
        async def submit_feedback(request: FeedbackRequest):
            """Submit user feedback."""
            return await self._submit_feedback(request)

        @app.get("/learning/stats", response_model=LearningStatsResponse)
        async def learning_stats(user_id: str):
            """Get learning statistics for a user."""
            return await self._get_learning_stats(user_id)

        @app.get("/sessions/{session_id}/history")
        async def session_history(session_id: str, limit: int = 50):
            """Get conversation history for a session."""
            return await self._get_session_history(session_id, limit)

        @app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str):
            """Delete a conversation session."""
            return await self._delete_session(session_id)

    def _register_exception_handlers(self, app: FastAPI) -> None:
        """Register exception handlers."""

        @app.exception_handler(AssistantError)
        async def assistant_error_handler(request, exc: AssistantError):
            """Handle assistant errors."""
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Assistant error",
                    detail=str(exc),
                    correlation_id=exc.correlation_id,
                    timestamp=datetime.now(),
                ).dict(),
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc: Exception):
            """Handle general exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(exc),
                    timestamp=datetime.now(),
                ).dict(),
            )

    async def _startup(self) -> None:
        """Initialize assistant on startup."""
        try:
            # Initialize vector DB if RAG is enabled
            vector_db = None
            if self.enable_rag:
                vector_db = QdrantClient(
                    url=self.vector_db_url,
                    collection_name=self.vector_db_collection,
                )

            # Initialize embedding service
            embedding_service = None
            if self.enable_rag:
                embedding_service = EmbeddingService(
                    model_name=self.embedding_model,
                )

            # Initialize reranking service
            reranking_service = None
            if self.enable_rag:
                reranking_service = RerankingService()

            # Create assistant
            self.assistant = MorganAssistant(
                storage_path=self.storage_path,
                llm_base_url=self.llm_base_url,
                llm_model=self.llm_model,
                vector_db=vector_db,
                embedding_service=embedding_service,
                reranking_service=reranking_service,
                enable_emotion_detection=self.enable_emotion_detection,
                enable_learning=self.enable_learning,
                enable_rag=self.enable_rag,
            )

            # Initialize
            await self.assistant.initialize()

            logger.info("Morgan Assistant initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}", exc_info=True)
            raise

    async def _shutdown(self) -> None:
        """Cleanup on shutdown."""
        if self.assistant:
            await self.assistant.cleanup()

    async def _health_check(self) -> HealthResponse:
        """Perform health check."""
        components = {}

        # Check assistant
        if self.assistant:
            components["assistant"] = {
                "status": "operational",
                "healthy": True,
            }

            # Check emotion detector
            if self.assistant.emotion_detector:
                try:
                    health = await self.assistant.emotion_detector.health_check()
                    components["emotion"] = {
                        "status": health.status.value,
                        "healthy": health.healthy,
                        "message": health.message,
                    }
                except Exception as e:
                    components["emotion"] = {
                        "status": "error",
                        "healthy": False,
                        "message": str(e),
                    }

            # Check learning engine
            if self.assistant.learning_engine:
                try:
                    health = await self.assistant.learning_engine.health_check()
                    components["learning"] = {
                        "status": health.status.value,
                        "healthy": health.healthy,
                        "message": health.message,
                    }
                except Exception as e:
                    components["learning"] = {
                        "status": "error",
                        "healthy": False,
                        "message": str(e),
                    }

            # Check RAG
            if self.assistant.search_engine:
                components["rag"] = {
                    "status": "operational",
                    "healthy": True,
                }
        else:
            components["assistant"] = {
                "status": "not_initialized",
                "healthy": False,
            }

        # Determine overall status
        all_healthy = all(c.get("healthy", False) for c in components.values())
        overall_status = "healthy" if all_healthy else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
        )

    async def _process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat message."""
        if not self.assistant:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assistant not initialized",
            )

        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())

            # Process message
            response = await self.assistant.process_message(
                user_id=request.user_id,
                message=request.message,
                session_id=session_id,
            )

            # Convert to response model
            return self._convert_response(
                response,
                include_sources=request.include_sources,
                include_emotion=request.include_emotion,
                include_metrics=request.include_metrics,
            )

        except Exception as e:
            logger.error(f"Chat processing failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    async def _stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Stream chat response."""
        if not self.assistant:
            yield f"data: {{'error': 'Assistant not initialized'}}\n\n"
            return

        try:
            import json

            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())

            # Send metadata
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            # Stream response
            async for chunk in self.assistant.stream_response(
                user_id=request.user_id,
                message=request.message,
                session_id=session_id,
            ):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            # Send completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def _submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Submit user feedback."""
        if not self.assistant or not self.assistant.learning_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning system not available",
            )

        try:
            # Create feedback signal
            feedback = FeedbackSignal(
                feedback_type=FeedbackType.EXPLICIT,
                signal_value=request.rating,
                timestamp=datetime.now(),
                response_id=request.response_id,
                user_id=request.user_id,
                session_id=request.session_id,
                context_data={"comment": request.comment} if request.comment else {},
            )

            # Process feedback
            await self.assistant.learning_engine.process_feedback(feedback)

            return FeedbackResponse(
                feedback_id=str(uuid.uuid4()),
                status="processed",
                message="Feedback received and processed",
            )

        except Exception as e:
            logger.error(f"Feedback submission failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    async def _get_learning_stats(self, user_id: str) -> LearningStatsResponse:
        """Get learning statistics."""
        if not self.assistant or not self.assistant.learning_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning system not available",
            )

        try:
            metrics = await self.assistant.learning_engine.get_metrics()

            return LearningStatsResponse(
                patterns_detected=metrics.patterns_detected,
                feedback_processed=metrics.feedback_processed,
                preferences_learned=metrics.preferences_learned,
                adaptations_made=metrics.adaptations_made,
                consolidations_performed=metrics.consolidations_performed,
                avg_confidence=metrics.avg_confidence,
            )

        except Exception as e:
            logger.error(f"Failed to get learning stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    async def _get_session_history(self, session_id: str, limit: int) -> Dict:
        """Get session history."""
        if not self.assistant:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assistant not initialized",
            )

        try:
            # Get messages from memory system
            messages = await self.assistant.memory_system.get_session_messages(
                session_id=session_id,
                limit=limit,
            )

            return {
                "session_id": session_id,
                "message_count": len(messages),
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                    for msg in messages
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get session history: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    async def _delete_session(self, session_id: str) -> Dict:
        """Delete session."""
        if not self.assistant:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assistant not initialized",
            )

        try:
            # Delete session from memory system
            await self.assistant.memory_system.delete_session(session_id)

            return {
                "session_id": session_id,
                "status": "deleted",
                "message": "Session deleted successfully",
            }

        except Exception as e:
            logger.error(f"Failed to delete session: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    def _convert_response(
        self,
        response: CoreAssistantResponse,
        include_sources: bool = True,
        include_emotion: bool = True,
        include_metrics: bool = False,
    ) -> ChatResponse:
        """Convert core response to API response."""
        # Convert emotion
        emotion_response = None
        if include_emotion and response.emotion:
            emotion_response = EmotionResponse(
                primary_emotion=response.emotion.primary_emotion,
                intensity=response.emotion.intensity,
                confidence=response.emotion.confidence,
                emotions=response.emotion.emotions,
            )

        # Convert sources
        sources_response = []
        if include_sources and response.sources:
            sources_response = [
                SourceResponse(
                    content=source.content,
                    source=source.source,
                    score=source.score,
                    document_id=source.document_id,
                )
                for source in response.sources[:5]  # Limit to top 5
            ]

        # Convert metrics
        metrics_response = None
        if include_metrics and hasattr(response, "metrics"):
            metrics_response = MetricsResponse(
                total_duration_ms=response.generation_time_ms,
                emotion_detection_ms=0.0,  # Would need to pass from assistant
                memory_retrieval_ms=0.0,
                rag_search_ms=0.0,
                response_generation_ms=response.generation_time_ms,
                learning_update_ms=0.0,
            )

        return ChatResponse(
            response_id=response.response_id,
            content=response.content,
            timestamp=response.timestamp,
            emotion=emotion_response,
            sources=sources_response,
            metrics=metrics_response,
            confidence=response.confidence,
        )


# ==================== Factory Functions ====================


def create_app(**kwargs) -> FastAPI:
    """
    Create Morgan Web Application.

    Args:
        **kwargs: Configuration parameters for MorganWebApp

    Returns:
        FastAPI application instance
    """
    web_app = MorganWebApp(**kwargs)
    return web_app.app


# ==================== Development Server ====================


if __name__ == "__main__":
    import uvicorn

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
