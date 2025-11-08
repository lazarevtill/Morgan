"""
FastAPI server for LLM service with streaming support
"""

import json
import logging
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shared.models.base import LLMRequest, LLMResponse
from shared.utils.logging import setup_logging
from shared.utils.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Generation request"""

    prompt: str = Field(..., description="Input prompt")
    context: Optional[list] = Field(None, description="Conversation context")
    model: Optional[str] = Field(None, description="Model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Temperature for sampling")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    stream: bool = Field(False, description="Enable streaming response")


class EmbeddingRequest(BaseModel):
    """Embedding request"""

    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


class ModelListResponse(BaseModel):
    """Model list response"""

    models: list
    total: int


class LLMAPIServer:
    """LLM API Server"""

    def __init__(self, llm_service, host: str = "0.0.0.0", port: int = 8001):
        self.llm_service = llm_service
        self.host = host
        self.port = port
        self.logger = setup_logging("llm_api", "INFO", "logs/llm_api.log")
        self.app = None

    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Morgan LLM Service",
            description="LLM service with OpenAI-compatible API",
            version="0.2.0",
        )

        # Add middleware
        app.add_middleware(RequestIDMiddleware)
        app.add_middleware(TimingMiddleware)
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_second=20.0,  # LLM service can handle more requests
            burst_size=40,
            exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"],
        )

        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health = await self.llm_service.health_check()
                return health
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

        @app.post("/generate", response_model=LLMResponse)
        async def generate(request: GenerateRequest):
            """Generate text completion"""
            try:
                # Convert to LLMRequest
                llm_request = LLMRequest(
                    prompt=request.prompt,
                    context=request.context,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                )

                # Check if streaming is requested
                if request.stream:
                    raise HTTPException(
                        status_code=400,
                        detail="Use /stream endpoint for streaming responses",
                    )

                response = await self.llm_service.generate(llm_request)
                return response

            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        @app.post("/stream")
        async def stream_generate(request: GenerateRequest):
            """Generate streaming text completion"""
            try:
                # Convert to LLMRequest
                llm_request = LLMRequest(
                    prompt=request.prompt,
                    context=request.context,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                )

                async def generate_stream():
                    """Generate streaming response"""
                    try:
                        async for chunk in self.llm_service.generate_stream(
                            llm_request
                        ):
                            # Send as SSE format
                            yield f"data: {json.dumps({'text': chunk})}\n\n"

                        # Send done signal
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    except Exception as e:
                        self.logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            except Exception as e:
                self.logger.error(f"Stream setup error: {e}")
                raise HTTPException(status_code=500, detail=f"Stream failed: {e}")

        @app.post("/embed")
        async def embed_text(request: EmbeddingRequest):
            """Generate text embeddings"""
            try:
                embedding = await self.llm_service.embed_text(
                    request.text, request.model
                )
                return {
                    "embedding": embedding,
                    "model": request.model
                    or self.llm_service.llm_config.embedding_model,
                    "dimensions": len(embedding),
                }
            except Exception as e:
                self.logger.error(f"Embedding error: {e}")
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

        @app.get("/models", response_model=ModelListResponse)
        async def list_models():
            """List available models"""
            try:
                models_data = await self.llm_service.list_models()
                return ModelListResponse(**models_data)
            except Exception as e:
                self.logger.error(f"List models error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to list models: {e}"
                )

        @app.get("/models/{model_name}")
        async def get_model_info(model_name: str):
            """Get model information"""
            try:
                model_info = await self.llm_service.get_model_info(model_name)
                return model_info
            except Exception as e:
                self.logger.error(f"Get model info error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to get model info: {e}"
                )

        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Morgan LLM Service",
                "version": "0.2.0",
                "status": "running",
                "docs": "/docs",
                "health": "/health",
            }

        self.app = app
        return app

    async def start(self):
        """Start the API server"""
        self.create_app()
        self.logger.info(f"Starting LLM API server on {self.host}:{self.port}")

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main(llm_service, host: str = "0.0.0.0", port: int = 8001):
    """Main entry point for LLM API server"""
    server = LLMAPIServer(llm_service, host, port)
    await server.start()
