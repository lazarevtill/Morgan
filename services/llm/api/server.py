"""
FastAPI server for LLM service
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ..service import LLMService, LLMConfig
from shared.config.base import ServiceConfig
from shared.models.base import LLMRequest, LLMResponse, ProcessingResult
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Text prompt for generation")
    model: Optional[str] = Field(None, description="Model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, le=8192, description="Maximum tokens to generate")
    context: Optional[List[Dict[str, str]]] = Field(None, description="Conversation context")
    stream: bool = Field(False, description="Enable streaming response")


class EmbedRequest(BaseModel):
    """Request model for text embedding"""
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Model to use")


class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    size: int
    modified: str
    template: str
    details: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: Optional[str]
    available_models: int
    ollama_url: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    config = ServiceConfig("llm")
    llm_config = LLMConfig(**config.all())

    logger = setup_logging(
        "llm_api",
        llm_config.log_level,
        "logs/llm_api.log"
    )

    logger.info("Starting LLM API server...")

    # Initialize LLM service
    app.state.llm_service = LLMService(config)
    await app.state.llm_service.start()

    logger.info("LLM API server started")

    yield

    # Shutdown
    logger.info("Shutting down LLM API server...")
    await app.state.llm_service.stop()
    logger.info("LLM API server stopped")


# Create FastAPI app
app = FastAPI(
    title="Morgan LLM Service",
    description="LLM service using Ollama with OpenAI compatibility",
    version="0.2.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        health = await app.state.llm_service.health_check()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/generate")
async def generate_text(request: GenerateRequest) -> Dict[str, Any]:
    """Generate text using LLM"""
    try:
        # Convert request to internal format
        llm_request = LLMRequest(
            prompt=request.prompt,
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            context=[Message(role=msg["role"], content=msg["content"])
                    for msg in request.context] if request.context else None,
            stream=request.stream
        )

        if request.stream:
            # Return streaming response
            async def generate_stream():
                try:
                    async for chunk in app.state.llm_service.generate_stream(llm_request):
                        yield f"data: {chunk}\n\n"
                except Exception as e:
                    yield f"data: ERROR: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return regular response
            response = await app.state.llm_service.generate(llm_request)
            return {
                "text": response.text,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "metadata": response.metadata
            }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/embed")
async def embed_text(request: EmbedRequest) -> Dict[str, Any]:
    """Generate text embeddings"""
    try:
        embeddings = await app.state.llm_service.embed_text(request.text, request.model)
        return {
            "embeddings": embeddings,
            "model": request.model or app.state.llm_service.llm_config.model,
            "dimensions": len(embeddings)
        }
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models"""
    try:
        models = await app.state.llm_service.list_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@app.get("/models/{model_name}")
async def get_model_info(model_name: str) -> ModelInfo:
    """Get information about a specific model"""
    try:
        info = await app.state.llm_service.get_model_info(model_name)

        return ModelInfo(
            name=info.get("model", model_name),
            size=info.get("size", 0),
            modified=info.get("modified_at", ""),
            template=info.get("template", ""),
            details=info.get("details", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@app.post("/models/{model_name}/pull")
async def pull_model(model_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Pull a model from registry"""
    try:
        # Add to background tasks
        background_tasks.add_task(app.state.llm_service.pull_model, model_name)

        return {
            "status": "pulling",
            "model": model_name,
            "message": "Model pull started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start model pull: {e}")


# OpenAI compatible endpoints
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI compatible chat completions endpoint"""
    try:
        # Extract parameters
        messages = request.get("messages", [])
        model = request.get("model", app.state.llm_service.llm_config.model)
        temperature = request.get("temperature", app.state.llm_service.llm_config.temperature)
        max_tokens = request.get("max_tokens", app.state.llm_service.llm_config.max_tokens)
        stream = request.get("stream", False)

        # Convert OpenAI format to internal format
        prompt = ""
        system_prompt = None
        context = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role in ["user", "assistant"]:
                context.append(Message(role=role, content=content))
                if role == "user":
                    prompt = content  # Last user message is the prompt

        llm_request = LLMRequest(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context,
            stream=stream
        )

        if stream:
            async def openai_stream():
                try:
                    async for chunk in app.state.llm_service.generate_stream(llm_request):
                        response_chunk = {
                            "id": "chatcmpl-123",
                            "object": "chat.completion.chunk",
                            "created": 1234567890,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                except Exception as e:
                    error_chunk = {
                        "error": {"message": str(e), "type": "internal_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                openai_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = await app.state.llm_service.generate(llm_request)

            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": response.finish_reason or "stop"
                }],
                "usage": response.usage or {
                    "prompt_tokens": 0,
                    "completion_tokens": len(response.text.split()),
                    "total_tokens": 0
                }
            }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"OpenAI compatible endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")


@app.post("/v1/embeddings")
async def openai_embeddings(request: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI compatible embeddings endpoint"""
    try:
        input_text = request.get("input", "")
        model = request.get("model", app.state.llm_service.llm_config.model)

        if isinstance(input_text, list):
            input_text = input_text[0]  # Take first text if list

        embeddings = await app.state.llm_service.embed_text(input_text, model)

        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embeddings,
                "index": 0
            }],
            "model": model,
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split())
            }
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"OpenAI embeddings error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.get("/v1/models")
async def openai_models() -> Dict[str, Any]:
    """OpenAI compatible models endpoint"""
    try:
        models_info = await app.state.llm_service.list_models()

        models = []
        for model in models_info.get("models", []):
            models.append({
                "id": model.get("name"),
                "object": "model",
                "created": 1234567890,
                "owned_by": "ollama"
            })

        return {
            "object": "list",
            "data": models
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


async def main():
    """Main entry point"""
    config = ServiceConfig("llm")
    llm_config = LLMConfig(**config.all())

    logger = setup_logging(
        "llm_api_main",
        llm_config.log_level
    )

    logger.info(f"Starting LLM API server on {llm_config.host}:{llm_config.port}")

    server_config = uvicorn.Config(
        app,
        host=llm_config.host,
        port=llm_config.port,
        log_level=llm_config.log_level.lower(),
        access_log=True
    )

    server = uvicorn.Server(server_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
