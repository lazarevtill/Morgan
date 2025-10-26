"""
FastAPI server for STT service
"""
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from service import STTService, STTConfig
from shared.config.base import ServiceConfig
from shared.models.base import STTRequest, STTResponse
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode
from shared.utils.middleware import RequestIDMiddleware, TimingMiddleware
from shared.utils.audio import AudioUtils


class TranscribeRequest(BaseModel):
    """Request model for transcription"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    language: Optional[str] = Field(None, description="Language code")
    model: Optional[str] = Field(None, description="Model to use")
    prompt: Optional[str] = Field(None, description="Context prompt")


class LanguageDetectionRequest(BaseModel):
    """Request model for language detection"""
    audio_data: str = Field(..., description="Base64 encoded audio data")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    device: str
    vad_enabled: bool
    sample_rate: int
    test_transcription: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    config = ServiceConfig("stt")
    stt_config = STTConfig(**config.all())

    logger = setup_logging(
        "stt_api",
        stt_config.log_level,
        "logs/stt_api.log"
    )

    logger.info("Starting STT API server...")

    # Initialize STT service
    app.state.stt_service = STTService(config)
    await app.state.stt_service.start()

    logger.info("STT API server started")

    yield

    # Shutdown
    logger.info("Shutting down STT API server...")
    await app.state.stt_service.stop()
    logger.info("STT API server stopped")


# Create FastAPI app
app = FastAPI(
    title="Morgan STT Service",
    description="Speech-to-text service with Silero VAD integration",
    version="0.2.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        health = await app.state.stt_service.health_check()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/transcribe")
async def transcribe_audio(request: TranscribeRequest) -> Dict[str, Any]:
    """Transcribe audio to text"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Create STT request
        stt_request = STTRequest(
            audio_data=audio_bytes,
            language=request.language,
            model=request.model,
            prompt=request.prompt
        )

        response = await app.state.stt_service.transcribe(stt_request)

        return {
            "text": response.text,
            "language": response.language,
            "confidence": response.confidence,
            "duration": response.duration,
            "segments": response.segments,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/transcribe/file")
async def transcribe_audio_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe uploaded audio file"""
    try:
        # Read file content
        audio_bytes = await file.read()

        # Validate audio data
        if not AudioUtils.validate_audio_data(audio_bytes):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        # Create STT request
        stt_request = STTRequest(audio_data=audio_bytes)

        response = await app.state.stt_service.transcribe(stt_request)

        return {
            "text": response.text,
            "language": response.language,
            "confidence": response.confidence,
            "duration": response.duration,
            "segments": response.segments,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"File transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"File transcription failed: {e}")


@app.post("/detect-language")
async def detect_language(request: LanguageDetectionRequest) -> Dict[str, Any]:
    """Detect language of audio"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Detect language
        language = await app.state.stt_service.detect_language(audio_bytes)

        return {
            "language": language,
            "confidence": 0.0  # Language detection confidence not available
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {e}")


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models"""
    try:
        models = await app.state.stt_service.list_models()
        return {
            "models": models,
            "current_model": app.state.stt_service.stt_config.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@app.post("/transcribe/stream")
async def transcribe_stream(request: TranscribeRequest) -> Dict[str, Any]:
    """Transcribe audio with streaming support"""
    try:
        # For now, just use regular transcription
        # In the future, this could support real-time streaming
        return await transcribe_audio(request)

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Stream transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Stream transcription failed: {e}")


async def main():
    """Main entry point"""
    config = ServiceConfig("stt")
    stt_config = STTConfig(**config.all())

    logger = setup_logging(
        "stt_api_main",
        stt_config.log_level
    )

    logger.info(f"Starting STT API server on {stt_config.host}:{stt_config.port}")

    server_config = uvicorn.Config(
        app,
        host=stt_config.host,
        port=stt_config.port,
        log_level=stt_config.log_level.lower(),
        access_log=True
    )

    server = uvicorn.Server(server_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
