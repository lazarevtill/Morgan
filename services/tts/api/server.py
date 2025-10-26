"""
FastAPI server for TTS service
"""
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from service import TTSService, TTSConfig
from shared.config.base import ServiceConfig
from shared.models.base import TTSRequest, TTSResponse
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode
from shared.utils.middleware import RequestIDMiddleware, TimingMiddleware


class GenerateSpeechRequest(BaseModel):
    """Request model for speech generation"""
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field(None, description="Voice to use")
    speed: Optional[float] = Field(None, ge=0.1, le=3.0, description="Speech speed")
    language: Optional[str] = Field(None, description="Language code")
    format: Optional[str] = Field("wav", description="Output audio format")


class VoiceInfo(BaseModel):
    """Voice information response"""
    name: str
    language: str
    gender: Optional[str] = None
    age: Optional[str] = None
    accent: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    device: str
    available_voices: int
    current_voice: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    config = ServiceConfig("tts")
    tts_config = TTSConfig(**config.all())

    logger = setup_logging(
        "tts_api",
        tts_config.log_level,
        "logs/tts_api.log"
    )

    logger.info("Starting TTS API server...")

    # Initialize TTS service
    app.state.tts_service = TTSService(config)
    await app.state.tts_service.start()

    logger.info("TTS API server started")

    yield

    # Shutdown
    logger.info("Shutting down TTS API server...")
    await app.state.tts_service.stop()
    logger.info("TTS API server stopped")


# Create FastAPI app
app = FastAPI(
    title="Morgan TTS Service",
    description="High-performance text-to-speech synthesis service",
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
        health = await app.state.tts_service.health_check()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/generate")
async def generate_speech(request: GenerateSpeechRequest) -> Dict[str, Any]:
    """Generate speech from text"""
    try:
        # Convert request to internal format
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            language=request.language,
            format=request.format
        )

        response = await app.state.tts_service.generate_speech(tts_request)

        return {
            "audio_data": response.audio_data.hex(),  # Convert to hex for JSON
            "format": response.format,
            "sample_rate": response.sample_rate,
            "duration": response.duration,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Speech generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}")


@app.get("/voices")
async def list_voices() -> Dict[str, Any]:
    """List available voices"""
    try:
        voices = await app.state.tts_service.list_voices()
        return voices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {e}")


@app.post("/generate/stream")
async def generate_speech_stream(request: GenerateSpeechRequest) -> StreamingResponse:
    """Generate speech and stream audio data"""
    try:
        # Convert request to internal format
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            language=request.language,
            format=request.format
        )

        response = await app.state.tts_service.generate_speech(tts_request)

        async def audio_stream():
            # Send audio data in chunks
            chunk_size = 8192
            audio_data = response.audio_data

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type=f"audio/{request.format or 'wav'}",
            headers={
                "Content-Length": str(len(response.audio_data)),
                "X-Sample-Rate": str(response.sample_rate),
                "X-Duration": str(response.duration or 0),
                "X-Model": response.metadata.get("model", "unknown") if response.metadata else "unknown"
            }
        )

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Streaming speech generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming speech generation failed: {e}")


@app.post("/preview")
async def preview_voice(request: GenerateSpeechRequest) -> Dict[str, Any]:
    """Preview voice with a sample text"""
    try:
        # Use a short sample text for preview
        preview_text = request.text[:100] + "..." if len(request.text) > 100 else request.text

        tts_request = TTSRequest(
            text=preview_text,
            voice=request.voice,
            speed=request.speed,
            language=request.language,
            format=request.format
        )

        response = await app.state.tts_service.generate_speech(tts_request)

        return {
            "audio_data": response.audio_data.hex(),
            "format": response.format,
            "sample_rate": response.sample_rate,
            "duration": response.duration,
            "preview_text": preview_text,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Voice preview error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice preview failed: {e}")


async def main():
    """Main entry point"""
    config = ServiceConfig("tts")
    tts_config = TTSConfig(**config.all())

    logger = setup_logging(
        "tts_api_main",
        tts_config.log_level
    )

    logger.info(f"Starting TTS API server on {tts_config.host}:{tts_config.port}")

    server_config = uvicorn.Config(
        app,
        host=tts_config.host,
        port=tts_config.port,
        log_level=tts_config.log_level.lower(),
        access_log=True
    )

    server = uvicorn.Server(server_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
