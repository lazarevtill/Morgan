"""
FastAPI server for VAD service
"""
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
import uvicorn

from ..service import VADService, VADConfig
from shared.config.base import ServiceConfig
from shared.models.base import VADRequest, VADResponse
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode


class DetectSpeechRequest(BaseModel):
    """Request model for speech detection"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection threshold")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    session_id: Optional[str] = Field("default", description="Session ID for state tracking")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    device: str
    active_states: int
    threshold: float
    test_detection: bool


class VADInfoResponse(BaseModel):
    """VAD information response"""
    model: str
    threshold: float
    min_speech_duration: float
    max_speech_duration: float
    window_size: int
    sample_rate: int
    device: str
    active_states: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    config = ServiceConfig("vad")
    vad_config = VADConfig(**config.all())

    logger = setup_logging(
        "vad_api",
        vad_config.log_level,
        "logs/vad_api.log"
    )

    logger.info("Starting VAD API server...")

    # Initialize VAD service
    app.state.vad_service = VADService(config)
    await app.state.vad_service.start()

    logger.info("VAD API server started")

    yield

    # Shutdown
    logger.info("Shutting down VAD API server...")
    await app.state.vad_service.stop()
    logger.info("VAD API server stopped")


# Create FastAPI app
app = FastAPI(
    title="Morgan VAD Service",
    description="Voice Activity Detection service using Silero VAD",
    version="0.2.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        health = await app.state.vad_service.health_check()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/detect")
async def detect_speech(request: DetectSpeechRequest) -> Dict[str, Any]:
    """Detect speech in audio data"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Create VAD request
        vad_request = VADRequest(
            audio_data=audio_bytes,
            threshold=request.threshold,
            sample_rate=request.sample_rate
        )

        # Add session ID for state tracking
        vad_request.session_id = request.session_id

        response = await app.state.vad_service.detect_speech(vad_request)

        return {
            "speech_detected": response.speech_detected,
            "confidence": response.confidence,
            "speech_segments": response.speech_segments,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Speech detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech detection failed: {e}")


@app.post("/detect/file")
async def detect_speech_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Detect speech in uploaded audio file"""
    try:
        # Read file content
        audio_bytes = await file.read()

        # Create VAD request
        vad_request = VADRequest(audio_data=audio_bytes)

        response = await app.state.vad_service.detect_speech(vad_request)

        return {
            "speech_detected": response.speech_detected,
            "confidence": response.confidence,
            "speech_segments": response.speech_segments,
            "metadata": response.metadata
        }

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"File speech detection error: {e}")
        raise HTTPException(status_code=500, detail=f"File speech detection failed: {e}")


@app.get("/info")
async def get_vad_info() -> VADInfoResponse:
    """Get VAD service information"""
    try:
        info = await app.state.vad_service.get_vad_info()
        return VADInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get VAD info: {e}")


@app.post("/reset/{session_id}")
async def reset_state(session_id: str):
    """Reset VAD state for a session"""
    try:
        await app.state.vad_service.reset_state(session_id)
        return {"message": f"VAD state reset for session: {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset VAD state: {e}")


@app.post("/reset")
async def reset_default_state():
    """Reset default VAD state"""
    try:
        await app.state.vad_service.reset_state("default")
        return {"message": "Default VAD state reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset default VAD state: {e}")


async def main():
    """Main entry point"""
    config = ServiceConfig("vad")
    vad_config = VADConfig(**config.all())

    logger = setup_logging(
        "vad_api_main",
        vad_config.log_level
    )

    logger.info(f"Starting VAD API server on {vad_config.host}:{vad_config.port}")

    server_config = uvicorn.Config(
        app,
        host=vad_config.host,
        port=vad_config.port,
        log_level=vad_config.log_level.lower(),
        access_log=True
    )

    server = uvicorn.Server(server_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
