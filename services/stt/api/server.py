"""
FastAPI server for STT service with WebSocket support
"""
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
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


class StreamStartRequest(BaseModel):
    """Request to start streaming session"""
    language: Optional[str] = Field("auto", description="Language for transcription")


class StreamChunkMessage(BaseModel):
    """WebSocket message for audio chunk"""
    type: str = Field("audio_chunk", description="Message type")
    audio_data: str = Field(..., description="Base64 encoded audio data")
    timestamp: Optional[float] = Field(None, description="Audio timestamp")


class StreamControlMessage(BaseModel):
    """WebSocket message for control commands"""
    type: str = Field("control", description="Message type")
    command: str = Field(..., description="Control command (start, stop, pause)")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


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


@app.post("/transcribe/chunk")
async def transcribe_chunk(request: TranscribeRequest) -> Dict[str, Any]:
    """Transcribe a single audio chunk for real-time streaming"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Use the new chunk transcription method for faster processing
        response = await app.state.stt_service.transcribe_chunk(audio_bytes, request.language)

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
        error_handler.logger.error(f"Chunk transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk transcription failed: {e}")


@app.post("/transcribe/realtime")
async def transcribe_realtime(request: TranscribeRequest) -> Dict[str, Any]:
    """Real-time transcription with VAD processing"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Use real-time processing with VAD
        response = await app.state.stt_service.process_realtime_audio(audio_bytes, request.language)

        return response

    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Real-time transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time transcription failed: {e}")


@app.post("/stream/start")
async def start_stream(request: StreamStartRequest) -> Dict[str, Any]:
    """Start a new audio streaming session"""
    try:
        session_id = str(uuid.uuid4())
        result = await app.state.stt_service.start_audio_stream(session_id, request.language)
        return result
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Start stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@app.post("/stream/{session_id}/chunk")
async def add_stream_chunk(session_id: str, request: TranscribeRequest) -> Dict[str, Any]:
    """Add audio chunk to streaming session"""
    try:
        # Decode base64 audio data
        audio_bytes = bytes.fromhex(request.audio_data)

        # Add chunk to session
        result = await app.state.stt_service.add_audio_chunk(session_id, audio_bytes)
        return result
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"Add chunk error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add chunk: {e}")


@app.post("/stream/{session_id}/end")
async def end_stream(session_id: str) -> Dict[str, Any]:
    """End streaming session and get final transcription"""
    try:
        result = await app.state.stt_service.end_audio_stream(session_id)
        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.duration,
            "segments": result.segments,
            "metadata": result.metadata
        }
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"End stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end stream: {e}")


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()

    try:
        # Start streaming session
        result = await app.state.stt_service.start_audio_stream(session_id, "auto")
        await websocket.send_text(json.dumps({
            "type": "session_started",
            "session_id": session_id,
            "sample_rate": result["sample_rate"],
            "chunk_size": result["chunk_size"]
        }))

        audio_buffer = []
        buffer_duration = 0.0

        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)

                if data["type"] == "audio_chunk":
                    # Decode base64 audio data
                    audio_bytes = bytes.fromhex(data["audio_data"])

                    # Add to buffer
                    audio_buffer.append(audio_bytes)
                    buffer_duration += 0.1  # Assume 100ms chunks

                    # Process if we have enough audio (at least 1 second)
                    if buffer_duration >= 1.0:
                        # Transcribe current buffer
                        result = await app.state.stt_service.transcribe_streaming(audio_buffer, "auto")

                        # Send transcription result
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": result.text,
                            "confidence": result.confidence,
                            "is_final": False,
                            "duration": result.duration
                        }))

                        # Clear buffer but keep last chunk for context
                        if len(audio_buffer) > 1:
                            audio_buffer = audio_buffer[-1:]
                            buffer_duration = 0.1

                elif data["type"] == "control":
                    if data["command"] == "stop":
                        # End session and send final result
                        final_result = await app.state.stt_service.end_audio_stream(session_id)
                        await websocket.send_text(json.dumps({
                            "type": "final_transcription",
                            "text": final_result.text,
                            "confidence": final_result.confidence,
                            "duration": final_result.duration
                        }))
                        break
                    elif data["command"] == "pause":
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "paused"
                        }))
                    elif data["command"] == "resume":
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "resumed"
                        }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))

    except WebSocketDisconnect:
        # Clean up session on disconnect
        try:
            await app.state.stt_service.end_audio_stream(session_id)
        except:
            pass
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Server error: {str(e)}"
        }))


@app.get("/stream/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List active streaming sessions"""
    try:
        sessions = list(app.state.stt_service.streaming_sessions.keys())
        return {
            "active_sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e}")


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
