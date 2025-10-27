"""
FastAPI server for STT service with real-time streaming support
"""
import logging
from typing import Dict, Any, Optional
import json
import base64
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import uvicorn

from shared.models.base import STTRequest, STTResponse
from shared.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class TranscribeRequest(BaseModel):
    """Transcription request"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    language: Optional[str] = Field("auto", description="Language code or 'auto'")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature")
    prompt: Optional[str] = Field(None, description="Initial prompt for model")


class StreamStartRequest(BaseModel):
    """Streaming session start request"""
    language: str = Field("auto", description="Language for transcription")
    session_id: Optional[str] = Field(None, description="Custom session ID")


class StreamChunkRequest(BaseModel):
    """Streaming audio chunk request"""
    audio_data: str = Field(..., description="Base64 encoded audio chunk")


class STTAPIServer:
    """STT API Server with real-time streaming"""

    def __init__(self, stt_service, host: str = "0.0.0.0", port: int = 8003):
        self.stt_service = stt_service
        self.host = host
        self.port = port
        self.logger = setup_logging("stt_api", "INFO", "logs/stt_api.log")
        self.app = None

    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Morgan STT Service",
            description="Speech-to-Text service with Faster Whisper and Silero VAD",
            version="0.2.0"
        )

        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health = await self.stt_service.health_check()
                return health
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

        @app.post("/transcribe", response_model=STTResponse)
        async def transcribe(request: TranscribeRequest):
            """Transcribe audio to text"""
            try:
                # Decode audio data
                audio_bytes = base64.b64decode(request.audio_data)

                # Convert to STTRequest
                stt_request = STTRequest(
                    audio_data=audio_bytes,
                    language=request.language,
                    temperature=request.temperature,
                    prompt=request.prompt
                )

                response = await self.stt_service.transcribe(stt_request)
                return response

            except Exception as e:
                self.logger.error(f"Transcription error: {e}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        @app.post("/transcribe/file")
        async def transcribe_file(file: UploadFile = File(...)):
            """Transcribe uploaded audio file"""
            try:
                # Read file
                audio_bytes = await file.read()

                # Create STT request
                stt_request = STTRequest(
                    audio_data=audio_bytes,
                    language="auto"
                )

                response = await self.stt_service.transcribe(stt_request)
                
                return {
                    "text": response.text,
                    "language": response.language,
                    "confidence": response.confidence,
                    "duration": response.duration,
                    "segments": response.segments,
                    "metadata": response.metadata
                }

            except Exception as e:
                self.logger.error(f"File transcription error: {e}")
                raise HTTPException(status_code=500, detail=f"File transcription failed: {e}")

        @app.post("/transcribe/realtime")
        async def transcribe_realtime(request: TranscribeRequest):
            """Transcribe audio with real-time VAD and processing"""
            try:
                # Decode audio data
                audio_bytes = base64.b64decode(request.audio_data)

                # Process with real-time VAD
                result = await self.stt_service.process_realtime_audio(
                    audio_bytes,
                    request.language
                )

                return result

            except Exception as e:
                self.logger.error(f"Real-time transcription error: {e}")
                raise HTTPException(status_code=500, detail=f"Real-time transcription failed: {e}")

        @app.post("/stream/start")
        async def start_streaming(request: StreamStartRequest):
            """Start a new streaming session"""
            try:
                session_id = request.session_id or str(uuid4())
                
                result = await self.stt_service.start_audio_stream(
                    session_id,
                    request.language
                )

                return result

            except Exception as e:
                self.logger.error(f"Stream start error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")

        @app.post("/stream/{session_id}/chunk")
        async def add_stream_chunk(session_id: str, request: StreamChunkRequest):
            """Add audio chunk to streaming session"""
            try:
                # Decode audio chunk
                audio_bytes = base64.b64decode(request.audio_data)

                result = await self.stt_service.add_audio_chunk(session_id, audio_bytes)
                return result

            except Exception as e:
                self.logger.error(f"Stream chunk error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process chunk: {e}")

        @app.post("/stream/{session_id}/end")
        async def end_streaming(session_id: str):
            """End streaming session and get final transcription"""
            try:
                result = await self.stt_service.end_audio_stream(session_id)
                
                return {
                    "text": result.text,
                    "language": result.language,
                    "confidence": result.confidence,
                    "duration": result.duration,
                    "segments": result.segments,
                    "metadata": result.metadata
                }

            except Exception as e:
                self.logger.error(f"Stream end error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to end stream: {e}")

        @app.websocket("/ws/stream/{session_id}")
        async def websocket_stream(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time audio streaming"""
            await websocket.accept()
            self.logger.info(f"WebSocket connection established for session: {session_id}")

            try:
                # Initialize streaming session
                await self.stt_service.start_audio_stream(session_id, "auto")

                while True:
                    # Receive audio chunk from client
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "audio":
                        # Decode and process audio chunk
                        audio_data = base64.b64decode(message["audio_data"])
                        
                        # Add chunk to session
                        result = await self.stt_service.add_audio_chunk(session_id, audio_data)

                        # Send transcription result if available
                        if result.get("transcription"):
                            await websocket.send_text(json.dumps({
                                "type": "transcription",
                                "text": result["transcription"],
                                "confidence": result["confidence"],
                                "is_final": result["is_final"]
                            }))
                        else:
                            # Send status update
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "buffering"
                            }))

                    elif message.get("type") == "end":
                        # End streaming session
                        final_result = await self.stt_service.end_audio_stream(session_id)
                        
                        await websocket.send_text(json.dumps({
                            "type": "final",
                            "text": final_result.text,
                            "language": final_result.language,
                            "confidence": final_result.confidence
                        }))
                        break

            except WebSocketDisconnect:
                self.logger.info(f"WebSocket disconnected: {session_id}")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            finally:
                # Clean up session
                try:
                    await self.stt_service.end_audio_stream(session_id)
                except:
                    pass

        @app.get("/sessions")
        async def list_sessions():
            """List active streaming sessions"""
            try:
                sessions = await self.stt_service.list_active_sessions()
                return sessions
            except Exception as e:
                self.logger.error(f"List sessions error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e}")

        @app.get("/models")
        async def list_models():
            """List available Whisper models"""
            try:
                models = await self.stt_service.list_models()
                return {"models": models, "total": len(models)}
            except Exception as e:
                self.logger.error(f"List models error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")

        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Morgan STT Service",
                "version": "0.2.0",
                "status": "running",
                "docs": "/docs",
                "health": "/health"
            }

        self.app = app
        return app

    async def start(self):
        """Start the API server"""
        self.create_app()
        self.logger.info(f"Starting STT API server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main(stt_service, host: str = "0.0.0.0", port: int = 8003):
    """Main entry point for STT API server"""
    server = STTAPIServer(stt_service, host, port)
    await server.start()
