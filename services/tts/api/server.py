"""
FastAPI server for TTS service with streaming support
"""
import logging
from typing import Dict, Any, Optional
import json
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import uvicorn

from shared.models.base import TTSRequest, TTSResponse
from shared.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class GenerateSpeechRequest(BaseModel):
    """Speech generation request"""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice to use")
    speed: Optional[float] = Field(None, description="Speech speed")
    output_format: Optional[str] = Field(None, description="Output audio format")
    stream: bool = Field(False, description="Enable streaming response")


class VoiceListResponse(BaseModel):
    """Voice list response"""
    voices: list
    presets: dict
    current_voice: str
    current_model: str


class TTSAPIServer:
    """TTS API Server"""

    def __init__(self, tts_service, host: str = "0.0.0.0", port: int = 8002):
        self.tts_service = tts_service
        self.host = host
        self.port = port
        self.logger = setup_logging("tts_api", "INFO", "logs/tts_api.log")
        self.app = None

    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Morgan TTS Service",
            description="Text-to-Speech service with multiple voice models",
            version="0.2.0"
        )

        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health = await self.tts_service.health_check()
                return health
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

        @app.post("/generate")
        async def generate_speech(request: GenerateSpeechRequest):
            """Generate speech from text"""
            try:
                # Convert to TTSRequest
                tts_request = TTSRequest(
                    text=request.text,
                    voice=request.voice,
                    speed=request.speed,
                    output_format=request.output_format
                )

                if request.stream:
                    # Stream audio chunks
                    async def stream_audio():
                        """Stream audio in chunks"""
                        try:
                            # Generate full audio first
                            response = await self.tts_service.generate_speech(tts_request)
                            
                            # Stream in chunks (16KB each)
                            chunk_size = 16384
                            audio_data = response.audio_data
                            
                            for i in range(0, len(audio_data), chunk_size):
                                chunk = audio_data[i:i + chunk_size]
                                yield chunk
                        except Exception as e:
                            self.logger.error(f"Streaming error: {e}")
                            raise

                    return StreamingResponse(
                        stream_audio(),
                        media_type="audio/wav",
                        headers={
                            "Content-Disposition": "attachment; filename=speech.wav",
                            "Cache-Control": "no-cache"
                        }
                    )
                else:
                    # Return full audio
                    response = await self.tts_service.generate_speech(tts_request)
                    
                    return {
                        "audio_data": base64.b64encode(response.audio_data).decode('utf-8'),
                        "format": response.format,
                        "sample_rate": response.sample_rate,
                        "duration": response.duration,
                        "metadata": response.metadata
                    }

            except Exception as e:
                self.logger.error(f"Speech generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}")

        @app.post("/generate/audio")
        async def generate_speech_audio(request: GenerateSpeechRequest):
            """Generate speech and return raw audio"""
            try:
                tts_request = TTSRequest(
                    text=request.text,
                    voice=request.voice,
                    speed=request.speed,
                    output_format=request.output_format
                )

                response = await self.tts_service.generate_speech(tts_request)
                
                return Response(
                    content=response.audio_data,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": "attachment; filename=speech.wav"
                    }
                )

            except Exception as e:
                self.logger.error(f"Audio generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")

        @app.get("/voices", response_model=VoiceListResponse)
        async def list_voices():
            """List available voices"""
            try:
                voices_data = await self.tts_service.list_voices()
                return VoiceListResponse(**voices_data)
            except Exception as e:
                self.logger.error(f"List voices error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list voices: {e}")

        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Morgan TTS Service",
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
        self.logger.info(f"Starting TTS API server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main(tts_service, host: str = "0.0.0.0", port: int = 8002):
    """Main entry point for TTS API server"""
    server = TTSAPIServer(tts_service, host, port)
    await server.start()
