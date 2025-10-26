"""
FastAPI server for Morgan Core Service
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json
import base64
import os

from shared.config.base import ServiceConfig
from shared.models.base import Response, ProcessingResult
from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode
from shared.utils.middleware import RequestIDMiddleware, TimingMiddleware
from shared.utils.audio import AudioCapture, DeviceAudioCapture
from shared.utils.http_client import service_registry


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


class DeviceAudioRequest(BaseModel):
    """Device audio input request"""
    device_id: Optional[str] = Field(None, description="Audio device ID")
    device_type: str = Field("microphone", description="Type of audio device")
    language: Optional[str] = Field("auto", description="Language for transcription")
    real_time: bool = Field(False, description="Enable real-time processing")
    chunk_duration_ms: int = Field(100, description="Audio chunk duration in milliseconds")


class StreamAudioRequest(BaseModel):
    """Streaming audio request"""
    session_id: str = Field(..., description="Streaming session ID")
    audio_data: str = Field(..., description="Base64 encoded audio data")
    timestamp: Optional[float] = Field(None, description="Audio timestamp")


class DeviceListResponse(BaseModel):
    """Device list response"""
    devices: List[Dict[str, Any]]
    count: int
    supported_formats: List[str]


class StreamAudioMessage(BaseModel):
    """WebSocket message for streaming audio"""
    type: str = Field(..., description="Message type: audio, start, stop, config")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    sample_rate: Optional[int] = Field(16000, description="Audio sample rate")
    language: Optional[str] = Field("auto", description="Language for transcription")
    user_id: Optional[str] = Field("default", description="User identifier")


class StreamResponse(BaseModel):
    """WebSocket response message"""
    type: str = Field(..., description="Response type: transcription, error, status")
    text: Optional[str] = Field(None, description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Confidence score")
    is_final: Optional[bool] = Field(False, description="Is this final transcription")
    error: Optional[str] = Field(None, description="Error message")


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

        # Add middleware
        self.app.add_middleware(RequestIDMiddleware)
        self.app.add_middleware(TimingMiddleware)

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Serve static files
        static_path = os.path.join(os.path.dirname(__file__), "..", "static")
        if os.path.exists(static_path):
            self.app.mount("/static", StaticFiles(directory=static_path), name="static")

        # Add HTTP middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all requests"""
            start_time = asyncio.get_event_loop().time()

            try:
                response = await call_next(request)
                process_time = asyncio.get_event_loop().time() - start_time

                # Log request details
                self.logger.info(
                    f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
                )

                # Add processing time header
                response.headers["X-Process-Time"] = str(process_time)

                return response

            except Exception as e:
                process_time = asyncio.get_event_loop().time() - start_time
                self.logger.error(
                    f"Request error: {request.method} {request.url.path} - {e} - {process_time:.3f}s"
                )
                raise

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
                "health": "/health",
                "voice": "/voice",
                "webrtc": "/webrtc",
                "audio_devices": "/audio/devices",
                "audio_streaming": "/audio/stream/start"
            }

        @self.app.get("/voice", response_class=HTMLResponse)
        async def voice_interface():
            """Voice interface page"""
            try:
                static_path = os.path.join(os.path.dirname(__file__), "..", "static", "voice_input.html")
                with open(static_path, 'r', encoding='utf-8') as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(
                    content="<h1>Voice Interface</h1><p>Voice interface not available. Please check server configuration.</p>",
                    status_code=404
                )

        @self.app.get("/audio/devices")
        async def list_audio_devices() -> DeviceListResponse:
            """List available audio input devices"""
            try:
                # Get available audio devices (this would typically use system APIs)
                # For now, return a mock list with common device types
                mock_devices = [
                    {
                        "id": "default_mic",
                        "name": "Default Microphone",
                        "type": "microphone",
                        "is_default": True,
                        "is_available": True
                    },
                    {
                        "id": "system_mic",
                        "name": "System Microphone",
                        "type": "microphone",
                        "is_default": False,
                        "is_available": True
                    },
                    {
                        "id": "line_in",
                        "name": "Line Input",
                        "type": "line_in",
                        "is_default": False,
                        "is_available": True
                    }
                ]

                formatted_devices = DeviceAudioCapture.format_device_list(mock_devices)

                return DeviceListResponse(
                    devices=formatted_devices,
                    count=len(formatted_devices),
                    supported_formats=DeviceAudioCapture.get_supported_audio_formats()
                )

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Device list error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list devices: {e}")


        @self.app.post("/audio/stream/start")
        async def start_audio_stream(request: DeviceAudioRequest) -> Dict[str, Any]:
            """Start audio streaming session"""
            try:
                # Start STT streaming session
                stt_client = await service_registry.get_service("stt")
                stt_result = await stt_client.post("/stream/start", json_data={
                    "language": request.language
                })

                if not stt_result.success:
                    raise HTTPException(status_code=500, detail="Failed to start stream")

                stt_response = stt_result.data

                session_id = stt_response.get("session_id")

                # Generate device configuration
                device_config = DeviceAudioCapture.get_recommended_settings(request.device_type)

                return {
                    "session_id": session_id,
                    "status": "active",
                    "device_config": device_config,
                    "sample_rate": device_config["sample_rate"],
                    "chunk_size": device_config["chunk_size"],
                    "websocket_url": f"ws://localhost:8003/ws/stream/{session_id}"
                }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Start audio stream error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start audio stream: {e}")


        @self.app.post("/audio/stream/{session_id}/chunk")
        async def send_audio_chunk(session_id: str, request: StreamAudioRequest) -> Dict[str, Any]:
            """Send audio chunk to streaming session"""
            try:
                # Forward to STT service
                stt_client = service_registry.get_service("stt")
                stt_response = await stt_client.post(f"/stream/{session_id}/chunk", json={
                    "audio_data": request.audio_data
                })

                if stt_response.get("status") == "error":
                    raise HTTPException(status_code=500, detail=stt_response.get("message", "Chunk processing failed"))

                return stt_response

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Audio chunk error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process audio chunk: {e}")


        @self.app.post("/audio/stream/{session_id}/end")
        async def end_audio_stream(session_id: str) -> Dict[str, Any]:
            """End audio streaming session"""
            try:
                # End STT streaming session
                stt_client = service_registry.get_service("stt")
                stt_response = await stt_client.post(f"/stream/{session_id}/end")

                if stt_response.get("status") == "error":
                    raise HTTPException(status_code=500, detail=stt_response.get("message", "Failed to end stream"))

                # Process the final transcription through the core orchestrator
                transcription = stt_response.get("text", "")
                if transcription:
                    # Create a text request from the transcription
                    text_request = TextRequest(text=transcription, user_id="streaming_user")

                    # Process through core orchestrator
                    response = await self.core_service.process_text_request(
                        text_request.text,
                        text_request.user_id,
                        {"source": "voice_streaming", "session_id": session_id}
                    )

                    return {
                        "transcription": transcription,
                        "ai_response": response.text,
                        "session_id": session_id,
                        "metadata": response.metadata
                    }
                else:
                    return {
                        "transcription": "",
                        "ai_response": "",
                        "session_id": session_id,
                        "metadata": {"no_speech_detected": True}
                    }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"End audio stream error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to end audio stream: {e}")


        @self.app.post("/audio/process")
        async def process_audio_file(
            file: UploadFile = File(...),
            device_type: str = Form("microphone"),
            language: str = Form("auto"),
            real_time: bool = Form(False)
        ) -> Dict[str, Any]:
            """Process uploaded audio file from device"""
            try:
                # Read audio file
                audio_bytes = await file.read()

                # Validate audio
                validation = AudioCapture.validate_audio_format(audio_bytes)
                if not validation.get("valid", False):
                    raise HTTPException(status_code=400, detail=f"Invalid audio format: {validation.get('error')}")

                # Get STT service client
                stt_client = service_registry.get_service("stt")

                if real_time:
                    # Use real-time processing
                    stt_response = await stt_client.post("/transcribe/realtime", json={
                        "audio_data": base64.b64encode(audio_bytes).decode('utf-8')
                    })
                else:
                    # Use standard transcription
                    stt_response = await stt_client.post("/transcribe", json={
                        "audio_data": base64.b64encode(audio_bytes).decode('utf-8'),
                        "language": language
                    })

                transcription = stt_response.get("text", "")

                # Process through core orchestrator
                response = await self.core_service.process_text_request(
                    transcription,
                    "device_audio",
                    {
                        "source": "device_upload",
                        "device_type": device_type,
                        "audio_format": validation,
                        "real_time": real_time
                    }
                )

                return {
                    "transcription": transcription,
                    "ai_response": response.text,
                    "audio_metadata": validation,
                    "processing_metadata": response.metadata
                }

            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.logger.error(f"Audio processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process audio: {e}")


        @self.app.get("/voice", response_class=HTMLResponse)
        async def voice_interface():
            """Voice input interface page"""
            try:
                static_path = os.path.join(os.path.dirname(__file__), "..", "static", "voice_input.html")
                with open(static_path, 'r', encoding='utf-8') as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(
                    content="<h1>Voice Interface</h1><p>Voice interface not available. Please check server configuration.</p>",
                    status_code=404
                )

        @self.app.get("/webrtc", response_class=HTMLResponse)
        async def webrtc_interface():
            """WebRTC voice interface page"""
            try:
                static_path = os.path.join(os.path.dirname(__file__), "..", "static", "webrtc.html")
                with open(static_path, 'r', encoding='utf-8') as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(
                    content="<h1>WebRTC Interface</h1><p>WebRTC interface not available. Please check server configuration.</p>",
                    status_code=404
                )

        @self.app.websocket("/ws/audio")
        async def websocket_audio_stream(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time audio streaming and transcription"""
            await websocket.accept()

            # Store active streaming sessions
            if not hasattr(self, 'streaming_sessions'):
                self.streaming_sessions = {}

            session_id = f"ws_{id(websocket)}"
            self.streaming_sessions[session_id] = {
                'websocket': websocket,
                'user_id': 'default',
                'is_active': True,
                'audio_buffer': b'',  # Buffer for audio chunks
                'last_transcription': '',
                'use_vad': True
            }

            self.logger.info(f"Audio streaming session started: {session_id}")

            try:
                while True:
                    # Receive message from client
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    # Handle different message types
                    if data.get('type') == 'start':
                        await self._handle_stream_start(websocket, data, session_id)
                    elif data.get('type') == 'audio':
                        await self._handle_audio_chunk(websocket, data, session_id)
                    elif data.get('type') == 'stop':
                        await self._handle_stream_stop(websocket, session_id)
                        break
                    elif data.get('type') == 'config':
                        await self._handle_stream_config(websocket, data, session_id)
                    elif data.get('type') == 'webrtc_offer':
                        await self._handle_webrtc_offer(websocket, data, session_id)
                    elif data.get('type') == 'webrtc_answer':
                        await self._handle_webrtc_answer(websocket, data, session_id)
                    elif data.get('type') == 'webrtc_ice':
                        await self._handle_webrtc_ice(websocket, data, session_id)

            except WebSocketDisconnect:
                self.logger.info(f"WebSocket disconnected: {session_id}")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await self._send_error(websocket, str(e))
            finally:
                if session_id in self.streaming_sessions:
                    del self.streaming_sessions[session_id]

        return None

    async def _handle_stream_start(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle streaming session start"""
        try:
            self.streaming_sessions[session_id]['user_id'] = data.get('user_id', 'default')
            self.streaming_sessions[session_id]['language'] = data.get('language', 'auto')
            self.streaming_sessions[session_id]['sample_rate'] = data.get('sample_rate', 16000)

            # Send confirmation
            response = StreamResponse(
                type="status",
                text="Streaming started"
            )
            await websocket.send_text(response.model_dump_json())

            self.logger.info(f"Streaming started for session {session_id}, user: {self.streaming_sessions[session_id]['user_id']}")

        except Exception as e:
            self.logger.error(f"Error starting stream: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_audio_chunk(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle incoming audio chunk"""
        try:
            # Decode base64 audio data
            audio_b64 = data.get('audio_data')
            if not audio_b64:
                return

            audio_bytes = base64.b64decode(audio_b64)

            # Get session info
            session = self.streaming_sessions.get(session_id)
            if not session or not session.get('is_active'):
                return

            # Buffer audio chunks for better processing
            session['audio_buffer'] += audio_bytes

            # Process if buffer is large enough (e.g., > 1 second of audio)
            sample_rate = session.get('sample_rate', 16000)
            chunk_duration = len(session['audio_buffer']) / (sample_rate * 2)  # 16-bit samples

            if chunk_duration >= 0.5:  # Process every 0.5 seconds
                await self._process_audio_chunk(session['audio_buffer'], websocket, session)
                session['audio_buffer'] = b''  # Clear buffer after processing

        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            await self._send_error(websocket, str(e))

    async def _process_audio_chunk(self, audio_bytes: bytes, websocket: WebSocket, session: Dict[str, Any]):
        """Process audio chunk through STT service with real-time VAD"""
        try:
            # Send to STT service (use real-time endpoint with VAD)
            stt_response = await self.core_service.service_orchestrator.transcribe_chunk(
                audio_bytes,
                session.get('language', 'auto')
            )

            # Check if we got a valid response with text
            if stt_response and stt_response.get('text', '').strip():
                current_text = stt_response['text'].strip()
                confidence = stt_response.get('confidence', 0.0)
                vad_result = stt_response.get('vad_result', 'unknown')

                # Only process if we have meaningful text or if VAD detected speech
                if confidence > 0.1 or vad_result == 'speech_detected':
                    # Send transcription result (partial for now)
                    response = StreamResponse(
                        type="transcription",
                        text=current_text,
                        confidence=confidence,
                        is_final=False  # Real-time, not final
                    )
                    await websocket.send_text(response.model_dump_json())

                    # Update session's last transcription
                    session['last_transcription'] = current_text

                    # Process through LLM if confidence is high enough or if it's been a while
                    if confidence > 0.7 or len(current_text.split()) > 3:
                        await self._process_transcription(current_text, websocket, session)
                else:
                    # Send VAD status if no speech detected
                    response = StreamResponse(
                        type="status",
                        text=f"VAD: {vad_result}"
                    )
                    await websocket.send_text(response.model_dump_json())

        except Exception as e:
            self.logger.error(f"Error in STT processing: {e}")
            await self._send_error(websocket, "Speech recognition failed")

    async def _process_transcription(self, text: str, websocket: WebSocket, session: Dict[str, Any]):
        """Process transcription through LLM and generate response"""
        try:
            # Process through core service
            response = await self.core_service.process_text_request(
                text,
                session.get('user_id', 'default'),
                metadata={"source": "voice", "session_id": session}
            )

            if response and response.text:
                # Send LLM response
                llm_response = StreamResponse(
                    type="response",
                    text=response.text
                )
                await websocket.send_text(llm_response.model_dump_json())

                # If there's audio response, send it too
                if response.audio_data:
                    audio_b64 = base64.b64encode(response.audio_data).decode('utf-8')
                    audio_response = StreamResponse(
                        type="audio",
                        text=audio_b64
                    )
                    await websocket.send_text(audio_response.model_dump_json())

        except Exception as e:
            self.logger.error(f"Error in LLM processing: {e}")
            await self._send_error(websocket, "Response generation failed")

    async def _handle_stream_stop(self, websocket: WebSocket, session_id: str):
        """Handle streaming session stop"""
        try:
            if session_id in self.streaming_sessions:
                self.streaming_sessions[session_id]['is_active'] = False

            response = StreamResponse(
                type="status",
                text="Streaming stopped"
            )
            await websocket.send_text(response.model_dump_json())

            self.logger.info(f"Streaming stopped for session {session_id}")

        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")

    async def _handle_stream_config(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle streaming configuration update"""
        try:
            # Update session configuration
            if session_id in self.streaming_sessions:
                session = self.streaming_sessions[session_id]
                session.update({
                    'language': data.get('language', session.get('language', 'auto')),
                    'sample_rate': data.get('sample_rate', session.get('sample_rate', 16000)),
                    'user_id': data.get('user_id', session.get('user_id', 'default'))
                })

            response = StreamResponse(
                type="status",
                text="Configuration updated"
            )
            await websocket.send_text(response.model_dump_json())

        except Exception as e:
            self.logger.error(f"Error updating stream config: {e}")
            await self._send_error(websocket, str(e))

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket client"""
        try:
            response = StreamResponse(
                type="error",
                error=error_message
            )
            await websocket.send_text(response.model_dump_json())
        except Exception:
            # If we can't send error, connection is probably closed
            pass

    async def _handle_webrtc_offer(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle WebRTC offer from client"""
        try:
            self.logger.info(f"Received WebRTC offer for session {session_id}")
            # For now, just acknowledge - full WebRTC implementation would be more complex
            response = StreamResponse(
                type="webrtc_status",
                text="WebRTC offer received - using fallback WebSocket mode"
            )
            await websocket.send_text(response.model_dump_json())
        except Exception as e:
            self.logger.error(f"Error handling WebRTC offer: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_webrtc_answer(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle WebRTC answer from client"""
        try:
            self.logger.info(f"Received WebRTC answer for session {session_id}")
            # Acknowledge WebRTC answer
            response = StreamResponse(
                type="webrtc_status",
                text="WebRTC connection established"
            )
            await websocket.send_text(response.model_dump_json())
        except Exception as e:
            self.logger.error(f"Error handling WebRTC answer: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_webrtc_ice(self, websocket: WebSocket, data: Dict[str, Any], session_id: str):
        """Handle WebRTC ICE candidate from client"""
        try:
            self.logger.debug(f"Received WebRTC ICE candidate for session {session_id}")
            # Process ICE candidate (simplified)
            pass
        except Exception as e:
            self.logger.error(f"Error handling WebRTC ICE: {e}")
            await self._send_error(websocket, str(e))


