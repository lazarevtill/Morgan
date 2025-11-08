"""
WebSocket handler for real-time streaming voice interactions
Integrates with StreamingOrchestrator for optimized STT → LLM → TTS pipeline
"""

import asyncio
import logging
import json
import base64
import time
from typing import Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from shared.utils.audio import safe_base64_decode, AudioValidationError

logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""

    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketHandler:
    """
    Handles WebSocket connections for real-time streaming

    Message types:
    - start: Initialize streaming session
    - audio: Audio chunk from client
    - text: Text message from client
    - stop: End streaming session
    - config: Update session configuration
    """

    def __init__(self, core_service):
        self.core_service = core_service
        self.streaming_orchestrator = core_service.streaming_orchestrator
        self.logger = logging.getLogger("websocket_handler")

        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}

    async def handle_connection(self, websocket: WebSocket, user_id: str = "default"):
        """Handle WebSocket connection lifecycle"""
        await websocket.accept()

        connection_id = f"ws_{id(websocket)}"
        self.active_connections[connection_id] = websocket

        session_id = None

        self.logger.info(
            f"WebSocket connection established: {connection_id} for user: {user_id}"
        )

        try:
            while True:
                # Receive message from client
                message_raw = await websocket.receive_text()

                # Validate JSON parsing with proper error handling
                try:
                    message_data = json.loads(message_raw)
                    if not isinstance(message_data, dict):
                        await self._send_error(
                            websocket, "Invalid message format: expected JSON object"
                        )
                        continue
                except json.JSONDecodeError as e:
                    await self._send_error(websocket, f"Invalid JSON: {str(e)}")
                    continue
                except Exception as e:
                    await self._send_error(
                        websocket, f"Failed to parse message: {str(e)}"
                    )
                    continue

                message_type = message_data.get("type")
                if not message_type:
                    await self._send_error(
                        websocket, "Message missing required 'type' field"
                    )
                    continue

                data = message_data.get("data", {})

                # Handle different message types
                if message_type == "start":
                    session_id = await self._handle_start(websocket, user_id, data)

                elif message_type == "audio":
                    if not session_id:
                        await self._send_error(
                            websocket, "No active session. Send 'start' first."
                        )
                        continue
                    await self._handle_audio_chunk(websocket, session_id, data)

                elif message_type == "text":
                    if not session_id:
                        await self._send_error(
                            websocket, "No active session. Send 'start' first."
                        )
                        continue
                    await self._handle_text_message(websocket, session_id, data)

                elif message_type == "utterance_end":
                    if not session_id:
                        await self._send_error(
                            websocket, "No active session. Send 'start' first."
                        )
                        continue
                    await self._handle_utterance_end(websocket, session_id, data)

                elif message_type == "stop":
                    if session_id:
                        await self._handle_stop(websocket, session_id)
                    break

                elif message_type == "config":
                    if session_id:
                        await self._handle_config(websocket, session_id, data)

                elif message_type == "ping":
                    await self._send_message(websocket, "pong", {})

                else:
                    await self._send_error(
                        websocket, f"Unknown message type: {message_type}"
                    )

        except WebSocketDisconnect:
            self.logger.info(f"WebSocket disconnected: {connection_id}")

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}", exc_info=True)
            await self._send_error(websocket, str(e))

        finally:
            # Cleanup
            if session_id:
                await self.streaming_orchestrator.end_streaming_session(session_id)

            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            self.logger.info(f"WebSocket connection closed: {connection_id}")

    async def _handle_start(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ) -> str:
        """Handle session start"""
        try:
            language = data.get("language", "auto")
            metadata = data.get("metadata", {})

            # Start streaming session
            session_result = await self.streaming_orchestrator.start_streaming_session(
                user_id=user_id,
                session_type="websocket",
                language=language,
                metadata=metadata,
            )

            session_id = session_result["session_id"]

            # Send confirmation
            await self._send_message(
                websocket,
                "session_started",
                {"session_id": session_id, "status": "active", "language": language},
            )

            self.logger.info(
                f"Started streaming session: {session_id} for user: {user_id}"
            )

            return session_id

        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            await self._send_error(websocket, f"Failed to start session: {e}")
            raise

    async def _handle_audio_chunk(
        self, websocket: WebSocket, session_id: str, data: Dict[str, Any]
    ):
        """Handle incoming audio chunk"""
        try:
            # Decode audio data
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                await self._send_error(websocket, "No audio_data in message")
                return

            # Safe base64 decode with validation
            try:
                audio_bytes = safe_base64_decode(audio_b64, max_size_mb=10)
            except AudioValidationError as e:
                await self._send_error(websocket, f"Invalid audio data: {str(e)}")
                return

            # Process through streaming orchestrator
            result = await self.streaming_orchestrator.process_audio_stream(
                session_id, audio_bytes
            )

            # Send transcription result if available
            if result["status"] == "success":
                await self._send_message(
                    websocket,
                    "transcription",
                    {
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "is_final": result["is_final"],
                    },
                )
            elif result["status"] == "no_speech":
                # Optionally send VAD status
                await self._send_message(
                    websocket,
                    "vad_status",
                    {"speech_detected": False, "vad_result": result["vad_result"]},
                )
            elif result["status"] == "error":
                await self._send_error(
                    websocket, result.get("message", "Processing error")
                )

        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            await self._send_error(websocket, f"Audio processing failed: {e}")

    async def _handle_text_message(
        self, websocket: WebSocket, session_id: str, data: Dict[str, Any]
    ):
        """Handle text message (e.g., from text input)"""
        try:
            text = data.get("text")
            if not text:
                await self._send_error(websocket, "No text in message")
                return

            # Process through complete utterance pipeline
            async for chunk in self.streaming_orchestrator.process_complete_utterance(
                session_id, text
            ):
                if chunk["type"] == "text":
                    # LLM text chunk
                    await self._send_message(
                        websocket,
                        "response_text",
                        {"text": chunk["text"], "is_final": chunk["is_final"]},
                    )

                elif chunk["type"] == "audio":
                    # TTS audio chunk
                    audio_b64 = base64.b64encode(chunk["audio_data"]).decode("utf-8")
                    await self._send_message(
                        websocket,
                        "response_audio",
                        {"audio_data": audio_b64, "is_final": chunk["is_final"]},
                    )

                elif chunk["type"] == "complete":
                    # Processing complete
                    await self._send_message(
                        websocket,
                        "response_complete",
                        {"text": chunk["text"], "is_final": True},
                    )

                elif chunk["type"] == "error":
                    await self._send_error(websocket, chunk["message"])

        except Exception as e:
            self.logger.error(f"Error processing text message: {e}")
            await self._send_error(websocket, f"Text processing failed: {e}")

    async def _handle_utterance_end(
        self, websocket: WebSocket, session_id: str, data: Dict[str, Any]
    ):
        """Handle end of utterance (complete sentence detected)"""
        try:
            # Get the buffered transcription
            transcription = data.get("transcription", "")

            if not transcription:
                return

            # Process the complete utterance
            await self._handle_text_message(
                websocket, session_id, {"text": transcription}
            )

        except Exception as e:
            self.logger.error(f"Error processing utterance end: {e}")
            await self._send_error(websocket, f"Utterance processing failed: {e}")

    async def _handle_stop(self, websocket: WebSocket, session_id: str):
        """Handle session stop"""
        try:
            # End streaming session
            result = await self.streaming_orchestrator.end_streaming_session(session_id)

            # Send confirmation
            await self._send_message(
                websocket,
                "session_ended",
                {
                    "session_id": session_id,
                    "status": "ended",
                    "duration": result.get("duration", 0),
                    "transcription": result.get("transcription", ""),
                },
            )

            self.logger.info(f"Ended streaming session: {session_id}")

        except Exception as e:
            self.logger.error(f"Error stopping session: {e}")
            await self._send_error(websocket, f"Failed to stop session: {e}")

    async def _handle_config(
        self, websocket: WebSocket, session_id: str, data: Dict[str, Any]
    ):
        """Handle configuration update"""
        try:
            # Update session configuration in Redis
            if self.streaming_orchestrator.redis:
                await self.streaming_orchestrator.redis.update_session(
                    session_id, **data
                )

            # Send confirmation
            await self._send_message(
                websocket, "config_updated", {"session_id": session_id, "config": data}
            )

        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            await self._send_error(websocket, f"Config update failed: {e}")

    async def _send_message(
        self, websocket: WebSocket, message_type: str, data: Dict[str, Any]
    ):
        """Send message to WebSocket client"""
        try:
            message = {"type": message_type, "data": data, "timestamp": time.time()}
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket client"""
        try:
            await self._send_message(websocket, "error", {"message": error_message})
        except Exception:
            pass  # Client probably disconnected

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    async def broadcast_message(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        for websocket in list(self.active_connections.values()):
            try:
                await self._send_message(websocket, message_type, data)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to client: {e}")
