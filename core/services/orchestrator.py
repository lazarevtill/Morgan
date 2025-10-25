"""
Service orchestrator for Morgan Core Service
"""
import asyncio
import logging
from typing import Dict, Any, Optional

import aiohttp
from shared.config.base import ServiceConfig
from shared.models.base import Response, Command, ConversationContext, Message, AudioChunk
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, ErrorCode
from shared.utils.http_client import service_registry


class ServiceOrchestrator:
    """Orchestrates communication between services"""

    def __init__(self, config: ServiceConfig, conversation_manager, handler_registry, integration_manager):
        self.config = config
        self.conversation_manager = conversation_manager
        self.handler_registry = handler_registry
        self.integration_manager = integration_manager

        self.error_handler = ErrorHandler()
        self.logger = setup_logging("service_orchestrator", "INFO", "logs/orchestrator.log")

        # Service endpoints
        self.llm_service_url = config.get("llm_service_url", "http://llm-service:8001")
        self.tts_service_url = config.get("tts_service_url", "http://tts-service:8002")
        self.stt_service_url = config.get("stt_service_url", "http://stt-service:8003")
        self.vad_service_url = config.get("vad_service_url", "http://vad-service:8004")

        self.logger.info("Service orchestrator initialized")

    async def start(self):
        """Start the service orchestrator"""
        try:
            # Verify all services are available
            await self._verify_services()

            self.logger.info("Service orchestrator started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start service orchestrator: {e}")
            raise

    async def stop(self):
        """Stop the service orchestrator"""
        self.logger.info("Service orchestrator stopping...")
        # Nothing specific to clean up for now
        self.logger.info("Service orchestrator stopped")

    async def _verify_services(self):
        """Verify all required services are available"""
        services_to_check = [
            ("llm", self.llm_service_url),
            ("tts", self.tts_service_url),
            ("stt", self.stt_service_url),
            ("vad", self.vad_service_url)
        ]

        for service_name, service_url in services_to_check:
            try:
                # Use service registry for health checks
                client = service_registry.clients.get(service_name)
                if client:
                    is_healthy = await client.health_check()
                    if not is_healthy:
                        self.logger.warning(f"Service {service_name} health check failed")
                    else:
                        self.logger.info(f"Service {service_name} is healthy")
                else:
                    self.logger.warning(f"Service client for {service_name} not found")

            except Exception as e:
                self.logger.error(f"Failed to verify service {service_name}: {e}")

    async def process_request(self, context: ConversationContext,
                            metadata: Optional[Dict[str, Any]] = None) -> Response:
        """Process a text request through the service pipeline"""
        with Timer(self.logger, f"Request processing for user {context.user_id}"):
            try:
                # Step 1: Generate LLM response
                llm_response = await self._generate_llm_response(context)

                # Step 2: Execute any commands/actions
                executed_commands = []
                if llm_response.get("actions"):
                    for action in llm_response["actions"]:
                        command = Command(action=action["action"], parameters=action.get("parameters", {}))
                        result = await self.handler_registry.process_command(command, context)
                        executed_commands.append(result)

                # Step 3: Generate TTS response if requested
                audio_data = None
                if metadata and metadata.get("generate_audio", False):
                    audio_data = await self._generate_tts_response(llm_response["text"])

                # Step 4: Create final response
                return Response(
                    text=llm_response["text"],
                    audio_data=audio_data,
                    actions=executed_commands,
                    metadata={
                        "llm_model": llm_response.get("model"),
                        "llm_usage": llm_response.get("usage"),
                        "processing_time": metadata.get("processing_time") if metadata else None
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in request processing: {e}")
                return Response(
                    text="I'm sorry, I encountered an error while processing your request. Please try again.",
                    metadata={"error": True, "error_message": str(e)}
                )

    async def process_audio_request(self, audio_data: bytes, context: ConversationContext,
                                  metadata: Optional[Dict[str, Any]] = None) -> Response:
        """Process an audio request through the service pipeline"""
        with Timer(self.logger, f"Audio request processing for user {context.user_id}"):
            try:
                # Step 1: Apply VAD if available
                vad_result = await self._apply_vad(audio_data)
                if not vad_result.get("speech_detected", True):
                    return Response(
                        text="I didn't detect any speech in the audio. Please try again.",
                        metadata={"vad_result": vad_result}
                    )

                # Step 2: Transcribe audio to text
                transcription = await self._transcribe_audio(audio_data)
                transcribed_text = transcription.get("text", "")

                if not transcribed_text:
                    return Response(
                        text="I couldn't understand the audio. Please try speaking more clearly.",
                        metadata={"transcription": transcription}
                    )

                # Add transcription to context
                transcription_message = Message(
                    role="user",
                    content=transcribed_text,
                    metadata={"transcribed": True, "original_audio_length": len(audio_data)}
                )
                context.add_message(transcription_message)

                # Step 3: Process as text request
                return await self.process_request(context, metadata)

            except Exception as e:
                self.logger.error(f"Error in audio request processing: {e}")
                return Response(
                    text="I'm sorry, I encountered an error while processing the audio. Please try again.",
                    metadata={"error": True, "error_message": str(e)}
                )

    async def _generate_llm_response(self, context: ConversationContext) -> Dict[str, Any]:
        """Generate response from LLM service"""
        try:
            # Get recent conversation context
            recent_messages = context.get_last_n_messages(10)

            # Prepare LLM request
            llm_request = {
                "prompt": recent_messages[-1].content if recent_messages else "",
                "context": [
                    {"role": msg.role, "content": msg.content}
                    for msg in recent_messages[:-1]
                ],
                "system_prompt": "You are Morgan, a helpful AI assistant. Respond naturally and helpfully.",
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Call LLM service
            client = await service_registry.get_service("llm")
            result = await client.post("/generate", json_data=llm_request)

            if result.success:
                return result.data
            else:
                raise Exception(f"LLM service error: {result.error}")

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise

    async def _generate_tts_response(self, text: str) -> Optional[bytes]:
        """Generate TTS audio response"""
        try:
            # Prepare TTS request
            tts_request = {
                "text": text,
                "voice": "default",
                "speed": 1.0,
                "format": "wav"
            }

            # Call TTS service
            client = await service_registry.get_service("tts")
            result = await client.post("/generate", json_data=tts_request)

            if result.success and result.data.get("audio_data"):
                # Convert hex back to bytes
                return bytes.fromhex(result.data["audio_data"])
            else:
                self.logger.warning(f"TTS generation failed or returned no audio: {result.error}")
                return None

        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return None

    async def _transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio using STT service"""
        try:
            # Prepare STT request
            stt_request = {
                "audio_data": audio_data.hex(),  # Convert to hex for JSON
                "language": "auto",
                "model": "whisper-large-v3"
            }

            # Call STT service
            client = await service_registry.get_service("stt")
            result = await client.post("/transcribe", json_data=stt_request)

            if result.success:
                return result.data
            else:
                raise Exception(f"STT service error: {result.error}")

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            raise

    async def _apply_vad(self, audio_data: bytes) -> Dict[str, Any]:
        """Apply Voice Activity Detection"""
        try:
            # Prepare VAD request
            vad_request = {
                "audio_data": audio_data.hex(),  # Convert to hex for JSON
                "threshold": 0.5,
                "sample_rate": 16000
            }

            # Call VAD service
            client = await service_registry.get_service("vad")
            result = await client.post("/detect", json_data=vad_request)

            if result.success:
                return result.data
            else:
                # VAD is optional, so don't fail if it doesn't work
                self.logger.warning(f"VAD service error: {result.error}")
                return {"speech_detected": True, "confidence": 1.0}

        except Exception as e:
            # VAD is optional, so don't fail if it doesn't work
            self.logger.warning(f"VAD processing failed: {e}")
            return {"speech_detected": True, "confidence": 1.0}

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the orchestrator"""
        try:
            # Check service connectivity
            service_health = await service_registry.health_check_all()

            # Check component health
            conversation_health = self.conversation_manager.get_status()
            integration_health = await self.integration_manager.health_check()

            overall_status = "healthy" if all(service_health.values()) else "degraded"

            return {
                "status": overall_status,
                "services": service_health,
                "conversations": conversation_health,
                "integrations": integration_health
            }

        except Exception as e:
            self.logger.error(f"Orchestrator health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
