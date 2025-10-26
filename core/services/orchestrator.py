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

    def __init__(self, config: ServiceConfig, conversation_manager, handler_registry, integration_manager,
                 memory_manager=None, tools_manager=None):
        self.config = config
        self.conversation_manager = conversation_manager
        self.handler_registry = handler_registry
        self.integration_manager = integration_manager
        self.memory_manager = memory_manager
        self.tools_manager = tools_manager

        self.error_handler = ErrorHandler()
        self.logger = setup_logging("service_orchestrator", "INFO", "logs/orchestrator.log")

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
        services_to_check = ["llm", "tts", "stt"]

        for service_name in services_to_check:
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
                # Get the last message text
                recent_messages = context.get_last_n_messages(1)
                if not recent_messages:
                    return Response(
                        text="No message to process.",
                        metadata={"error": True}
                    )

                user_text = recent_messages[0].content

                # Step 1: Check if this is a "remember" command
                if self.handler_registry.remember_handler and self.handler_registry.remember_handler.can_handle(user_text):
                    self.logger.info(f"Processing remember command: {user_text[:50]}...")
                    result = await self.handler_registry.remember_handler.handle(
                        user_text,
                        context.user_id,
                        metadata=metadata
                    )

                    response_text = result.get("response", "I've stored that information.")

                    # Generate audio if requested
                    audio_data = None
                    if metadata and metadata.get("generate_audio", False):
                        audio_data = await self._generate_tts_response(response_text)

                    return Response(
                        text=response_text,
                        audio_data=audio_data,
                        metadata=result.get("metadata", {})
                    )

                # Step 2: Retrieve relevant memories
                relevant_memories = []
                if self.memory_manager:
                    try:
                        # Get memory search parameters from config
                        memory_limit = self.config.get("memory_search_limit", 5)
                        min_importance = self.config.get("memory_min_importance", 3)

                        memories = await self.memory_manager.search_memories(
                            user_id=context.user_id,
                            query=user_text,
                            limit=memory_limit,
                            min_importance=min_importance
                        )
                        relevant_memories = [
                            {"content": mem.content, "category": mem.category, "importance": mem.importance}
                            for mem in memories
                        ]
                        if relevant_memories:
                            self.logger.info(f"Retrieved {len(relevant_memories)} relevant memories")
                    except Exception as e:
                        self.logger.error(f"Error retrieving memories: {e}", exc_info=True)

                # Step 3: Generate LLM response with memory context and tools
                llm_response = await self._generate_llm_response(context, relevant_memories)

                # Step 4: Execute any commands/actions
                executed_commands = []
                if isinstance(llm_response, dict) and llm_response.get("actions"):
                    for action in llm_response["actions"]:
                        command = Command(action=action["action"], parameters=action.get("parameters", {}))
                        result = await self.handler_registry.process_command(command, context)
                        executed_commands.append(result)

                # Step 5: Generate TTS response if requested
                audio_data = None
                if metadata and metadata.get("generate_audio", False):
                    text = llm_response.get("text", "") if isinstance(llm_response, dict) else getattr(llm_response, "text", "")
                    audio_data = await self._generate_tts_response(text)

                # Step 6: Create final response
                text = llm_response.get("text", "") if isinstance(llm_response, dict) else getattr(llm_response, "text", "")
                return Response(
                    text=text,
                    audio_data=audio_data,
                    actions=executed_commands,
                    metadata={
                        "llm_model": llm_response.get("model") if isinstance(llm_response, dict) else getattr(llm_response, "model", None),
                        "llm_usage": llm_response.get("usage") if isinstance(llm_response, dict) else getattr(llm_response, "usage", None),
                        "processing_time": metadata.get("processing_time") if metadata else None,
                        "memories_used": len(relevant_memories)
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in request processing: {e}", exc_info=True)
                return Response(
                    text="I'm sorry, I encountered an error while processing your request. Please try again.",
                    metadata={"error": True, "error_message": str(e)}
                )

    async def process_audio_request(self, audio_data: bytes, context: ConversationContext,
                                  metadata: Optional[Dict[str, Any]] = None) -> Response:
        """Process an audio request through the service pipeline"""
        with Timer(self.logger, f"Audio request processing for user {context.user_id}"):
            try:
                # Transcribe audio to text (STT service has VAD integrated)
                transcription = await self._transcribe_audio(audio_data)
                transcribed_text = transcription.get("text", "") if isinstance(transcription, dict) else getattr(transcription, "text", "")

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

    async def _generate_llm_response(self, context: ConversationContext, memories: list = None) -> Dict[str, Any]:
        """Generate response from LLM service with memory context and tools"""
        try:
            # Get recent conversation context
            recent_messages = context.get_last_n_messages(10)

            # Build system prompt with memories and tools
            system_prompt_parts = ["You are Morgan, a helpful AI assistant. Respond naturally and helpfully."]

            # Add memory context if available
            if memories:
                memory_context = "\n\nRelevant information I remember about you:\n"
                for mem in memories:
                    memory_context += f"- {mem['content']}"
                    if mem.get('category'):
                        memory_context += f" (Category: {mem['category']})"
                    memory_context += "\n"
                system_prompt_parts.append(memory_context)

            # Add available tools if tools manager exists
            if self.tools_manager:
                try:
                    tools_description = self.tools_manager.get_tool_descriptions_for_llm()
                    if tools_description:
                        system_prompt_parts.append("\n\nAvailable tools you can use:\n" + tools_description)
                        system_prompt_parts.append("\nTo use a tool, respond with: USE_TOOL: tool_name with parameters: {param: value}")
                except Exception as e:
                    self.logger.error(f"Error getting tool descriptions: {e}")

            system_prompt = "\n".join(system_prompt_parts)

            # Prepare LLM request
            llm_request = {
                "prompt": recent_messages[-1].content if recent_messages else "",
                "context": [
                    {"role": msg.role, "content": msg.content}
                    for msg in recent_messages[:-1]
                ],
                "system_prompt": system_prompt,
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Call LLM service
            client = await service_registry.get_service("llm")
            result = await client.post("/generate", json_data=llm_request)

            if result.success:
                # Handle response data - LLM service returns dict
                response_data = result.data
                if isinstance(response_data, dict):
                    # Check for tool usage in response
                    response_text = response_data.get("text", "")
                    if "USE_TOOL:" in response_text and self.tools_manager:
                        response_data = await self._execute_tool_from_llm_response(
                            response_text,
                            context.user_id,
                            context.conversation_id
                        )
                    return response_data
                else:
                    # Fallback: convert to dict if it's an object
                    if hasattr(response_data, '__dict__'):
                        return response_data.__dict__
                    else:
                        # Last resort: create a basic response
                        return {"text": str(response_data), "model": "unknown"}
            else:
                raise Exception(f"LLM service error: {result.error}")

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise

    async def _execute_tool_from_llm_response(self, response_text: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Parse and execute tool calls from LLM response"""
        import re
        import json

        try:
            # Parse tool call from response
            # Format: USE_TOOL: tool_name with parameters: {param: value}
            match = re.search(r'USE_TOOL:\s*(\w+)\s+with parameters:\s*(\{.*\})', response_text, re.IGNORECASE)
            if not match:
                return {"text": response_text, "error": "Could not parse tool call"}

            tool_name = match.group(1)
            params_str = match.group(2)

            try:
                parameters = json.loads(params_str)
            except json.JSONDecodeError:
                return {"text": response_text, "error": "Invalid tool parameters format"}

            # Execute tool
            self.logger.info(f"Executing tool: {tool_name} with params: {parameters}")
            result = await self.tools_manager.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                user_id=user_id,
                conversation_id=conversation_id
            )

            # Format result as response
            if result.get("status") == "success":
                tool_result = result.get("result", {})
                return {
                    "text": f"I used the {tool_name} tool. Result: {json.dumps(tool_result, indent=2)}",
                    "tool_execution": result
                }
            else:
                return {
                    "text": f"I tried to use the {tool_name} tool but encountered an error: {result.get('error')}",
                    "tool_execution": result
                }

        except Exception as e:
            self.logger.error(f"Error executing tool from LLM response: {e}", exc_info=True)
            return {"text": response_text, "error": str(e)}

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

            if result.success:
                # Handle response data - TTS service returns dict with hex audio_data
                response_data = result.data
                if isinstance(response_data, dict) and response_data.get("audio_data"):
                    # Convert hex back to bytes
                    return bytes.fromhex(response_data["audio_data"])
                else:
                    self.logger.warning(f"TTS generation failed or returned no audio: {result.error}")
                    return None
            else:
                self.logger.warning(f"TTS generation failed: {result.error}")
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
                # Handle response data - STT service returns dict
                response_data = result.data
                if isinstance(response_data, dict):
                    return response_data
                else:
                    # Fallback: convert to dict if it's an object
                    if hasattr(response_data, '__dict__'):
                        return response_data.__dict__
                    else:
                        raise Exception("Invalid STT response format")
            else:
                raise Exception(f"STT service error: {result.error}")

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            raise

    async def transcribe_chunk(self, audio_data: bytes, language: str = "auto") -> Dict[str, Any]:
        """Transcribe a single audio chunk for real-time streaming"""
        try:
            # Prepare STT request for real-time endpoint (with VAD)
            stt_request = {
                "audio_data": audio_data.hex(),  # Convert to hex for JSON
                "language": language
            }

            # Call STT service real-time endpoint for better VAD integration
            client = await service_registry.get_service("stt")
            result = await client.post("/transcribe/realtime", json_data=stt_request)

            if result.success:
                # Handle response data - could be dict or ProcessingResult
                response_data = result.data
                if hasattr(response_data, 'get'):
                    return response_data
                else:
                    # Convert ProcessingResult or other object to dict
                    if hasattr(response_data, '__dict__'):
                        return response_data.__dict__
                    else:
                        # Fallback response
                        return {"text": "", "confidence": 0.0, "error": "Invalid response format"}
            else:
                # Fallback to chunk endpoint if real-time fails
                result = await client.post("/transcribe/chunk", json_data=stt_request)
                if result.success:
                    response_data = result.data
                    if hasattr(response_data, 'get'):
                        return response_data
                    else:
                        if hasattr(response_data, '__dict__'):
                            return response_data.__dict__
                        else:
                            return {"text": "", "confidence": 0.0, "error": "Invalid response format"}
                else:
                    raise Exception(f"STT service error: {result.error}")

        except Exception as e:
            self.logger.error(f"Audio chunk transcription failed: {e}")
            raise

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
