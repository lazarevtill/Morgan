"""
Real-time Streaming Orchestrator for Morgan AI Assistant
Optimized for low-latency STT → LLM → TTS pipeline with Redis caching and PostgreSQL persistence
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator
from uuid import UUID, uuid4

from shared.config.base import ServiceConfig
from shared.models.base import (
    Response,
    Message,
    ConversationContext,
    AudioChunk,
    Command,
)
from shared.models.database import (
    ConversationModel,
    MessageModel,
    StreamingSessionModel,
    AudioTranscriptionModel,
    TTSGenerationModel,
)
from shared.utils.logging import setup_logging, Timer
from shared.utils.exceptions import MorganException, ErrorCategory
from shared.utils.http_client import service_registry
from shared.utils.redis_client import RedisClient
from shared.utils.database import DatabaseClient

logger = logging.getLogger(__name__)


class StreamingOrchestrator:
    """
    Optimized streaming orchestrator for real-time voice interactions

    Pipeline: Audio → STT (streaming) → LLM (streaming) → TTS (streaming) → Audio
    Features:
    - Real-time streaming with minimal latency
    - Redis for session state and caching
    - PostgreSQL for conversation persistence
    - Proper error handling and recovery
    """

    def __init__(
        self,
        config: ServiceConfig,
        conversation_manager,
        handler_registry,
        integration_manager,
        memory_manager=None,
        tools_manager=None,
        redis_client: Optional[RedisClient] = None,
        db_client: Optional[DatabaseClient] = None,
    ):
        self.config = config
        self.conversation_manager = conversation_manager
        self.handler_registry = handler_registry
        self.integration_manager = integration_manager
        self.memory_manager = memory_manager
        self.tools_manager = tools_manager
        self.redis = redis_client
        self.db = db_client

        self.logger = setup_logging(
            "streaming_orchestrator", "INFO", "logs/orchestrator.log"
        )

        # Streaming configuration
        self.stream_enabled = config.get("streaming_enabled", True)
        self.real_time_mode = config.get("real_time_processing", True)
        self.stream_timeout = config.get("stream_timeout", 60)

        # Active streaming sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        self.logger.info("Streaming orchestrator initialized")

    async def start(self):
        """Start the streaming orchestrator"""
        try:
            # Verify all services are available
            await self._verify_services()
            self.logger.info("Streaming orchestrator started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start streaming orchestrator: {e}")
            raise

    async def stop(self):
        """Stop the streaming orchestrator"""
        self.logger.info("Streaming orchestrator stopping...")

        # Close all active streaming sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_streaming_session(session_id)

        self.logger.info("Streaming orchestrator stopped")

    async def _verify_services(self):
        """Verify all required services are available"""
        services_to_check = ["llm", "tts", "stt"]

        for service_name in services_to_check:
            try:
                client = service_registry.clients.get(service_name)
                if client:
                    is_healthy = await client.health_check()
                    if not is_healthy:
                        self.logger.warning(
                            f"Service {service_name} health check failed"
                        )
                    else:
                        self.logger.info(f"Service {service_name} is healthy")
                else:
                    self.logger.warning(f"Service client for {service_name} not found")
            except Exception as e:
                self.logger.error(f"Failed to verify service {service_name}: {e}")

    # === Streaming Session Management ===

    async def start_streaming_session(
        self,
        user_id: str,
        session_type: str = "websocket",
        language: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a new streaming session"""
        try:
            session_id = str(uuid4())

            # Get or create conversation
            conversation = await self.conversation_manager.get_or_create_context(
                user_id
            )

            # Create session data
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "conversation_id": conversation.conversation_id,
                "status": "active",
                "session_type": session_type,
                "language": language,
                "created_at": time.time(),
                "metadata": metadata or {},
                "audio_buffer": [],
                "transcription_buffer": "",
                "last_activity": time.time(),
            }

            # Store session in memory
            self.active_sessions[session_id] = session_data

            # Cache session in Redis if available
            if self.redis:
                await self.redis.set_session(session_id, session_data, expire=3600)
                self.logger.debug(f"Session {session_id} cached in Redis")

            # Persist session to database if available
            if self.db and conversation.id:
                db_session = StreamingSessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    conversation_id=conversation.id,
                    status="active",
                    session_type=session_type,
                    metadata=session_data["metadata"],
                )
                await self.db.create_streaming_session(db_session)
                self.logger.debug(f"Session {session_id} persisted to database")

            self.logger.info(
                f"Started streaming session: {session_id} for user: {user_id}"
            )

            return {
                "session_id": session_id,
                "status": "active",
                "conversation_id": conversation.conversation_id,
                "language": language,
            }

        except Exception as e:
            self.logger.error(f"Failed to start streaming session: {e}")
            raise

    async def end_streaming_session(self, session_id: str) -> Dict[str, Any]:
        """End a streaming session"""
        try:
            # Get session from memory or Redis
            session = self.active_sessions.get(session_id)
            if not session and self.redis:
                session = await self.redis.get_session(session_id)

            if not session:
                return {"status": "error", "message": "Session not found"}

            # Process any remaining buffered data
            transcription = session.get("transcription_buffer", "")

            # Update session status
            session["status"] = "ended"
            session["ended_at"] = time.time()

            # Update in database if available
            if self.db:
                await self.db.update_streaming_session(
                    session_id, status="ended", ended_at=session["ended_at"]
                )

            # Clear from Redis and memory
            if self.redis:
                await self.redis.delete_session(session_id)

            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            self.logger.info(f"Ended streaming session: {session_id}")

            return {
                "status": "ended",
                "session_id": session_id,
                "transcription": transcription,
                "duration": session["ended_at"] - session["created_at"],
            }

        except Exception as e:
            self.logger.error(f"Failed to end streaming session: {e}")
            return {"status": "error", "message": str(e)}

    # === Real-time Streaming Pipeline ===

    async def process_audio_stream(
        self, session_id: str, audio_chunk: bytes
    ) -> Dict[str, Any]:
        """
        Process audio chunk in real-time streaming mode

        Pipeline: Audio → STT (VAD + transcription) → intermediate results
        """
        try:
            # Get session
            session = self.active_sessions.get(session_id)
            if not session:
                if self.redis:
                    session = await self.redis.get_session(session_id)
                    if session:
                        self.active_sessions[session_id] = session

            if not session:
                return {"status": "error", "message": "Session not found"}

            # Update last activity
            session["last_activity"] = time.time()

            # Buffer audio chunk in Redis for potential replay
            if self.redis:
                import base64

                chunk_data = base64.b64encode(audio_chunk).decode("utf-8")
                await self.redis.buffer_audio_chunk(session_id, chunk_data)

            # Transcribe chunk using STT service (with VAD)
            stt_result = await self._transcribe_chunk_realtime(
                audio_chunk, session.get("language", "auto")
            )

            # Handle transcription result
            if stt_result and stt_result.get("text"):
                transcription = stt_result["text"].strip()
                confidence = stt_result.get("confidence", 0.0)

                # Append to session buffer
                session["transcription_buffer"] += " " + transcription

                # Update session in Redis
                if self.redis:
                    await self.redis.update_session(
                        session_id,
                        transcription_buffer=session["transcription_buffer"],
                        last_activity=session["last_activity"],
                    )

                return {
                    "status": "success",
                    "text": transcription,
                    "confidence": confidence,
                    "is_final": False,
                    "vad_result": stt_result.get("vad_result", "unknown"),
                }

            # No speech detected
            return {
                "status": "no_speech",
                "vad_result": (
                    stt_result.get("vad_result", "no_speech")
                    if stt_result
                    else "no_speech"
                ),
            }

        except Exception as e:
            self.logger.error(f"Error processing audio stream: {e}")
            return {"status": "error", "message": str(e)}

    async def process_complete_utterance(
        self, session_id: str, text: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process complete utterance through LLM → TTS streaming pipeline

        Pipeline: Text → LLM (streaming) → TTS (streaming) → Audio chunks
        Yields: Stream of response chunks (text + audio)
        """
        try:
            # Get session
            session = self.active_sessions.get(session_id)
            if not session and self.redis:
                session = await self.redis.get_session(session_id)

            if not session:
                yield {"status": "error", "message": "Session not found"}
                return

            user_id = session["user_id"]
            conversation_id = session["conversation_id"]

            # Get conversation context
            context = await self.conversation_manager.get_or_create_context(user_id)

            # Add user message to context
            user_message = Message(role="user", content=text)
            context.add_message(user_message)

            # Cache in Redis
            if self.redis:
                messages_data = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    }
                    for msg in context.messages
                ]
                await self.redis.cache_conversation_messages(
                    conversation_id, messages_data
                )

            # Persist to database
            if self.db and context.id:
                db_message = MessageModel(
                    conversation_id=context.id,
                    role="user",
                    content=text,
                    sequence_number=len(context.messages),
                )
                await self.db.add_message(db_message)

            # Get LLM streaming response
            llm_chunks = []
            async for chunk_data in self._generate_llm_stream(context):
                llm_chunks.append(chunk_data["text"])

                # Yield intermediate text chunks
                yield {"type": "text", "text": chunk_data["text"], "is_final": False}

            # Combine LLM response
            full_response = "".join(llm_chunks)

            # Add assistant message to context
            assistant_message = Message(role="assistant", content=full_response)
            context.add_message(assistant_message)

            # Persist assistant message
            if self.db and context.id:
                db_message = MessageModel(
                    conversation_id=context.id,
                    role="assistant",
                    content=full_response,
                    sequence_number=len(context.messages),
                )
                await self.db.add_message(db_message)

            # Update conversation in Redis
            if self.redis:
                messages_data = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    }
                    for msg in context.messages
                ]
                await self.redis.cache_conversation_messages(
                    conversation_id, messages_data
                )

            # Generate TTS streaming audio
            async for audio_chunk in self._generate_tts_stream(full_response):
                yield {"type": "audio", "audio_data": audio_chunk, "is_final": False}

            # Send final marker
            yield {"type": "complete", "text": full_response, "is_final": True}

        except Exception as e:
            self.logger.error(f"Error in complete utterance processing: {e}")
            yield {"type": "error", "message": str(e)}

    # === Internal Service Communication ===

    async def _transcribe_chunk_realtime(
        self, audio_data: bytes, language: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """Transcribe audio chunk with VAD in real-time"""
        try:
            client = await service_registry.get_service("stt")

            # Use real-time endpoint
            result = await client.post(
                "/transcribe/realtime",
                json_data={"audio_data": audio_data.hex(), "language": language},
            )

            if result.success:
                return (
                    result.data
                    if isinstance(result.data, dict)
                    else result.data.__dict__
                )
            else:
                self.logger.warning(f"STT transcription failed: {result.error}")
                return None

        except Exception as e:
            self.logger.error(f"Error in real-time transcription: {e}")
            return None

    async def _generate_llm_stream(
        self, context: ConversationContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming LLM response"""
        try:
            client = await service_registry.get_service("llm")

            # Prepare request
            recent_messages = context.get_last_n_messages(10)
            messages_data = [
                {"role": msg.role, "content": msg.content} for msg in recent_messages
            ]

            # Call LLM streaming endpoint
            result = await client.post(
                "/generate/stream",
                json_data={"messages": messages_data, "stream": True},
            )

            if result.success:
                # Handle streaming response
                data = result.data
                if hasattr(data, "__aiter__"):
                    # It's an async generator
                    async for chunk in data:
                        yield {"text": chunk, "delta": chunk}
                elif isinstance(data, dict) and "text" in data:
                    # Single response
                    yield {"text": data["text"], "delta": data["text"]}
                else:
                    # Fallback
                    yield {"text": str(data), "delta": str(data)}
            else:
                self.logger.error(f"LLM generation failed: {result.error}")
                yield {
                    "text": "I encountered an error processing your request.",
                    "delta": "",
                }

        except Exception as e:
            self.logger.error(f"Error in LLM streaming: {e}")
            yield {
                "text": "I encountered an error processing your request.",
                "delta": "",
            }

    async def _generate_tts_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate streaming TTS audio"""
        try:
            client = await service_registry.get_service("tts")

            # Call TTS streaming endpoint
            result = await client.post(
                "/generate/stream",
                json_data={"text": text, "voice": "default", "stream": True},
            )

            if result.success:
                data = result.data
                if hasattr(data, "__aiter__"):
                    # It's an async generator
                    async for chunk in data:
                        if isinstance(chunk, bytes):
                            yield chunk
                        elif isinstance(chunk, dict) and "audio_data" in chunk:
                            # Decode base64 if needed
                            import base64

                            audio_bytes = base64.b64decode(chunk["audio_data"])
                            yield audio_bytes
                elif isinstance(data, dict) and "audio_data" in data:
                    # Single response
                    import base64

                    audio_bytes = base64.b64decode(data["audio_data"])
                    yield audio_bytes
                elif isinstance(data, bytes):
                    yield data
            else:
                self.logger.error(f"TTS generation failed: {result.error}")

        except Exception as e:
            self.logger.error(f"Error in TTS streaming: {e}")

    # === Legacy Non-Streaming Methods (for backward compatibility) ===

    async def process_request(
        self, context: ConversationContext, metadata: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Process a text request through the service pipeline"""
        with Timer(self.logger, f"Request processing for user {context.user_id}"):
            try:
                # Get the last message text
                recent_messages = context.get_last_n_messages(1)
                if not recent_messages:
                    return Response(
                        text="No message to process.", metadata={"error": True}
                    )

                user_text = recent_messages[0].content

                # Step 1: Check if this is a "remember" command
                if (
                    self.handler_registry.remember_handler
                    and self.handler_registry.remember_handler.can_handle(user_text)
                ):
                    self.logger.info(
                        f"Processing remember command: {user_text[:50]}..."
                    )
                    result = await self.handler_registry.remember_handler.handle(
                        user_text, context.user_id, metadata=metadata
                    )

                    response_text = result.get(
                        "response", "I've stored that information."
                    )

                    # Generate audio if requested
                    audio_data = None
                    if metadata and metadata.get("generate_audio", False):
                        audio_data = await self._generate_tts_response(response_text)

                    return Response(
                        text=response_text,
                        audio_data=audio_data,
                        metadata=result.get("metadata", {}),
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
                            min_importance=min_importance,
                        )
                        relevant_memories = [
                            {
                                "content": mem.content,
                                "category": mem.category,
                                "importance": mem.importance,
                            }
                            for mem in memories
                        ]
                        if relevant_memories:
                            self.logger.info(
                                f"Retrieved {len(relevant_memories)} relevant memories"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Error retrieving memories: {e}", exc_info=True
                        )

                # Step 3: Generate LLM response with memory context and tools
                llm_response = await self._generate_llm_response(
                    context, relevant_memories
                )

                # Step 4: Execute any commands/actions
                executed_commands = []
                if isinstance(llm_response, dict) and llm_response.get("actions"):
                    for action in llm_response["actions"]:
                        command = Command(
                            action=action["action"],
                            parameters=action.get("parameters", {}),
                        )
                        result = await self.handler_registry.process_command(
                            command, context
                        )
                        executed_commands.append(result)

                # Step 5: Generate TTS response if requested
                audio_data = None
                if metadata and metadata.get("generate_audio", False):
                    text = (
                        llm_response.get("text", "")
                        if isinstance(llm_response, dict)
                        else getattr(llm_response, "text", "")
                    )
                    audio_data = await self._generate_tts_response(text)

                # Step 6: Create final response
                text = (
                    llm_response.get("text", "")
                    if isinstance(llm_response, dict)
                    else getattr(llm_response, "text", "")
                )
                return Response(
                    text=text,
                    audio_data=audio_data,
                    actions=executed_commands,
                    metadata={
                        "llm_model": (
                            llm_response.get("model")
                            if isinstance(llm_response, dict)
                            else getattr(llm_response, "model", None)
                        ),
                        "llm_usage": (
                            llm_response.get("usage")
                            if isinstance(llm_response, dict)
                            else getattr(llm_response, "usage", None)
                        ),
                        "processing_time": (
                            metadata.get("processing_time") if metadata else None
                        ),
                        "memories_used": len(relevant_memories),
                    },
                )

            except Exception as e:
                self.logger.error(f"Error in request processing: {e}", exc_info=True)
                return Response(
                    text="I'm sorry, I encountered an error while processing your request. Please try again.",
                    metadata={"error": True, "error_message": str(e)},
                )

    async def process_audio_request(
        self,
        audio_data: bytes,
        context: ConversationContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Process an audio request through the service pipeline"""
        with Timer(self.logger, f"Audio request processing for user {context.user_id}"):
            try:
                # Transcribe audio to text (STT service has VAD integrated)
                transcription = await self._transcribe_audio(audio_data)
                transcribed_text = (
                    transcription.get("text", "")
                    if isinstance(transcription, dict)
                    else getattr(transcription, "text", "")
                )

                if not transcribed_text:
                    return Response(
                        text="I couldn't understand the audio. Please try speaking more clearly.",
                        metadata={"transcription": transcription},
                    )

                # Add transcription to context
                transcription_message = Message(
                    role="user",
                    content=transcribed_text,
                    metadata={
                        "transcribed": True,
                        "original_audio_length": len(audio_data),
                    },
                )
                context.add_message(transcription_message)

                # Step 3: Process as text request
                return await self.process_request(context, metadata)

            except Exception as e:
                self.logger.error(f"Error in audio request processing: {e}")
                return Response(
                    text="I'm sorry, I encountered an error while processing the audio. Please try again.",
                    metadata={"error": True, "error_message": str(e)},
                )

    async def transcribe_chunk(
        self, audio_data: bytes, language: str = "auto"
    ) -> Dict[str, Any]:
        """Transcribe a single audio chunk for real-time streaming"""
        try:
            # Prepare STT request for real-time endpoint (with VAD)
            stt_request = {
                "audio_data": audio_data.hex(),  # Convert to hex for JSON
                "language": language,
            }

            # Call STT service real-time endpoint for better VAD integration
            client = await service_registry.get_service("stt")
            result = await client.post("/transcribe/realtime", json_data=stt_request)

            if result.success:
                # Handle response data - could be dict or ProcessingResult
                response_data = result.data
                if hasattr(response_data, "get"):
                    return response_data
                else:
                    # Convert ProcessingResult or other object to dict
                    if hasattr(response_data, "__dict__"):
                        return response_data.__dict__
                    else:
                        # Fallback response
                        return {
                            "text": "",
                            "confidence": 0.0,
                            "error": "Invalid response format",
                        }
            else:
                # Fallback to chunk endpoint if real-time fails
                result = await client.post("/transcribe/chunk", json_data=stt_request)
                if result.success:
                    response_data = result.data
                    if hasattr(response_data, "get"):
                        return response_data
                    else:
                        if hasattr(response_data, "__dict__"):
                            return response_data.__dict__
                        else:
                            return {
                                "text": "",
                                "confidence": 0.0,
                                "error": "Invalid response format",
                            }
                else:
                    raise Exception(f"STT service error: {result.error}")

        except Exception as e:
            self.logger.error(f"Audio chunk transcription failed: {e}")
            raise

    async def _generate_llm_response(
        self, context: ConversationContext, memories: list = None
    ) -> Dict[str, Any]:
        """Generate response from LLM service with memory context and tools"""
        try:
            # Get recent conversation context
            recent_messages = context.get_last_n_messages(10)

            # Build system prompt with memories and tools
            system_prompt_parts = [
                "You are Morgan, a helpful AI assistant. Respond naturally and helpfully."
            ]

            # Add memory context if available
            if memories:
                memory_context = "\n\nRelevant information I remember about you:\n"
                for mem in memories:
                    memory_context += f"- {mem['content']}"
                    if mem.get("category"):
                        memory_context += f" (Category: {mem['category']})"
                    memory_context += "\n"
                system_prompt_parts.append(memory_context)

            # Add available tools if tools manager exists
            if self.tools_manager:
                try:
                    tools_description = (
                        self.tools_manager.get_tool_descriptions_for_llm()
                    )
                    if tools_description:
                        system_prompt_parts.append(
                            "\n\nAvailable tools you can use:\n" + tools_description
                        )
                        system_prompt_parts.append(
                            "\nTo use a tool, respond with: USE_TOOL: tool_name with parameters: {param: value}"
                        )
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
                "max_tokens": 1000,
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
                            response_text, context.user_id, context.conversation_id
                        )
                    return response_data
                else:
                    # Fallback: convert to dict if it's an object
                    if hasattr(response_data, "__dict__"):
                        return response_data.__dict__
                    else:
                        # Last resort: create a basic response
                        return {"text": str(response_data), "model": "unknown"}
            else:
                raise Exception(f"LLM service error: {result.error}")

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise

    async def _execute_tool_from_llm_response(
        self, response_text: str, user_id: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Parse and execute tool calls from LLM response"""
        import re
        import json

        try:
            # Parse tool call from response
            # Format: USE_TOOL: tool_name with parameters: {param: value}
            match = re.search(
                r"USE_TOOL:\s*(\w+)\s+with parameters:\s*(\{.*\})",
                response_text,
                re.IGNORECASE,
            )
            if not match:
                return {"text": response_text, "error": "Could not parse tool call"}

            tool_name = match.group(1)
            params_str = match.group(2)

            try:
                parameters = json.loads(params_str)
            except json.JSONDecodeError:
                return {
                    "text": response_text,
                    "error": "Invalid tool parameters format",
                }

            # Execute tool
            self.logger.info(f"Executing tool: {tool_name} with params: {parameters}")
            result = await self.tools_manager.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            # Format result as response
            if result.get("status") == "success":
                tool_result = result.get("result", {})
                return {
                    "text": f"I used the {tool_name} tool. Result: {json.dumps(tool_result, indent=2)}",
                    "tool_execution": result,
                }
            else:
                return {
                    "text": f"I tried to use the {tool_name} tool but encountered an error: {result.get('error')}",
                    "tool_execution": result,
                }

        except Exception as e:
            self.logger.error(
                f"Error executing tool from LLM response: {e}", exc_info=True
            )
            return {"text": response_text, "error": str(e)}

    async def _generate_tts_response(self, text: str) -> Optional[bytes]:
        """Generate TTS audio response using CSM-streaming"""
        try:
            # Prepare TTS request for CSM-streaming
            tts_request = {
                "text": text,
                "voice": "default",  # CSM-streaming default voice
                "speed": 1.0,
                "output_format": "wav",
            }

            # Call TTS service
            client = await service_registry.get_service("tts")
            result = await client.post("/generate", json_data=tts_request)

            if result.success:
                # Handle response data - TTS service returns dict with base64 audio_data
                response_data = result.data
                if isinstance(response_data, dict) and response_data.get("audio_data"):
                    # Convert base64 back to bytes
                    import base64

                    return base64.b64decode(response_data["audio_data"])
                else:
                    self.logger.warning(
                        f"TTS generation failed or returned no audio: {result.error}"
                    )
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
                "model": "whisper-distil-large-v3.5 ",
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
                    if hasattr(response_data, "__dict__"):
                        return response_data.__dict__
                    else:
                        raise Exception("Invalid STT response format")
            else:
                raise Exception(f"STT service error: {result.error}")

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            raise

    # === Health Check ===

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the streaming orchestrator"""
        try:
            # Check service connectivity
            service_health = await service_registry.health_check_all()

            # Check component health
            conversation_health = self.conversation_manager.get_status()
            integration_health = await self.integration_manager.health_check()

            # Check Redis connection
            redis_healthy = False
            if self.redis:
                redis_healthy = await self.redis.health_check()

            # Check database connection
            db_healthy = False
            if self.db:
                db_healthy = await self.db.health_check()

            overall_status = "healthy" if all(service_health.values()) else "degraded"

            return {
                "status": overall_status,
                "services": service_health,
                "conversations": conversation_health,
                "integrations": integration_health,
                "redis": redis_healthy,
                "database": db_healthy,
                "active_sessions": len(self.active_sessions),
                "streaming_enabled": self.stream_enabled,
                "real_time_mode": self.real_time_mode,
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
