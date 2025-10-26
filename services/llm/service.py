"""
LLM Service implementation using Ollama with OpenAI compatibility
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from shared.config.base import ServiceConfig
from shared.models.base import LLMRequest, LLMResponse, Message, ProcessingResult
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, ModelError, ErrorCode

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """LLM service configuration"""
    host: str = "0.0.0.0"
    port: int = 8001
    model: str = "llama3.2:latest"
    openai_api_base: str = "https://gpt.lazarev.cloud/ollama/v1"
    api_key: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0
    gpu_layers: int = -1
    context_window: int = 4096
    system_prompt: str = "You are Morgan, a helpful AI assistant."
    log_level: str = "INFO"
    embedding_model: str = "qwen3-embedding:latest"
    max_context_messages: int = 10
    enable_streaming: bool = True


class LLMService:
    """LLM Service using OpenAI-compatible API"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("llm")
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "llm_service",
            self.config.get("log_level", "INFO"),
            "logs/llm_service.log"
        )

        # Load configuration
        self.llm_config = LLMConfig(**self.config.all())
        self.openai_client = None

        # Model and conversation management
        self.current_model = None
        self.conversation_cache = {}

        self.logger.info(f"LLM Service initialized with model: {self.llm_config.model}")
        self.logger.info(f"Using OpenAI-compatible API at: {self.llm_config.openai_api_base}")

    async def start(self):
        """Start the LLM service"""
        try:
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(
                base_url=self.llm_config.openai_api_base,
                api_key=self.llm_config.api_key
            )

            # Test connection to OpenAI-compatible API only if API key is provided
            if self.llm_config.api_key:
                await self._test_api_connection()
                self.current_model = self.llm_config.model
                self.logger.info("LLM Service started successfully")
            else:
                self.logger.warning("No API key provided - LLM service started in offline mode")
                self.current_model = self.llm_config.model

        except Exception as e:
            self.logger.error(f"Failed to start LLM service: {e}")
            raise

    async def stop(self):
        """Stop the LLM service"""
        self.logger.info("LLM Service stopping...")

        # Close clients
        if self.openai_client:
            await self.openai_client.close()

        self.logger.info("LLM Service stopped")

    async def _test_api_connection(self):
        """Test connection to OpenAI-compatible API service"""
        if not self.llm_config.api_key:
            self.logger.warning("No API key provided - skipping connection test")
            return

        try:
            # Test basic connectivity by listing models
            response = await self.openai_client.models.list()
            available_models = [model.id for model in response.data]

            self.logger.info(f"Available models from external API: {available_models}")

            if self.llm_config.model not in available_models:
                self.logger.warning(f"Configured model {self.llm_config.model} not found. Available models: {available_models}")
                if available_models:
                    # Use the first available model as fallback
                    self.llm_config.model = available_models[0]
                    self.current_model = self.llm_config.model
                    self.logger.info(f"Using fallback model: {self.llm_config.model}")
                else:
                    raise ModelError("No models available in OpenAI-compatible API service", ErrorCode.MODEL_LOAD_ERROR)
            else:
                self.current_model = self.llm_config.model
                self.logger.info(f"Model {self.llm_config.model} is available")

        except Exception as e:
            self.logger.error(f"Error testing OpenAI-compatible API connection: {e}")
            raise ModelError(f"Failed to connect to OpenAI-compatible API: {e}", ErrorCode.MODEL_LOAD_ERROR)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text response using LLM"""
        with Timer(self.logger, f"LLM generation for prompt length {len(request.prompt)}"):
            try:
                # Check if API key is available
                if not self.llm_config.api_key:
                    return LLMResponse(
                        text="I'm currently offline and cannot process your request. Please provide an API key to enable LLM functionality.",
                        model=self.llm_config.model,
                        metadata={"offline": True, "error": "No API key provided"}
                    )

                # Prepare messages for OpenAI format
                messages = []

                # Add system prompt
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                else:
                    messages.append({"role": "system", "content": self.llm_config.system_prompt})

                # Add conversation context if provided
                if request.context:
                    # Use configured max context messages or default to 10
                    max_context = getattr(self.llm_config, 'max_context_messages', 10)
                    for msg in request.context[-max_context:]:  # Keep last N messages for context
                        messages.append({
                            "role": msg.role,
                            "content": msg.content
                        })

                # Add current prompt
                messages.append({"role": "user", "content": request.prompt})

                # Generate response
                response = await self.openai_client.chat.completions.create(
                    model=request.model or self.llm_config.model,
                    messages=messages,
                    max_tokens=request.max_tokens or self.llm_config.max_tokens,
                    temperature=request.temperature or self.llm_config.temperature,
                    stream=False
                )

                generated_text = response.choices[0].message.content

                # Extract usage information
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                return LLMResponse(
                    text=generated_text,
                    model=request.model or self.llm_config.model,
                    usage=usage,
                    finish_reason=response.choices[0].finish_reason,
                    metadata={"service": "ollama", "provider": "openai_compatible"}
                )

            except Exception as e:
                self.logger.error(f"Error in text generation: {e}")
                raise ModelError(f"Text generation failed: {e}", ErrorCode.MODEL_INFERENCE_ERROR)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming text response"""
        try:
            # Check if API key is available
            if not self.llm_config.api_key:
                yield "I'm currently offline and cannot process your request. Please provide an API key to enable LLM functionality."
                return

            # Prepare messages
            messages = []

            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            else:
                messages.append({"role": "system", "content": self.llm_config.system_prompt})

            if request.context:
                # Use configured max context messages or default to 10
                max_context = getattr(self.llm_config, 'max_context_messages', 10)
                for msg in request.context[-max_context:]:
                    messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": request.prompt})

            # Generate streaming response
            stream = await self.openai_client.chat.completions.create(
                model=request.model or self.llm_config.model,
                messages=messages,
                max_tokens=request.max_tokens or self.llm_config.max_tokens,
                temperature=request.temperature or self.llm_config.temperature,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Error in streaming generation: {e}")
            raise ModelError(f"Streaming generation failed: {e}", ErrorCode.MODEL_INFERENCE_ERROR)

    async def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for text using OpenAI-compatible API"""
        try:
            # Check if API key is available
            if not self.llm_config.api_key:
                # Return a dummy embedding vector when offline
                import hashlib
                hash_obj = hashlib.sha256(text.encode())
                hash_bytes = hash_obj.digest()
                # Create a 384-dimensional embedding from hash (common embedding size)
                embedding = []
                for i in range(384):
                    embedding.append((hash_bytes[i % len(hash_bytes)] / 255.0 - 0.5) * 2)
                return embedding

            # Use configured embedding model or fallback to specified model
            embedding_model = model or self.llm_config.embedding_model

            # Create a separate client for embeddings to avoid connection issues
            embedding_client = AsyncOpenAI(
                base_url=self.llm_config.openai_api_base,
                api_key=self.llm_config.api_key
            )

            response = await embedding_client.embeddings.create(
                input=text,
                model=embedding_model
            )

            # Close the embedding client to avoid connection leaks
            await embedding_client.close()

            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise ModelError(f"Embedding generation failed: {e}", ErrorCode.MODEL_INFERENCE_ERROR)

    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            if not self.llm_config.api_key:
                # Return offline model list
                return {
                    "models": [
                        {"name": self.llm_config.model, "size": 0, "modified": "offline"},
                        {"name": self.llm_config.embedding_model, "size": 0, "modified": "offline"}
                    ],
                    "total": 2,
                    "offline": True
                }

            response = await self.openai_client.models.list()
            models = [{"name": model.id, "size": 0, "modified": ""} for model in response.data]

            return {
                "models": models,
                "total": len(models)
            }
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise ModelError(f"Failed to list models: {e}", ErrorCode.MODEL_LOAD_ERROR)

    async def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            model_name = model or self.llm_config.model

            # Since OpenAI API doesn't provide detailed model info, return basic info
            return {
                "model": model_name,
                "modified_at": "unknown",
                "size": 0,
                "template": "unknown",
                "details": {
                    "family": "llama",
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                    "provider": "openai_compatible"
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            raise ModelError(f"Failed to get model info: {e}", ErrorCode.MODEL_LOAD_ERROR)

    async def pull_model(self, model: str) -> AsyncGenerator[str, None]:
        """Pull a model from OpenAI-compatible API"""
        # Since we're using an external OpenAI-compatible service, model management should be done there
        yield json.dumps({
            "status": "info",
            "message": f"Model management should be done on the external OpenAI-compatible API service at {self.llm_config.openai_api_base}"
        })
        yield json.dumps({
            "status": "info",
            "message": "Please manage models through the OpenAI-compatible API service"
        })

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            if not self.llm_config.api_key:
                return {
                    "status": "offline",
                    "model": self.current_model,
                    "available_models": 0,
                    "message": "No API key provided - running in offline mode",
                    "api_base": self.llm_config.openai_api_base
                }

            # Test basic connectivity by listing models
            response = await self.openai_client.models.list()
            available_models = len(response.data)

            return {
                "status": "healthy",
                "model": self.current_model,
                "available_models": available_models,
                "api_base": self.llm_config.openai_api_base
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_base": self.llm_config.openai_api_base
            }
