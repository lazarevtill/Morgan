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
    ollama_url: str = "http://192.168.101.3:11434"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0
    gpu_layers: int = -1
    context_window: int = 4096
    system_prompt: str = "You are Morgan, a helpful AI assistant."
    log_level: str = "INFO"


class LLMService:
    """LLM Service using Ollama with OpenAI compatibility"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("llm")
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "llm_service",
            self.config.get("log_level", "INFO"),
            f"logs/llm_service.log"
        )

        # Load configuration
        self.llm_config = LLMConfig(**self.config.all())
        self.openai_client = None

        # Model and conversation management
        self.current_model = None
        self.conversation_cache = {}

        self.logger.info(f"LLM Service initialized with model: {self.llm_config.model}")
        self.logger.info(f"Using external Ollama at: {self.llm_config.ollama_url}")

    async def start(self):
        """Start the LLM service"""
        try:
            # Initialize OpenAI client for Ollama
            self.openai_client = AsyncOpenAI(
                base_url=f"{self.llm_config.ollama_url}/v1",
                api_key="ollama"  # Ollama doesn't require a real API key
            )

            # Test connection to external Ollama
            await self._test_ollama_connection()

            self.current_model = self.llm_config.model
            self.logger.info("LLM Service started successfully")

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

    async def _test_ollama_connection(self):
        """Test connection to external Ollama service"""
        try:
            # Test basic connectivity by listing models
            response = await self.openai_client.models.list()
            available_models = [model.id for model in response.data]

            if self.llm_config.model not in available_models:
                self.logger.warning(f"Model {self.llm_config.model} not found. Available models: {available_models}")
                if available_models:
                    # Use the first available model as fallback
                    self.llm_config.model = available_models[0]
                    self.current_model = self.llm_config.model
                    self.logger.info(f"Using fallback model: {self.llm_config.model}")
                else:
                    raise ModelError(f"No models available in Ollama service", ErrorCode.MODEL_LOAD_ERROR)
            else:
                self.current_model = self.llm_config.model
                self.logger.info(f"Model {self.llm_config.model} is available")

        except Exception as e:
            self.logger.error(f"Error testing Ollama connection: {e}")
            raise ModelError(f"Failed to connect to Ollama: {e}", ErrorCode.MODEL_LOAD_ERROR)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text response using LLM"""
        with Timer(self.logger, f"LLM generation for prompt length {len(request.prompt)}"):
            try:
                # Prepare messages for OpenAI format
                messages = []

                # Add system prompt
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                else:
                    messages.append({"role": "system", "content": self.llm_config.system_prompt})

                # Add conversation context if provided
                if request.context:
                    for msg in request.context[-10:]:  # Keep last 10 messages for context
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
            # Prepare messages
            messages = []

            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            else:
                messages.append({"role": "system", "content": self.llm_config.system_prompt})

            if request.context:
                for msg in request.context[-10:]:
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
        """Generate embeddings for text"""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=model or self.llm_config.model
            )

            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise ModelError(f"Embedding generation failed: {e}", ErrorCode.MODEL_INFERENCE_ERROR)

    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
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
                    "quantization_level": "unknown"
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            raise ModelError(f"Failed to get model info: {e}", ErrorCode.MODEL_LOAD_ERROR)

    async def pull_model(self, model: str) -> AsyncGenerator[str, None]:
        """Pull a model from Ollama registry"""
        # Since we're using an external Ollama service, model management should be done there
        yield json.dumps({
            "status": "info",
            "message": f"Model management should be done on the external Ollama service at {self.llm_config.ollama_url}"
        })
        yield json.dumps({
            "status": "info",
            "message": f"Please run 'ollama pull {model}' on the external Ollama service"
        })

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test basic connectivity by listing models
            response = await self.openai_client.models.list()
            available_models = len(response.data)

            return {
                "status": "healthy",
                "model": self.current_model,
                "available_models": available_models,
                "ollama_url": self.llm_config.ollama_url
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "ollama_url": self.llm_config.ollama_url
            }
