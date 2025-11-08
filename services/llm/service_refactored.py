"""
Production-quality LLM Service implementation using Open WebUI / Ollama API

Features:
- Async HTTP client with connection pooling
- Circuit breaker pattern for fault tolerance
- Rate limiting for API protection
- Retry with exponential backoff and jitter
- Health monitoring
- Comprehensive error handling
- Structured logging and metrics

Reference: https://docs.openwebui.com/getting-started/api-endpoints/
"""
import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import httpx

from shared.config.base import ServiceConfig
from shared.models.base import LLMRequest, LLMResponse, Message
from shared.utils.logging import setup_logging, Timer
from shared.utils.errors import ErrorHandler, ModelError, ErrorCode, ServiceError
from shared.infrastructure import (
    EnhancedHTTPClient,
    ConnectionPoolConfig,
    RetryConfig,
    TimeoutConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    HealthStatus
)

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

    # Connection pool settings
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0

    # Retry settings
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Rate limiting (requests per second)
    rate_limit_rps: float = 10.0
    rate_limit_burst: int = 20

    # Health monitoring
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0


class ProductionLLMService:
    """
    Production-quality LLM Service using Open WebUI / Ollama API

    Implements enterprise-grade patterns for reliability and performance:
    - Connection pooling for efficient resource usage
    - Circuit breaker to prevent cascading failures
    - Rate limiting to protect backend services
    - Retry logic with exponential backoff and jitter
    - Comprehensive health monitoring
    - Structured error handling and logging
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("llm")
        self.error_handler = ErrorHandler(logger)
        self.logger = setup_logging(
            "production_llm_service",
            self.config.get("log_level", "INFO"),
            "logs/production_llm_service.log"
        )

        # Load configuration with environment variable overrides
        config_dict = self._load_config()
        self.llm_config = LLMConfig(**config_dict)

        # HTTP clients (initialized in start())
        self.openai_client: Optional[AsyncOpenAI] = None
        self.embedding_client: Optional[EnhancedHTTPClient] = None

        # Model and conversation management
        self.current_model = None
        self.conversation_cache = {}

        # Metrics
        self.generation_count = 0
        self.embedding_count = 0
        self.total_tokens_used = 0

        self.logger.info(
            f"Production LLM Service initialized: "
            f"model={self.llm_config.model}, "
            f"api_base={self.llm_config.openai_api_base}"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with environment variable overrides"""
        config_dict = self.config.all()

        # Environment variable overrides
        env_overrides = {
            "OLLAMA_BASE_URL": "openai_api_base",
            "MORGAN_LLM_API_KEY": "api_key",
            "MORGAN_EMBEDDING_MODEL": "embedding_model",
            "MORGAN_LLM_MODEL": "model",
        }

        for env_var, config_key in env_overrides.items():
            value = os.getenv(env_var)
            if value:
                config_dict[config_key] = value
                self.logger.debug(f"Config override from {env_var}: {config_key}")

        return config_dict

    async def start(self):
        """Start the LLM service with all production components"""
        try:
            # Initialize OpenAI client with production-grade HTTP client
            pool_config = ConnectionPoolConfig(
                max_connections=self.llm_config.max_connections,
                max_keepalive_connections=self.llm_config.max_keepalive_connections,
                keepalive_expiry=self.llm_config.keepalive_expiry
            )

            timeout_config = TimeoutConfig(
                connect=5.0,
                read=self.llm_config.timeout,
                write=10.0,
                pool=5.0
            )

            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.llm_config.timeout),
                limits=httpx.Limits(
                    max_connections=pool_config.max_connections,
                    max_keepalive_connections=pool_config.max_keepalive_connections,
                    keepalive_expiry=pool_config.keepalive_expiry
                ),
                follow_redirects=True
            )

            self.openai_client = AsyncOpenAI(
                base_url=self.llm_config.openai_api_base,
                api_key=self.llm_config.api_key or "sk-dummy-key",
                http_client=http_client
            )

            # Initialize enhanced HTTP client for embeddings
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=self.llm_config.circuit_breaker_failure_threshold,
                timeout=self.llm_config.circuit_breaker_timeout
            )

            rate_limit_config = RateLimitConfig(
                requests_per_second=self.llm_config.rate_limit_rps,
                burst_size=self.llm_config.rate_limit_burst
            )

            retry_config = RetryConfig(
                max_retries=self.llm_config.max_retries,
                base_delay=self.llm_config.base_retry_delay,
                max_delay=self.llm_config.max_retry_delay,
                jitter=True
            )

            # Extract base URL for embeddings (remove /v1 suffix)
            embed_base_url = self.llm_config.openai_api_base.rstrip('/').rsplit('/v1', 1)[0]

            self.embedding_client = EnhancedHTTPClient(
                service_name="ollama_embeddings",
                base_url=embed_base_url,
                pool_config=pool_config,
                retry_config=retry_config,
                timeout_config=timeout_config,
                circuit_breaker_config=circuit_breaker_config,
                rate_limit_config=rate_limit_config,
                enable_health_monitoring=self.llm_config.enable_health_monitoring,
                health_check_interval=self.llm_config.health_check_interval
            )

            await self.embedding_client.start()

            # Test connection if API key is provided
            if self.llm_config.api_key:
                await self._test_api_connection()
                self.current_model = self.llm_config.model
                self.logger.info("Production LLM Service started successfully")
            else:
                self.logger.warning(
                    "No API key provided - LLM service started in offline mode"
                )
                self.current_model = self.llm_config.model

        except Exception as e:
            self.logger.error(f"Failed to start Production LLM service: {e}")
            raise

    async def stop(self):
        """Stop the LLM service and cleanup resources"""
        self.logger.info("Production LLM Service stopping...")

        # Stop embedding client
        if self.embedding_client:
            await self.embedding_client.stop()

        # Close OpenAI client
        if self.openai_client:
            await self.openai_client.close()

        self.logger.info("Production LLM Service stopped")

    async def _test_api_connection(self):
        """Test connection to OpenAI-compatible API service"""
        if not self.llm_config.api_key:
            self.logger.warning("No API key provided - skipping connection test")
            return

        try:
            with Timer(self.logger, "API connection test"):
                response = await self.openai_client.models.list()
                available_models = [model.id for model in response.data]

                self.logger.info(
                    f"Available models: {len(available_models)} models found"
                )
                self.logger.debug(f"Model list: {available_models}")

                if self.llm_config.model not in available_models:
                    self.logger.warning(
                        f"Configured model '{self.llm_config.model}' not found"
                    )
                    if available_models:
                        fallback_model = available_models[0]
                        self.llm_config.model = fallback_model
                        self.current_model = fallback_model
                        self.logger.info(f"Using fallback model: {fallback_model}")
                    else:
                        raise ModelError(
                            "No models available in API service",
                            ErrorCode.MODEL_LOAD_ERROR
                        )
                else:
                    self.current_model = self.llm_config.model
                    self.logger.info(f"Model '{self.llm_config.model}' verified")

        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            raise ModelError(
                f"Failed to connect to API: {e}",
                ErrorCode.MODEL_LOAD_ERROR
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text response using LLM

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLMResponse with generated text

        Raises:
            ModelError: On generation failure
        """
        with Timer(
            self.logger,
            f"LLM generation (prompt_len={len(request.prompt)})"
        ):
            try:
                # Check if API key is available
                if not self.llm_config.api_key:
                    return self._create_offline_response()

                # Prepare messages
                messages = self._prepare_messages(request)

                # Generate response
                response = await self.openai_client.chat.completions.create(
                    model=request.model or self.llm_config.model,
                    messages=messages,
                    max_tokens=request.max_tokens or self.llm_config.max_tokens,
                    temperature=request.temperature or self.llm_config.temperature,
                    stream=False
                )

                generated_text = response.choices[0].message.content

                # Track metrics
                self.generation_count += 1
                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens

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
                    metadata={
                        "service": "production_llm",
                        "provider": "openai_compatible",
                        "generation_count": self.generation_count
                    }
                )

            except Exception as e:
                self.logger.error(f"Text generation failed: {e}")
                raise ModelError(
                    f"Text generation failed: {e}",
                    ErrorCode.MODEL_INFERENCE_ERROR
                )

    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text response

        Args:
            request: LLM request with prompt and parameters

        Yields:
            Generated text chunks

        Raises:
            ModelError: On generation failure
        """
        try:
            # Check if API key is available
            if not self.llm_config.api_key:
                yield "I'm currently offline and cannot process your request."
                return

            # Prepare messages
            messages = self._prepare_messages(request)

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

            # Track metrics
            self.generation_count += 1

        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise ModelError(
                f"Streaming generation failed: {e}",
                ErrorCode.MODEL_INFERENCE_ERROR
            )

    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings for text using Ollama API

        Args:
            text: Text to embed
            model: Optional embedding model (defaults to config)

        Returns:
            List of embedding values

        Raises:
            ModelError: On embedding failure
        """
        try:
            embedding_model = model or self.llm_config.embedding_model

            self.logger.debug(
                f"Generating embeddings: model={embedding_model}, "
                f"text_len={len(text)}"
            )

            # Prepare request
            payload = {
                'model': embedding_model,
                'input': [text]
            }

            headers = {'Content-Type': 'application/json'}
            if self.llm_config.api_key:
                headers['Authorization'] = f'Bearer {self.llm_config.api_key}'

            # Make request through enhanced HTTP client
            response = await self.embedding_client.post(
                '/api/embed',
                json=payload,
                headers=headers
            )

            result = response.json()

            # Track metrics
            self.embedding_count += 1

            # Extract embeddings
            if 'embeddings' in result and len(result['embeddings']) > 0:
                return result['embeddings'][0]
            elif 'embedding' in result:
                return result['embedding']
            else:
                raise ModelError(
                    f"Unexpected embedding response format",
                    ErrorCode.MODEL_INFERENCE_ERROR,
                    {"response": result}
                )

        except ServiceError as e:
            # Already a ServiceError, re-raise as ModelError
            self.logger.error(f"Embedding generation failed: {e}")
            raise ModelError(
                f"Embedding generation failed: {e}",
                ErrorCode.MODEL_INFERENCE_ERROR
            )
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise ModelError(
                f"Embedding generation failed: {e}",
                ErrorCode.MODEL_INFERENCE_ERROR
            )

    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        messages = []

        # Add system prompt
        system_prompt = request.system_prompt or self.llm_config.system_prompt
        messages.append({"role": "system", "content": system_prompt})

        # Add conversation context
        if request.context:
            max_context = self.llm_config.max_context_messages
            for msg in request.context[-max_context:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add current prompt
        messages.append({"role": "user", "content": request.prompt})

        return messages

    def _create_offline_response(self) -> LLMResponse:
        """Create response for offline mode"""
        return LLMResponse(
            text=(
                "I'm currently offline and cannot process your request. "
                "Please provide an API key to enable LLM functionality."
            ),
            model=self.llm_config.model,
            metadata={
                "offline": True,
                "error": "No API key provided"
            }
        )

    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            if not self.llm_config.api_key:
                return self._create_offline_models_list()

            response = await self.openai_client.models.list()
            models = [
                {"name": model.id, "size": 0, "modified": ""}
                for model in response.data
            ]

            return {
                "models": models,
                "total": len(models),
                "service": "production_llm"
            }

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise ModelError(
                f"Failed to list models: {e}",
                ErrorCode.MODEL_LOAD_ERROR
            )

    def _create_offline_models_list(self) -> Dict[str, Any]:
        """Create models list for offline mode"""
        return {
            "models": [
                {
                    "name": self.llm_config.model,
                    "size": 0,
                    "modified": "offline"
                },
                {
                    "name": self.llm_config.embedding_model,
                    "size": 0,
                    "modified": "offline"
                }
            ],
            "total": 2,
            "offline": True
        }

    async def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_name = model or self.llm_config.model

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

    async def pull_model(self, model: str) -> AsyncGenerator[str, None]:
        """Pull a model (not supported for external API)"""
        yield json.dumps({
            "status": "info",
            "message": (
                f"Model management should be done on the external API service "
                f"at {self.llm_config.openai_api_base}"
            )
        })

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            health_status = {
                "service": "production_llm",
                "model": self.current_model,
                "api_base": self.llm_config.openai_api_base,
                "metrics": {
                    "generation_count": self.generation_count,
                    "embedding_count": self.embedding_count,
                    "total_tokens_used": self.total_tokens_used
                }
            }

            if not self.llm_config.api_key:
                health_status.update({
                    "status": "offline",
                    "available_models": 0,
                    "message": "No API key provided"
                })
                return health_status

            # Test connectivity
            response = await self.openai_client.models.list()
            available_models = len(response.data)

            # Get embedding client status
            embedding_status = self.embedding_client.get_status()

            health_status.update({
                "status": "healthy",
                "available_models": available_models,
                "embedding_client": embedding_status,
                "circuit_breaker": embedding_status.get("circuit_breaker"),
                "rate_limiter": embedding_status.get("rate_limiter")
            })

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "service": "production_llm",
                "status": "unhealthy",
                "error": str(e),
                "api_base": self.llm_config.openai_api_base
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        metrics = {
            "generation_count": self.generation_count,
            "embedding_count": self.embedding_count,
            "total_tokens_used": self.total_tokens_used
        }

        if self.embedding_client:
            metrics["embedding_client"] = self.embedding_client.get_status()

        return metrics
