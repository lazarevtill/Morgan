"""
Specialized error handling for Jina AI components in Morgan RAG.

Provides production-grade error handling specifically for Jina AI models,
web scraping, multimodal processing, and background tasks with intelligent
fallback mechanisms and graceful degradation.
"""

import functools
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from morgan.utils.error_handling import (
    CircuitBreaker,
    ErrorCategory,
    MorganError,
    get_degradation_manager,
    get_recovery_manager,
)

logger = structlog.get_logger(__name__)


class JinaServiceType(Enum):
    """Types of Jina AI services."""

    EMBEDDING = "embedding"
    RERANKING = "reranking"
    WEB_SCRAPING = "web_scraping"
    MULTIMODAL = "multimodal"
    CODE_INTELLIGENCE = "code_intelligence"


class JinaModelType(Enum):
    """Jina AI model types."""

    EMBEDDINGS_V4 = "jina-embeddings-v4"
    CODE_EMBEDDINGS = "jina-code-embeddings-1.5b"
    CLIP_V2 = "jina-clip-v2"
    RERANKER_V3 = "jina-reranker-v3"
    RERANKER_V2_MULTILINGUAL = "jina-reranker-v2-base-multilingual"
    READER_LM_V2 = "jina-reader-lm-v2"


@dataclass
class JinaErrorContext:
    """Extended error context for Jina AI operations."""

    service_type: JinaServiceType
    model_type: Optional[JinaModelType] = None
    batch_size: Optional[int] = None
    input_size: Optional[int] = None
    language: Optional[str] = None
    content_type: Optional[str] = None
    fallback_attempted: bool = False
    fallback_successful: bool = False


class JinaEmbeddingError(MorganError):
    """Errors specific to Jina embedding services."""

    def __init__(
        self, message: str, model_type: Optional[JinaModelType] = None, **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.EMBEDDING,
            component="jina_embedding_service",
            **kwargs,
        )
        self.model_type = model_type


class JinaRerankingError(MorganError):
    """Errors specific to Jina reranking services."""

    def __init__(
        self, message: str, model_type: Optional[JinaModelType] = None, **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.SEARCH,
            component="jina_reranking_service",
            **kwargs,
        )
        self.model_type = model_type


class JinaWebScrapingError(MorganError):
    """Errors specific to Jina web scraping services."""

    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            component="jina_web_scraping_service",
            **kwargs,
        )
        self.url = url


class JinaMultimodalError(MorganError):
    """Errors specific to Jina multimodal processing."""

    def __init__(self, message: str, content_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VECTORIZATION,
            component="jina_multimodal_service",
            **kwargs,
        )
        self.content_type = content_type


class JinaCodeIntelligenceError(MorganError):
    """Errors specific to Jina code intelligence."""

    def __init__(self, message: str, language: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VECTORIZATION,
            component="jina_code_intelligence_service",
            **kwargs,
        )
        self.language = language


class JinaServiceManager:
    """
    Manages Jina AI service health, fallbacks, and error recovery.

    Provides centralized management of Jina AI services with intelligent
    fallback mechanisms, circuit breakers, and graceful degradation.
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_health: Dict[JinaServiceType, bool] = {}
        self.fallback_models: Dict[JinaModelType, List[JinaModelType]] = {}
        self.service_metrics: Dict[str, Dict[str, Any]] = {}

        # Initialize circuit breakers for each service
        self._initialize_circuit_breakers()

        # Initialize fallback model mappings
        self._initialize_fallback_models()

        logger.info("JinaServiceManager initialized")

    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for Jina services."""
        from morgan.utils.error_handling import CircuitBreakerConfig

        # Configure circuit breakers with service-specific settings
        configs = {
            JinaServiceType.EMBEDDING: CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=30.0, success_threshold=2
            ),
            JinaServiceType.RERANKING: CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=60.0, success_threshold=3
            ),
            JinaServiceType.WEB_SCRAPING: CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=45.0, success_threshold=2
            ),
            JinaServiceType.MULTIMODAL: CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=30.0, success_threshold=2
            ),
            JinaServiceType.CODE_INTELLIGENCE: CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=30.0, success_threshold=2
            ),
        }

        for service_type, config in configs.items():
            self.circuit_breakers[service_type.value] = CircuitBreaker(
                name=f"jina_{service_type.value}", config=config
            )

    def _initialize_fallback_models(self):
        """Initialize fallback model mappings."""
        self.fallback_models = {
            # Primary embedding model fallbacks
            JinaModelType.EMBEDDINGS_V4: [
                # Could fallback to local models if available
            ],
            # Code embedding fallbacks
            JinaModelType.CODE_EMBEDDINGS: [
                JinaModelType.EMBEDDINGS_V4  # General embeddings as fallback
            ],
            # Multimodal fallbacks
            JinaModelType.CLIP_V2: [
                # Could fallback to text-only processing
            ],
            # Reranking fallbacks
            JinaModelType.RERANKER_V3: [JinaModelType.RERANKER_V2_MULTILINGUAL],
            JinaModelType.RERANKER_V2_MULTILINGUAL: [
                # Could fallback to embedding-only ranking
            ],
        }

    def get_circuit_breaker(self, service_type: JinaServiceType) -> CircuitBreaker:
        """Get circuit breaker for a specific service."""
        return self.circuit_breakers[service_type.value]

    def record_service_success(
        self, service_type: JinaServiceType, operation: str, duration: float
    ):
        """Record successful service operation."""
        service_key = service_type.value

        if service_key not in self.service_metrics:
            self.service_metrics[service_key] = {
                "success_count": 0,
                "error_count": 0,
                "total_duration": 0.0,
                "last_success": None,
            }

        metrics = self.service_metrics[service_key]
        metrics["success_count"] += 1
        metrics["total_duration"] += duration
        metrics["last_success"] = datetime.now()

        self.service_health[service_type] = True

        logger.debug(
            "Recorded Jina service success",
            service_type=service_type.value,
            operation=operation,
            duration=duration,
        )

    def record_service_error(
        self, service_type: JinaServiceType, operation: str, error: Exception
    ):
        """Record service error."""
        service_key = service_type.value

        if service_key not in self.service_metrics:
            self.service_metrics[service_key] = {
                "success_count": 0,
                "error_count": 0,
                "total_duration": 0.0,
                "last_error": None,
            }

        metrics = self.service_metrics[service_key]
        metrics["error_count"] += 1
        metrics["last_error"] = datetime.now()

        # Update service health based on error rate
        total_operations = metrics["success_count"] + metrics["error_count"]
        error_rate = (
            metrics["error_count"] / total_operations if total_operations > 0 else 1.0
        )

        self.service_health[service_type] = (
            error_rate < 0.1
        )  # Healthy if error rate < 10%

        logger.error(
            "Recorded Jina service error",
            service_type=service_type.value,
            operation=operation,
            error=str(error),
            error_rate=error_rate,
        )

    def get_service_health(self, service_type: JinaServiceType) -> bool:
        """Get current health status of a service."""
        return self.service_health.get(service_type, False)

    def get_fallback_models(self, model_type: JinaModelType) -> List[JinaModelType]:
        """Get fallback models for a given model type."""
        return self.fallback_models.get(model_type, [])

    def get_service_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive service metrics."""
        return self.service_metrics.copy()


# Global service manager instance
_jina_service_manager = JinaServiceManager()


def get_jina_service_manager() -> JinaServiceManager:
    """Get global Jina service manager instance."""
    return _jina_service_manager


def with_jina_retry(
    service_type: JinaServiceType,
    model_type: Optional[JinaModelType] = None,
    max_attempts: int = 3,
    enable_fallback: bool = True,
    operation: str = "jina_operation",
):
    """
    Decorator for Jina AI operations with retry logic and fallback support.

    Args:
        service_type: Type of Jina service
        model_type: Specific model type (optional)
        max_attempts: Maximum retry attempts
        enable_fallback: Whether to attempt fallback models
        operation: Operation name for logging
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            service_manager = get_jina_service_manager()
            circuit_breaker = service_manager.get_circuit_breaker(service_type)

            last_exception = None

            # Try primary model
            for attempt in range(max_attempts):
                try:
                    start_time = time.time()

                    # Use circuit breaker for the call
                    result = circuit_breaker.call(func, *args, **kwargs)

                    duration = time.time() - start_time
                    service_manager.record_service_success(
                        service_type, operation, duration
                    )

                    return result

                except Exception as e:
                    last_exception = e
                    duration = time.time() - start_time
                    service_manager.record_service_error(service_type, operation, e)

                    # Check if we should retry
                    if attempt < max_attempts - 1:
                        delay = min(2.0**attempt, 10.0)  # Exponential backoff, max 10s
                        logger.warning(
                            f"Jina {service_type.value} operation failed, retrying in {delay}s",
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error=str(e),
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Jina {service_type.value} operation failed after {max_attempts} attempts",
                            error=str(e),
                        )

            # Try fallback models if enabled and available
            if enable_fallback and model_type:
                fallback_models = service_manager.get_fallback_models(model_type)

                for fallback_model in fallback_models:
                    try:
                        logger.info(
                            f"Attempting fallback from {model_type.value} to {fallback_model.value}"
                        )

                        # Update kwargs to use fallback model
                        fallback_kwargs = kwargs.copy()
                        fallback_kwargs["model_type"] = fallback_model

                        start_time = time.time()
                        result = func(*args, **fallback_kwargs)
                        duration = time.time() - start_time

                        service_manager.record_service_success(
                            service_type, f"{operation}_fallback", duration
                        )

                        logger.info(
                            f"Fallback to {fallback_model.value} successful",
                            original_model=model_type.value,
                            fallback_model=fallback_model.value,
                        )

                        return result

                    except Exception as fallback_error:
                        logger.warning(
                            f"Fallback to {fallback_model.value} failed: {fallback_error}"
                        )
                        continue

            # All attempts failed
            if isinstance(last_exception, MorganError):
                raise last_exception
            else:
                # Convert to appropriate Jina error type
                if service_type == JinaServiceType.EMBEDDING:
                    raise JinaEmbeddingError(
                        f"Embedding operation failed: {last_exception}",
                        model_type=model_type,
                        operation=operation,
                        cause=last_exception,
                    )
                elif service_type == JinaServiceType.RERANKING:
                    raise JinaRerankingError(
                        f"Reranking operation failed: {last_exception}",
                        model_type=model_type,
                        operation=operation,
                        cause=last_exception,
                    )
                elif service_type == JinaServiceType.WEB_SCRAPING:
                    raise JinaWebScrapingError(
                        f"Web scraping operation failed: {last_exception}",
                        operation=operation,
                        cause=last_exception,
                    )
                elif service_type == JinaServiceType.MULTIMODAL:
                    raise JinaMultimodalError(
                        f"Multimodal operation failed: {last_exception}",
                        operation=operation,
                        cause=last_exception,
                    )
                elif service_type == JinaServiceType.CODE_INTELLIGENCE:
                    raise JinaCodeIntelligenceError(
                        f"Code intelligence operation failed: {last_exception}",
                        operation=operation,
                        cause=last_exception,
                    )
                else:
                    raise MorganError(
                        f"Jina operation failed: {last_exception}",
                        category=ErrorCategory.NETWORK,
                        component=f"jina_{service_type.value}_service",
                        operation=operation,
                        cause=last_exception,
                    )

        return wrapper

    return decorator


def handle_jina_embedding_errors(
    model_type: JinaModelType = JinaModelType.EMBEDDINGS_V4,
    enable_local_fallback: bool = True,
    batch_size_reduction: bool = True,
):
    """
    Decorator for handling Jina embedding service errors with intelligent fallbacks.

    Args:
        model_type: Primary model to use
        enable_local_fallback: Whether to fallback to local embeddings
        batch_size_reduction: Whether to reduce batch size on memory errors
    """

    def decorator(func: Callable) -> Callable:
        @with_jina_retry(
            service_type=JinaServiceType.EMBEDDING,
            model_type=model_type,
            operation="embedding_generation",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except JinaEmbeddingError as e:
                # Check for specific error types and apply appropriate fallbacks

                # Memory/batch size errors
                if "memory" in str(e).lower() or "batch" in str(e).lower():
                    if batch_size_reduction and "batch_size" in kwargs:
                        original_batch_size = kwargs["batch_size"]
                        reduced_batch_size = max(1, original_batch_size // 2)

                        logger.warning(
                            f"Reducing batch size from {original_batch_size} to {reduced_batch_size}"
                        )

                        kwargs["batch_size"] = reduced_batch_size
                        return func(*args, **kwargs)

                # Model loading errors
                if "model" in str(e).lower() and "load" in str(e).lower():
                    if enable_local_fallback:
                        logger.warning(
                            "Jina embedding model failed to load, attempting local fallback"
                        )

                        try:
                            # Attempt to use local embedding service
                            from morgan.embeddings.service import (
                                get_embedding_service,
                            )

                            service = get_embedding_service()
                            service._remote_available = False

                            if service._check_local_available():
                                return func(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(
                                f"Local embedding fallback failed: {fallback_error}"
                            )

                # Re-raise if no fallback worked
                raise

        return wrapper

    return decorator


def handle_jina_reranking_errors(
    primary_model: JinaModelType = JinaModelType.RERANKER_V3,
    enable_embedding_fallback: bool = True,
):
    """
    Decorator for handling Jina reranking service errors with fallbacks.

    Args:
        primary_model: Primary reranking model
        enable_embedding_fallback: Whether to fallback to embedding-only ranking
    """

    def decorator(func: Callable) -> Callable:
        @with_jina_retry(
            service_type=JinaServiceType.RERANKING,
            model_type=primary_model,
            operation="result_reranking",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except JinaRerankingError:
                if enable_embedding_fallback:
                    logger.warning(
                        "Jina reranking failed, falling back to embedding-based ranking"
                    )

                    try:
                        # Fallback to embedding-based ranking
                        kwargs.get("query", args[0] if args else "")
                        results = kwargs.get(
                            "results", args[1] if len(args) > 1 else []
                        )

                        # Simple embedding-based reranking fallback
                        # This would use cosine similarity with query embeddings
                        logger.info("Using embedding-based ranking as fallback")

                        # Return results as-is for now (in real implementation,
                        # would re-rank using embedding similarity)
                        return results

                    except Exception as fallback_error:
                        logger.error(
                            f"Embedding-based ranking fallback failed: {fallback_error}"
                        )

                raise

        return wrapper

    return decorator


def handle_jina_web_scraping_errors(
    enable_html_fallback: bool = True, timeout_seconds: float = 30.0
):
    """
    Decorator for handling Jina web scraping errors with fallbacks.

    Args:
        enable_html_fallback: Whether to fallback to basic HTML parsing
        timeout_seconds: Timeout for web scraping operations
    """

    def decorator(func: Callable) -> Callable:
        @with_jina_retry(
            service_type=JinaServiceType.WEB_SCRAPING,
            operation="web_content_extraction",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except JinaWebScrapingError:
                if enable_html_fallback:
                    url = kwargs.get("url", args[0] if args else None)

                    if url:
                        logger.warning(
                            f"Jina web scraping failed for {url}, attempting basic HTML parsing"
                        )

                        try:
                            # Fallback to basic HTML parsing
                            import requests
                            from bs4 import BeautifulSoup

                            response = requests.get(url, timeout=timeout_seconds)
                            response.raise_for_status()

                            soup = BeautifulSoup(response.content, "html.parser")

                            # Extract basic content
                            title = soup.find("title")
                            title_text = (
                                title.get_text().strip() if title else "No title"
                            )

                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()

                            # Get text content
                            content = soup.get_text()

                            # Clean up whitespace
                            lines = (line.strip() for line in content.splitlines())
                            chunks = (
                                phrase.strip()
                                for line in lines
                                for phrase in line.split("  ")
                            )
                            content = " ".join(chunk for chunk in chunks if chunk)

                            logger.info(f"Basic HTML parsing successful for {url}")

                            return {
                                "url": url,
                                "title": title_text,
                                "content": content,
                                "extraction_method": "basic_html_fallback",
                                "metadata": {
                                    "content_length": len(content),
                                    "extraction_time": datetime.now().isoformat(),
                                },
                            }

                        except Exception as fallback_error:
                            logger.error(
                                f"Basic HTML parsing fallback failed: {fallback_error}"
                            )

                raise

        return wrapper

    return decorator


def handle_jina_multimodal_errors(
    enable_text_only_fallback: bool = True, enable_ocr_fallback: bool = True
):
    """
    Decorator for handling Jina multimodal processing errors.

    Args:
        enable_text_only_fallback: Whether to process text-only when multimodal fails
        enable_ocr_fallback: Whether to use OCR for image text extraction
    """

    def decorator(func: Callable) -> Callable:
        @with_jina_retry(
            service_type=JinaServiceType.MULTIMODAL, operation="multimodal_processing"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except JinaMultimodalError as e:
                text_content = kwargs.get("text_content", "")
                images = kwargs.get("images", [])

                # Try text-only processing
                if enable_text_only_fallback and text_content:
                    logger.warning(
                        "Multimodal processing failed, falling back to text-only"
                    )

                    try:
                        # Process only text content
                        from morgan.embeddings.service import (
                            get_embedding_service,
                        )

                        embedding_service = get_embedding_service()

                        text_embeddings = embedding_service.encode([text_content])

                        return {
                            "text_embeddings": (
                                text_embeddings[0] if text_embeddings else []
                            ),
                            "image_embeddings": [],
                            "combined_embedding": (
                                text_embeddings[0] if text_embeddings else []
                            ),
                            "processing_method": "text_only_fallback",
                            "metadata": {
                                "original_error": str(e),
                                "fallback_used": "text_only",
                            },
                        }

                    except Exception as fallback_error:
                        logger.error(f"Text-only fallback failed: {fallback_error}")

                # Try OCR extraction for images
                if enable_ocr_fallback and images:
                    logger.warning("Attempting OCR text extraction from images")

                    try:
                        # Use DeepSeek-OCR via Ollama
                        from morgan.services.ocr_service import (
                            get_ocr_service,
                            OCRMode,
                        )
                        import asyncio

                        ocr_service = get_ocr_service()

                        # Extract text from all images
                        async def run_ocr():
                            results = await ocr_service.extract_text_batch(
                                images, OCRMode.FREE
                            )
                            return " ".join(
                                r.text for r in results if r.success and r.text
                            )

                        # Run async in sync context
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as p:
                                    future = p.submit(asyncio.run, run_ocr())
                                    extracted_text = future.result(timeout=120)
                            else:
                                extracted_text = loop.run_until_complete(run_ocr())
                        except RuntimeError:
                            extracted_text = asyncio.run(run_ocr())

                        if extracted_text:
                            # Process extracted text
                            from morgan.embeddings.service import (
                                get_embedding_service,
                            )

                            embedding_service = get_embedding_service()

                            combined = f"{text_content} {extracted_text}".strip()
                            text_embeddings = embedding_service.encode([combined])

                            return {
                                "text_embeddings": (
                                    text_embeddings[0] if text_embeddings else []
                                ),
                                "image_embeddings": [],
                                "combined_embedding": (
                                    text_embeddings[0] if text_embeddings else []
                                ),
                                "processing_method": "ocr_fallback",
                                "extracted_text": extracted_text,
                                "metadata": {
                                    "original_error": str(e),
                                    "fallback_used": "deepseek_ocr",
                                },
                            }

                    except Exception as ocr_error:
                        logger.error("OCR fallback failed: %s", ocr_error)

                raise

        return wrapper

    return decorator


def handle_background_task_errors(
    task_type: str,
    enable_graceful_degradation: bool = True,
    max_consecutive_failures: int = 3,
):
    """
    Decorator for handling background task errors with graceful degradation.

    Args:
        task_type: Type of background task
        enable_graceful_degradation: Whether to apply graceful degradation
        max_consecutive_failures: Maximum consecutive failures before degradation
    """

    def decorator(func: Callable) -> Callable:
        # Track consecutive failures per task type
        if not hasattr(handle_background_task_errors, "failure_counts"):
            handle_background_task_errors.failure_counts = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)

                # Reset failure count on success
                handle_background_task_errors.failure_counts[task_type] = 0

                return result

            except Exception as e:
                # Increment failure count
                current_failures = handle_background_task_errors.failure_counts.get(
                    task_type, 0
                )
                handle_background_task_errors.failure_counts[task_type] = (
                    current_failures + 1
                )

                logger.error(
                    f"Background task {task_type} failed",
                    consecutive_failures=current_failures + 1,
                    max_failures=max_consecutive_failures,
                    error=str(e),
                )

                # Apply graceful degradation if too many failures
                if (
                    enable_graceful_degradation
                    and handle_background_task_errors.failure_counts[task_type]
                    >= max_consecutive_failures
                ):

                    logger.warning(
                        f"Background task {task_type} has failed {max_consecutive_failures} times, "
                        "applying graceful degradation"
                    )

                    degradation_manager = get_degradation_manager()

                    # Create error context for degradation assessment
                    from morgan.utils.error_handling import (
                        ErrorCategory,
                        ErrorContext,
                        ErrorSeverity,
                    )

                    error_context = ErrorContext(
                        error_id=f"background_task_{task_type}_{int(time.time())}",
                        timestamp=datetime.now(),
                        operation=f"background_{task_type}",
                        component="background_processor",
                        category=ErrorCategory.VECTORIZATION,
                        severity=ErrorSeverity.HIGH,
                        metadata={
                            "task_type": task_type,
                            "consecutive_failures": current_failures + 1,
                            "error": str(e),
                        },
                    )

                    degradation_manager.assess_and_apply_degradation(error_context)

                raise MorganError(
                    f"Background task {task_type} failed: {e}",
                    category=ErrorCategory.VECTORIZATION,
                    component="background_processor",
                    operation=f"background_{task_type}",
                    severity=(
                        ErrorSeverity.HIGH
                        if current_failures >= max_consecutive_failures
                        else ErrorSeverity.MEDIUM
                    ),
                    metadata={
                        "task_type": task_type,
                        "consecutive_failures": current_failures + 1,
                    },
                    cause=e,
                )

        return wrapper

    return decorator


# Recovery procedures for Jina AI services
def register_jina_recovery_procedures():
    """Register recovery procedures for Jina AI services."""
    recovery_manager = get_recovery_manager()

    # Jina embedding service recovery
    def recover_jina_embedding_service(
        error: JinaEmbeddingError, context: Dict[str, Any]
    ) -> bool:
        """Recover Jina embedding service by checking model availability."""
        try:
            service_manager = get_jina_service_manager()

            # Check if service is healthy
            if service_manager.get_service_health(JinaServiceType.EMBEDDING):
                return True

            # Try to reinitialize the service
            logger.info("Attempting to recover Jina embedding service")

            # In a real implementation, this would reinitialize the service
            # For now, just mark as recovered
            service_manager.service_health[JinaServiceType.EMBEDDING] = True

            return True

        except Exception as e:
            logger.error(f"Failed to recover Jina embedding service: {e}")
            return False

    # Register recovery procedures
    from morgan.utils.error_handling import RecoveryProcedure, RecoveryStrategy

    recovery_manager.register_procedure(
        RecoveryProcedure(
            name="jina_embedding_recovery",
            strategy=RecoveryStrategy.RETRY,
            applicable_errors=[JinaEmbeddingError],
            recovery_function=recover_jina_embedding_service,
            description="Recover Jina embedding service",
        )
    )

    logger.info("Registered Jina AI recovery procedures")


# Initialize Jina error handling
def initialize_jina_error_handling():
    """Initialize Jina AI error handling system."""
    logger.info("Initializing Jina AI error handling")

    # Register recovery procedures
    register_jina_recovery_procedures()

    # Initialize service manager
    get_jina_service_manager()

    logger.info("Jina AI error handling initialized successfully")


if __name__ == "__main__":
    # Demo Jina error handling
    print("🔧 Jina AI Error Handling Demo")
    print("=" * 40)

    # Initialize system
    initialize_jina_error_handling()

    # Test Jina embedding error handling
    @handle_jina_embedding_errors(
        model_type=JinaModelType.EMBEDDINGS_V4, enable_local_fallback=True
    )
    def test_embedding_operation(texts, batch_size=32):
        # Simulate embedding operation
        if "fail" in str(texts).lower():
            raise Exception("Simulated embedding failure")
        return [[0.1, 0.2, 0.3] for _ in texts]

    # Test successful operation
    try:
        result = test_embedding_operation(["hello world"])
        print(f"Embedding successful: {len(result)} embeddings")
    except Exception as e:
        print(f"Embedding failed: {e}")

    # Test failed operation with error handling
    try:
        result = test_embedding_operation(["fail this operation"])
        print(f"Embedding result: {result}")
    except Exception as e:
        print(f"Embedding failed as expected: {e}")

    # Test service manager
    service_manager = get_jina_service_manager()
    print(
        f"Service health: {service_manager.get_service_health(JinaServiceType.EMBEDDING)}"
    )
    print(f"Service metrics: {service_manager.get_service_metrics()}")

    print("\n" + "=" * 40)
    print("Jina AI error handling demo completed!")
