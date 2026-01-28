"""
Error handling decorators for Morgan RAG components.

Provides convenient decorators for adding production-grade error handling
to existing methods and functions with minimal code changes.
"""

import functools
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from morgan.utils.error_handling import (
    CacheError,
    CompanionError,
    EmbeddingError,
    EmotionalProcessingError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    MemoryProcessingError,
    NetworkError,
    RetryConfig,
    SearchError,
    StorageError,
    ValidationError,
    error_context,
    get_degradation_manager,
    get_health_monitor,
    with_retry,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def handle_vectorization_errors(
    operation: str = "vectorization_operation",
    component: str = "vectorization_service",
    retry_config: Optional[RetryConfig] = None,
    enable_recovery: bool = True,
    enable_degradation: bool = True,
):
    """
    Decorator for handling vectorization-related errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        retry_config: Optional retry configuration
        enable_recovery: Whether to attempt error recovery
        enable_degradation: Whether to apply graceful degradation

    Example:
        @handle_vectorization_errors("process_documents", "document_processor")
        def process_documents(self, documents):
            return self._process_documents_impl(documents)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id and request_id from kwargs if available
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            with error_context(
                operation=operation,
                component=component,
                category=ErrorCategory.VECTORIZATION,
                user_id=user_id,
                request_id=request_id,
            ):
                # Apply retry if configured
                if retry_config:
                    retry_wrapper = with_retry(retry_config, operation, component)
                    return retry_wrapper(func)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_embedding_errors(
    operation: str = "embedding_operation",
    component: str = "embedding_service",
    retry_config: Optional[RetryConfig] = None,
    fallback_to_local: bool = True,
):
    """
    Decorator for handling embedding service errors with automatic fallback.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        retry_config: Optional retry configuration
        fallback_to_local: Whether to fallback to local embeddings

    Example:
        @handle_embedding_errors("encode_batch", "embedding_service",
                                 RetryConfig(max_attempts=3))
        def encode_batch(self, texts):
            return self._encode_batch_remote(texts)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.EMBEDDING,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    if retry_config:
                        retry_wrapper = with_retry(retry_config, operation, component)
                        return retry_wrapper(func)(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

            except (EmbeddingError, NetworkError) as e:
                if fallback_to_local:
                    logger.warning(
                        f"Embedding operation failed, attempting local fallback: {e}"
                    )

                    # Try to get the embedding service and force local mode
                    try:
                        from morgan.services.embeddings import (
                            get_embedding_service,
                        )

                        service = get_embedding_service()

                        # Force local availability check
                        service._remote_available = False

                        if service._check_local_available():
                            logger.info(
                                "Successfully switched to local embedding model"
                            )
                            return func(*args, **kwargs)
                        else:
                            logger.error("Local embedding model not available")
                            raise

                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback to local embeddings failed: {fallback_error}"
                        )
                        raise e from fallback_error
                else:
                    raise

        return wrapper

    return decorator


def handle_storage_errors(
    operation: str = "storage_operation",
    component: str = "vector_db_client",
    retry_config: Optional[RetryConfig] = None,
    enable_circuit_breaker: bool = True,
):
    """
    Decorator for handling vector database storage errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        retry_config: Optional retry configuration
        enable_circuit_breaker: Whether to use circuit breaker pattern

    Example:
        @handle_storage_errors("upsert_points", "vector_db_client",
                               RetryConfig(max_attempts=5))
        def upsert_points(self, collection_name, points):
            return self._upsert_points_impl(collection_name, points)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            with error_context(
                operation=operation,
                component=component,
                category=ErrorCategory.STORAGE,
                user_id=user_id,
                request_id=request_id,
            ):
                if retry_config:
                    # Use custom retry config for storage operations
                    storage_retry_config = RetryConfig(
                        max_attempts=retry_config.max_attempts,
                        base_delay=retry_config.base_delay,
                        max_delay=retry_config.max_delay,
                        retryable_exceptions=(StorageError, NetworkError),
                    )
                    retry_wrapper = with_retry(
                        storage_retry_config, operation, component
                    )
                    return retry_wrapper(func)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_search_errors(
    operation: str = "search_operation",
    component: str = "search_engine",
    fallback_to_basic: bool = True,
    min_results: int = 0,
):
    """
    Decorator for handling search errors with fallback strategies.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        fallback_to_basic: Whether to fallback to basic search
        min_results: Minimum number of results to return

    Example:
        @handle_search_errors("multi_stage_search", "search_engine")
        def search(self, query, strategies):
            return self._multi_stage_search_impl(query, strategies)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.SEARCH,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except SearchError as e:
                if fallback_to_basic:
                    logger.warning(
                        f"Advanced search failed, falling back to basic search: {e}"
                    )

                    try:
                        # Try to perform basic search as fallback
                        from morgan.core.search import SmartSearch

                        basic_search = SmartSearch()

                        # Extract query from args/kwargs
                        query = args[1] if len(args) > 1 else kwargs.get("query", "")
                        max_results = kwargs.get("max_results", 10)

                        if query:
                            results = basic_search._search_hybrid(
                                query, max_results, 0.5
                            )

                            if len(results) >= min_results:
                                logger.info(
                                    f"Basic search fallback successful: {len(results)} results"
                                )
                                return results
                            else:
                                logger.warning(
                                    f"Basic search returned insufficient results: {len(results)}"
                                )

                    except Exception as fallback_error:
                        logger.error(f"Basic search fallback failed: {fallback_error}")

                # If fallback fails or is disabled, return empty results
                logger.error(f"All search strategies failed for operation: {operation}")
                return []

        return wrapper

    return decorator


def handle_companion_errors(
    operation: str = "companion_operation",
    component: str = "companion_manager",
    degrade_gracefully: bool = True,
    skip_on_failure: bool = True,
):
    """
    Decorator for handling companion feature errors with graceful degradation.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        degrade_gracefully: Whether to apply graceful degradation
        skip_on_failure: Whether to skip operation on failure

    Example:
        @handle_companion_errors("update_relationship", "companion_manager")
        def update_relationship(self, user_id, interaction_data):
            return self._update_relationship_impl(user_id, interaction_data)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            # Check if companion features are enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("companion"):
                logger.debug(f"Companion features disabled, skipping {operation}")
                return None

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.COMPANION,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except (CompanionError, EmotionalProcessingError) as e:
                if degrade_gracefully:
                    logger.warning(
                        f"Companion operation failed, applying degradation: {e}"
                    )

                    # Apply graceful degradation
                    if hasattr(e, 'get_context'):
                        degradation_manager.assess_and_apply_degradation(e.get_context())
                    else:
                        # Base exception without ErrorHandlingMixin
                        ctx = ErrorContext(
                            error_id=f"companion_{id(e)}",
                            timestamp=datetime.now(timezone.utc),
                            operation=operation,
                            component=component,
                            category=ErrorCategory.COMPANION,
                            severity=ErrorSeverity.MEDIUM,
                        )
                        degradation_manager.assess_and_apply_degradation(ctx)

                if skip_on_failure:
                    logger.info(
                        f"Skipping companion operation due to failure: {operation}"
                    )
                    return None
                else:
                    raise

        return wrapper

    return decorator


def handle_emotional_processing_errors(
    operation: str = "emotional_operation",
    component: str = "emotional_intelligence",
    fallback_to_neutral: bool = True,
    skip_on_failure: bool = True,
):
    """
    Decorator for handling emotional processing errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        fallback_to_neutral: Whether to return neutral emotional state on failure
        skip_on_failure: Whether to skip operation on failure

    Example:
        @handle_emotional_processing_errors("analyze_emotion", "emotional_intelligence")
        def analyze_emotion(self, text, context):
            return self._analyze_emotion_impl(text, context)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            # Check if emotional processing is enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("emotional"):
                logger.debug(f"Emotional processing disabled, skipping {operation}")
                return None

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.EMOTIONAL,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except EmotionalProcessingError as e:
                if fallback_to_neutral:
                    logger.warning(
                        f"Emotional processing failed, returning neutral state: {e}"
                    )

                    # Return neutral emotional state
                    from dataclasses import dataclass

                    @dataclass
                    class NeutralEmotionalState:
                        primary_emotion: str = "neutral"
                        intensity: float = 0.0
                        confidence: float = 0.0
                        secondary_emotions: list = None
                        emotional_indicators: list = None

                        def __post_init__(self):
                            if self.secondary_emotions is None:
                                self.secondary_emotions = []
                            if self.emotional_indicators is None:
                                self.emotional_indicators = []

                    return NeutralEmotionalState()

                if skip_on_failure:
                    logger.info(
                        f"Skipping emotional processing due to failure: {operation}"
                    )
                    return None
                else:
                    raise

        return wrapper

    return decorator


def handle_memory_processing_errors(
    operation: str = "memory_operation",
    component: str = "memory_processor",
    skip_on_failure: bool = True,
    log_failure: bool = True,
):
    """
    Decorator for handling memory processing errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        skip_on_failure: Whether to skip operation on failure
        log_failure: Whether to log the failure

    Example:
        @handle_memory_processing_errors("extract_memories", "memory_processor")
        def extract_memories(self, conversation_turn):
            return self._extract_memories_impl(conversation_turn)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            # Check if memory processing is enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("memory"):
                logger.debug(f"Memory processing disabled, skipping {operation}")
                return []

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.MEMORY,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except MemoryProcessingError as e:
                if log_failure:
                    logger.warning(f"Memory processing failed: {e}")

                if skip_on_failure:
                    logger.info(
                        f"Skipping memory processing due to failure: {operation}"
                    )
                    return []
                else:
                    raise

        return wrapper

    return decorator


def handle_cache_errors(
    operation: str = "cache_operation",
    component: str = "cache_manager",
    fallback_to_no_cache: bool = True,
    disable_cache_on_failure: bool = False,
):
    """
    Decorator for handling cache-related errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        fallback_to_no_cache: Whether to continue without cache on failure
        disable_cache_on_failure: Whether to disable caching after failure

    Example:
        @handle_cache_errors("get_cached_embedding", "cache_manager")
        def get_cached_embedding(self, cache_key):
            return self._get_cached_embedding_impl(cache_key)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            # Check if caching is enabled
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_feature_enabled("caching"):
                logger.debug(f"Caching disabled, skipping {operation}")
                return None

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.CACHE,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except CacheError as e:
                if fallback_to_no_cache:
                    logger.warning(
                        f"Cache operation failed, continuing without cache: {e}"
                    )

                    if disable_cache_on_failure:
                        # Temporarily disable caching
                        from morgan.utils.error_handling import DegradationLevel

                        degradation_manager.force_degradation_level(
                            DegradationLevel.MINIMAL, f"cache_failure_{operation}"
                        )

                    return None
                else:
                    raise

        return wrapper

    return decorator


def handle_validation_errors(
    operation: str = "validation_operation",
    component: str = "validator",
    return_default: Any = None,
    log_validation_failures: bool = True,
):
    """
    Decorator for handling validation errors.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        return_default: Default value to return on validation failure
        log_validation_failures: Whether to log validation failures

    Example:
        @handle_validation_errors("validate_user_input", "input_validator",
                                  return_default={})
        def validate_user_input(self, user_input):
            return self._validate_user_input_impl(user_input)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            request_id = kwargs.get("request_id")

            try:
                with error_context(
                    operation=operation,
                    component=component,
                    category=ErrorCategory.VALIDATION,
                    user_id=user_id,
                    request_id=request_id,
                ):
                    return func(*args, **kwargs)

            except ValidationError as e:
                if log_validation_failures:
                    logger.warning(f"Validation failed for {operation}: {e}")

                return return_default

        return wrapper

    return decorator


def monitor_performance(
    operation: str = "operation",
    component: str = "component",
    log_slow_operations: bool = True,
    slow_threshold: float = 5.0,
    track_metrics: bool = True,
):
    """
    Decorator for monitoring operation performance and health.

    Args:
        operation: Operation name for logging
        component: Component name for logging
        log_slow_operations: Whether to log slow operations
        slow_threshold: Threshold in seconds for slow operation logging
        track_metrics: Whether to track performance metrics

    Example:
        @monitor_performance("encode_batch", "embedding_service", slow_threshold=2.0)
        def encode_batch(self, texts):
            return self._encode_batch_impl(texts)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Log slow operations
                if log_slow_operations and duration > slow_threshold:
                    logger.warning(
                        f"Slow operation detected: {operation} in {component} "
                        f"took {duration:.2f}s (threshold: {slow_threshold}s)"
                    )

                # Track metrics
                if track_metrics:
                    health_monitor = get_health_monitor()
                    # In a full implementation, this would track detailed metrics
                    logger.debug(f"Operation {operation} completed in {duration:.3f}s")

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation {operation} in {component} failed after {duration:.3f}s: {e}"
                )

                # Track error metrics
                if track_metrics:
                    health_monitor = get_health_monitor()
                    health_monitor.record_error(component, e)

                raise

        return wrapper

    return decorator


# Convenience decorator combinations
def robust_vectorization_operation(
    operation: str,
    component: str = "vectorization_service",
    max_retries: int = 3,
    enable_monitoring: bool = True,
):
    """
    Comprehensive decorator for robust vectorization operations.

    Combines error handling, retry logic, and performance monitoring.
    """

    def decorator(func: Callable) -> Callable:
        # Apply multiple decorators
        decorated_func = func

        # Add performance monitoring
        if enable_monitoring:
            decorated_func = monitor_performance(operation, component)(decorated_func)

        # Add error handling with retry
        retry_config = (
            RetryConfig(max_attempts=max_retries) if max_retries > 1 else None
        )
        decorated_func = handle_vectorization_errors(
            operation, component, retry_config
        )(decorated_func)

        return decorated_func

    return decorator


def robust_companion_operation(
    operation: str, component: str = "companion_manager", enable_monitoring: bool = True
):
    """
    Comprehensive decorator for robust companion operations.

    Combines companion error handling with graceful degradation.
    """

    def decorator(func: Callable) -> Callable:
        decorated_func = func

        # Add performance monitoring
        if enable_monitoring:
            decorated_func = monitor_performance(operation, component)(decorated_func)

        # Add companion error handling
        decorated_func = handle_companion_errors(operation, component)(decorated_func)

        return decorated_func

    return decorator


if __name__ == "__main__":
    # Demo error handling decorators
    print("🎯 Morgan RAG Error Decorators Demo")
    print("=" * 40)

    # Example usage of decorators
    class DemoService:

        @handle_embedding_errors("demo_encode", "demo_service")
        def encode_text(self, text: str):
            # Simulate embedding operation
            if "fail" in text.lower():
                raise EmbeddingError("Simulated embedding failure")
            return [0.1, 0.2, 0.3]

        @handle_companion_errors("demo_companion", "demo_service")
        def update_companion_data(self, user_id: str, data: dict):
            # Simulate companion operation
            if data.get("force_error"):
                raise CompanionError("Simulated companion failure")
            return {"status": "updated"}

        @monitor_performance("demo_slow", "demo_service", slow_threshold=1.0)
        def slow_operation(self, delay: float = 0.5):
            import time

            time.sleep(delay)
            return "completed"

    # Test the decorated methods
    service = DemoService()

    # Test successful operation
    try:
        result = service.encode_text("hello world")
        print(f"Encoding successful: {result}")
    except Exception as e:
        print(f"Encoding failed: {e}")

    # Test failed operation with error handling
    try:
        result = service.encode_text("fail this operation")
        print(f"Encoding result: {result}")
    except Exception as e:
        print(f"Encoding failed as expected: {e}")

    # Test companion operation
    try:
        result = service.update_companion_data("user123", {"mood": "happy"})
        print(f"Companion update successful: {result}")
    except Exception as e:
        print(f"Companion update failed: {e}")

    # Test performance monitoring
    result = service.slow_operation(0.2)  # Fast operation
    print(f"Fast operation: {result}")

    result = service.slow_operation(1.5)  # Slow operation
    print(f"Slow operation: {result}")

    print("\n" + "=" * 40)
    print("Error decorators demo completed!")
