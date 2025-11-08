"""
Production-grade error handling decorators for Morgan AI Assistant

Provides:
- Retry decorators with exponential backoff
- Circuit breaker pattern
- Error recovery strategies
- Error aggregation for batch operations
- Correlation ID propagation
"""
import asyncio
import functools
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple, List, Dict
from enum import Enum

from shared.utils.exceptions import (
    MorganException,
    ErrorSeverity,
    NetworkTimeoutError,
    ServiceTimeoutError,
    ServiceUnavailableError
)


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    Prevents cascading failures by:
    - Tracking failure rate
    - Opening circuit after threshold
    - Allowing periodic test requests
    - Closing circuit when service recovers
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}': HALF_OPEN (testing recovery)")
            else:
                raise ServiceUnavailableError(
                    service_name=self.name,
                    message=f"Circuit breaker is OPEN (too many failures)",
                    context={"circuit_breaker": self.name}
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}': HALF_OPEN (testing recovery)")
            else:
                raise ServiceUnavailableError(
                    service_name=self.name,
                    message=f"Circuit breaker is OPEN (too many failures)",
                    context={"circuit_breaker": self.name}
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}': CLOSED (service recovered)")
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker '{self.name}': OPEN "
                f"(failure threshold {self.failure_threshold} reached)"
            )


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2
) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            name=name
        )
    return _circuit_breakers[name]


def retry_on_transient_error(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff for transient errors

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback function called on each retry

    Example:
        @retry_on_transient_error(max_attempts=3, base_delay=1.0)
        async def fetch_data():
            # This will retry up to 3 times with exponential backoff
            return await external_api.get_data()
    """
    if retryable_exceptions is None:
        # Default: retry on all transient MorganException errors
        retryable_exceptions = (MorganException,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            attempt = 0

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e

                    # Check if exception is retryable
                    is_retryable = False
                    if isinstance(e, MorganException):
                        is_retryable = e.is_retryable
                    elif isinstance(e, retryable_exceptions):
                        is_retryable = True

                    if not is_retryable or attempt >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts: {e}",
                            extra={"correlation_id": getattr(e, "correlation_id", None)}
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                    logger.warning(
                        f"Retrying {func.__name__} (attempt {attempt}/{max_attempts}) "
                        f"after {delay:.2f}s due to: {e}",
                        extra={"correlation_id": getattr(e, "correlation_id", None)}
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            attempt = 0

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e

                    # Check if exception is retryable
                    is_retryable = False
                    if isinstance(e, MorganException):
                        is_retryable = e.is_retryable
                    elif isinstance(e, retryable_exceptions):
                        is_retryable = True

                    if not is_retryable or attempt >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts: {e}",
                            extra={"correlation_id": getattr(e, "correlation_id", None)}
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                    logger.warning(
                        f"Retrying {func.__name__} (attempt {attempt}/{max_attempts}) "
                        f"after {delay:.2f}s due to: {e}",
                        extra={"correlation_id": getattr(e, "correlation_id", None)}
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2
):
    """
    Circuit breaker decorator

    Args:
        name: Circuit breaker name (use service/operation identifier)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Number of successes needed to close circuit

    Example:
        @with_circuit_breaker(name="llm_service", failure_threshold=5)
        async def call_llm_service():
            return await llm_service.generate()
    """
    def decorator(func: Callable) -> Callable:
        circuit_breaker = get_circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await circuit_breaker.call_async(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return circuit_breaker.call(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def with_timeout(timeout_seconds: float, error_message: Optional[str] = None):
    """
    Timeout decorator for async functions

    Args:
        timeout_seconds: Maximum execution time in seconds
        error_message: Custom error message

    Example:
        @with_timeout(timeout_seconds=30.0)
        async def slow_operation():
            await asyncio.sleep(60)  # Will raise ServiceTimeoutError
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                msg = error_message or f"Operation timed out after {timeout_seconds}s"
                raise ServiceTimeoutError(
                    service_name=func.__name__,
                    timeout_seconds=timeout_seconds,
                    message=msg
                )

        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@with_timeout can only be applied to async functions")

        return wrapper

    return decorator


def handle_errors(
    logger_instance: Optional[logging.Logger] = None,
    default_error_message: str = "An error occurred",
    return_value_on_error: Any = None,
    suppress_errors: bool = False,
    transform_exception: Optional[Callable[[Exception], Exception]] = None
):
    """
    Error handling decorator with logging and transformation

    Args:
        logger_instance: Logger to use for error logging
        default_error_message: Default error message if not provided
        return_value_on_error: Value to return on error (if suppress_errors=True)
        suppress_errors: If True, suppress errors and return default value
        transform_exception: Function to transform exceptions

    Example:
        @handle_errors(logger_instance=my_logger, suppress_errors=True, return_value_on_error={})
        async def fetch_optional_data():
            return await external_service.get_data()
    """
    log = logger_instance or logger

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log the error
                correlation_id = getattr(e, "correlation_id", "none")
                log.error(
                    f"Error in {func.__name__}: {e}",
                    exc_info=True,
                    extra={"correlation_id": correlation_id}
                )

                # Transform exception if transformer provided
                if transform_exception:
                    e = transform_exception(e)

                # Suppress or re-raise
                if suppress_errors:
                    return return_value_on_error
                else:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                correlation_id = getattr(e, "correlation_id", "none")
                log.error(
                    f"Error in {func.__name__}: {e}",
                    exc_info=True,
                    extra={"correlation_id": correlation_id}
                )

                # Transform exception if transformer provided
                if transform_exception:
                    e = transform_exception(e)

                # Suppress or re-raise
                if suppress_errors:
                    return return_value_on_error
                else:
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def aggregate_errors(operation_name: str = "batch_operation"):
    """
    Decorator for batch operations that aggregates errors

    Collects all errors from batch operations and raises an
    AggregateError containing all failures.

    Example:
        @aggregate_errors(operation_name="process_batch")
        async def process_items(items):
            results = []
            errors = []
            for item in items:
                try:
                    result = await process_item(item)
                    results.append(result)
                except Exception as e:
                    errors.append((item, e))
            return results, errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                result = await func(*args, **kwargs)

                # Check if result contains errors (tuple of results and errors)
                if isinstance(result, tuple) and len(result) == 2:
                    results, errors = result
                    if errors:
                        logger.warning(
                            f"{operation_name}: {len(errors)} errors occurred in batch operation",
                            extra={"error_count": len(errors)}
                        )
                        # Return results and errors for caller to handle
                        return results, errors

                return result
            except Exception as e:
                logger.error(f"{operation_name}: Batch operation failed completely: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                # Check if result contains errors (tuple of results and errors)
                if isinstance(result, tuple) and len(result) == 2:
                    results, errors = result
                    if errors:
                        logger.warning(
                            f"{operation_name}: {len(errors)} errors occurred in batch operation",
                            extra={"error_count": len(errors)}
                        )
                        return results, errors

                return result
            except Exception as e:
                logger.error(f"{operation_name}: Batch operation failed completely: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def with_fallback(fallback_func: Callable, log_fallback: bool = True):
    """
    Fallback decorator - executes fallback function on error

    Args:
        fallback_func: Function to call on error
        log_fallback: Whether to log fallback execution

    Example:
        async def fallback_response():
            return "Service temporarily unavailable"

        @with_fallback(fallback_response)
        async def get_llm_response():
            return await llm_service.generate()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"Using fallback for {func.__name__} due to error: {e}",
                        extra={"correlation_id": getattr(e, "correlation_id", None)}
                    )

                # Execute fallback
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"Using fallback for {func.__name__} due to error: {e}",
                        extra={"correlation_id": getattr(e, "correlation_id", None)}
                    )

                # Execute fallback
                return fallback_func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def propagate_correlation_id(func: Callable) -> Callable:
    """
    Decorator to propagate correlation ID through function calls

    Extracts correlation_id from kwargs and ensures it's passed to
    all MorganException instances.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        correlation_id = kwargs.get("correlation_id")

        try:
            return await func(*args, **kwargs)
        except MorganException as e:
            # Update correlation ID if not set
            if not e.correlation_id and correlation_id:
                e.correlation_id = correlation_id
            raise
        except Exception as e:
            # Wrap non-Morgan exceptions with correlation ID
            if correlation_id:
                raise MorganException(
                    message=str(e),
                    correlation_id=correlation_id,
                    cause=e
                )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        correlation_id = kwargs.get("correlation_id")

        try:
            return func(*args, **kwargs)
        except MorganException as e:
            # Update correlation ID if not set
            if not e.correlation_id and correlation_id:
                e.correlation_id = correlation_id
            raise
        except Exception as e:
            # Wrap non-Morgan exceptions with correlation ID
            if correlation_id:
                raise MorganException(
                    message=str(e),
                    correlation_id=correlation_id,
                    cause=e
                )
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
