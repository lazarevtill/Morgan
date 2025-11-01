"""
Production-grade error handling for Morgan RAG vectorization system.

Provides comprehensive error management including:
- Custom exception hierarchy for different error types
- Retry logic with exponential backoff and jitter
- Graceful degradation for companion features
- Error recovery procedures for emotional processing
- Circuit breaker pattern for external services
- Error context tracking and structured logging
"""

import time
import random
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different types of failures."""
    VECTORIZATION = "vectorization"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    SEARCH = "search"
    CACHE = "cache"
    NETWORK = "network"
    COMPANION = "companion"
    EMOTIONAL = "emotional"
    MEMORY = "memory"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring."""
    error_id: str
    timestamp: datetime
    operation: str
    component: str
    category: ErrorCategory
    severity: ErrorSeverity
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            "category": self.category.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful
        }


# ========================================
# Custom Exception Hierarchy
# ========================================

class MorganError(Exception):
    """Base exception for all Morgan RAG errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.VECTORIZATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        operation: str = "unknown",
        component: str = "unknown",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.operation = operation
        self.component = component
        self.user_id = user_id
        self.request_id = request_id
        self.metadata = metadata or {}
        self.cause = cause
        self.error_id = self._generate_error_id()
        self.timestamp = datetime.utcnow()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return f"morgan_{self.category.value}_{uuid.uuid4().hex[:8]}"
    
    def get_context(self) -> ErrorContext:
        """Get structured error context."""
        import traceback
        
        return ErrorContext(
            error_id=self.error_id,
            timestamp=self.timestamp,
            operation=self.operation,
            component=self.component,
            category=self.category,
            severity=self.severity,
            user_id=self.user_id,
            request_id=self.request_id,
            metadata=self.metadata,
            stack_trace=traceback.format_exc() if self.cause else None
        )


class VectorizationError(MorganError):
    """Errors related to document vectorization operations."""
    
    def __init__(self, message: str, **kwargs):
        # Set default component if not provided
        if 'component' not in kwargs:
            kwargs['component'] = "vectorization_service"
        super().__init__(
            message,
            category=ErrorCategory.VECTORIZATION,
            **kwargs
        )


class EmbeddingError(MorganError):
    """Errors related to embedding generation."""
    
    def __init__(self, message: str, **kwargs):
        # Set default component if not provided
        if 'component' not in kwargs:
            kwargs['component'] = "embedding_service"
        super().__init__(
            message,
            category=ErrorCategory.EMBEDDING,
            **kwargs
        )


class StorageError(MorganError):
    """Errors related to vector database operations."""
    
    def __init__(self, message: str, **kwargs):
        # Set default component if not provided
        if 'component' not in kwargs:
            kwargs['component'] = "vector_db_client"
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            **kwargs
        )


class SearchError(MorganError):
    """Errors related to search operations."""
    
    def __init__(self, message: str, **kwargs):
        # Set default component if not provided
        if 'component' not in kwargs:
            kwargs['component'] = "search_engine"
        super().__init__(
            message,
            category=ErrorCategory.SEARCH,
            **kwargs
        )


class CacheError(MorganError):
    """Errors related to caching operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CACHE,
            component="cache_manager",
            **kwargs
        )


class NetworkError(MorganError):
    """Errors related to network operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class CompanionError(MorganError):
    """Errors related to companion features."""
    
    def __init__(self, message: str, **kwargs):
        # Set default component if not provided
        if 'component' not in kwargs:
            kwargs['component'] = "companion_manager"
        super().__init__(
            message,
            category=ErrorCategory.COMPANION,
            **kwargs
        )


class EmotionalProcessingError(MorganError):
    """Errors related to emotional intelligence processing."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EMOTIONAL,
            component="emotional_intelligence",
            **kwargs
        )


class MemoryProcessingError(MorganError):
    """Errors related to memory processing operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            component="memory_processor",
            **kwargs
        )


class ValidationError(MorganError):
    """Errors related to data validation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class ConfigurationError(MorganError):
    """Errors related to system configuration."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


# ========================================
# Retry Logic with Exponential Backoff
# ========================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.3
    retryable_exceptions: tuple = (NetworkError, StorageError, EmbeddingError)
    non_retryable_exceptions: tuple = (ValidationError, ConfigurationError)


class RetryExhaustedError(MorganError):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, original_error: Exception, attempts: int, **kwargs):
        message = f"Retry exhausted after {attempts} attempts: {original_error}"
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            metadata={"original_error": str(original_error), "attempts": attempts},
            cause=original_error,
            **kwargs
        )


def with_retry(
    config: Optional[RetryConfig] = None,
    operation: str = "unknown",
    component: str = "unknown"
):
    """
    Decorator for adding retry logic with exponential backoff and jitter.
    
    Args:
        config: Retry configuration (uses default if None)
        operation: Operation name for logging
        component: Component name for logging
        
    Example:
        @with_retry(RetryConfig(max_attempts=5), "encode_batch", "embedding_service")
        def encode_batch(self, texts):
            return self._encode_batch_remote(texts)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except config.non_retryable_exceptions as e:
                    # Don't retry these exceptions
                    logger.error(
                        f"Non-retryable error in {operation} (component: {component}): {e}"
                    )
                    raise
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        jitter_amount = delay * config.jitter_range
                        delay += random.uniform(-jitter_amount, jitter_amount)
                        delay = max(0, delay)  # Ensure non-negative
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {operation} "
                        f"(component: {component}), retrying in {delay:.2f}s: {e}"
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(
                        f"Unexpected error in {operation} (component: {component}): {e}"
                    )
                    raise
            
            # All retries exhausted
            raise RetryExhaustedError(
                original_error=last_exception,
                attempts=config.max_attempts,
                operation=operation,
                component=component
            )
        
        return wrapper
    return decorator


# ========================================
# Circuit Breaker Pattern
# ========================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreakerError(MorganError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, **kwargs):
        message = f"Circuit breaker open for service: {service_name}"
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            metadata={"service_name": service_name},
            **kwargs
        )


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    Prevents cascading failures by temporarily stopping calls to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise CircuitBreakerError(
                        service_name=self.name,
                        operation="circuit_breaker_call",
                        component="circuit_breaker"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (
            datetime.utcnow() - self.last_failure_time
        ).total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} reopened due to failure")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# ========================================
# Graceful Degradation Manager
# ========================================

class DegradationLevel(Enum):
    """Levels of service degradation."""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    companion_features_enabled: bool = True
    emotional_processing_enabled: bool = True
    memory_processing_enabled: bool = True
    advanced_search_enabled: bool = True
    caching_enabled: bool = True
    
    def apply_degradation(self, level: DegradationLevel) -> 'DegradationConfig':
        """Apply degradation level to configuration."""
        if level == DegradationLevel.MINIMAL:
            return DegradationConfig(
                companion_features_enabled=True,
                emotional_processing_enabled=True,
                memory_processing_enabled=True,
                advanced_search_enabled=False,  # Disable advanced search
                caching_enabled=True
            )
        elif level == DegradationLevel.MODERATE:
            return DegradationConfig(
                companion_features_enabled=True,
                emotional_processing_enabled=False,  # Disable emotional processing
                memory_processing_enabled=True,
                advanced_search_enabled=False,
                caching_enabled=True
            )
        elif level == DegradationLevel.SEVERE:
            return DegradationConfig(
                companion_features_enabled=False,  # Disable companion features
                emotional_processing_enabled=False,
                memory_processing_enabled=False,  # Disable memory processing
                advanced_search_enabled=False,
                caching_enabled=True
            )
        elif level == DegradationLevel.CRITICAL:
            return DegradationConfig(
                companion_features_enabled=False,
                emotional_processing_enabled=False,
                memory_processing_enabled=False,
                advanced_search_enabled=False,
                caching_enabled=False  # Disable all caching
            )
        else:
            return self


class GracefulDegradationManager:
    """
    Manages graceful degradation of Morgan's features during failures.
    
    Automatically disables non-essential features to maintain core functionality
    when system resources are constrained or services are failing.
    """
    
    def __init__(self):
        self.current_level = DegradationLevel.NONE
        self.config = DegradationConfig()
        self.degradation_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def assess_and_apply_degradation(
        self,
        error_context: ErrorContext,
        system_metrics: Optional[Dict[str, Any]] = None
    ) -> DegradationLevel:
        """
        Assess system state and apply appropriate degradation level.
        
        Args:
            error_context: Current error context
            system_metrics: Optional system performance metrics
            
        Returns:
            Applied degradation level
        """
        with self._lock:
            # Determine appropriate degradation level
            new_level = self._calculate_degradation_level(error_context, system_metrics)
            
            if new_level != self.current_level:
                self._apply_degradation(new_level, error_context)
            
            return new_level
    
    def _calculate_degradation_level(
        self,
        error_context: ErrorContext,
        system_metrics: Optional[Dict[str, Any]] = None
    ) -> DegradationLevel:
        """Calculate appropriate degradation level based on error and metrics."""
        # Start with current level
        suggested_level = self.current_level
        
        # Escalate based on error severity and category
        if error_context.severity == ErrorSeverity.CRITICAL:
            suggested_level = DegradationLevel.SEVERE
        elif error_context.severity == ErrorSeverity.HIGH:
            if error_context.category in [ErrorCategory.STORAGE, ErrorCategory.NETWORK]:
                suggested_level = DegradationLevel.MODERATE
            else:
                suggested_level = DegradationLevel.MINIMAL
        
        # Consider system metrics if available
        if system_metrics:
            memory_usage = system_metrics.get("memory_usage_percent", 0)
            cpu_usage = system_metrics.get("cpu_usage_percent", 0)
            error_rate = system_metrics.get("error_rate_percent", 0)
            
            if memory_usage > 90 or cpu_usage > 95 or error_rate > 50:
                suggested_level = max(suggested_level, DegradationLevel.SEVERE)
            elif memory_usage > 80 or cpu_usage > 85 or error_rate > 25:
                suggested_level = max(suggested_level, DegradationLevel.MODERATE)
            elif memory_usage > 70 or cpu_usage > 75 or error_rate > 10:
                suggested_level = max(suggested_level, DegradationLevel.MINIMAL)
        
        return suggested_level
    
    def _apply_degradation(self, level: DegradationLevel, error_context: ErrorContext):
        """Apply degradation level and update configuration."""
        old_level = self.current_level
        self.current_level = level
        self.config = DegradationConfig().apply_degradation(level)
        
        # Record degradation event
        degradation_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "old_level": old_level.value,
            "new_level": level.value,
            "trigger_error": error_context.error_id,
            "trigger_category": error_context.category.value,
            "trigger_severity": error_context.severity.value
        }
        
        self.degradation_history.append(degradation_event)
        
        # Keep only last 100 events
        if len(self.degradation_history) > 100:
            self.degradation_history = self.degradation_history[-100:]
        
        logger.warning(
            f"Graceful degradation applied: {old_level.value} -> {level.value} "
            f"(trigger: {error_context.error_id})"
        )
    
    def get_current_config(self) -> DegradationConfig:
        """Get current degradation configuration."""
        return self.config
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is currently enabled."""
        feature_map = {
            "companion": self.config.companion_features_enabled,
            "emotional": self.config.emotional_processing_enabled,
            "memory": self.config.memory_processing_enabled,
            "advanced_search": self.config.advanced_search_enabled,
            "caching": self.config.caching_enabled
        }
        
        return feature_map.get(feature, True)
    
    def force_degradation_level(self, level: DegradationLevel, reason: str = "manual"):
        """Manually force a specific degradation level."""
        with self._lock:
            old_level = self.current_level
            self.current_level = level
            self.config = DegradationConfig().apply_degradation(level)
            
            logger.info(f"Forced degradation: {old_level.value} -> {level.value} (reason: {reason})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            "current_level": self.current_level.value,
            "config": {
                "companion_features_enabled": self.config.companion_features_enabled,
                "emotional_processing_enabled": self.config.emotional_processing_enabled,
                "memory_processing_enabled": self.config.memory_processing_enabled,
                "advanced_search_enabled": self.config.advanced_search_enabled,
                "caching_enabled": self.config.caching_enabled
            },
            "recent_events": self.degradation_history[-10:] if self.degradation_history else []
        }


# ========================================
# Error Recovery Procedures
# ========================================

class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    SKIP = "skip"
    MANUAL = "manual"


@dataclass
class RecoveryProcedure:
    """Definition of an error recovery procedure."""
    name: str
    strategy: RecoveryStrategy
    applicable_errors: List[Type[MorganError]]
    recovery_function: Callable
    max_attempts: int = 1
    timeout: float = 30.0
    description: str = ""


class ErrorRecoveryManager:
    """
    Manages error recovery procedures for different types of failures.
    
    Automatically attempts recovery based on error type and context,
    with fallback strategies for graceful degradation.
    """
    
    def __init__(self):
        self.procedures: Dict[str, RecoveryProcedure] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self._register_default_procedures()
    
    def register_procedure(self, procedure: RecoveryProcedure):
        """Register a recovery procedure."""
        self.procedures[procedure.name] = procedure
        logger.debug(f"Registered recovery procedure: {procedure.name}")
    
    def attempt_recovery(
        self,
        error: MorganError,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt recovery for the given error.
        
        Args:
            error: The error to recover from
            context: Additional context for recovery
            
        Returns:
            True if recovery was successful
        """
        applicable_procedures = self._find_applicable_procedures(error)
        
        if not applicable_procedures:
            logger.warning(f"No recovery procedures found for error: {error.error_id}")
            return False
        
        for procedure in applicable_procedures:
            try:
                logger.info(f"Attempting recovery with procedure: {procedure.name}")
                
                recovery_start = datetime.utcnow()
                success = self._execute_recovery_procedure(procedure, error, context)
                recovery_duration = (datetime.utcnow() - recovery_start).total_seconds()
                
                # Record recovery attempt
                recovery_record = {
                    "timestamp": recovery_start.isoformat(),
                    "error_id": error.error_id,
                    "procedure_name": procedure.name,
                    "strategy": procedure.strategy.value,
                    "success": success,
                    "duration": recovery_duration
                }
                
                self.recovery_history.append(recovery_record)
                
                # Keep only last 1000 records
                if len(self.recovery_history) > 1000:
                    self.recovery_history = self.recovery_history[-1000:]
                
                if success:
                    logger.info(
                        f"Recovery successful with procedure {procedure.name} "
                        f"for error {error.error_id} in {recovery_duration:.2f}s"
                    )
                    return True
                else:
                    logger.warning(
                        f"Recovery failed with procedure {procedure.name} "
                        f"for error {error.error_id}"
                    )
                    
            except Exception as recovery_error:
                logger.error(
                    f"Recovery procedure {procedure.name} raised exception: {recovery_error}"
                )
        
        logger.error(f"All recovery attempts failed for error: {error.error_id}")
        return False
    
    def _find_applicable_procedures(self, error: MorganError) -> List[RecoveryProcedure]:
        """Find recovery procedures applicable to the given error."""
        applicable = []
        
        for procedure in self.procedures.values():
            if any(isinstance(error, error_type) for error_type in procedure.applicable_errors):
                applicable.append(procedure)
        
        # Sort by strategy priority (retry first, then fallback, etc.)
        strategy_priority = {
            RecoveryStrategy.RETRY: 1,
            RecoveryStrategy.FALLBACK: 2,
            RecoveryStrategy.DEGRADE: 3,
            RecoveryStrategy.SKIP: 4,
            RecoveryStrategy.MANUAL: 5
        }
        
        applicable.sort(key=lambda p: strategy_priority.get(p.strategy, 999))
        return applicable
    
    def _execute_recovery_procedure(
        self,
        procedure: RecoveryProcedure,
        error: MorganError,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Execute a specific recovery procedure."""
        try:
            # Set timeout for recovery procedure
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Recovery procedure {procedure.name} timed out")
            
            # Set timeout (only on Unix systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(procedure.timeout))
            
            try:
                # Execute recovery function
                result = procedure.recovery_function(error, context or {})
                return bool(result)
                
            finally:
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    
        except TimeoutError:
            logger.error(f"Recovery procedure {procedure.name} timed out after {procedure.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Recovery procedure {procedure.name} failed: {e}")
            return False
    
    def _register_default_procedures(self):
        """Register default recovery procedures for common error types."""
        
        # Embedding service recovery
        def recover_embedding_service(error: EmbeddingError, context: Dict[str, Any]) -> bool:
            """Attempt to recover embedding service by switching to fallback."""
            try:
                from morgan.services.embedding_service import get_embedding_service
                service = get_embedding_service()
                
                # Force check of service availability
                service._remote_available = None
                service._local_available = None
                
                return service.is_available()
            except Exception:
                return False
        
        self.register_procedure(RecoveryProcedure(
            name="embedding_service_fallback",
            strategy=RecoveryStrategy.FALLBACK,
            applicable_errors=[EmbeddingError],
            recovery_function=recover_embedding_service,
            description="Switch to local embedding model when remote fails"
        ))
        
        # Vector database recovery
        def recover_vector_db(error: StorageError, context: Dict[str, Any]) -> bool:
            """Attempt to recover vector database connection."""
            try:
                from morgan.vector_db.client import VectorDBClient
                client = VectorDBClient()
                return client.health_check()
            except Exception:
                return False
        
        self.register_procedure(RecoveryProcedure(
            name="vector_db_reconnect",
            strategy=RecoveryStrategy.RETRY,
            applicable_errors=[StorageError],
            recovery_function=recover_vector_db,
            max_attempts=3,
            description="Reconnect to vector database"
        ))
        
        # Companion feature recovery
        def recover_companion_features(error: CompanionError, context: Dict[str, Any]) -> bool:
            """Recover companion features by graceful degradation."""
            try:
                # Get degradation manager and apply minimal degradation
                degradation_manager = get_degradation_manager()
                degradation_manager.force_degradation_level(
                    DegradationLevel.MINIMAL,
                    "companion_error_recovery"
                )
                return True
            except Exception:
                return False
        
        self.register_procedure(RecoveryProcedure(
            name="companion_graceful_degradation",
            strategy=RecoveryStrategy.DEGRADE,
            applicable_errors=[CompanionError, EmotionalProcessingError],
            recovery_function=recover_companion_features,
            description="Gracefully degrade companion features"
        ))
        
        # Memory processing recovery
        def recover_memory_processing(error: MemoryProcessingError, context: Dict[str, Any]) -> bool:
            """Recover memory processing by skipping current operation."""
            try:
                # Log the memory processing failure and continue
                logger.warning(f"Skipping memory processing due to error: {error.error_id}")
                return True
            except Exception:
                return False
        
        self.register_procedure(RecoveryProcedure(
            name="memory_processing_skip",
            strategy=RecoveryStrategy.SKIP,
            applicable_errors=[MemoryProcessingError],
            recovery_function=recover_memory_processing,
            description="Skip failed memory processing operations"
        ))
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for record in self.recovery_history if record["success"])
        success_rate = (successful_attempts / total_attempts) * 100
        
        # Group by procedure
        procedure_stats = {}
        for record in self.recovery_history:
            proc_name = record["procedure_name"]
            if proc_name not in procedure_stats:
                procedure_stats[proc_name] = {"attempts": 0, "successes": 0}
            
            procedure_stats[proc_name]["attempts"] += 1
            if record["success"]:
                procedure_stats[proc_name]["successes"] += 1
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "procedure_stats": procedure_stats,
            "recent_attempts": self.recovery_history[-10:]
        }


# ========================================
# Context Managers and Utilities
# ========================================

@contextmanager
def error_context(
    operation: str,
    component: str,
    category: ErrorCategory = ErrorCategory.VECTORIZATION,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for comprehensive error handling and logging.
    
    Args:
        operation: Operation being performed
        component: Component performing the operation
        category: Error category
        user_id: Optional user ID
        request_id: Optional request ID
        metadata: Optional metadata
        
    Example:
        with error_context("encode_batch", "embedding_service", ErrorCategory.EMBEDDING):
            embeddings = service.encode_batch(texts)
    """
    start_time = datetime.utcnow()
    
    try:
        yield
        
    except MorganError as e:
        # Morgan error - already has context, just log and re-raise
        logger.error(f"Morgan error in {operation}: {e.get_context().to_dict()}")
        raise
        
    except Exception as e:
        # Convert to Morgan error with context
        morgan_error = MorganError(
            message=str(e),
            category=category,
            operation=operation,
            component=component,
            user_id=user_id,
            request_id=request_id,
            metadata=metadata,
            cause=e
        )
        
        # Log structured error
        logger.error(f"Error in {operation}: {morgan_error.get_context().to_dict()}")
        
        # Attempt recovery
        recovery_manager = get_recovery_manager()
        recovery_successful = recovery_manager.attempt_recovery(morgan_error)
        
        if not recovery_successful:
            # Apply graceful degradation
            degradation_manager = get_degradation_manager()
            degradation_manager.assess_and_apply_degradation(morgan_error.get_context())
        
        raise morgan_error from e
    
    finally:
        # Log operation completion time
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.debug(f"Operation {operation} completed in {duration:.3f}s")


# ========================================
# Singleton Instances
# ========================================

_degradation_manager_instance = None
_recovery_manager_instance = None
_degradation_lock = threading.Lock()
_recovery_lock = threading.Lock()


def get_degradation_manager() -> GracefulDegradationManager:
    """Get singleton graceful degradation manager instance."""
    global _degradation_manager_instance
    
    if _degradation_manager_instance is None:
        with _degradation_lock:
            if _degradation_manager_instance is None:
                _degradation_manager_instance = GracefulDegradationManager()
    
    return _degradation_manager_instance


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get singleton error recovery manager instance."""
    global _recovery_manager_instance
    
    if _recovery_manager_instance is None:
        with _recovery_lock:
            if _recovery_manager_instance is None:
                _recovery_manager_instance = ErrorRecoveryManager()
    
    return _recovery_manager_instance


# ========================================
# Monitoring and Health Checks
# ========================================

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemHealthMonitor:
    """
    Monitors system health and provides health check endpoints.
    
    Tracks component health, error rates, and system performance
    to provide comprehensive health status.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_health_check = None
        self._lock = threading.Lock()
    
    def register_health_check(
        self,
        component: str,
        check_function: Callable[[], bool],
        timeout: float = 10.0
    ):
        """Register a health check function for a component."""
        try:
            # Execute health check with timeout
            start_time = datetime.utcnow()
            
            try:
                is_healthy = check_function()
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                message = "OK" if is_healthy else "Health check failed"
                
            except Exception as e:
                status = HealthStatus.CRITICAL
                message = f"Health check error: {e}"
            
            # Record health check result
            health_check = HealthCheck(
                component=component,
                status=status,
                message=message,
                timestamp=start_time,
                metadata={"duration": (datetime.utcnow() - start_time).total_seconds()}
            )
            
            with self._lock:
                self.health_checks[component] = health_check
            
            logger.debug(f"Health check for {component}: {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to execute health check for {component}: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self._lock:
            if not self.health_checks:
                return {
                    "overall_status": HealthStatus.UNKNOWN.value,
                    "message": "No health checks registered",
                    "components": {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Determine overall status
            component_statuses = [check.status for check in self.health_checks.values()]
            
            if all(status == HealthStatus.HEALTHY for status in component_statuses):
                overall_status = HealthStatus.HEALTHY
            elif any(status == HealthStatus.CRITICAL for status in component_statuses):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.UNHEALTHY for status in component_statuses):
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
            
            # Get degradation status
            degradation_manager = get_degradation_manager()
            degradation_status = degradation_manager.get_status()
            
            # Get recovery statistics
            recovery_manager = get_recovery_manager()
            recovery_stats = recovery_manager.get_recovery_stats()
            
            return {
                "overall_status": overall_status.value,
                "message": f"System is {overall_status.value}",
                "components": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "metadata": check.metadata
                    }
                    for name, check in self.health_checks.items()
                },
                "degradation": degradation_status,
                "recovery": recovery_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def record_error(self, component: str, error: MorganError):
        """Record an error for monitoring."""
        with self._lock:
            self.error_counts[component] = self.error_counts.get(component, 0) + 1
    
    def get_error_rates(self) -> Dict[str, int]:
        """Get error counts by component."""
        with self._lock:
            return self.error_counts.copy()


# Global health monitor instance
_health_monitor = SystemHealthMonitor()


def get_health_monitor() -> SystemHealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


# ========================================
# Initialization and Setup
# ========================================

def initialize_error_handling():
    """Initialize error handling system with default configurations."""
    logger.info("Initializing Morgan RAG error handling system")
    
    # Initialize managers
    degradation_manager = get_degradation_manager()
    recovery_manager = get_recovery_manager()
    health_monitor = get_health_monitor()
    
    # Register default health checks
    def check_embedding_service():
        try:
            from morgan.services.embedding_service import get_embedding_service
            service = get_embedding_service()
            return service.is_available()
        except Exception:
            return False
    
    def check_vector_db():
        try:
            from morgan.vector_db.client import VectorDBClient
            client = VectorDBClient()
            return client.health_check()
        except Exception:
            return False
    
    health_monitor.register_health_check("embedding_service", check_embedding_service)
    health_monitor.register_health_check("vector_database", check_vector_db)
    
    logger.info("Error handling system initialized successfully")


if __name__ == "__main__":
    # Demo error handling capabilities
    print("🛡️ Morgan RAG Error Handling Demo")
    print("=" * 50)
    
    # Initialize system
    initialize_error_handling()
    
    # Test custom exceptions
    try:
        raise EmbeddingError(
            "Remote embedding service unavailable",
            operation="encode_batch",
            user_id="demo_user",
            metadata={"batch_size": 100}
        )
    except MorganError as e:
        print(f"Caught Morgan error: {e.error_id}")
        print(f"Context: {e.get_context().to_dict()}")
    
    # Test retry decorator
    @with_retry(RetryConfig(max_attempts=3), "demo_operation", "demo_component")
    def failing_operation():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise NetworkError("Simulated network failure")
        return "Success!"
    
    try:
        result = failing_operation()
        print(f"Operation result: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")
    
    # Test circuit breaker
    circuit = CircuitBreaker("demo_service")
    
    def unreliable_service():
        import random
        if random.random() < 0.8:  # 80% failure rate
            raise Exception("Service unavailable")
        return "Service response"
    
    for i in range(10):
        try:
            result = circuit.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: {type(e).__name__}")
    
    print(f"Circuit state: {circuit.get_state()}")
    
    # Test graceful degradation
    degradation_manager = get_degradation_manager()
    print(f"Current degradation: {degradation_manager.get_status()}")
    
    # Test health monitoring
    health_monitor = get_health_monitor()
    health_status = health_monitor.get_system_health()
    print(f"System health: {health_status['overall_status']}")
    
    print("\n" + "=" * 50)
    print("Error handling demo completed!")