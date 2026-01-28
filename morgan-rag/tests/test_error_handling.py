"""
Tests for Morgan RAG error handling system.

Tests comprehensive error management including:
- Custom exception hierarchy
- Retry logic with exponential backoff
- Graceful degradation
- Error recovery procedures
- Circuit breaker pattern
- Performance monitoring
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from morgan.utils.error_handling import (
    MorganError,
    VectorizationError,
    EmbeddingError,
    StorageError,
    SearchError,
    CacheError,
    NetworkError,
    CompanionError,
    EmotionalProcessingError,
    MemoryProcessingError,
    ValidationError,
    ConfigurationError,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    RetryConfig,
    with_retry,
    RetryExhaustedError,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    CircuitBreakerConfig,
    GracefulDegradationManager,
    DegradationLevel,
    DegradationConfig,
    ErrorRecoveryManager,
    RecoveryStrategy,
    RecoveryProcedure,
    error_context,
    get_degradation_manager,
    get_recovery_manager,
    SystemHealthMonitor,
    HealthStatus,
    HealthCheck,
    initialize_error_handling,
)

from morgan.utils.error_decorators import (
    handle_vectorization_errors,
    handle_embedding_errors,
    handle_storage_errors,
    handle_search_errors,
    handle_companion_errors,
    handle_emotional_processing_errors,
    handle_memory_processing_errors,
    handle_cache_errors,
    handle_validation_errors,
    monitor_performance,
    robust_vectorization_operation,
    robust_companion_operation,
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_morgan_error_creation(self):
        """Test MorganError creation with context."""
        error = MorganError(
            "Test error message",
            category=ErrorCategory.EMBEDDING,
            severity=ErrorSeverity.HIGH,
            operation="test_operation",
            component="test_component",
            user_id="test_user",
            request_id="test_request",
            metadata={"key": "value"},
        )

        assert error.message == "Test error message"
        assert error.category == ErrorCategory.EMBEDDING
        assert error.severity == ErrorSeverity.HIGH
        assert error.operation == "test_operation"
        assert error.component == "test_component"
        assert error.user_id == "test_user"
        assert error.request_id == "test_request"
        assert error.metadata == {"key": "value"}
        assert error.error_id.startswith("morgan_embedding_")
        assert isinstance(error.timestamp, datetime)

    def test_error_context_generation(self):
        """Test error context generation."""
        error = EmbeddingError(
            "Embedding service failed", operation="encode_batch", user_id="user123"
        )

        context = error.get_context()

        assert isinstance(context, ErrorContext)
        assert context.error_id == error.error_id
        assert context.category == ErrorCategory.EMBEDDING
        assert context.operation == "encode_batch"
        assert context.component == "embedding_service"
        assert context.user_id == "user123"

    def test_specialized_exceptions(self):
        """Test specialized exception types."""
        # Test VectorizationError
        vec_error = VectorizationError("Vectorization failed")
        assert vec_error.category == ErrorCategory.VECTORIZATION
        assert vec_error.component == "vectorization_service"

        # Test StorageError
        storage_error = StorageError("Database connection failed")
        assert storage_error.category == ErrorCategory.STORAGE
        assert storage_error.component == "vector_db_client"

        # Test CompanionError
        companion_error = CompanionError("Companion feature failed")
        assert companion_error.category == ErrorCategory.COMPANION
        assert companion_error.component == "companion_manager"


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_retry_success_after_failures(self):
        """Test successful retry after initial failures."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3), "test_op", "test_component")
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Simulated network failure")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion after max attempts."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=2), "test_op", "test_component")
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_failing_function()

        assert call_count == 2
        assert "Retry exhausted after 2 attempts" in str(exc_info.value)

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3), "test_op", "test_component")
        def validation_error_function():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Invalid input")

        with pytest.raises(ValidationError):
            validation_error_function()

        assert call_count == 1  # Should not retry

    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable testing
        )

        call_times = []

        @with_retry(config, "test_op", "test_component")
        def timing_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise NetworkError("Timing test")
            return "success"

        start_time = time.time()
        result = timing_function()

        assert result == "success"
        assert len(call_times) == 3

        # Check delays (approximately 1s, 2s between calls)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.9 <= delay1 <= 1.1  # ~1 second
        assert 1.9 <= delay2 <= 2.1  # ~2 seconds


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        circuit = CircuitBreaker("test_service")

        def successful_operation():
            return "success"

        result = circuit.call(successful_operation)
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failure threshold."""
        circuit = CircuitBreaker("test_service")

        def failing_operation():
            raise Exception("Service failure")

        # Trigger failures to open circuit
        for _ in range(5):  # Default failure threshold is 5
            with pytest.raises(Exception):
                circuit.call(failing_operation)

        assert circuit.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            circuit.call(failing_operation)

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
            success_threshold=2,
        )
        circuit = CircuitBreaker("test_service", config)

        def operation(should_fail=True):
            if should_fail:
                raise Exception("Service failure")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(lambda: operation(True))

        assert circuit.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # First call after timeout should enter HALF_OPEN
        result = circuit.call(lambda: operation(False))
        assert result == "success"
        assert circuit.state == CircuitState.HALF_OPEN

        # Another successful call should close the circuit
        result = circuit.call(lambda: operation(False))
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED


class TestGracefulDegradation:
    """Test graceful degradation manager."""

    def test_degradation_level_application(self):
        """Test application of different degradation levels."""
        manager = GracefulDegradationManager()

        # Test minimal degradation
        manager.force_degradation_level(DegradationLevel.MINIMAL, "test")
        config = manager.get_current_config()

        assert config.companion_features_enabled is True
        assert config.emotional_processing_enabled is True
        assert config.advanced_search_enabled is False

        # Test severe degradation
        manager.force_degradation_level(DegradationLevel.SEVERE, "test")
        config = manager.get_current_config()

        assert config.companion_features_enabled is False
        assert config.emotional_processing_enabled is False
        assert config.memory_processing_enabled is False

    def test_feature_enablement_check(self):
        """Test feature enablement checking."""
        manager = GracefulDegradationManager()

        # Initially all features should be enabled
        assert manager.is_feature_enabled("companion") is True
        assert manager.is_feature_enabled("emotional") is True

        # Apply degradation and check again
        manager.force_degradation_level(DegradationLevel.MODERATE, "test")

        assert manager.is_feature_enabled("companion") is True
        assert manager.is_feature_enabled("emotional") is False

    def test_degradation_assessment(self):
        """Test automatic degradation assessment."""
        manager = GracefulDegradationManager()

        # Create high severity error
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.utcnow(),
            operation="test_operation",
            component="test_component",
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
        )

        # Assess degradation
        level = manager.assess_and_apply_degradation(error_context)

        assert level in [DegradationLevel.MINIMAL, DegradationLevel.MODERATE]
        assert manager.current_level == level


class TestErrorRecovery:
    """Test error recovery procedures."""

    def test_recovery_procedure_registration(self):
        """Test registration of recovery procedures."""
        manager = ErrorRecoveryManager()

        def test_recovery(error, context):
            return True

        procedure = RecoveryProcedure(
            name="test_recovery",
            strategy=RecoveryStrategy.RETRY,
            applicable_errors=[NetworkError],
            recovery_function=test_recovery,
        )

        manager.register_procedure(procedure)
        assert "test_recovery" in manager.procedures

    def test_successful_recovery(self):
        """Test successful error recovery."""
        manager = ErrorRecoveryManager()

        def successful_recovery(error, context):
            return True

        procedure = RecoveryProcedure(
            name="test_recovery",
            strategy=RecoveryStrategy.FALLBACK,
            applicable_errors=[NetworkError],
            recovery_function=successful_recovery,
        )

        manager.register_procedure(procedure)

        error = NetworkError("Network failure", operation="test")
        success = manager.attempt_recovery(error)

        assert success is True

    def test_failed_recovery(self):
        """Test failed error recovery."""
        manager = ErrorRecoveryManager()

        def failing_recovery(error, context):
            return False

        procedure = RecoveryProcedure(
            name="test_recovery",
            strategy=RecoveryStrategy.RETRY,
            applicable_errors=[NetworkError],
            recovery_function=failing_recovery,
        )

        manager.register_procedure(procedure)

        error = NetworkError("Network failure", operation="test")
        success = manager.attempt_recovery(error)

        assert success is False


class TestErrorDecorators:
    """Test error handling decorators."""

    def test_embedding_error_decorator(self):
        """Test embedding error handling decorator."""

        @handle_embedding_errors("test_encode", "test_service")
        def encode_function(text, should_fail=False):
            if should_fail:
                raise Exception("Encoding failed")
            return [0.1, 0.2, 0.3]

        # Test successful operation
        result = encode_function("test text")
        assert result == [0.1, 0.2, 0.3]

        # Test error handling - the decorator converts exceptions to MorganError
        with pytest.raises(MorganError):
            encode_function("test text", should_fail=True)

    def test_companion_error_decorator(self):
        """Test companion error handling decorator."""

        @handle_companion_errors("test_companion", "test_service")
        def companion_function(data, should_fail=False):
            if should_fail:
                raise CompanionError("Companion failed")
            return {"status": "success"}

        # Test successful operation
        result = companion_function({"test": "data"})
        assert result == {"status": "success"}

        # Test graceful failure (should return None)
        result = companion_function({"test": "data"}, should_fail=True)
        assert result is None

    def test_performance_monitoring_decorator(self):
        """Test performance monitoring decorator."""

        @monitor_performance("test_operation", "test_component", slow_threshold=0.1)
        def slow_operation(delay=0.05):
            time.sleep(delay)
            return "completed"

        # Test fast operation (should not log warning)
        result = slow_operation(0.05)
        assert result == "completed"

        # Test slow operation (would log warning in real scenario)
        result = slow_operation(0.15)
        assert result == "completed"

    def test_robust_operation_decorators(self):
        """Test combined robust operation decorators."""

        @robust_vectorization_operation("test_vectorization", max_retries=2)
        def vectorization_function(should_fail_count=0):
            if hasattr(vectorization_function, "call_count"):
                vectorization_function.call_count += 1
            else:
                vectorization_function.call_count = 1

            if vectorization_function.call_count <= should_fail_count:
                raise VectorizationError("Vectorization failed")

            return "vectorized"

        # Test successful operation
        result = vectorization_function()
        assert result == "vectorized"

        # Reset call count
        vectorization_function.call_count = 0

        # Test retry on failure - should raise error since VectorizationError is not retryable by default
        with pytest.raises(VectorizationError):
            vectorization_function(should_fail_count=1)


class TestErrorContext:
    """Test error context manager."""

    def test_successful_operation_context(self):
        """Test error context with successful operation."""

        with error_context(
            operation="test_operation",
            component="test_component",
            category=ErrorCategory.VECTORIZATION,
        ):
            result = "success"

        assert result == "success"

    def test_error_context_with_exception(self):
        """Test error context with exception handling."""

        with pytest.raises(MorganError) as exc_info:
            with error_context(
                operation="test_operation",
                component="test_component",
                category=ErrorCategory.EMBEDDING,
                user_id="test_user",
            ):
                raise ValueError("Test error")

        error = exc_info.value
        assert isinstance(error, MorganError)
        assert error.category == ErrorCategory.EMBEDDING
        assert error.operation == "test_operation"
        assert error.component == "test_component"
        assert error.user_id == "test_user"


class TestSystemHealthMonitor:
    """Test system health monitoring."""

    def test_health_check_registration(self):
        """Test health check registration and execution."""
        monitor = SystemHealthMonitor()

        def healthy_check():
            return True

        def unhealthy_check():
            return False

        monitor.register_health_check("service1", healthy_check)
        monitor.register_health_check("service2", unhealthy_check)

        health_status = monitor.get_system_health()

        assert "service1" in health_status["components"]
        assert "service2" in health_status["components"]
        assert (
            health_status["components"]["service1"]["status"]
            == HealthStatus.HEALTHY.value
        )
        assert (
            health_status["components"]["service2"]["status"]
            == HealthStatus.UNHEALTHY.value
        )

    def test_overall_health_status(self):
        """Test overall system health status calculation."""
        monitor = SystemHealthMonitor()

        def healthy_check():
            return True

        def critical_check():
            raise Exception("Critical failure")

        monitor.register_health_check("healthy_service", healthy_check)
        monitor.register_health_check("critical_service", critical_check)

        health_status = monitor.get_system_health()

        # Should be critical due to one critical component
        assert health_status["overall_status"] == HealthStatus.CRITICAL.value


class TestIntegration:
    """Integration tests for error handling system."""

    def test_full_error_handling_flow(self):
        """Test complete error handling flow with all components."""

        # Initialize error handling system
        initialize_error_handling()

        # Get managers
        degradation_manager = get_degradation_manager()
        recovery_manager = get_recovery_manager()

        # Test that managers are properly initialized
        assert degradation_manager is not None
        assert recovery_manager is not None

        # Test degradation status
        status = degradation_manager.get_status()
        assert "current_level" in status
        assert "config" in status

        # Test recovery stats
        stats = recovery_manager.get_recovery_stats()
        assert "total_attempts" in stats
        assert "success_rate" in stats

    @patch("morgan.embeddings.service.get_embedding_service")
    def test_embedding_service_error_handling(self, mock_get_service):
        """Test error handling in embedding service integration."""

        # Mock embedding service that fails
        mock_service = Mock()
        mock_service.is_available.return_value = False
        mock_get_service.return_value = mock_service

        # Test that recovery procedure is triggered
        recovery_manager = get_recovery_manager()

        error = EmbeddingError(
            "Service unavailable", operation="encode", component="embedding_service"
        )

        # This should attempt recovery (though it may fail in test environment)
        recovery_attempted = recovery_manager.attempt_recovery(error)

        # Recovery may fail in test environment, but should not raise exception
        assert isinstance(recovery_attempted, bool)

    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access."""

        degradation_manager = get_degradation_manager()
        errors = []

        def create_error_thread():
            try:
                error = NetworkError(
                    "Concurrent error",
                    operation="concurrent_test",
                    component="test_component",
                )

                # Apply degradation
                degradation_manager.assess_and_apply_degradation(error.get_context())

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_error_thread)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should not have any errors from concurrent access
        assert len(errors) == 0

        # Degradation manager should still be functional
        status = degradation_manager.get_status()
        assert status is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
