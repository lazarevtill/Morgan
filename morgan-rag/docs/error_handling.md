# Morgan RAG Error Handling System

## Overview

The Morgan RAG error handling system provides comprehensive, production-grade error management for all vectorization operations. It implements industry best practices including structured error hierarchies, retry logic with exponential backoff, circuit breaker patterns, graceful degradation, and automatic error recovery.

## Key Features

### 🛡️ Comprehensive Error Management
- **Structured Exception Hierarchy**: Custom exceptions with rich context and metadata
- **Error Categorization**: Organized by component and severity for better handling
- **Contextual Logging**: Detailed error context with operation tracking and user identification

### 🔄 Retry Logic with Exponential Backoff
- **Intelligent Retries**: Automatic retry for transient failures with exponential backoff
- **Jitter Support**: Prevents thundering herd problems with randomized delays
- **Configurable Policies**: Customizable retry attempts, delays, and exception types

### ⚡ Circuit Breaker Pattern
- **Failure Protection**: Prevents cascading failures by temporarily stopping calls to failing services
- **Automatic Recovery**: Self-healing with configurable recovery timeouts and success thresholds
- **State Monitoring**: Real-time circuit state tracking and metrics

### 🎭 Graceful Degradation
- **Feature Disabling**: Automatically disables non-essential features during failures
- **Performance Preservation**: Maintains core functionality while reducing system load
- **Configurable Levels**: Multiple degradation levels from minimal to critical

### 🔧 Error Recovery Procedures
- **Automatic Recovery**: Registered recovery procedures for common failure scenarios
- **Fallback Strategies**: Multiple recovery strategies including retry, fallback, and degradation
- **Recovery Tracking**: Comprehensive statistics and success rate monitoring

### 📊 Health Monitoring
- **Component Health Checks**: Real-time health status for all system components
- **System-wide Status**: Overall health assessment with detailed component breakdown
- **Performance Metrics**: Operation timing and error rate tracking

## Architecture

### Exception Hierarchy

```python
MorganError (Base)
├── VectorizationError
├── EmbeddingError
├── StorageError
├── SearchError
├── CacheError
├── NetworkError
├── CompanionError
├── EmotionalProcessingError
├── MemoryProcessingError
├── ValidationError
└── ConfigurationError
```

### Core Components

1. **Error Context Manager**: Provides structured error handling with automatic context capture
2. **Retry Manager**: Implements exponential backoff retry logic with configurable policies
3. **Circuit Breaker**: Prevents cascading failures with automatic recovery
4. **Degradation Manager**: Manages graceful feature degradation during failures
5. **Recovery Manager**: Orchestrates error recovery procedures
6. **Health Monitor**: Tracks system and component health status

## Usage Guide

### Basic Error Handling

```python
from morgan.utils.error_handling import error_context, EmbeddingError

# Use error context for automatic error handling
with error_context("encode_batch", "embedding_service", ErrorCategory.EMBEDDING):
    embeddings = service.encode_batch(texts)
```

### Decorators for Easy Integration

```python
from morgan.utils.error_decorators import handle_embedding_errors, RetryConfig

@handle_embedding_errors("encode_text", "embedding_service", 
                         RetryConfig(max_attempts=3, base_delay=2.0))
def encode_text(self, text: str):
    return self._encode_text_impl(text)
```

### Robust Operations

```python
from morgan.utils.error_decorators import robust_vectorization_operation

@robust_vectorization_operation("process_documents", max_retries=3)
def process_documents(self, documents):
    return self._process_documents_impl(documents)
```

### Circuit Breaker Usage

```python
from morgan.utils.error_handling import CircuitBreaker

circuit = CircuitBreaker("external_service")

def call_external_service():
    return circuit.call(lambda: external_api.request())
```

### Graceful Degradation

```python
from morgan.utils.error_handling import get_degradation_manager

degradation_manager = get_degradation_manager()

if degradation_manager.is_feature_enabled("companion"):
    # Execute companion features
    result = companion_service.process(data)
else:
    # Skip companion features during degradation
    result = None
```

## Configuration

### Retry Configuration

```python
from morgan.utils.error_handling import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,           # Maximum retry attempts
    base_delay=1.0,          # Base delay in seconds
    max_delay=60.0,          # Maximum delay cap
    exponential_base=2.0,    # Exponential backoff base
    jitter=True,             # Enable jitter
    jitter_range=0.3,        # Jitter range (30%)
    retryable_exceptions=(NetworkError, StorageError),
    non_retryable_exceptions=(ValidationError, ConfigurationError)
)
```

### Circuit Breaker Configuration

```python
from morgan.utils.error_handling import CircuitBreakerConfig

circuit_config = CircuitBreakerConfig(
    failure_threshold=5,      # Failures before opening
    recovery_timeout=60.0,    # Seconds before attempting recovery
    success_threshold=3,      # Successes needed to close circuit
    timeout=30.0             # Operation timeout
)
```

### Degradation Configuration

```python
from morgan.utils.error_handling import DegradationConfig, DegradationLevel

# Apply specific degradation level
degradation_manager.force_degradation_level(
    DegradationLevel.MODERATE, 
    "manual_maintenance"
)

# Check current configuration
config = degradation_manager.get_current_config()
print(f"Companion features: {config.companion_features_enabled}")
print(f"Emotional processing: {config.emotional_processing_enabled}")
```

## Error Recovery Procedures

### Built-in Recovery Procedures

1. **Embedding Service Fallback**: Switches to local embeddings when remote service fails
2. **Vector Database Reconnect**: Attempts to reconnect to vector database
3. **Companion Graceful Degradation**: Disables companion features during failures
4. **Memory Processing Skip**: Skips failed memory processing operations

### Custom Recovery Procedures

```python
from morgan.utils.error_handling import RecoveryProcedure, RecoveryStrategy

def custom_recovery(error, context):
    # Implement custom recovery logic
    return True  # Return True if recovery successful

procedure = RecoveryProcedure(
    name="custom_recovery",
    strategy=RecoveryStrategy.FALLBACK,
    applicable_errors=[CustomError],
    recovery_function=custom_recovery,
    description="Custom recovery for specific errors"
)

recovery_manager = get_recovery_manager()
recovery_manager.register_procedure(procedure)
```

## Monitoring and Observability

### Health Checks

```python
from morgan.utils.error_handling import get_health_monitor

health_monitor = get_health_monitor()

# Register custom health check
def check_service_health():
    return service.is_healthy()

health_monitor.register_health_check("my_service", check_service_health)

# Get system health status
health_status = health_monitor.get_system_health()
print(f"Overall status: {health_status['overall_status']}")
```

### Performance Monitoring

```python
from morgan.utils.error_decorators import monitor_performance

@monitor_performance("expensive_operation", "my_component", slow_threshold=2.0)
def expensive_operation(self, data):
    # Operation implementation
    return result
```

### Error Statistics

```python
# Get recovery statistics
recovery_stats = recovery_manager.get_recovery_stats()
print(f"Success rate: {recovery_stats['success_rate']:.1f}%")

# Get degradation status
degradation_status = degradation_manager.get_status()
print(f"Current level: {degradation_status['current_level']}")
```

## Integration with Existing Code

### Embedding Service Integration

The embedding service has been enhanced with comprehensive error handling:

```python
# Automatic retry for network failures
@handle_embedding_errors("encode_batch", "embedding_service", 
                         RetryConfig(max_attempts=3))
def encode_batch(self, texts):
    return self._encode_batch_impl(texts)

# Structured error reporting
try:
    embeddings = service.encode_batch(texts)
except EmbeddingError as e:
    logger.error(f"Embedding failed: {e.get_context().to_dict()}")
```

### Vector Database Integration

Vector database operations include automatic retry and circuit breaker protection:

```python
@handle_storage_errors("upsert_points", "vector_db_client",
                      RetryConfig(max_attempts=3, base_delay=2.0))
def upsert_points(self, collection_name, points):
    return self._upsert_points_impl(collection_name, points)
```

### Search Engine Integration

Search operations include fallback strategies and graceful degradation:

```python
@handle_search_errors("find_relevant_info", "smart_search", 
                      fallback_to_basic=True)
def find_relevant_info(self, query, **kwargs):
    return self._find_relevant_info_impl(query, **kwargs)
```

## Best Practices

### 1. Use Appropriate Error Types
- Choose specific error types (EmbeddingError, StorageError) over generic exceptions
- Include relevant metadata and context in error messages
- Set appropriate severity levels for different error conditions

### 2. Configure Retry Policies Appropriately
- Use shorter delays and fewer attempts for user-facing operations
- Use longer delays and more attempts for background processing
- Don't retry validation errors or configuration errors

### 3. Implement Graceful Degradation
- Identify which features are essential vs. nice-to-have
- Design fallback behaviors for non-essential features
- Communicate degradation status to users when appropriate

### 4. Monitor and Alert
- Set up alerts for high error rates or circuit breaker activations
- Monitor recovery success rates and degradation events
- Track performance metrics to identify trends

### 5. Test Error Scenarios
- Test retry logic with simulated failures
- Verify graceful degradation behavior
- Validate recovery procedures work as expected

## Troubleshooting

### Common Issues

1. **High Retry Rates**: Check network connectivity and service health
2. **Circuit Breaker Constantly Open**: Investigate underlying service issues
3. **Frequent Degradation**: Review system resource usage and scaling
4. **Recovery Failures**: Check recovery procedure implementations

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger('morgan.utils.error_handling').setLevel(logging.DEBUG)

# Get detailed error context
try:
    operation()
except MorganError as e:
    context = e.get_context()
    print(f"Error details: {context.to_dict()}")

# Check system health
health_status = get_health_monitor().get_system_health()
print(f"System health: {health_status}")
```

## Performance Impact

The error handling system is designed for minimal performance overhead:

- **Error Context**: ~0.1ms overhead per operation
- **Retry Logic**: Only activates on failures
- **Circuit Breaker**: ~0.01ms overhead per call
- **Health Checks**: Configurable frequency, default 30s intervals
- **Degradation Manager**: Thread-safe with minimal locking

## Security Considerations

- **API Key Masking**: Automatically masks API keys in error messages and logs
- **Sensitive Data**: Avoids logging sensitive user data in error contexts
- **Error Information**: Provides detailed errors in development, sanitized in production

## Migration Guide

### From Basic Exception Handling

```python
# Before
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise

# After
@handle_vectorization_errors("risky_operation", "my_component")
def risky_operation(self):
    return self._risky_operation_impl()
```

### Adding Retry Logic

```python
# Before
def unreliable_operation(self):
    return external_service.call()

# After
@handle_embedding_errors("unreliable_operation", "my_service",
                         RetryConfig(max_attempts=3, base_delay=1.0))
def unreliable_operation(self):
    return external_service.call()
```

## Conclusion

The Morgan RAG error handling system provides enterprise-grade reliability and observability for vectorization operations. By implementing structured error management, intelligent retry logic, and graceful degradation, it ensures robust operation even under adverse conditions while maintaining excellent performance and user experience.

For more examples and advanced usage patterns, see the `examples/error_handling_demo.py` file and the comprehensive test suite in `tests/test_error_handling.py`.