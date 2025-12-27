#!/usr/bin/env python3
"""
Morgan RAG Error Handling System Demo

Demonstrates the comprehensive error handling capabilities including:
- Custom exception hierarchy
- Retry logic with exponential backoff
- Circuit breaker pattern
- Graceful degradation
- Error recovery procedures
- Performance monitoring
"""

import time
import random
from typing import List

# Import Morgan's error handling system
from morgan.utils.error_handling import (
    initialize_error_handling,
    get_degradation_manager,
    get_recovery_manager,
    get_health_monitor,
    EmbeddingError,
    NetworkError,
    CompanionError,
    CircuitBreaker,
    ErrorSeverity,
    DegradationLevel,
)

from morgan.utils.error_decorators import (
    handle_embedding_errors,
    handle_companion_errors,
    monitor_performance,
    robust_vectorization_operation,
    RetryConfig,
)

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class DemoEmbeddingService:
    """Demo embedding service that simulates various failure scenarios."""

    def __init__(self):
        self.failure_rate = 0.3  # 30% failure rate
        self.call_count = 0

    @handle_embedding_errors(
        "encode_text",
        "demo_embedding_service",
        RetryConfig(max_attempts=3, base_delay=0.5),
    )
    @monitor_performance("encode_text", "demo_embedding_service")
    def encode_text(self, text: str) -> List[float]:
        """Encode text with simulated failures."""
        self.call_count += 1

        # Simulate network failures
        if random.random() < self.failure_rate:
            raise NetworkError(
                f"Simulated network failure (attempt {self.call_count})",
                operation="encode_text",
                component="demo_embedding_service",
            )

        # Simulate successful encoding
        time.sleep(0.1)  # Simulate processing time
        return [random.random() for _ in range(384)]  # Mock embedding

    @handle_embedding_errors(
        "encode_batch",
        "demo_embedding_service",
        RetryConfig(max_attempts=2, base_delay=1.0),
    )
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode batch with potential failures."""
        embeddings = []
        for text in texts:
            embedding = self.encode_text(text)
            embeddings.append(embedding)
        return embeddings


class DemoCompanionService:
    """Demo companion service with graceful degradation."""

    def __init__(self):
        self.emotional_processing_enabled = True

    @handle_companion_errors("analyze_emotion", "demo_companion_service")
    @monitor_performance("analyze_emotion", "demo_companion_service")
    def analyze_emotion(self, text: str, force_failure: bool = False):
        """Analyze emotion with potential graceful degradation."""
        if force_failure:
            raise CompanionError(
                "Emotional processing service unavailable",
                operation="analyze_emotion",
                component="demo_companion_service",
                severity=ErrorSeverity.MEDIUM,
            )

        # Simulate emotion analysis
        emotions = ["joy", "sadness", "anger", "fear", "surprise"]
        return {
            "primary_emotion": random.choice(emotions),
            "intensity": random.uniform(0.1, 1.0),
            "confidence": random.uniform(0.7, 0.95),
        }

    @robust_companion_operation("update_relationship", "demo_companion_service")
    def update_relationship(self, user_id: str, interaction_data: dict):
        """Update relationship data with error handling."""
        # Check if companion features are enabled
        degradation_manager = get_degradation_manager()
        if not degradation_manager.is_feature_enabled("companion"):
            logger.info("Companion features disabled, skipping relationship update")
            return None

        # Simulate relationship update
        return {
            "user_id": user_id,
            "relationship_score": random.uniform(0.5, 1.0),
            "interaction_count": random.randint(1, 100),
            "last_updated": time.time(),
        }


def demo_basic_error_handling():
    """Demonstrate basic error handling with retries."""
    print("\n🔄 Demo: Basic Error Handling with Retries")
    print("=" * 50)

    service = DemoEmbeddingService()
    service.failure_rate = 0.7  # High failure rate to trigger retries

    try:
        # This should succeed after retries
        embedding = service.encode_text("Hello, world!")
        print(f"✅ Successfully encoded text after {service.call_count} attempts")
        print(f"   Embedding dimension: {len(embedding)}")

    except Exception as e:
        print(f"❌ Failed to encode text: {e}")

    # Reset for next demo
    service.call_count = 0


def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n⚡ Demo: Circuit Breaker Pattern")
    print("=" * 50)

    circuit = CircuitBreaker("demo_service")

    def unreliable_service():
        if random.random() < 0.8:  # 80% failure rate
            raise Exception("Service temporarily unavailable")
        return "Service response"

    # Make multiple calls to trigger circuit breaker
    for i in range(10):
        try:
            result = circuit.call(unreliable_service)
            print(f"Call {i+1}: ✅ {result}")
        except Exception as e:
            print(f"Call {i+1}: ❌ {type(e).__name__}")

    print(f"\nCircuit breaker state: {circuit.get_state()}")


def demo_graceful_degradation():
    """Demonstrate graceful degradation of companion features."""
    print("\n🎭 Demo: Graceful Degradation")
    print("=" * 50)

    companion_service = DemoCompanionService()
    degradation_manager = get_degradation_manager()

    # Show initial state
    print(f"Initial degradation level: {degradation_manager.current_level.value}")
    print(
        f"Companion features enabled: {degradation_manager.is_feature_enabled('companion')}"
    )

    # Simulate successful emotion analysis
    try:
        emotion = companion_service.analyze_emotion("I'm feeling great today!")
        print(f"✅ Emotion analysis: {emotion}")
    except Exception as e:
        print(f"❌ Emotion analysis failed: {e}")

    # Force failure to trigger degradation
    try:
        emotion = companion_service.analyze_emotion(
            "This will fail", force_failure=True
        )
        print(f"✅ Emotion analysis: {emotion}")
    except Exception as e:
        print(f"❌ Emotion analysis failed (expected): {type(e).__name__}")

    # Check degradation status after failure
    print(f"\nAfter failure:")
    print(f"Degradation level: {degradation_manager.current_level.value}")
    print(
        f"Companion features enabled: {degradation_manager.is_feature_enabled('companion')}"
    )

    # Try relationship update (should be affected by degradation)
    result = companion_service.update_relationship("user123", {"mood": "happy"})
    if result:
        print(f"✅ Relationship updated: {result}")
    else:
        print("⚠️  Relationship update skipped due to degradation")


def demo_error_recovery():
    """Demonstrate error recovery procedures."""
    print("\n🔧 Demo: Error Recovery Procedures")
    print("=" * 50)

    recovery_manager = get_recovery_manager()

    # Create a test error
    error = EmbeddingError(
        "Remote embedding service unavailable",
        operation="encode_batch",
        component="embedding_service",
        severity=ErrorSeverity.HIGH,
    )

    print(f"Attempting recovery for error: {error.error_id}")

    # Attempt recovery
    recovery_successful = recovery_manager.attempt_recovery(error)

    if recovery_successful:
        print("✅ Error recovery successful")
    else:
        print("❌ Error recovery failed")

    # Show recovery statistics
    stats = recovery_manager.get_recovery_stats()
    print(f"\nRecovery Statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")


def demo_health_monitoring():
    """Demonstrate system health monitoring."""
    print("\n🏥 Demo: System Health Monitoring")
    print("=" * 50)

    health_monitor = get_health_monitor()

    # Register some demo health checks
    def embedding_service_health():
        return random.random() > 0.2  # 80% healthy

    def vector_db_health():
        return random.random() > 0.1  # 90% healthy

    def companion_service_health():
        return random.random() > 0.3  # 70% healthy

    health_monitor.register_health_check("embedding_service", embedding_service_health)
    health_monitor.register_health_check("vector_database", vector_db_health)
    health_monitor.register_health_check("companion_service", companion_service_health)

    # Get system health status
    health_status = health_monitor.get_system_health()

    print(f"Overall system status: {health_status['overall_status']}")
    print(f"Components:")

    for component, status in health_status["components"].items():
        status_icon = "✅" if status["status"] == "healthy" else "❌"
        print(f"  {status_icon} {component}: {status['status']}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n📊 Demo: Performance Monitoring")
    print("=" * 50)

    service = DemoEmbeddingService()
    service.failure_rate = 0.1  # Low failure rate for performance demo

    # Perform multiple operations to collect performance data
    print("Performing operations to collect performance metrics...")

    for i in range(5):
        try:
            # Vary the processing time
            time.sleep(random.uniform(0.05, 0.2))
            embedding = service.encode_text(f"Sample text {i+1}")
            print(f"  Operation {i+1}: ✅ Completed")
        except Exception as e:
            print(f"  Operation {i+1}: ❌ Failed - {type(e).__name__}")

    print("\nPerformance monitoring data collected (check logs for details)")


def main():
    """Run all error handling demos."""
    print("🛡️  Morgan RAG Error Handling System Demo")
    print("=" * 60)

    # Initialize the error handling system
    print("Initializing error handling system...")
    initialize_error_handling()
    print("✅ Error handling system initialized\n")

    # Run all demos
    demo_basic_error_handling()
    demo_circuit_breaker()
    demo_graceful_degradation()
    demo_error_recovery()
    demo_health_monitoring()
    demo_performance_monitoring()

    print("\n" + "=" * 60)
    print("🎉 Error handling demo completed!")
    print("\nKey features demonstrated:")
    print("  ✅ Retry logic with exponential backoff")
    print("  ✅ Circuit breaker pattern")
    print("  ✅ Graceful degradation")
    print("  ✅ Error recovery procedures")
    print("  ✅ Health monitoring")
    print("  ✅ Performance monitoring")
    print("\nThe error handling system is now ready for production use!")


if __name__ == "__main__":
    main()
