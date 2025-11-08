"""
Tests for production infrastructure components

Demonstrates the new production-quality patterns:
- Circuit breaker
- Rate limiting
- Enhanced HTTP client
- Health monitoring
"""

import asyncio
import time
from typing import Dict, Any

# Mock imports for testing without dependencies
try:
    from shared.infrastructure import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        TokenBucketRateLimiter,
        RateLimitConfig,
        HealthMonitor,
        HealthStatus,
    )

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    print("⚠️  Infrastructure components not available - tests will be skipped")


class TestProductionInfrastructure:
    """Test suite for production infrastructure"""

    async def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        print("\n" + "=" * 60)
        print("Testing Circuit Breaker Pattern")
        print("=" * 60)

        config = CircuitBreakerConfig(
            failure_threshold=3, success_threshold=2, timeout=2.0
        )
        cb = CircuitBreaker(config)

        # Test normal operation (CLOSED)
        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        print("✅ Circuit breaker CLOSED state working")

        # Trigger failures to open circuit
        async def failing_func():
            raise Exception("Service unavailable")

        for i in range(3):
            try:
                await cb.call(failing_func)
            except Exception:
                pass

        assert cb.state == CircuitBreakerState.OPEN
        print("✅ Circuit breaker OPEN after failures")

        # Try to call while OPEN
        try:
            from shared.infrastructure.circuit_breaker import CircuitBreakerError

            await cb.call(success_func)
            assert False, "Should have raised CircuitBreakerError"
        except CircuitBreakerError:
            print("✅ Circuit breaker rejects calls when OPEN")

        # Wait for timeout and test HALF_OPEN
        await asyncio.sleep(2.1)
        result = await cb.call(success_func)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        print("✅ Circuit breaker transitions to HALF_OPEN")

        # Success in HALF_OPEN should close circuit
        result = await cb.call(success_func)
        assert cb.state == CircuitBreakerState.CLOSED
        print("✅ Circuit breaker closes after successes in HALF_OPEN")

        # Get state
        state = cb.get_state()
        print(f"\nFinal state: {state}")

    async def test_rate_limiter(self):
        """Test rate limiting"""
        print("\n" + "=" * 60)
        print("Testing Rate Limiter")
        print("=" * 60)

        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = TokenBucketRateLimiter(config)

        # Test burst
        start = time.time()
        for i in range(10):
            await limiter.acquire()
        burst_time = time.time() - start

        assert burst_time < 0.5, "Burst should be fast"
        print(f"✅ Burst of 10 requests: {burst_time:.3f}s")

        # Test rate limiting
        start = time.time()
        for i in range(5):
            await limiter.acquire()
        limited_time = time.time() - start

        assert limited_time >= 0.8, "Should be rate limited"
        print(f"✅ Rate limited 5 requests: {limited_time:.3f}s")

        # Get state
        state = limiter.get_state()
        print(f"\nRate limiter state: {state}")

    async def test_health_monitor(self):
        """Test health monitoring"""
        print("\n" + "=" * 60)
        print("Testing Health Monitor")
        print("=" * 60)

        monitor = HealthMonitor(
            name="test_service",
            check_interval=1.0,
            degraded_threshold=2,
            unhealthy_threshold=3,
        )

        # Test healthy checks
        async def healthy_check():
            return True

        result = await monitor.check_health(healthy_check)
        assert result.status == HealthStatus.HEALTHY
        print("✅ Health check: HEALTHY")

        # Test unhealthy checks
        async def unhealthy_check():
            raise Exception("Service down")

        for i in range(3):
            result = await monitor.check_health(unhealthy_check)

        assert monitor.get_status() == HealthStatus.UNHEALTHY
        print("✅ Health status: UNHEALTHY after failures")

        # Get metrics
        metrics = monitor.get_metrics()
        print(f"\nHealth metrics: {metrics}")

        # Get summary
        summary = monitor.get_summary()
        print(f"\nHealth summary:")
        print(f"  Status: {summary['status']}")
        print(f"  Total checks: {summary['metrics']['total_checks']}")
        print(f"  Uptime: {summary['metrics']['uptime_percentage']:.1f}%")

    async def test_integration(self):
        """Test integration of components"""
        print("\n" + "=" * 60)
        print("Testing Component Integration")
        print("=" * 60)

        # Create a mock service with all components
        class MockService:
            def __init__(self):
                self.circuit_breaker = CircuitBreaker(
                    CircuitBreakerConfig(failure_threshold=3)
                )
                self.rate_limiter = TokenBucketRateLimiter(
                    RateLimitConfig(requests_per_second=10)
                )
                self.health_monitor = HealthMonitor(
                    name="mock_service", check_interval=30.0
                )
                self.request_count = 0

            async def make_request(self):
                # Apply rate limiting
                await self.rate_limiter.acquire()

                # Use circuit breaker
                result = await self.circuit_breaker.call(self._execute_request)

                return result

            async def _execute_request(self):
                self.request_count += 1
                # Simulate request
                await asyncio.sleep(0.01)
                return {"status": "success", "count": self.request_count}

            async def health_check(self):
                return await self.health_monitor.check_health(self._health_check_impl)

            async def _health_check_impl(self):
                # Service is healthy if circuit is not open
                if self.circuit_breaker.state == CircuitBreakerState.OPEN:
                    raise Exception("Circuit breaker open")
                return True

        # Test the service
        service = MockService()

        # Make some requests
        for i in range(5):
            result = await service.make_request()
            print(f"✅ Request {i+1}: {result}")

        # Check health
        health = await service.health_check()
        assert health.status == HealthStatus.HEALTHY
        print(f"✅ Service health: {health.status.value}")

        print(f"\nService Stats:")
        print(f"  Total requests: {service.request_count}")
        print(f"  Circuit breaker: {service.circuit_breaker.state.value}")
        print(f"  Rate limiter: {service.rate_limiter.get_state()}")

    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "🧪 " + "=" * 58)
        print("Production Infrastructure Test Suite")
        print("=" * 60 + "\n")

        if not INFRASTRUCTURE_AVAILABLE:
            print("❌ Infrastructure components not available")
            print("\nTo run tests, ensure shared.infrastructure is importable")
            return

        try:
            await self.test_circuit_breaker()
            await self.test_rate_limiter()
            await self.test_health_monitor()
            await self.test_integration()

            print("\n" + "=" * 60)
            print("✅ All tests passed!")
            print("=" * 60 + "\n")

        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            raise


async def main():
    """Main test runner"""
    tester = TestProductionInfrastructure()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
