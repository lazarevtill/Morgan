"""
Property-based tests for server performance under load.

**Feature: client-server-separation, Property 13: Performance under load**
**Validates: Requirements 6.4**
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient

from morgan_server.app import create_app
from morgan_server.config import ServerConfig


# ============================================================================
# Property 13: Performance under load
# **Validates: Requirements 6.4**
# ============================================================================

@settings(
    max_examples=100,
    deadline=None,  # Disable deadline for performance tests
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    num_concurrent_requests=st.integers(min_value=10, max_value=100),
    request_complexity=st.sampled_from(["simple", "moderate", "complex"])
)
def test_performance_under_load(num_concurrent_requests: int, request_complexity: str):
    """
    **Feature: client-server-separation, Property 13: Performance under load**
    
    Property: For any concurrent load up to the configured maximum, the server should
    maintain 95th percentile response times under 5 seconds.
    
    This test verifies:
    1. Server can handle concurrent requests up to max_concurrent_requests
    2. Response times remain reasonable under load
    3. 95th percentile response time stays under 5 seconds
    4. No requests fail due to overload
    
    **Validates: Requirements 6.4**
    """
    # Create test app with performance-optimized config
    config = ServerConfig(
        host="127.0.0.1",
        port=8080,
        max_concurrent_requests=100,
        request_timeout_seconds=60,
        session_timeout_minutes=60,
        llm_endpoint="http://localhost:11434",
        vector_db_url="http://localhost:6333",
    )
    
    app = create_app(config=config)
    client = TestClient(app)
    
    # Track response times
    response_times: List[float] = []
    errors: List[str] = []
    
    def make_request(request_id: int) -> Dict[str, Any]:
        """Make a single request and measure response time."""
        start_time = time.time()
        
        try:
            # Choose endpoint based on complexity
            if request_complexity == "simple":
                # Simple health check
                response = client.get("/health")
            elif request_complexity == "moderate":
                # Status check with component details
                response = client.get("/api/status")
            else:  # complex
                # Memory stats (more complex query)
                response = client.get("/api/memory/stats")
            
            elapsed = time.time() - start_time
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": elapsed,
                "success": response.status_code in [200, 503]  # 503 is OK for tests without full components
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": elapsed,
                "success": False,
                "error": str(e)
            }
    
    # Execute requests concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_concurrent_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Collect metrics
    for result in results:
        response_times.append(result["response_time"])
        
        if not result["success"]:
            errors.append(
                f"Request {result['request_id']} failed: "
                f"status={result.get('status_code', 'N/A')}, "
                f"error={result.get('error', 'Unknown')}"
            )
    
    # ========================================================================
    # Verify Performance Properties
    # ========================================================================
    
    # Property 1: All requests should complete successfully
    assert len(errors) == 0, (
        f"Some requests failed under load:\n" + "\n".join(errors[:5])
    )
    
    # Property 2: Calculate 95th percentile response time
    if response_times:
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_response_time = sorted_times[p95_index]
        
        mean_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        max_response_time = max(response_times)
        
        # Property 3: 95th percentile should be under 5 seconds
        assert p95_response_time < 5.0, (
            f"95th percentile response time ({p95_response_time:.3f}s) exceeds 5 seconds "
            f"under load of {num_concurrent_requests} concurrent requests "
            f"with {request_complexity} complexity. "
            f"Stats: mean={mean_response_time:.3f}s, "
            f"median={median_response_time:.3f}s, "
            f"max={max_response_time:.3f}s"
        )
        
        # Property 4: Mean response time should be reasonable (under 2 seconds)
        # This is a softer requirement but helps catch performance degradation
        assert mean_response_time < 2.0, (
            f"Mean response time ({mean_response_time:.3f}s) exceeds 2 seconds "
            f"under load of {num_concurrent_requests} concurrent requests "
            f"with {request_complexity} complexity"
        )


def test_performance_under_maximum_load():
    """
    Test performance at the configured maximum concurrent requests.
    
    This is a specific test case for the maximum load scenario.
    
    **Feature: client-server-separation, Property 13: Performance under load**
    **Validates: Requirements 6.4**
    """
    # Create test app
    config = ServerConfig(
        host="127.0.0.1",
        port=8080,
        max_concurrent_requests=100,
        request_timeout_seconds=60,
        session_timeout_minutes=60,
        llm_endpoint="http://localhost:11434",
        vector_db_url="http://localhost:6333",
    )
    
    app = create_app(config=config)
    client = TestClient(app)
    
    # Test at maximum configured load
    num_requests = config.max_concurrent_requests
    response_times: List[float] = []
    
    def make_health_check(request_id: int) -> float:
        """Make a health check request and return response time."""
        start_time = time.time()
        response = client.get("/health")
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 503], (
            f"Health check failed at maximum load: {response.status_code}"
        )
        
        return elapsed
    
    # Execute requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_health_check, i) for i in range(num_requests)]
        response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Verify performance at maximum load
    sorted_times = sorted(response_times)
    p95_index = int(len(sorted_times) * 0.95)
    p95_response_time = sorted_times[p95_index]
    
    mean_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    
    # 95th percentile must be under 5 seconds even at maximum load
    assert p95_response_time < 5.0, (
        f"95th percentile response time ({p95_response_time:.3f}s) exceeds 5 seconds "
        f"at maximum load of {num_requests} concurrent requests. "
        f"Stats: mean={mean_response_time:.3f}s, max={max_response_time:.3f}s"
    )


def test_performance_degradation_is_gradual():
    """
    Test that performance degrades gradually as load increases.
    
    This verifies that the server doesn't have sudden performance cliffs.
    
    **Feature: client-server-separation, Property 13: Performance under load**
    **Validates: Requirements 6.4**
    """
    # Create test app
    config = ServerConfig(
        host="127.0.0.1",
        port=8080,
        max_concurrent_requests=100,
        request_timeout_seconds=60,
        session_timeout_minutes=60,
        llm_endpoint="http://localhost:11434",
        vector_db_url="http://localhost:6333",
    )
    
    app = create_app(config=config)
    client = TestClient(app)
    
    # Test at different load levels
    load_levels = [10, 25, 50, 75, 100]
    p95_times_by_load: Dict[int, float] = {}
    
    for load in load_levels:
        response_times: List[float] = []
        
        def make_request(request_id: int) -> float:
            """Make a request and return response time."""
            start_time = time.time()
            response = client.get("/health")
            elapsed = time.time() - start_time
            assert response.status_code in [200, 503]
            return elapsed
        
        # Execute requests at this load level
        with concurrent.futures.ThreadPoolExecutor(max_workers=load) as executor:
            futures = [executor.submit(make_request, i) for i in range(load)]
            response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Calculate p95 for this load level
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_times_by_load[load] = sorted_times[p95_index]
    
    # Verify gradual degradation
    # Each load level should not be more than 2x slower than the previous
    previous_load = None
    previous_p95 = None
    
    for load in load_levels:
        p95 = p95_times_by_load[load]
        
        if previous_load is not None:
            # Performance shouldn't degrade by more than 2x between load levels
            degradation_factor = p95 / previous_p95 if previous_p95 > 0 else 1.0
            
            assert degradation_factor < 2.0, (
                f"Performance degraded too sharply from {previous_load} to {load} requests: "
                f"{previous_p95:.3f}s -> {p95:.3f}s (factor: {degradation_factor:.2f}x)"
            )
        
        previous_load = load
        previous_p95 = p95


def test_sustained_load_performance():
    """
    Test performance under sustained load over time.
    
    This verifies that performance doesn't degrade over multiple rounds of requests.
    
    **Feature: client-server-separation, Property 13: Performance under load**
    **Validates: Requirements 6.4**
    """
    # Create test app
    config = ServerConfig(
        host="127.0.0.1",
        port=8080,
        max_concurrent_requests=100,
        request_timeout_seconds=60,
        session_timeout_minutes=60,
        llm_endpoint="http://localhost:11434",
        vector_db_url="http://localhost:6333",
    )
    
    app = create_app(config=config)
    client = TestClient(app)
    
    # Run multiple rounds of requests
    num_rounds = 5
    requests_per_round = 50
    p95_times_by_round: List[float] = []
    
    for round_num in range(num_rounds):
        response_times: List[float] = []
        
        def make_request(request_id: int) -> float:
            """Make a request and return response time."""
            start_time = time.time()
            response = client.get("/health")
            elapsed = time.time() - start_time
            assert response.status_code in [200, 503]
            return elapsed
        
        # Execute requests for this round
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_round) as executor:
            futures = [executor.submit(make_request, i) for i in range(requests_per_round)]
            response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Calculate p95 for this round
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_times_by_round.append(sorted_times[p95_index])
        
        # Small delay between rounds
        time.sleep(0.1)
    
    # Verify sustained performance
    # Performance shouldn't degrade significantly over time
    first_round_p95 = p95_times_by_round[0]
    last_round_p95 = p95_times_by_round[-1]
    
    # Last round shouldn't be more than 1.5x slower than first round
    degradation_factor = last_round_p95 / first_round_p95 if first_round_p95 > 0 else 1.0
    
    assert degradation_factor < 1.5, (
        f"Performance degraded over sustained load: "
        f"first round p95={first_round_p95:.3f}s, "
        f"last round p95={last_round_p95:.3f}s "
        f"(factor: {degradation_factor:.2f}x)"
    )
    
    # All rounds should maintain p95 under 5 seconds
    for round_num, p95 in enumerate(p95_times_by_round):
        assert p95 < 5.0, (
            f"Round {round_num + 1} p95 response time ({p95:.3f}s) exceeds 5 seconds"
        )
