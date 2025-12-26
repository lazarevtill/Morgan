"""
Property-Based Tests for Metrics Endpoint

This module contains property-based tests for the metrics endpoint
using Hypothesis to verify correctness properties across many inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings
from morgan_server.health import HealthCheckSystem


# Feature: client-server-separation, Property 11: Metrics accuracy
# Validates: Requirements 4.5
@settings(max_examples=100)
@given(
    requests=st.lists(
        st.tuples(
            st.booleans(),  # success
            st.floats(min_value=0.1, max_value=10000.0),  # response_time_ms
        ),
        min_size=1,
        max_size=100,
    )
)
def test_metrics_accuracy(requests):
    """
    Property: For any sequence of API requests, the metrics endpoint should
    return counts and statistics that accurately reflect the actual requests
    processed (request count, error count, response times).

    This test verifies that:
    1. Total request count matches the number of recorded requests
    2. Success count matches the number of successful requests
    3. Error count matches the number of failed requests
    4. Error rate is correctly calculated
    5. Average response time is correctly calculated
    """
    # Create a fresh health system for this test
    health_system = HealthCheckSystem(version="test")

    # Track expected values
    expected_total = len(requests)
    expected_success = sum(1 for success, _ in requests if success)
    expected_error = sum(1 for success, _ in requests if not success)
    expected_response_times = [rt for _, rt in requests]
    expected_avg_response_time = sum(expected_response_times) / len(
        expected_response_times
    )
    expected_error_rate = expected_error / expected_total if expected_total > 0 else 0.0

    # Record all requests
    for success, response_time_ms in requests:
        health_system.record_request(success, response_time_ms)

    # Get metrics
    metrics = health_system.get_system_metrics()

    # Verify request counts
    assert metrics.requests_total == expected_total, (
        f"Total requests mismatch: expected {expected_total}, "
        f"got {metrics.requests_total}"
    )

    assert health_system.requests_success == expected_success, (
        f"Success count mismatch: expected {expected_success}, "
        f"got {health_system.requests_success}"
    )

    assert health_system.requests_error == expected_error, (
        f"Error count mismatch: expected {expected_error}, "
        f"got {health_system.requests_error}"
    )

    # Verify error rate (with small tolerance for floating point)
    assert abs(metrics.error_rate - expected_error_rate) < 0.0001, (
        f"Error rate mismatch: expected {expected_error_rate}, "
        f"got {metrics.error_rate}"
    )

    # Verify average response time (with small tolerance for floating point)
    assert abs(metrics.average_response_time_ms - expected_avg_response_time) < 0.01, (
        f"Average response time mismatch: expected {expected_avg_response_time}, "
        f"got {metrics.average_response_time_ms}"
    )

    # Verify that requests_per_second is non-negative and reasonable
    # Note: We don't verify exact RPS calculation because uptime can be very small
    # in tests (microseconds), making the calculation extremely sensitive to timing
    assert (
        metrics.requests_per_second >= 0.0
    ), f"Requests per second should be non-negative, got {metrics.requests_per_second}"


@settings(max_examples=100)
@given(
    initial_requests=st.lists(
        st.tuples(st.booleans(), st.floats(min_value=0.1, max_value=1000.0)),
        min_size=1,
        max_size=50,
    ),
    additional_requests=st.lists(
        st.tuples(st.booleans(), st.floats(min_value=0.1, max_value=1000.0)),
        min_size=1,
        max_size=50,
    ),
)
def test_metrics_accumulation(initial_requests, additional_requests):
    """
    Property: Metrics should accumulate correctly over time.

    This test verifies that when requests are recorded in batches,
    the metrics correctly accumulate and reflect the total of all requests.
    """
    health_system = HealthCheckSystem(version="test")

    # Record initial batch
    for success, response_time_ms in initial_requests:
        health_system.record_request(success, response_time_ms)

    initial_metrics = health_system.get_system_metrics()
    initial_total = initial_metrics.requests_total

    # Record additional batch
    for success, response_time_ms in additional_requests:
        health_system.record_request(success, response_time_ms)

    final_metrics = health_system.get_system_metrics()

    # Verify accumulation
    expected_total = len(initial_requests) + len(additional_requests)
    assert final_metrics.requests_total == expected_total, (
        f"Total should accumulate: expected {expected_total}, "
        f"got {final_metrics.requests_total}"
    )

    # Verify that total increased by the number of additional requests
    assert final_metrics.requests_total == initial_total + len(additional_requests), (
        f"Total should increase by {len(additional_requests)}, "
        f"but went from {initial_total} to {final_metrics.requests_total}"
    )


@settings(max_examples=100)
@given(
    session_changes=st.lists(
        st.sampled_from(["increment", "decrement"]), min_size=1, max_size=50
    )
)
def test_active_sessions_tracking(session_changes):
    """
    Property: Active sessions should be tracked accurately.

    This test verifies that incrementing and decrementing active sessions
    results in the correct count, and that the count never goes below zero.
    """
    health_system = HealthCheckSystem(version="test")

    expected_sessions = 0

    for change in session_changes:
        if change == "increment":
            health_system.increment_active_sessions()
            expected_sessions += 1
        else:  # decrement
            health_system.decrement_active_sessions()
            expected_sessions = max(0, expected_sessions - 1)  # Can't go below 0

    metrics = health_system.get_system_metrics()

    assert metrics.active_sessions == expected_sessions, (
        f"Active sessions mismatch: expected {expected_sessions}, "
        f"got {metrics.active_sessions}"
    )

    # Verify sessions never go negative
    assert (
        metrics.active_sessions >= 0
    ), f"Active sessions should never be negative, got {metrics.active_sessions}"


@settings(max_examples=100)
@given(
    requests=st.lists(
        st.tuples(st.booleans(), st.floats(min_value=0.1, max_value=1000.0)),
        min_size=1,
        max_size=1500,  # More than the 1000 limit
    )
)
def test_response_time_window(requests):
    """
    Property: Response time tracking should maintain a sliding window.

    This test verifies that the health system only keeps the last 1000
    response times for calculating averages, preventing unbounded memory growth.
    """
    health_system = HealthCheckSystem(version="test")

    # Record all requests
    for success, response_time_ms in requests:
        health_system.record_request(success, response_time_ms)

    # Verify that response_times list doesn't exceed 1000 entries
    assert len(health_system.response_times) <= 1000, (
        f"Response times should be limited to 1000 entries, "
        f"got {len(health_system.response_times)}"
    )

    # If we recorded more than 1000 requests, verify we kept the most recent
    if len(requests) > 1000:
        assert len(health_system.response_times) == 1000, (
            f"Should keep exactly 1000 entries when more are recorded, "
            f"got {len(health_system.response_times)}"
        )

        # Verify the average is calculated from the last 1000 entries
        expected_avg = sum(rt for _, rt in requests[-1000:]) / 1000
        metrics = health_system.get_system_metrics()

        assert abs(metrics.average_response_time_ms - expected_avg) < 0.01, (
            f"Average should be calculated from last 1000 entries: "
            f"expected {expected_avg}, got {metrics.average_response_time_ms}"
        )


@settings(max_examples=100)
@given(
    num_successes=st.integers(min_value=0, max_value=100),
    num_errors=st.integers(min_value=0, max_value=100),
)
def test_error_rate_calculation(num_successes, num_errors):
    """
    Property: Error rate should be correctly calculated as errors / total.

    This test verifies that the error rate is always between 0 and 1,
    and correctly represents the proportion of failed requests.
    """
    # Skip if no requests
    if num_successes == 0 and num_errors == 0:
        return

    health_system = HealthCheckSystem(version="test")

    # Record successes
    for _ in range(num_successes):
        health_system.record_request(True, 100.0)

    # Record errors
    for _ in range(num_errors):
        health_system.record_request(False, 100.0)

    metrics = health_system.get_system_metrics()

    # Calculate expected error rate
    total = num_successes + num_errors
    expected_error_rate = num_errors / total if total > 0 else 0.0

    # Verify error rate
    assert abs(metrics.error_rate - expected_error_rate) < 0.0001, (
        f"Error rate mismatch: expected {expected_error_rate}, "
        f"got {metrics.error_rate}"
    )

    # Verify error rate is in valid range
    assert (
        0.0 <= metrics.error_rate <= 1.0
    ), f"Error rate should be between 0 and 1, got {metrics.error_rate}"

    # Verify edge cases
    if num_errors == 0:
        assert metrics.error_rate == 0.0, "Error rate should be 0 when no errors"

    if num_successes == 0 and num_errors > 0:
        assert metrics.error_rate == 1.0, "Error rate should be 1 when all errors"
