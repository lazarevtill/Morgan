"""
Property-based tests for middleware.

This module tests middleware logging completeness and error logging detail
using property-based testing with Hypothesis.
"""

import json
import logging
from io import StringIO
from datetime import datetime
from typing import Optional

from hypothesis import given, strategies as st, settings
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from morgan_server.middleware import (
    LoggingMiddleware,
    ErrorHandlingMiddleware,
    RequestValidationMiddleware,
    configure_logging,
    JSONFormatter,
)
from morgan_server.api.models import ErrorResponse


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()

    # Add test endpoints
    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"message": "success"}

    @app.post("/test")
    async def test_post_endpoint(request: Request):
        body = await request.json()
        return {"received": body}

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    @app.get("/timeout")
    async def timeout_endpoint():
        raise TimeoutError("Test timeout")

    @app.get("/not-found")
    async def not_found_endpoint():
        raise FileNotFoundError("Resource not found")

    return app


@pytest.fixture
def app_with_logging(app):
    """Create app with logging middleware."""
    app.add_middleware(LoggingMiddleware)
    return app


@pytest.fixture
def app_with_error_handling(app):
    """Create app with error handling middleware."""
    app.add_middleware(ErrorHandlingMiddleware)
    return app


@pytest.fixture
def app_with_all_middleware(app):
    """Create app with all middleware."""
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestValidationMiddleware)
    return app


class LogCapture:
    """Helper to capture log output."""

    def __init__(self, logger_name: str = "morgan"):
        self.logger_name = logger_name
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(JSONFormatter())
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()

    def __enter__(self):
        """Start capturing logs."""
        self.logger.handlers.clear()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *args):
        """Stop capturing logs."""
        self.logger.removeHandler(self.handler)
        self.logger.handlers = self.original_handlers
        self.logger.setLevel(self.original_level)

    def get_logs(self) -> list[dict]:
        """Get captured logs as list of dicts."""
        logs = []
        for line in self.stream.getvalue().strip().split("\n"):
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return logs


# ============================================================================
# Property 23: Request Logging Completeness
# ============================================================================


class TestRequestLoggingCompleteness:
    """
    Property-based tests for request logging completeness.

    **Feature: client-server-separation, Property 23: Request logging completeness**

    For any API request processed by the server, a log entry should be created
    containing the timestamp, user ID (if available), endpoint, and response time.

    **Validates: Requirements 10.1**
    """

    @given(
        method=st.sampled_from(["GET", "POST", "PUT", "DELETE"]),
        path=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=50
        ).map(lambda x: f"/{x.replace('/', '_')}"),
        user_id=st.one_of(
            st.none(),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
                min_size=1,
                max_size=50,
            ),
        ),
        conversation_id=st.one_of(
            st.none(),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
                min_size=1,
                max_size=50,
            ),
        ),
    )
    @settings(max_examples=100)
    def test_property_request_logging_completeness(
        self, method, path, user_id, conversation_id
    ):
        """
        Property: All requests are logged with complete information.

        For any request (method, path, user_id, conversation_id), the logging
        middleware should create log entries containing:
        - timestamp
        - request_id
        - method
        - path
        - user_id (if provided)
        - conversation_id (if provided)
        - response_time_ms
        - status_code
        """
        # Create app with logging middleware
        app = FastAPI()

        @app.api_route(path, methods=[method])
        async def dynamic_endpoint(request: Request):
            return {"message": "success"}

        app.add_middleware(LoggingMiddleware)

        client = TestClient(app)

        # Capture logs
        with LogCapture("morgan.middleware.logging") as log_capture:
            # Make request with headers
            headers = {}
            if user_id:
                headers["X-User-ID"] = user_id
            if conversation_id:
                headers["X-Conversation-ID"] = conversation_id

            response = client.request(method, path, headers=headers)

            # Get captured logs
            logs = log_capture.get_logs()

            # Should have at least 2 log entries (request received, request completed)
            assert len(logs) >= 2, f"Expected at least 2 log entries, got {len(logs)}"

            # Check request received log
            request_log = logs[0]
            assert "timestamp" in request_log, "Request log missing timestamp"
            assert "request_id" in request_log, "Request log missing request_id"
            assert (
                request_log.get("method") == method
            ), f"Expected method {method}, got {request_log.get('method')}"
            assert (
                request_log.get("path") == path
            ), f"Expected path {path}, got {request_log.get('path')}"

            if user_id:
                assert (
                    request_log.get("user_id") == user_id
                ), "Request log missing or incorrect user_id"

            if conversation_id:
                assert (
                    request_log.get("conversation_id") == conversation_id
                ), "Request log missing or incorrect conversation_id"

            # Check request completed log
            response_log = logs[1]
            assert "timestamp" in response_log, "Response log missing timestamp"
            assert "request_id" in response_log, "Response log missing request_id"
            assert (
                "response_time_ms" in response_log
            ), "Response log missing response_time_ms"
            assert "status_code" in response_log, "Response log missing status_code"
            assert (
                response_log.get("method") == method
            ), f"Expected method {method} in response log"
            assert (
                response_log.get("path") == path
            ), f"Expected path {path} in response log"

            # Verify response time is a positive number
            assert isinstance(
                response_log["response_time_ms"], (int, float)
            ), "response_time_ms should be numeric"
            assert (
                response_log["response_time_ms"] >= 0
            ), "response_time_ms should be non-negative"

            # Verify status code is valid
            assert isinstance(
                response_log["status_code"], int
            ), "status_code should be integer"
            assert (
                100 <= response_log["status_code"] < 600
            ), "status_code should be valid HTTP status"

            # Verify request IDs match
            assert (
                request_log["request_id"] == response_log["request_id"]
            ), "Request IDs should match"

    @given(
        status_code=st.integers(min_value=200, max_value=599),
    )
    @settings(max_examples=100)
    def test_property_response_status_logged(self, status_code):
        """
        Property: Response status codes are always logged.

        For any response status code, the logging middleware should include
        it in the log entry.
        """
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return JSONResponse(content={"status": "ok"}, status_code=status_code)

        app.add_middleware(LoggingMiddleware)

        client = TestClient(app)

        with LogCapture("morgan.middleware.logging") as log_capture:
            response = client.get("/test")

            logs = log_capture.get_logs()

            # Find the response log (should be the last one)
            response_log = logs[-1]

            assert "status_code" in response_log, "Response log missing status_code"
            assert (
                response_log["status_code"] == status_code
            ), f"Expected status {status_code}, got {response_log['status_code']}"


# ============================================================================
# Property 24: Error Logging Detail
# ============================================================================


class TestErrorLoggingDetail:
    """
    Property-based tests for error logging detail.

    **Feature: client-server-separation, Property 24: Error logging detail**

    For any error that occurs during request processing, a log entry should be
    created at the ERROR level containing the stack trace and relevant context
    (request details, user ID, conversation ID).

    **Validates: Requirements 10.2**
    """

    @given(
        error_message=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_",
            min_size=1,
            max_size=200,
        ),
        user_id=st.one_of(
            st.none(),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
                min_size=1,
                max_size=50,
            ),
        ),
        conversation_id=st.one_of(
            st.none(),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
                min_size=1,
                max_size=50,
            ),
        ),
    )
    @settings(max_examples=100)
    def test_property_error_logging_detail(
        self, error_message, user_id, conversation_id
    ):
        """
        Property: All errors are logged with complete context.

        For any error that occurs, the error handling middleware should log:
        - timestamp
        - request_id
        - method
        - path
        - error_code
        - error_type
        - error_message
        - status_code
        - user_id (if provided)
        - conversation_id (if provided)
        - stack_trace
        """
        app = FastAPI()

        @app.get("/error")
        async def error_endpoint():
            raise ValueError(error_message)

        app.add_middleware(ErrorHandlingMiddleware)
        app.add_middleware(LoggingMiddleware)

        client = TestClient(app)

        # Capture logs
        with LogCapture("morgan.middleware.error") as log_capture:
            # Make request with headers
            headers = {}
            if user_id:
                headers["X-User-ID"] = user_id
            if conversation_id:
                headers["X-Conversation-ID"] = conversation_id

            response = client.get("/error", headers=headers)

            # Get captured logs
            logs = log_capture.get_logs()

            # Should have at least 1 error log entry
            assert (
                len(logs) >= 1
            ), f"Expected at least 1 error log entry, got {len(logs)}"

            # Check error log
            error_log = logs[0]

            # Verify required fields
            assert "timestamp" in error_log, "Error log missing timestamp"
            assert "level" in error_log, "Error log missing level"
            assert (
                error_log["level"] == "ERROR"
            ), f"Expected ERROR level, got {error_log['level']}"

            assert "method" in error_log, "Error log missing method"
            assert (
                error_log["method"] == "GET"
            ), f"Expected GET method, got {error_log['method']}"

            assert "path" in error_log, "Error log missing path"
            assert (
                error_log["path"] == "/error"
            ), f"Expected /error path, got {error_log['path']}"

            assert "error_code" in error_log, "Error log missing error_code"
            assert "error_type" in error_log, "Error log missing error_type"
            assert (
                error_log["error_type"] == "ValueError"
            ), f"Expected ValueError, got {error_log['error_type']}"

            assert "error_message" in error_log, "Error log missing error_message"
            assert (
                error_message in error_log["error_message"]
            ), "Error message not in log"

            assert "status_code" in error_log, "Error log missing status_code"
            assert (
                error_log["status_code"] == 400
            ), f"Expected 400 status, got {error_log['status_code']}"

            assert "stack_trace" in error_log, "Error log missing stack_trace"
            assert len(error_log["stack_trace"]) > 0, "Stack trace should not be empty"

            # Verify user context if provided
            if user_id:
                assert (
                    error_log.get("user_id") == user_id
                ), "Error log missing or incorrect user_id"

            if conversation_id:
                assert (
                    error_log.get("conversation_id") == conversation_id
                ), "Error log missing or incorrect conversation_id"

            # Verify exception info is present
            assert "exception" in error_log, "Error log missing exception info"

    @given(
        error_type=st.sampled_from(
            [
                (ValueError, "INVALID_REQUEST", 400),
                (FileNotFoundError, "NOT_FOUND", 404),
                (TimeoutError, "TIMEOUT", 504),
                (RuntimeError, "INTERNAL_ERROR", 500),
            ]
        ),
    )
    @settings(max_examples=100)
    def test_property_different_error_types_logged(self, error_type):
        """
        Property: Different error types are logged with appropriate details.

        For any error type, the error handling middleware should log the
        error with the correct error_type, error_code, and status_code.
        """
        exception_class, expected_code, expected_status = error_type

        app = FastAPI()

        @app.get("/error")
        async def error_endpoint():
            raise exception_class("Test error")

        app.add_middleware(ErrorHandlingMiddleware)

        client = TestClient(app)

        with LogCapture("morgan.middleware.error") as log_capture:
            response = client.get("/error")

            logs = log_capture.get_logs()

            assert len(logs) >= 1, "Expected at least 1 error log"

            error_log = logs[0]

            # Verify error type
            assert (
                error_log.get("error_type") == exception_class.__name__
            ), f"Expected {exception_class.__name__}, got {error_log.get('error_type')}"

            # Verify error code
            assert (
                error_log.get("error_code") == expected_code
            ), f"Expected {expected_code}, got {error_log.get('error_code')}"

            # Verify status code
            assert (
                error_log.get("status_code") == expected_status
            ), f"Expected {expected_status}, got {error_log.get('status_code')}"

            # Verify stack trace is present
            assert "stack_trace" in error_log, "Error log missing stack_trace"
            assert len(error_log["stack_trace"]) > 0, "Stack trace should not be empty"

    @given(
        path=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                min_codepoint=ord("a"),
                max_codepoint=ord("z"),
            ),
            min_size=1,
            max_size=50,
        ).map(lambda x: f"/{x.replace('/', '_')}"),
    )
    @settings(max_examples=100)
    def test_property_error_context_includes_path(self, path):
        """
        Property: Error logs include the request path.

        For any request path that results in an error, the error log should
        include the path in the context.
        """
        app = FastAPI()

        @app.api_route(path, methods=["GET"])
        async def error_endpoint():
            raise ValueError("Test error")

        app.add_middleware(ErrorHandlingMiddleware)

        client = TestClient(app)

        with LogCapture("morgan.middleware.error") as log_capture:
            response = client.get(path)

            logs = log_capture.get_logs()

            assert len(logs) >= 1, "Expected at least 1 error log"

            error_log = logs[0]

            assert "path" in error_log, "Error log missing path"
            assert (
                error_log["path"] == path
            ), f"Expected path {path}, got {error_log['path']}"
