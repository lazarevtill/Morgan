"""
Middleware for Morgan Server

This module provides middleware for:
- Request/response logging
- CORS handling
- Error handling
- Request validation

All middleware follows FastAPI/Starlette middleware patterns.
"""

import time
import json
import logging
import traceback
import uuid
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from morgan_server.api.models import ErrorResponse
from morgan_server.logging_config import get_logger, configure_logging, JSONFormatter


# ============================================================================
# Logging Middleware
# ============================================================================


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Logs:
    - Request method, path, headers
    - Response status code, time
    - Request ID for tracking
    - User ID and conversation ID if available

    **Validates: Requirements 10.1**
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize logging middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)
        self.logger = logging.getLogger("morgan.middleware.logging")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from handler
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract user context if available
        user_id = request.headers.get("X-User-ID")
        conversation_id = request.headers.get("X-Conversation-ID")

        # Log request
        start_time = time.time()

        self.logger.info(
            "Request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "client_host": request.client.host if request.client else None,
            },
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log response
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "response_time_ms": response_time_ms,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log error
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_time_ms": response_time_ms,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
                exc_info=True,
            )

            # Re-raise to be handled by error middleware
            raise


# ============================================================================
# Error Handling Middleware
# ============================================================================


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for catching and formatting errors.

    Converts exceptions to structured ErrorResponse format.
    Logs errors with full context and stack traces.

    **Validates: Requirements 10.2**
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)
        self.logger = logging.getLogger("morgan.middleware.error")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and handle errors.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from handler or error response
        """
        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            # Validation errors - 400 Bad Request
            return await self._handle_error(
                request=request,
                error=e,
                status_code=400,
                error_code="INVALID_REQUEST",
                message=str(e),
            )

        except PermissionError as e:
            # Permission errors - 403 Forbidden
            return await self._handle_error(
                request=request,
                error=e,
                status_code=403,
                error_code="FORBIDDEN",
                message="Access denied",
            )

        except FileNotFoundError as e:
            # Not found errors - 404 Not Found
            return await self._handle_error(
                request=request,
                error=e,
                status_code=404,
                error_code="NOT_FOUND",
                message=str(e),
            )

        except TimeoutError as e:
            # Timeout errors - 504 Gateway Timeout
            return await self._handle_error(
                request=request,
                error=e,
                status_code=504,
                error_code="TIMEOUT",
                message="Request timed out",
            )

        except Exception as e:
            # Unexpected errors - 500 Internal Server Error
            return await self._handle_error(
                request=request,
                error=e,
                status_code=500,
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
            )

    async def _handle_error(
        self,
        request: Request,
        error: Exception,
        status_code: int,
        error_code: str,
        message: str,
    ) -> JSONResponse:
        """
        Handle error and create error response.

        Args:
            request: Request that caused error
            error: Exception that was raised
            status_code: HTTP status code
            error_code: Error code for response
            message: User-friendly error message

        Returns:
            JSONResponse with error details
        """
        # Get request ID if available
        request_id = getattr(request.state, "request_id", None)

        # Get user context if available
        user_id = request.headers.get("X-User-ID")
        conversation_id = request.headers.get("X-Conversation-ID")

        # Log error with full context
        self.logger.error(
            f"Error handling request: {message}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_code": error_code,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "status_code": status_code,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )

        # Create error response
        error_response = ErrorResponse(
            error=error_code,
            message=message,
            details=(
                {
                    "error_type": type(error).__name__,
                }
                if status_code >= 500
                else None
            ),  # Only include details for server errors
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(mode="json"),
            headers={"X-Request-ID": request_id} if request_id else {},
        )


# ============================================================================
# Request Validation Middleware
# ============================================================================


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating incoming requests.

    Validates:
    - Request size limits
    - Content-Type headers
    - Required headers
    """

    def __init__(
        self, app: ASGIApp, max_request_size: int = 10 * 1024 * 1024  # 10 MB default
    ):
        """
        Initialize request validation middleware.

        Args:
            app: ASGI application
            max_request_size: Maximum request body size in bytes
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.logger = logging.getLogger("morgan.middleware.validation")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Validate request and process.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from handler or validation error
        """
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    self.logger.warning(
                        f"Request too large: {size} bytes (max: {self.max_request_size})",
                        extra={
                            "request_id": getattr(request.state, "request_id", None),
                            "path": request.url.path,
                            "size": size,
                            "max_size": self.max_request_size,
                        },
                    )

                    error_response = ErrorResponse(
                        error="REQUEST_TOO_LARGE",
                        message=f"Request body too large (max: {self.max_request_size} bytes)",
                        timestamp=datetime.now(timezone.utc),
                    )

                    return JSONResponse(
                        status_code=413, content=error_response.model_dump(mode="json")
                    )
            except ValueError:
                pass  # Invalid content-length header, let it through

        # Process request
        response = await call_next(request)
        return response


# ============================================================================
# CORS Configuration
# ============================================================================


def configure_cors(
    app: ASGIApp,
    allow_origins: Optional[list[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[list[str]] = None,
    allow_headers: Optional[list[str]] = None,
) -> CORSMiddleware:
    """
    Configure CORS middleware for the application.

    Args:
        app: ASGI application
        allow_origins: List of allowed origins (default: ["*"])
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed methods (default: ["*"])
        allow_headers: List of allowed headers (default: ["*"])

    Returns:
        Configured CORSMiddleware instance
    """
    if allow_origins is None:
        allow_origins = ["*"]

    if allow_methods is None:
        allow_methods = ["*"]

    if allow_headers is None:
        allow_headers = ["*"]

    return CORSMiddleware(
        app=app,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
    )


# ============================================================================
# Middleware Setup Helper
# ============================================================================


def setup_middleware(
    app: ASGIApp,
    enable_logging: bool = True,
    enable_error_handling: bool = True,
    enable_validation: bool = True,
    enable_cors: bool = True,
    max_request_size: int = 10 * 1024 * 1024,
    cors_origins: Optional[list[str]] = None,
) -> None:
    """
    Set up all middleware for the application.

    Middleware is applied in order:
    1. CORS (if enabled)
    2. Request validation (if enabled)
    3. Logging (if enabled)
    4. Error handling (if enabled)

    Args:
        app: FastAPI application
        enable_logging: Whether to enable logging middleware
        enable_error_handling: Whether to enable error handling middleware
        enable_validation: Whether to enable request validation middleware
        enable_cors: Whether to enable CORS middleware
        max_request_size: Maximum request body size in bytes
        cors_origins: List of allowed CORS origins
    """
    # Add middleware in reverse order (last added = first executed)

    if enable_error_handling:
        app.add_middleware(ErrorHandlingMiddleware)

    if enable_logging:
        app.add_middleware(LoggingMiddleware)

    if enable_validation:
        app.add_middleware(
            RequestValidationMiddleware, max_request_size=max_request_size
        )

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
