"""
Middleware utilities for FastAPI services
"""

import logging
import time
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from shared.infrastructure.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing to response headers"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}"

        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log error
            request_id = getattr(request.state, "request_id", "unknown")

            # Return JSON error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "detail": str(exc),
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm

    Implements per-IP rate limiting with configurable limits.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_second: float = 10.0,
        burst_size: Optional[int] = None,
        exempt_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.config = RateLimitConfig(
            requests_per_second=requests_per_second, burst_size=burst_size
        )
        # Store rate limiters per IP address
        self.limiters: dict[str, TokenBucketRateLimiter] = {}
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        logger.info(
            f"Rate limit middleware initialized: {requests_per_second} req/s, "
            f"burst={burst_size or int(requests_per_second)}"
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header first (for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_or_create_limiter(self, client_ip: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for client IP"""
        if client_ip not in self.limiters:
            self.limiters[client_ip] = TokenBucketRateLimiter(self.config)
            logger.debug(f"Created rate limiter for IP: {client_ip}")

        return self.limiters[client_ip]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if path is exempt from rate limiting
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client IP and rate limiter
        client_ip = self._get_client_ip(request)
        limiter = self._get_or_create_limiter(client_ip)

        try:
            # Acquire token (this will wait if rate limit exceeded)
            await limiter.acquire(tokens=1)

            # Process request
            response = await call_next(request)

            # Add rate limit info to response headers
            state = limiter.get_state()
            response.headers["X-RateLimit-Limit"] = str(
                int(self.config.requests_per_second)
            )
            response.headers["X-RateLimit-Remaining"] = str(
                int(state["available_tokens"])
            )

            return response

        except Exception as e:
            # If rate limiting fails for any reason, log and continue
            logger.error(f"Rate limiting error for {client_ip}: {e}")
            return await call_next(request)
