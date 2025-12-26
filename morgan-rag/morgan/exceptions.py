# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Morgan Exception Hierarchy.

Provides a consistent exception hierarchy for all Morgan services.
All custom exceptions inherit from MorganError for unified error handling.

Usage:
    from morgan.exceptions import LLMServiceError, EmbeddingServiceError

    try:
        response = llm_service.generate("Hello")
    except LLMServiceError as e:
        logger.error(f"LLM failed: {e.message}, operation: {e.operation}")
"""

from typing import Any, Dict, Optional


class MorganError(Exception):
    """
    Base exception for all Morgan errors.

    Provides consistent error context across all services.

    Attributes:
        message: Human-readable error message
        service: Name of the service that raised the error
        operation: The operation that failed
        details: Additional error context
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.service = service or "morgan"
        self.operation = operation
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.service:
            parts.append(f"service={self.service}")
        if self.operation:
            parts.append(f"operation={self.operation}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "service": self.service,
            "operation": self.operation,
            "details": self.details,
        }


# =============================================================================
# Service-Specific Exceptions
# =============================================================================


class LLMServiceError(MorganError):
    """
    Error from LLM service operations.

    Raised when LLM generation, chat, or streaming fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="llm",
            operation=operation,
            details=details,
        )


class EmbeddingServiceError(MorganError):
    """
    Error from embedding service operations.

    Raised when text encoding or batch embedding fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="embeddings",
            operation=operation,
            details=details,
        )


class RerankingServiceError(MorganError):
    """
    Error from reranking service operations.

    Raised when document reranking fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="reranking",
            operation=operation,
            details=details,
        )


class VectorDBError(MorganError):
    """
    Error from vector database operations.

    Raised when Qdrant operations fail.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="vector_db",
            operation=operation,
            details=details,
        )


class MemoryServiceError(MorganError):
    """
    Error from memory service operations.

    Raised when memory storage or retrieval fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="memory",
            operation=operation,
            details=details,
        )


class SearchServiceError(MorganError):
    """
    Error from search service operations.

    Raised when multi-stage search fails.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="search",
            operation=operation,
            details=details,
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(MorganError):
    """
    Error in configuration.

    Raised when configuration is invalid or missing.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message=message,
            service="config",
            operation="load",
            details=details,
        )


class ValidationError(MorganError):
    """
    Error in input validation.

    Raised when input data fails validation.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(
            message=message,
            service="validation",
            operation="validate",
            details=details,
        )


# =============================================================================
# Infrastructure Exceptions
# =============================================================================


class InfrastructureError(MorganError):
    """
    Error in infrastructure layer.

    Raised when distributed infrastructure operations fail.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if component:
            details["component"] = component
        super().__init__(
            message=message,
            service="infrastructure",
            operation=component,
            details=details,
        )


class ConnectionError(InfrastructureError):
    """
    Error connecting to a service.

    Raised when network connections fail.
    """

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(
            message=message,
            component="connection",
            details=details,
        )


class TimeoutError(InfrastructureError):
    """
    Operation timed out.

    Raised when an operation exceeds its timeout.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(
            message=message,
            component="timeout",
            details=details,
        )


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Base
    "MorganError",
    # Services
    "LLMServiceError",
    "EmbeddingServiceError",
    "RerankingServiceError",
    "VectorDBError",
    "MemoryServiceError",
    "SearchServiceError",
    # Configuration
    "ConfigurationError",
    "ValidationError",
    # Infrastructure
    "InfrastructureError",
    "ConnectionError",
    "TimeoutError",
]
