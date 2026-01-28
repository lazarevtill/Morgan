"""Shared models for Morgan AI Assistant."""

from shared.models.base import Message, Response
from shared.models.enums import (
    HostRole,
    HostStatus,
    ServiceType,
    GPURole,
    LoadBalancingStrategy,
    ConnectionStatus,
    ServiceStatus,
    ModelType,
)

__all__ = [
    # Base models
    "Message",
    "Response",
    # Enums
    "HostRole",
    "HostStatus",
    "ServiceType",
    "GPURole",
    "LoadBalancingStrategy",
    "ConnectionStatus",
    "ServiceStatus",
    "ModelType",
]
