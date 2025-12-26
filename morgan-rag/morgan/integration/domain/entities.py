"""
Domain entities for the Integration Service.

This module contains the core business objects and data structures
used throughout the integration service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class SystemValidationResult:
    """Result of system validation against performance targets."""

    component: str
    target_metric: str
    target_value: float
    actual_value: float
    achieved: bool
    performance_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationWorkflowResult:
    """Result of an end-to-end integration workflow execution."""

    workflow_name: str
    total_processing_time: float
    components_involved: List[str]
    documents_processed: int
    companion_interactions: int
    search_operations: int
    background_tasks: int
    success_rate: float
    performance_targets_met: int
    total_performance_targets: int
    error_message: Optional[str] = None


@dataclass
class IntegrationEvent:
    """Base class for integration lifecycle events."""

    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)
