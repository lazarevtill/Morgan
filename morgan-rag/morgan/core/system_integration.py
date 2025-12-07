"""
Comprehensive System Integration for Morgan RAG.

Integrates all components into a cohesive companion-aware system:
- Jina AI models (embeddings, reranking, web scraping, code intelligence)
- Emotional intelligence and companion features
- Background processing with foreground operations
- Multimodal processing capabilities
- Performance validation and monitoring

Key Features:
- End-to-end workflows from document ingestion to companion responses
- Comprehensive validation for all performance targets
- Real-time companion interaction optimization
- Integrated error handling and monitoring

> [!NOTE]
> This module has been refactored to use a Facade pattern. 
> Core logic has moved to `morgan.integration`.
"""

from typing import Any, Dict, List

from morgan.config import get_settings
from morgan.utils.logger import get_logger

# Import new DDD components
from ..integration.application.services import IntegrationOrchestrator
from ..integration.domain.entities import (
    IntegrationWorkflowResult,
    SystemValidationResult,
)

logger = get_logger(__name__)


# Re-export classes for backward compatibility
SystemValidationResult = SystemValidationResult
IntegrationWorkflowResult = IntegrationWorkflowResult


class ComprehensiveSystemIntegration:
    """
    Comprehensive system integration for Morgan RAG.
    
    Acts as a Facade to the IntegrationOrchestrator.

    Provides:
    - End-to-end workflow integration
    - Performance validation against all targets
    - Companion-aware system coordination
    - Real-time optimization monitoring
    """

    def __init__(self):
        """Initialize comprehensive system integration facade."""
        self.settings = get_settings()
        
        # Initialize the orchestrator (which initializes the container)
        self.orchestrator = IntegrationOrchestrator()
        
        # Expose services for backward compatibility
        services = self.orchestrator.services
        self.batch_optimizer = services.batch_optimizer
        self.embedding_service = services.embedding_service
        self.reranking_service = services.reranking_service
        self.scraping_service = services.scraping_service
        self.multimodal_service = services.multimodal_service
        self.code_service = services.code_service
        
        self.emotional_engine = services.emotional_engine
        self.companion_manager = services.companion_manager
        self.memory_processor = services.memory_processor
        
        self.search_engine = services.search_engine
        self.background_service = services.background_service
        
        # Facade performance targets (read-only view)
        self.performance_targets = self.orchestrator.performance_targets

        logger.info("ComprehensiveSystemIntegration (Facade) initialized")

    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """
        Run comprehensive integration test covering all components.

        Returns:
            Dictionary with integration test results
        """
        # Delegate to orchestrator
        return await self.orchestrator.run_comprehensive_integration_test()

    # Legacy private methods are removed as they are now encapsulated in the orchestrator.
    # The public API is preserved via the run_comprehensive_integration_test method.

