"""
Infrastructure adapters for Morgan Integration Service.

This module handles the initialization and access to external services
required by the integration workflows.
"""


from ...background import BackgroundProcessingService
from ...companion.relationship_manager import CompanionRelationshipManager
from ...intelligence.core.intelligence_engine import EmotionalIntelligenceEngine
from ...jina.embeddings.code_service import (
    CodeIntelligenceService as CodeIntelligenceEngine,
)
from ...jina.embeddings.multimodal_service import MultimodalContentProcessor
from ...jina.embeddings.service import JinaEmbeddingService
from ...jina.reranking.service import JinaRerankingService
from ...jina.scraping.service import JinaWebScrapingService
from ...memory.memory_processor import MemoryProcessor
from ...optimization.comprehensive_batch_optimizer import (
    get_comprehensive_batch_optimizer,
)
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ServiceContainer:
    """
    Container for all services required by the integration workflows.
    Handles initialization and dependency management.
    """

    def __init__(self):
        self._initialized = False

        # Jina AI Services
        self.embedding_service = None
        self.reranking_service = None
        self.scraping_service = None
        self.multimodal_service = None
        self.code_service = None

        # Optimization
        self.batch_optimizer = None

        # Core & Companion
        self.emotional_engine = None
        self.companion_manager = None
        self.memory_processor = None
        self.search_engine = None
        self.background_service = None

        # Core Components
        self.knowledge_base = None
        self.conversation_memory = None
        self.assistant = None
        self.smart_search = None

    def initialize(self):
        """Initialize all services if not already initialized."""
        if self._initialized:
            return

        logger.info("Initializing Integration ServiceContainer...")

        # Optimization
        self.batch_optimizer = get_comprehensive_batch_optimizer()

        # Jina AI services
        self.embedding_service = JinaEmbeddingService()
        self.reranking_service = JinaRerankingService()
        self.scraping_service = JinaWebScrapingService()
        self.multimodal_service = MultimodalContentProcessor()
        self.code_service = CodeIntelligenceEngine()

        # Companion and emotional services
        self.emotional_engine = EmotionalIntelligenceEngine()
        self.companion_manager = CompanionRelationshipManager()
        self.memory_processor = MemoryProcessor()

        # Core components
        from ...core.knowledge import KnowledgeService
        from ...core.memory import MemoryService
        from ...core.assistant import MorganAssistant
        from ...core.search import SmartSearch
        from ...search.multi_stage_search import MultiStageSearchEngine

        self.knowledge_base = KnowledgeService()
        self.conversation_memory = MemoryService()
        self.assistant = MorganAssistant()
        self.smart_search = SmartSearch()

        # Search and background services
        self.search_engine = MultiStageSearchEngine()
        self.background_service = BackgroundProcessingService()

        self._initialized = True
        logger.info("ServiceContainer initialized successfully.")
