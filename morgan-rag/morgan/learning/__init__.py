"""
Learning and personalization module for Morgan RAG.

Provides continuous learning capabilities that analyze user interactions,
extract preferences, adapt behavior, and process feedback to improve
personalization over time.
"""

from .adaptation import (
    AdaptationResult,
    AdaptationStrategy,
    BehavioralAdaptationEngine,
    ContentSelectionAdapter,
    ResponseStyleAdapter,
)
from .consolidation import (
    ConsolidatedKnowledge,
    ConsolidatedPattern,
    ConsolidationEngine,
    ConsolidationMetrics,
    ConsolidationResult,
    ConsolidationScheduler,
    ConsolidationType,
    KnowledgeStability,
    KnowledgeSynthesizer,
    MetaPattern,
    MetaPatternExtractor,
    PatternConsolidator,
    get_consolidation_engine,
)
from .engine import LearningEngine, get_learning_engine
from .feedback import (
    FeedbackAnalysis,
    FeedbackProcessor,
    FeedbackType,
    LearningUpdate,
    UserFeedback,
)
from .patterns import (
    BehavioralPattern,
    CommunicationPattern,
    InteractionPatternAnalyzer,
    InteractionPatterns,
    TimingPattern,
    TopicPattern,
)
from .preferences import (
    PreferenceCategory,
    PreferenceExtractor,
    PreferenceStorage,
    PreferenceUpdate,
    UserPreferenceProfile,
)

__all__ = [
    # Core engine
    "LearningEngine",
    "get_learning_engine",
    # Pattern analysis
    "InteractionPatternAnalyzer",
    "InteractionPatterns",
    "CommunicationPattern",
    "TopicPattern",
    "TimingPattern",
    "BehavioralPattern",
    # Preference management
    "PreferenceExtractor",
    "PreferenceStorage",
    "UserPreferenceProfile",
    "PreferenceCategory",
    "PreferenceUpdate",
    # Behavioral adaptation
    "BehavioralAdaptationEngine",
    "AdaptationStrategy",
    "AdaptationResult",
    "ResponseStyleAdapter",
    "ContentSelectionAdapter",
    # Feedback processing
    "FeedbackProcessor",
    "UserFeedback",
    "FeedbackType",
    "FeedbackAnalysis",
    "LearningUpdate",
    # Consolidation
    "ConsolidationEngine",
    "get_consolidation_engine",
    "ConsolidatedPattern",
    "ConsolidatedKnowledge",
    "MetaPattern",
    "ConsolidationResult",
    "ConsolidationMetrics",
    "KnowledgeStability",
    "ConsolidationType",
    "PatternConsolidator",
    "KnowledgeSynthesizer",
    "MetaPatternExtractor",
    "ConsolidationScheduler",
]
