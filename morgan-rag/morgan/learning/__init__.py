"""
Learning and personalization module for Morgan RAG.

Provides continuous learning capabilities that analyze user interactions,
extract preferences, adapt behavior, and process feedback to improve
personalization over time.
"""

from .engine import (
    LearningEngine,
    get_learning_engine
)
from .patterns import (
    InteractionPatternAnalyzer,
    InteractionPatterns,
    CommunicationPattern,
    TopicPattern,
    TimingPattern,
    BehavioralPattern
)
from .preferences import (
    PreferenceExtractor,
    PreferenceStorage,
    UserPreferenceProfile,
    PreferenceCategory,
    PreferenceUpdate
)
from .adaptation import (
    BehavioralAdaptationEngine,
    AdaptationStrategy,
    AdaptationResult,
    ResponseStyleAdapter,
    ContentSelectionAdapter
)
from .feedback import (
    FeedbackProcessor,
    UserFeedback,
    FeedbackType,
    FeedbackAnalysis,
    LearningUpdate
)

__all__ = [
    # Core engine
    'LearningEngine',
    'get_learning_engine',
    
    # Pattern analysis
    'InteractionPatternAnalyzer',
    'InteractionPatterns',
    'CommunicationPattern',
    'TopicPattern',
    'TimingPattern',
    'BehavioralPattern',
    
    # Preference management
    'PreferenceExtractor',
    'PreferenceStorage',
    'UserPreferenceProfile',
    'PreferenceCategory',
    'PreferenceUpdate',
    
    # Behavioral adaptation
    'BehavioralAdaptationEngine',
    'AdaptationStrategy',
    'AdaptationResult',
    'ResponseStyleAdapter',
    'ContentSelectionAdapter',
    
    # Feedback processing
    'FeedbackProcessor',
    'UserFeedback',
    'FeedbackType',
    'FeedbackAnalysis',
    'LearningUpdate'
]