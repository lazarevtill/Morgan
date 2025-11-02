"""
Preference Extraction and Storage for Morgan RAG.

Extracts user preferences from interactions and stores them for personalization.
Handles preference categories, confidence scoring, and preference evolution over time.
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from ..utils.logger import get_logger
from ..config import get_settings
from ..emotional.models import (
    InteractionData, UserPreferences, CommunicationStyle, 
    ResponseLength, ConversationContext
)

logger = get_logger(__name__)


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    COMMUNICATION = "communication"
    CONTENT = "content"
    TOPICS = "topics"
    LEARNING = "learning"
    INTERACTION = "interaction"
    EMOTIONAL = "emotional"
    TECHNICAL = "technical"


class PreferenceSource(Enum):
    """Sources of preference information."""
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    CONVERSATION_ANALYSIS = "conversation_analysis"
    PATTERN_INFERENCE = "pattern_inference"


@dataclass
class PreferenceUpdate:
    """Represents an update to a user preference."""
    update_id: str
    user_id: str
    category: PreferenceCategory
    preference_key: str
    preference_value: Any
    confidence_score: float  # 0.0 to 1.0
    source: PreferenceSource
    evidence: List[str]  # Supporting evidence for the preference
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize update ID if not provided."""
        if not self.update_id:
            self.update_id = str(uuid.uuid4())


@dataclass
class UserPreferenceProfile:
    """Complete preference profile for a user."""
    user_id: str
    preferences: Dict[str, Dict[str, Any]]  # category -> {key: value}
    confidence_scores: Dict[str, Dict[str, float]]  # category -> {key: confidence}
    preference_sources: Dict[str, Dict[str, PreferenceSource]]  # category -> {key: source}
    preference_history: List[PreferenceUpdate] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    profile_version: int = 1
    
    def get_preference(
        self,
        category: PreferenceCategory,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a specific preference value."""
        cat_str = category.value
        return self.preferences.get(cat_str, {}).get(key, default)
    
    def get_confidence(
        self,
        category: PreferenceCategory,
        key: str
    ) -> float:
        """Get confidence score for a preference."""
        cat_str = category.value
        return self.confidence_scores.get(cat_str, {}).get(key, 0.0)
    
    def set_preference(
        self,
        category: PreferenceCategory,
        key: str,
        value: Any,
        confidence: float,
        source: PreferenceSource
    ):
        """Set a preference value."""
        cat_str = category.value
        
        if cat_str not in self.preferences:
            self.preferences[cat_str] = {}
            self.confidence_scores[cat_str] = {}
            self.preference_sources[cat_str] = {}
        
        self.preferences[cat_str][key] = value
        self.confidence_scores[cat_str][key] = confidence
        self.preference_sources[cat_str][key] = source
        self.last_updated = datetime.utcnow()


class PreferenceExtractor:
    """
    Extracts user preferences from interaction data.
    
    Analyzes conversations, feedback, and behavior patterns to identify
    user preferences across different categories.
    """
    
    # Preference extraction patterns
    COMMUNICATION_PATTERNS = {
        'formal_language': [
            r'\b(please|thank you|would you|could you|may I)\b',
            r'\b(appreciate|grateful|kindly|respectfully)\b'
        ],
        'casual_language': [
            r'\b(hey|hi|yeah|yep|cool|awesome)\b',
            r'[!]{2,}|[?]{2,}'
        ],
        'technical_preference': [
            r'\b(technical|detailed|specific|precise)\b',
            r'\b(algorithm|implementation|architecture)\b'
        ],
        'simple_preference': [
            r'\b(simple|easy|basic|straightforward)\b',
            r'\b(explain simply|in simple terms)\b'
        ]
    }
    
    CONTENT_PATTERNS = {
        'brief_responses': [
            r'\b(brief|short|quick|summary)\b',
            r'\b(just the|only the|main points)\b'
        ],
        'detailed_responses': [
            r'\b(detailed|comprehensive|thorough|complete)\b',
            r'\b(explain everything|full explanation)\b'
        ],
        'examples_preferred': [
            r'\b(example|for instance|show me|demonstrate)\b',
            r'\b(sample|illustration|case study)\b'
        ]
    }
    
    TOPIC_PATTERNS = {
        'learning_topics': [
            r'\b(learn|understand|study|master)\b',
            r'\b(how to|tutorial|guide|course)\b'
        ],
        'work_topics': [
            r'\b(work|job|project|business|professional)\b',
            r'\b(career|workplace|office|meeting)\b'
        ],
        'personal_topics': [
            r'\b(personal|family|friend|hobby|interest)\b',
            r'\b(free time|weekend|vacation|fun)\b'
        ]
    }
    
    def __init__(self):
        """Initialize the preference extractor."""
        logger.info("Preference extractor initialized")
    
    def extract_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """
        Extract preferences from user interactions.
        
        Args:
            user_id: User identifier
            interactions: List of interactions to analyze
            
        Returns:
            List[PreferenceUpdate]: Extracted preference updates
        """
        logger.info(f"Extracting preferences for user {user_id} from {len(interactions)} interactions")
        
        preference_updates = []
        
        # Extract communication preferences
        comm_updates = self._extract_communication_preferences(user_id, interactions)
        preference_updates.extend(comm_updates)
        
        # Extract content preferences
        content_updates = self._extract_content_preferences(user_id, interactions)
        preference_updates.extend(content_updates)
        
        # Extract topic preferences
        topic_updates = self._extract_topic_preferences(user_id, interactions)
        preference_updates.extend(topic_updates)
        
        # Extract learning preferences
        learning_updates = self._extract_learning_preferences(user_id, interactions)
        preference_updates.extend(learning_updates)
        
        # Extract interaction preferences
        interaction_updates = self._extract_interaction_preferences(user_id, interactions)
        preference_updates.extend(interaction_updates)
        
        # Extract emotional preferences
        emotional_updates = self._extract_emotional_preferences(user_id, interactions)
        preference_updates.extend(emotional_updates)
        
        logger.info(f"Extracted {len(preference_updates)} preference updates for user {user_id}")
        return preference_updates
    
    def _extract_communication_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract communication style preferences."""
        updates = []
        
        # Collect all messages
        messages = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'message_text'):
                messages.append(interaction.conversation_context.message_text)
        
        if not messages:
            return updates
        
        # Analyze formality preference
        formal_score = self._calculate_pattern_score(messages, self.COMMUNICATION_PATTERNS['formal_language'])
        casual_score = self._calculate_pattern_score(messages, self.COMMUNICATION_PATTERNS['casual_language'])
        
        if formal_score > casual_score and formal_score > 0.3:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.COMMUNICATION,
                preference_key="formality_level",
                preference_value="formal",
                confidence_score=min(formal_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Formal language patterns detected in {len(messages)} messages"]
            ))
        elif casual_score > 0.3:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.COMMUNICATION,
                preference_key="formality_level",
                preference_value="casual",
                confidence_score=min(casual_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Casual language patterns detected in {len(messages)} messages"]
            ))
        
        # Analyze technical preference
        technical_score = self._calculate_pattern_score(messages, self.COMMUNICATION_PATTERNS['technical_preference'])
        simple_score = self._calculate_pattern_score(messages, self.COMMUNICATION_PATTERNS['simple_preference'])
        
        if technical_score > simple_score and technical_score > 0.2:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.COMMUNICATION,
                preference_key="technical_depth",
                preference_value="high",
                confidence_score=min(technical_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Technical language preference detected"]
            ))
        elif simple_score > 0.2:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.COMMUNICATION,
                preference_key="technical_depth",
                preference_value="low",
                confidence_score=min(simple_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Simple language preference detected"]
            ))
        
        return updates
    
    def _extract_content_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract content format and length preferences."""
        updates = []
        
        messages = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'message_text'):
                messages.append(interaction.conversation_context.message_text)
        
        if not messages:
            return updates
        
        # Analyze response length preference
        brief_score = self._calculate_pattern_score(messages, self.CONTENT_PATTERNS['brief_responses'])
        detailed_score = self._calculate_pattern_score(messages, self.CONTENT_PATTERNS['detailed_responses'])
        
        if brief_score > detailed_score and brief_score > 0.2:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.CONTENT,
                preference_key="response_length",
                preference_value="brief",
                confidence_score=min(brief_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Brief response preference detected"]
            ))
        elif detailed_score > 0.2:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.CONTENT,
                preference_key="response_length",
                preference_value="detailed",
                confidence_score=min(detailed_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Detailed response preference detected"]
            ))
        
        # Analyze example preference
        example_score = self._calculate_pattern_score(messages, self.CONTENT_PATTERNS['examples_preferred'])
        if example_score > 0.2:
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.CONTENT,
                preference_key="examples_preferred",
                preference_value=True,
                confidence_score=min(example_score, 1.0),
                source=PreferenceSource.CONVERSATION_ANALYSIS,
                evidence=[f"Examples preference detected in conversations"]
            ))
        
        return updates
    
    def _extract_topic_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract topic interest preferences."""
        updates = []
        
        # Collect topics from interactions
        all_topics = []
        for interaction in interactions:
            topics = getattr(interaction, 'topics_discussed', [])
            all_topics.extend(topics)
        
        if all_topics:
            # Count topic frequencies
            from collections import Counter
            topic_counter = Counter(all_topics)
            
            # Add top topics as preferences
            for topic, count in topic_counter.most_common(5):
                confidence = min(count / len(interactions), 1.0)
                if confidence > 0.2:  # Only include topics with reasonable frequency
                    updates.append(PreferenceUpdate(
                        update_id=str(uuid.uuid4()),
                        user_id=user_id,
                        category=PreferenceCategory.TOPICS,
                        preference_key=f"interest_{topic}",
                        preference_value=confidence,
                        confidence_score=confidence,
                        source=PreferenceSource.IMPLICIT_BEHAVIOR,
                        evidence=[f"Topic '{topic}' discussed {count} times"]
                    ))
        
        # Analyze topic category preferences from messages
        messages = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'message_text'):
                messages.append(interaction.conversation_context.message_text)
        
        if messages:
            learning_score = self._calculate_pattern_score(messages, self.TOPIC_PATTERNS['learning_topics'])
            work_score = self._calculate_pattern_score(messages, self.TOPIC_PATTERNS['work_topics'])
            personal_score = self._calculate_pattern_score(messages, self.TOPIC_PATTERNS['personal_topics'])
            
            if learning_score > 0.2:
                updates.append(PreferenceUpdate(
                    update_id=str(uuid.uuid4()),
                    user_id=user_id,
                    category=PreferenceCategory.TOPICS,
                    preference_key="learning_focus",
                    preference_value=True,
                    confidence_score=min(learning_score, 1.0),
                    source=PreferenceSource.CONVERSATION_ANALYSIS,
                    evidence=[f"Learning-focused conversations detected"]
                ))
            
            if work_score > 0.2:
                updates.append(PreferenceUpdate(
                    update_id=str(uuid.uuid4()),
                    user_id=user_id,
                    category=PreferenceCategory.TOPICS,
                    preference_key="work_focus",
                    preference_value=True,
                    confidence_score=min(work_score, 1.0),
                    source=PreferenceSource.CONVERSATION_ANALYSIS,
                    evidence=[f"Work-focused conversations detected"]
                ))
            
            if personal_score > 0.2:
                updates.append(PreferenceUpdate(
                    update_id=str(uuid.uuid4()),
                    user_id=user_id,
                    category=PreferenceCategory.TOPICS,
                    preference_key="personal_focus",
                    preference_value=True,
                    confidence_score=min(personal_score, 1.0),
                    source=PreferenceSource.CONVERSATION_ANALYSIS,
                    evidence=[f"Personal conversations detected"]
                ))
        
        return updates
    
    def _extract_learning_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract learning style and goal preferences."""
        updates = []
        
        # Analyze learning indicators from interactions
        learning_indicators = []
        for interaction in interactions:
            indicators = getattr(interaction, 'learning_indicators', [])
            learning_indicators.extend(indicators)
        
        if learning_indicators:
            from collections import Counter
            indicator_counter = Counter(learning_indicators)
            
            for indicator, count in indicator_counter.most_common(3):
                confidence = min(count / len(interactions), 1.0)
                if confidence > 0.1:
                    updates.append(PreferenceUpdate(
                        update_id=str(uuid.uuid4()),
                        user_id=user_id,
                        category=PreferenceCategory.LEARNING,
                        preference_key=f"learning_style_{indicator}",
                        preference_value=confidence,
                        confidence_score=confidence,
                        source=PreferenceSource.IMPLICIT_BEHAVIOR,
                        evidence=[f"Learning indicator '{indicator}' observed {count} times"]
                    ))
        
        return updates
    
    def _extract_interaction_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract interaction style preferences."""
        updates = []
        
        if not interactions:
            return updates
        
        # Analyze feedback frequency
        feedback_count = 0
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'user_feedback'):
                if interaction.conversation_context.user_feedback is not None:
                    feedback_count += 1
        
        feedback_frequency = feedback_count / len(interactions)
        if feedback_frequency > 0.1:  # User provides feedback regularly
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.INTERACTION,
                preference_key="feedback_frequency",
                preference_value=feedback_frequency,
                confidence_score=min(feedback_frequency * 2, 1.0),
                source=PreferenceSource.IMPLICIT_BEHAVIOR,
                evidence=[f"User provides feedback {feedback_frequency:.1%} of the time"]
            ))
        
        # Analyze session length preferences
        session_durations = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, 'session_duration'):
                duration = interaction.conversation_context.session_duration
                if duration:
                    session_durations.append(duration.total_seconds())
        
        if session_durations:
            avg_duration = sum(session_durations) / len(session_durations)
            if avg_duration < 300:  # Less than 5 minutes
                session_pref = "short"
            elif avg_duration > 1800:  # More than 30 minutes
                session_pref = "long"
            else:
                session_pref = "medium"
            
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.INTERACTION,
                preference_key="session_length_preference",
                preference_value=session_pref,
                confidence_score=0.7,
                source=PreferenceSource.IMPLICIT_BEHAVIOR,
                evidence=[f"Average session duration: {avg_duration:.0f} seconds"]
            ))
        
        return updates
    
    def _extract_emotional_preferences(
        self,
        user_id: str,
        interactions: List[InteractionData]
    ) -> List[PreferenceUpdate]:
        """Extract emotional interaction preferences."""
        updates = []
        
        # Analyze emotional states from interactions
        emotional_states = []
        for interaction in interactions:
            if interaction.emotional_state:
                emotional_states.append(interaction.emotional_state)
        
        if emotional_states:
            # Analyze dominant emotions
            from collections import Counter
            emotion_counter = Counter(state.primary_emotion for state in emotional_states)
            
            # Analyze emotional intensity preferences
            avg_intensity = sum(state.intensity for state in emotional_states) / len(emotional_states)
            
            if avg_intensity > 0.7:
                intensity_pref = "high"
            elif avg_intensity < 0.3:
                intensity_pref = "low"
            else:
                intensity_pref = "medium"
            
            updates.append(PreferenceUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                category=PreferenceCategory.EMOTIONAL,
                preference_key="emotional_intensity_comfort",
                preference_value=intensity_pref,
                confidence_score=0.6,
                source=PreferenceSource.IMPLICIT_BEHAVIOR,
                evidence=[f"Average emotional intensity: {avg_intensity:.2f}"]
            ))
            
            # Add dominant emotion preferences
            for emotion, count in emotion_counter.most_common(2):
                confidence = min(count / len(emotional_states), 1.0)
                if confidence > 0.2:
                    updates.append(PreferenceUpdate(
                        update_id=str(uuid.uuid4()),
                        user_id=user_id,
                        category=PreferenceCategory.EMOTIONAL,
                        preference_key=f"comfortable_with_{emotion.value}",
                        preference_value=confidence,
                        confidence_score=confidence,
                        source=PreferenceSource.IMPLICIT_BEHAVIOR,
                        evidence=[f"Emotion '{emotion.value}' observed {count} times"]
                    ))
        
        return updates
    
    def _calculate_pattern_score(self, messages: List[str], patterns: List[str]) -> float:
        """Calculate pattern match score for messages."""
        import re
        
        total_matches = 0
        total_words = 0
        
        for message in messages:
            words = message.lower().split()
            total_words += len(words)
            
            for pattern in patterns:
                matches = re.findall(pattern, message.lower())
                total_matches += len(matches)
        
        if total_words == 0:
            return 0.0
        
        return min(total_matches / (total_words / 100), 1.0)  # Normalize per 100 words


class PreferenceStorage:
    """
    Stores and manages user preference profiles.
    
    Handles persistence, retrieval, and updating of user preferences
    with confidence tracking and history management.
    """
    
    def __init__(self):
        """Initialize preference storage."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_dir) / "preferences"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.preference_cache: Dict[str, UserPreferenceProfile] = {}
        
        logger.info(f"Preference storage initialized at {self.storage_path}")
    
    def get_user_preferences(self, user_id: str) -> UserPreferenceProfile:
        """
        Get user preference profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferenceProfile: User's preference profile
        """
        # Check cache first
        if user_id in self.preference_cache:
            return self.preference_cache[user_id]
        
        # Load from storage
        profile_path = self.storage_path / f"{user_id}_preferences.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert back to dataclass
                profile = UserPreferenceProfile(
                    user_id=data['user_id'],
                    preferences=data['preferences'],
                    confidence_scores=data['confidence_scores'],
                    preference_sources=data.get('preference_sources', {}),
                    preference_history=[],  # History not loaded for performance
                    last_updated=datetime.fromisoformat(data['last_updated']),
                    profile_version=data.get('profile_version', 1)
                )
                
                # Cache the profile
                self.preference_cache[user_id] = profile
                return profile
                
            except Exception as e:
                logger.error(f"Error loading preferences for user {user_id}: {e}")
        
        # Create new profile if none exists
        profile = UserPreferenceProfile(
            user_id=user_id,
            preferences={},
            confidence_scores={},
            preference_sources={}
        )
        
        self.preference_cache[user_id] = profile
        return profile
    
    def update_preference(self, user_id: str, update: PreferenceUpdate):
        """
        Update a user preference.
        
        Args:
            user_id: User identifier
            update: Preference update to apply
        """
        profile = self.get_user_preferences(user_id)
        
        # Apply the update
        profile.set_preference(
            category=update.category,
            key=update.preference_key,
            value=update.preference_value,
            confidence=update.confidence_score,
            source=update.source
        )
        
        # Add to history
        profile.preference_history.append(update)
        
        # Update cache
        self.preference_cache[user_id] = profile
        
        # Save to storage
        self._save_profile(profile)
        
        logger.debug(f"Updated preference {update.preference_key} for user {user_id}")
    
    def save_user_preferences(self, profile: UserPreferenceProfile):
        """
        Save user preference profile.
        
        Args:
            profile: User preference profile to save
        """
        self.preference_cache[profile.user_id] = profile
        self._save_profile(profile)
    
    def _save_profile(self, profile: UserPreferenceProfile):
        """Save profile to storage."""
        profile_path = self.storage_path / f"{profile.user_id}_preferences.json"
        
        try:
            # Convert sources enum to string for JSON serialization
            serializable_sources = {}
            for category, sources in profile.preference_sources.items():
                serializable_sources[category] = {
                    key: source.value if hasattr(source, 'value') else str(source)
                    for key, source in sources.items()
                }
            
            data = {
                'user_id': profile.user_id,
                'preferences': profile.preferences,
                'confidence_scores': profile.confidence_scores,
                'preference_sources': serializable_sources,
                'last_updated': profile.last_updated.isoformat(),
                'profile_version': profile.profile_version
            }
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving preferences for user {profile.user_id}: {e}")
    
    def get_preference_history(self, user_id: str) -> List[PreferenceUpdate]:
        """
        Get preference update history for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List[PreferenceUpdate]: Preference update history
        """
        profile = self.get_user_preferences(user_id)
        return profile.preference_history
    
    def clear_user_preferences(self, user_id: str):
        """
        Clear all preferences for a user.
        
        Args:
            user_id: User identifier
        """
        # Remove from cache
        self.preference_cache.pop(user_id, None)
        
        # Remove from storage
        profile_path = self.storage_path / f"{user_id}_preferences.json"
        if profile_path.exists():
            profile_path.unlink()
        
        logger.info(f"Cleared preferences for user {user_id}")