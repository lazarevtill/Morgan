"""
Emotional context building module.

Provides focused emotional context construction, relationship awareness,
and contextual emotional intelligence for enhanced user interactions.
"""

import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotions.memory import get_emotional_memory_storage, EmotionalMemory
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext, CompanionProfile,
    RelationshipMilestone, MilestoneType
)

logger = get_logger(__name__)


class EmotionalContext:
    """
    Represents the current emotional context for a user interaction.
    
    Features:
    - Multi-layered emotional awareness
    - Relationship context integration
    - Historical emotional patterns
    - Contextual adaptation signals
    """
    
    def __init__(
        self,
        user_id: str,
        current_emotion: EmotionalState,
        conversation_context: ConversationContext,
        recent_emotions: List[EmotionalState],
        relevant_memories: List[EmotionalMemory],
        relationship_context: Dict[str, Any],
        contextual_factors: Dict[str, Any]
    ):
        """Initialize emotional context."""
        self.user_id = user_id
        self.current_emotion = current_emotion
        self.conversation_context = conversation_context
        self.recent_emotions = recent_emotions
        self.relevant_memories = relevant_memories
        self.relationship_context = relationship_context
        self.contextual_factors = contextual_factors
        self.created_at = datetime.utcnow()
        
        # Calculate derived metrics
        self.emotional_stability = self._calculate_emotional_stability()
        self.relationship_depth = self._calculate_relationship_depth()
        self.context_confidence = self._calculate_context_confidence()
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability from recent emotions."""
        if len(self.recent_emotions) < 2:
            return 0.5
        
        # Calculate variance in emotional intensity
        intensities = [e.intensity for e in self.recent_emotions]
        mean_intensity = sum(intensities) / len(intensities)
        variance = sum((i - mean_intensity) ** 2 for i in intensities) / len(intensities)
        
        # Stability is inverse of variance (normalized)
        stability = max(0.0, 1.0 - (variance * 2.0))
        return min(1.0, stability)
    
    def _calculate_relationship_depth(self) -> float:
        """Calculate relationship depth from context."""
        depth_factors = []
        
        # Memory significance
        if self.relevant_memories:
            avg_significance = sum(
                m.relationship_significance for m in self.relevant_memories
            ) / len(self.relevant_memories)
            depth_factors.append(avg_significance)
        
        # Relationship milestones
        milestone_count = self.relationship_context.get('milestone_count', 0)
        milestone_factor = min(1.0, milestone_count / 10.0)
        depth_factors.append(milestone_factor)
        
        # Interaction history
        interaction_count = self.relationship_context.get('interaction_count', 0)
        interaction_factor = min(1.0, interaction_count / 100.0)
        depth_factors.append(interaction_factor)
        
        return sum(depth_factors) / len(depth_factors) if depth_factors else 0.0
    
    def _calculate_context_confidence(self) -> float:
        """Calculate confidence in the emotional context."""
        confidence_factors = []
        
        # Current emotion confidence
        confidence_factors.append(self.current_emotion.confidence)
        
        # Recent emotion sample size
        sample_factor = min(1.0, len(self.recent_emotions) / 10.0)
        confidence_factors.append(sample_factor)
        
        # Memory relevance
        if self.relevant_memories:
            avg_importance = sum(
                m.calculate_current_importance() for m in self.relevant_memories
            ) / len(self.relevant_memories)
            confidence_factors.append(avg_importance)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_emotional_trajectory(self) -> str:
        """
        Determine emotional trajectory from recent emotions.
        
        Returns:
            Trajectory description: "improving", "declining", "stable", "volatile"
        """
        if len(self.recent_emotions) < 3:
            return "insufficient_data"
        
        # Calculate emotional valence over time
        valence_map = {
            EmotionType.JOY: 1.0,
            EmotionType.SURPRISE: 0.3,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.FEAR: -0.4,
            EmotionType.DISGUST: -0.6,
            EmotionType.SADNESS: -0.8,
            EmotionType.ANGER: -0.9
        }
        
        valences = [
            valence_map.get(e.primary_emotion, 0.0) * e.intensity
            for e in self.recent_emotions
        ]
        
        # Calculate trend
        recent_avg = sum(valences[-3:]) / 3
        older_avg = sum(valences[:-3]) / max(1, len(valences) - 3)
        
        # Check volatility
        variance = sum((v - sum(valences) / len(valences)) ** 2 for v in valences) / len(valences)
        
        if variance > 0.3:
            return "volatile"
        elif recent_avg > older_avg + 0.2:
            return "improving"
        elif recent_avg < older_avg - 0.2:
            return "declining"
        else:
            return "stable"
    
    def get_contextual_recommendations(self) -> Dict[str, Any]:
        """
        Get contextual recommendations for interaction adaptation.
        
        Returns:
            Dictionary of recommendations for response adaptation
        """
        recommendations = {
            'empathy_level': 'medium',
            'response_tone': 'neutral',
            'personalization_level': 'medium',
            'relationship_references': False,
            'emotional_validation': False,
            'support_offering': False
        }
        
        # Adjust based on current emotion
        if self.current_emotion.primary_emotion in [EmotionType.SADNESS, EmotionType.FEAR]:
            recommendations['empathy_level'] = 'high'
            recommendations['response_tone'] = 'supportive'
            recommendations['emotional_validation'] = True
            recommendations['support_offering'] = True
        elif self.current_emotion.primary_emotion == EmotionType.JOY:
            recommendations['response_tone'] = 'enthusiastic'
            recommendations['relationship_references'] = True
        elif self.current_emotion.primary_emotion == EmotionType.ANGER:
            recommendations['empathy_level'] = 'high'
            recommendations['response_tone'] = 'calm'
            recommendations['emotional_validation'] = True
        
        # Adjust based on relationship depth
        if self.relationship_depth > 0.7:
            recommendations['personalization_level'] = 'high'
            recommendations['relationship_references'] = True
        elif self.relationship_depth < 0.3:
            recommendations['personalization_level'] = 'low'
        
        # Adjust based on emotional stability
        if self.emotional_stability < 0.4:
            recommendations['empathy_level'] = 'high'
            recommendations['support_offering'] = True
        
        return recommendations


class EmotionalContextBuilder:
    """
    Builds comprehensive emotional context for user interactions.
    
    Features:
    - Multi-source context integration
    - Temporal emotional analysis
    - Relationship-aware context building
    - Adaptive context depth
    """
    
    def __init__(self):
        """Initialize emotional context builder."""
        self.settings = get_settings()
        self.memory_storage = get_emotional_memory_storage()
        
        # Context caching for performance
        self._context_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info("Emotional Context Builder initialized")
    
    def build_context(
        self,
        user_id: str,
        current_emotion: EmotionalState,
        conversation_context: ConversationContext,
        companion_profile: Optional[CompanionProfile] = None,
        context_depth: str = "medium"
    ) -> EmotionalContext:
        """
        Build comprehensive emotional context for a user interaction.
        
        Args:
            user_id: User identifier
            current_emotion: Current emotional state
            conversation_context: Current conversation context
            companion_profile: Optional companion profile
            context_depth: Context depth ("shallow", "medium", "deep")
            
        Returns:
            Comprehensive emotional context
        """
        # Check cache first
        cache_key = f"{user_id}:{conversation_context.conversation_id}:{context_depth}"
        cached_context = self._get_cached_context(cache_key)
        if cached_context:
            logger.debug(f"Using cached emotional context for user {user_id}")
            return cached_context
        
        # Gather recent emotions
        recent_emotions = self._gather_recent_emotions(user_id, context_depth)
        
        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(
            user_id, conversation_context, context_depth
        )
        
        # Build relationship context
        relationship_context = self._build_relationship_context(
            user_id, companion_profile, context_depth
        )
        
        # Analyze contextual factors
        contextual_factors = self._analyze_contextual_factors(
            current_emotion, recent_emotions, conversation_context
        )
        
        # Create emotional context
        emotional_context = EmotionalContext(
            user_id=user_id,
            current_emotion=current_emotion,
            conversation_context=conversation_context,
            recent_emotions=recent_emotions,
            relevant_memories=relevant_memories,
            relationship_context=relationship_context,
            contextual_factors=contextual_factors
        )
        
        # Cache the context
        self._cache_context(cache_key, emotional_context)
        
        logger.debug(f"Built emotional context for user {user_id} "
                    f"(depth: {context_depth}, confidence: {emotional_context.context_confidence:.2f})")
        
        return emotional_context
    
    def update_context_with_feedback(
        self,
        context: EmotionalContext,
        user_feedback: int,
        response_effectiveness: float
    ) -> EmotionalContext:
        """
        Update emotional context based on user feedback.
        
        Args:
            context: Original emotional context
            user_feedback: User feedback rating (1-5)
            response_effectiveness: Measured response effectiveness
            
        Returns:
            Updated emotional context
        """
        # Update contextual factors with feedback
        context.contextual_factors['user_feedback'] = user_feedback
        context.contextual_factors['response_effectiveness'] = response_effectiveness
        
        # Adjust context confidence based on feedback
        feedback_factor = (user_feedback - 3) / 2.0  # Normalize to -1 to 1
        context.context_confidence = max(0.0, min(1.0, 
            context.context_confidence + (feedback_factor * 0.1)
        ))
        
        # Update memory significance for relevant memories
        if user_feedback >= 4:  # Positive feedback
            for memory in context.relevant_memories:
                new_significance = min(1.0, memory.relationship_significance + 0.1)
                self.memory_storage.update_memory_significance(
                    memory.memory_id, new_significance
                )
        
        logger.debug(f"Updated context with feedback: {user_feedback}/5, "
                    f"effectiveness: {response_effectiveness:.2f}")
        
        return context
    
    def analyze_emotional_patterns(
        self,
        user_id: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze emotional patterns for context building.
        
        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days
            
        Returns:
            Emotional pattern analysis
        """
        # Retrieve memories within timeframe
        memories = self.memory_storage.retrieve_memories(
            user_id=user_id,
            max_age_days=timeframe_days,
            limit=100
        )
        
        if not memories:
            return {'pattern': 'insufficient_data'}
        
        # Analyze emotion frequency
        emotion_counts = defaultdict(int)
        emotion_intensities = defaultdict(list)
        
        for memory in memories:
            emotion = memory.emotional_state.primary_emotion
            emotion_counts[emotion] += 1
            emotion_intensities[emotion].append(memory.emotional_state.intensity)
        
        # Calculate dominant emotions
        total_memories = len(memories)
        dominant_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Calculate average intensities
        avg_intensities = {}
        for emotion, intensities in emotion_intensities.items():
            avg_intensities[emotion] = sum(intensities) / len(intensities)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(memories)
        
        return {
            'timeframe_days': timeframe_days,
            'total_memories': total_memories,
            'dominant_emotions': [
                {'emotion': e.value, 'frequency': count / total_memories}
                for e, count in dominant_emotions
            ],
            'average_intensities': {
                e.value: intensity for e, intensity in avg_intensities.items()
            },
            'temporal_patterns': temporal_patterns,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _gather_recent_emotions(
        self,
        user_id: str,
        context_depth: str
    ) -> List[EmotionalState]:
        """Gather recent emotional states for context."""
        # Determine how many recent emotions to include
        emotion_limits = {
            'shallow': 3,
            'medium': 7,
            'deep': 15
        }
        limit = emotion_limits.get(context_depth, 7)
        
        # Retrieve recent memories and extract emotions
        recent_memories = self.memory_storage.retrieve_memories(
            user_id=user_id,
            max_age_days=7,
            limit=limit
        )
        
        return [memory.emotional_state for memory in recent_memories]
    
    def _retrieve_relevant_memories(
        self,
        user_id: str,
        conversation_context: ConversationContext,
        context_depth: str
    ) -> List[EmotionalMemory]:
        """Retrieve memories relevant to current conversation."""
        # Determine memory limits
        memory_limits = {
            'shallow': 3,
            'medium': 5,
            'deep': 10
        }
        limit = memory_limits.get(context_depth, 5)
        
        # Search for semantically relevant memories
        relevant_memories = self.memory_storage.retrieve_memories(
            user_id=user_id,
            query=conversation_context.message_text,
            min_importance=0.3,
            max_age_days=90,
            limit=limit
        )
        
        return relevant_memories
    
    def _build_relationship_context(
        self,
        user_id: str,
        companion_profile: Optional[CompanionProfile],
        context_depth: str
    ) -> Dict[str, Any]:
        """Build relationship context information."""
        if not companion_profile:
            return {
                'relationship_age_days': 0,
                'interaction_count': 0,
                'trust_level': 0.0,
                'milestone_count': 0,
                'recent_milestones': []
            }
        
        # Basic relationship metrics
        relationship_context = {
            'relationship_age_days': companion_profile.get_relationship_age_days(),
            'interaction_count': companion_profile.interaction_count,
            'trust_level': companion_profile.trust_level,
            'engagement_score': companion_profile.engagement_score,
            'milestone_count': len(companion_profile.relationship_milestones)
        }
        
        # Include recent milestones for deeper context
        if context_depth in ['medium', 'deep']:
            recent_milestones = sorted(
                companion_profile.relationship_milestones,
                key=lambda m: m.timestamp,
                reverse=True
            )[:5]
            
            relationship_context['recent_milestones'] = [
                {
                    'type': milestone.milestone_type.value,
                    'description': milestone.description,
                    'significance': milestone.emotional_significance,
                    'timestamp': milestone.timestamp.isoformat()
                }
                for milestone in recent_milestones
            ]
        
        return relationship_context
    
    def _analyze_contextual_factors(
        self,
        current_emotion: EmotionalState,
        recent_emotions: List[EmotionalState],
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze various contextual factors."""
        factors = {}
        
        # Emotional consistency
        if recent_emotions:
            consistent_emotions = sum(
                1 for e in recent_emotions
                if e.primary_emotion == current_emotion.primary_emotion
            )
            factors['emotional_consistency'] = consistent_emotions / len(recent_emotions)
        
        # Conversation continuity
        factors['has_previous_context'] = bool(conversation_context.previous_messages)
        factors['conversation_length'] = len(conversation_context.previous_messages)
        
        # Temporal factors
        now = datetime.utcnow()
        factors['time_of_day'] = now.hour
        factors['day_of_week'] = now.weekday()
        
        # Session factors
        if conversation_context.session_duration:
            factors['session_duration_minutes'] = conversation_context.session_duration.total_seconds() / 60
        
        return factors
    
    def _analyze_temporal_patterns(
        self,
        memories: List[EmotionalMemory]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in emotional memories."""
        if not memories:
            return {}
        
        # Group by time periods
        hourly_emotions = defaultdict(list)
        daily_emotions = defaultdict(list)
        
        for memory in memories:
            hour = memory.created_at.hour
            day = memory.created_at.weekday()
            
            hourly_emotions[hour].append(memory.emotional_state.primary_emotion)
            daily_emotions[day].append(memory.emotional_state.primary_emotion)
        
        # Find patterns
        patterns = {}
        
        # Most emotional hours
        if hourly_emotions:
            hour_counts = {hour: len(emotions) for hour, emotions in hourly_emotions.items()}
            most_active_hour = max(hour_counts, key=hour_counts.get)
            patterns['most_active_hour'] = most_active_hour
        
        # Most emotional days
        if daily_emotions:
            day_counts = {day: len(emotions) for day, emotions in daily_emotions.items()}
            most_active_day = max(day_counts, key=day_counts.get)
            patterns['most_active_day'] = most_active_day
        
        return patterns
    
    def _get_cached_context(self, cache_key: str) -> Optional[EmotionalContext]:
        """Get cached emotional context if valid."""
        if cache_key in self._context_cache:
            context, timestamp = self._context_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                return context
            else:
                # Remove expired cache entry
                del self._context_cache[cache_key]
        
        return None
    
    def _cache_context(self, cache_key: str, context: EmotionalContext):
        """Cache emotional context."""
        self._context_cache[cache_key] = (context, datetime.utcnow())
        
        # Clean up old cache entries
        if len(self._context_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self._context_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_items[:20]:
                del self._context_cache[key]


# Singleton instance
_context_builder_instance = None
_context_builder_lock = threading.Lock()


def get_emotional_context_builder() -> EmotionalContextBuilder:
    """
    Get singleton emotional context builder instance.
    
    Returns:
        Shared EmotionalContextBuilder instance
    """
    global _context_builder_instance
    
    if _context_builder_instance is None:
        with _context_builder_lock:
            if _context_builder_instance is None:
                _context_builder_instance = EmotionalContextBuilder()
    
    return _context_builder_instance