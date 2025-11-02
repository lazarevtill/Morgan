"""
User preference learning module.

Learns and adapts to user communication preferences through interaction
analysis, feedback processing, and behavioral pattern recognition.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.llm_service import get_llm_service
from morgan.utils.cache import FileCache
from morgan.emotional.models import (
    EmotionalState, ConversationContext, UserPreferences,
    CommunicationStyle, ResponseLength, CompanionProfile
)

logger = get_logger(__name__)


@dataclass
class PreferenceLearningResult:
    """Result of preference learning analysis."""
    learned_preferences: UserPreferences
    confidence_score: float
    learning_sources: List[str]
    preference_changes: Dict[str, Any]
    learning_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InteractionPattern:
    """Pattern detected from user interactions."""
    pattern_type: str
    pattern_description: str
    frequency: int
    confidence: float
    examples: List[str]


class UserPreferenceLearner:
    """
    User preference learning system.
    
    Features:
    - Communication style preference detection
    - Response length preference learning
    - Topic interest identification
    - Interaction timing pattern analysis
    - Feedback-based preference refinement
    - Cultural communication preference adaptation
    """
    
    def __init__(self):
        """Initialize user preference learner."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()
        
        # Setup cache for preference data
        cache_dir = self.settings.morgan_data_dir / "cache" / "preferences"
        self.cache = FileCache(cache_dir)
        
        # User interaction data storage
        self.user_interactions: Dict[str, List[ConversationContext]] = defaultdict(list)
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.learning_history: Dict[str, List[PreferenceLearningResult]] = defaultdict(list)
        
        logger.info("User Preference Learner initialized")
    
    def learn_preferences(
        self,
        user_id: str,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
        user_feedback: Optional[float] = None
    ) -> PreferenceLearningResult:
        """
        Learn user preferences from interaction data.
        
        Args:
            user_id: User identifier
            conversation_context: Current conversation context
            emotional_state: User's emotional state
            user_feedback: Optional feedback score (0.0-1.0)
            
        Returns:
            Preference learning result
        """
        # Store interaction data
        self.user_interactions[user_id].append(conversation_context)
        
        # Keep only recent interactions (last 100)
        if len(self.user_interactions[user_id]) > 100:
            self.user_interactions[user_id] = self.user_interactions[user_id][-100:]
        
        # Get or create user preferences
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self._create_initial_preferences()
        
        current_preferences = self.user_preferences[user_id]
        
        # Analyze interaction patterns
        patterns = self._analyze_interaction_patterns(user_id)
        
        # Learn communication style preferences
        style_preferences = self._learn_communication_style(
            user_id, conversation_context, emotional_state, user_feedback
        )
        
        # Learn response length preferences
        length_preferences = self._learn_response_length_preferences(
            user_id, conversation_context, user_feedback
        )
        
        # Learn topic interests
        topic_interests = self._learn_topic_interests(
            user_id, conversation_context, emotional_state
        )
        
        # Learn timing preferences
        timing_preferences = self._learn_timing_preferences(user_id)
        
        # Update preferences
        updated_preferences, changes = self._update_preferences(
            current_preferences,
            style_preferences,
            length_preferences,
            topic_interests,
            timing_preferences
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_learning_confidence(
            user_id, patterns, user_feedback
        )
        
        # Create learning result
        learning_sources = [
            "interaction_patterns",
            "communication_analysis",
            "topic_analysis"
        ]
        
        if user_feedback is not None:
            learning_sources.append("user_feedback")
        
        result = PreferenceLearningResult(
            learned_preferences=updated_preferences,
            confidence_score=confidence_score,
            learning_sources=learning_sources,
            preference_changes=changes
        )
        
        # Store updated preferences
        self.user_preferences[user_id] = updated_preferences
        self.learning_history[user_id].append(result)
        
        # Cache preferences
        self._cache_preferences(user_id, updated_preferences)
        
        logger.debug(
            f"Learned preferences for user {user_id}: "
            f"confidence={confidence_score:.2f}, "
            f"changes={len(changes)}"
        )
        
        return result
    
    def get_user_preferences(
        self,
        user_id: str,
        use_cache: bool = True
    ) -> Optional[UserPreferences]:
        """
        Get current user preferences.
        
        Args:
            user_id: User identifier
            use_cache: Whether to use cached preferences
            
        Returns:
            User preferences or None if not available
        """
        # Check cache first
        if use_cache:
            cached_prefs = self._get_cached_preferences(user_id)
            if cached_prefs:
                return cached_prefs
        
        # Return in-memory preferences
        return self.user_preferences.get(user_id)
    
    def analyze_preference_evolution(
        self,
        user_id: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze how user preferences have evolved over time.
        
        Args:
            user_id: User identifier
            timeframe_days: Number of days to analyze
            
        Returns:
            Preference evolution analysis
        """
        if user_id not in self.learning_history:
            return {"error": "No learning history available"}
        
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_learning = [
            result for result in self.learning_history[user_id]
            if result.learning_timestamp >= cutoff_date
        ]
        
        if not recent_learning:
            return {"error": "No recent learning data available"}
        
        # Analyze preference stability
        stability_analysis = self._analyze_preference_stability(recent_learning)
        
        # Analyze learning confidence trends
        confidence_trends = self._analyze_confidence_trends(recent_learning)
        
        # Identify preference drift
        preference_drift = self._identify_preference_drift(recent_learning)
        
        return {
            "timeframe_days": timeframe_days,
            "learning_sessions": len(recent_learning),
            "stability_analysis": stability_analysis,
            "confidence_trends": confidence_trends,
            "preference_drift": preference_drift,
            "latest_preferences": recent_learning[-1].learned_preferences.__dict__
        }
    
    def predict_user_needs(
        self,
        user_id: str,
        current_context: ConversationContext,
        emotional_state: EmotionalState
    ) -> List[Dict[str, Any]]:
        """
        Predict user needs based on learned preferences and current context.
        
        Args:
            user_id: User identifier
            current_context: Current conversation context
            emotional_state: Current emotional state
            
        Returns:
            List of predicted needs with confidence scores
        """
        preferences = self.get_user_preferences(user_id)
        if not preferences:
            return []
        
        predictions = []
        
        # Predict communication style needs
        style_prediction = self._predict_communication_style_needs(
            preferences, current_context, emotional_state
        )
        if style_prediction:
            predictions.append(style_prediction)
        
        # Predict response length needs
        length_prediction = self._predict_response_length_needs(
            preferences, current_context
        )
        if length_prediction:
            predictions.append(length_prediction)
        
        # Predict topic interest alignment
        topic_prediction = self._predict_topic_interest_alignment(
            preferences, current_context
        )
        if topic_prediction:
            predictions.append(topic_prediction)
        
        # Predict emotional support needs
        support_prediction = self._predict_emotional_support_needs(
            preferences, emotional_state
        )
        if support_prediction:
            predictions.append(support_prediction)
        
        # Sort by confidence score
        predictions.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return predictions
    
    def _create_initial_preferences(self) -> UserPreferences:
        """Create initial user preferences with defaults."""
        return UserPreferences(
            topics_of_interest=[],
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length=ResponseLength.MODERATE,
            learning_goals=[],
            personal_context={}
        )
    
    def _analyze_interaction_patterns(
        self,
        user_id: str
    ) -> List[InteractionPattern]:
        """Analyze patterns in user interactions."""
        interactions = self.user_interactions[user_id]
        if len(interactions) < 3:
            return []
        
        patterns = []
        
        # Analyze message length patterns
        message_lengths = [len(ctx.message_text) for ctx in interactions]
        avg_length = sum(message_lengths) / len(message_lengths)
        
        if avg_length > 150:
            patterns.append(InteractionPattern(
                pattern_type="detailed_communicator",
                pattern_description="User tends to provide detailed messages",
                frequency=len([l for l in message_lengths if l > 150]),
                confidence=0.8,
                examples=[]
            ))
        elif avg_length < 50:
            patterns.append(InteractionPattern(
                pattern_type="brief_communicator",
                pattern_description="User prefers brief communication",
                frequency=len([l for l in message_lengths if l < 50]),
                confidence=0.8,
                examples=[]
            ))
        
        # Analyze timing patterns
        if len(interactions) > 5:
            time_gaps = []
            for i in range(1, len(interactions)):
                gap = (interactions[i].timestamp - interactions[i-1].timestamp).total_seconds()
                time_gaps.append(gap)
            
            avg_gap = sum(time_gaps) / len(time_gaps)
            if avg_gap < 300:  # Less than 5 minutes
                patterns.append(InteractionPattern(
                    pattern_type="rapid_interaction",
                    pattern_description="User engages in rapid back-and-forth",
                    frequency=len([g for g in time_gaps if g < 300]),
                    confidence=0.7,
                    examples=[]
                ))
        
        # Analyze topic consistency
        all_text = " ".join([ctx.message_text for ctx in interactions])
        topic_patterns = self._extract_topic_patterns(all_text)
        patterns.extend(topic_patterns)
        
        return patterns
    
    def _learn_communication_style(
        self,
        user_id: str,
        context: ConversationContext,
        emotional_state: EmotionalState,
        feedback: Optional[float]
    ) -> Dict[str, Any]:
        """Learn communication style preferences."""
        interactions = self.user_interactions[user_id]
        
        # Analyze formality level
        formality_indicators = {
            "formal": ["please", "thank you", "would you", "could you", "appreciate"],
            "casual": ["hey", "yeah", "ok", "cool", "awesome", "thanks"]
        }
        
        formal_count = 0
        casual_count = 0
        
        for interaction in interactions[-10:]:  # Last 10 interactions
            text_lower = interaction.message_text.lower()
            formal_count += sum(1 for indicator in formality_indicators["formal"] if indicator in text_lower)
            casual_count += sum(1 for indicator in formality_indicators["casual"] if indicator in text_lower)
        
        preferred_formality = "formal" if formal_count > casual_count else "casual"
        
        # Analyze emotional expression preferences
        emotional_expression = "expressive" if emotional_state.intensity > 0.6 else "reserved"
        
        return {
            "preferred_formality": preferred_formality,
            "emotional_expression": emotional_expression,
            "confidence": 0.7 if len(interactions) > 5 else 0.4
        }
    
    def _learn_response_length_preferences(
        self,
        user_id: str,
        context: ConversationContext,
        feedback: Optional[float]
    ) -> Dict[str, Any]:
        """Learn response length preferences."""
        interactions = self.user_interactions[user_id]
        
        # Analyze user message lengths
        message_lengths = [len(ctx.message_text) for ctx in interactions[-20:]]
        avg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 100
        
        # Determine preferred response length
        if avg_length > 200:
            preferred_length = ResponseLength.DETAILED
        elif avg_length < 50:
            preferred_length = ResponseLength.BRIEF
        else:
            preferred_length = ResponseLength.MODERATE
        
        # Adjust based on feedback
        confidence = 0.6
        if feedback is not None:
            if feedback > 0.8:
                confidence = 0.9
            elif feedback < 0.4:
                confidence = 0.3
        
        return {
            "preferred_length": preferred_length,
            "average_user_message_length": avg_length,
            "confidence": confidence
        }
    
    def _learn_topic_interests(
        self,
        user_id: str,
        context: ConversationContext,
        emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Learn topic interests from conversations."""
        interactions = self.user_interactions[user_id]
        
        # Extract topics from recent interactions
        all_text = " ".join([ctx.message_text for ctx in interactions[-20:]])
        topics = self._extract_topics_from_text(all_text)
        
        # Weight topics by emotional engagement
        weighted_topics = {}
        for topic in topics:
            # Higher emotional intensity suggests higher interest
            weight = 1.0 + (emotional_state.intensity * 0.5)
            weighted_topics[topic] = weighted_topics.get(topic, 0) + weight
        
        # Get top interests
        top_interests = sorted(
            weighted_topics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "topic_interests": [topic for topic, _ in top_interests],
            "topic_weights": dict(top_interests),
            "confidence": 0.7 if len(interactions) > 10 else 0.4
        }
    
    def _learn_timing_preferences(self, user_id: str) -> Dict[str, Any]:
        """Learn timing preferences from interaction patterns."""
        interactions = self.user_interactions[user_id]
        
        if len(interactions) < 5:
            return {"confidence": 0.0}
        
        # Analyze interaction times
        hours = [ctx.timestamp.hour for ctx in interactions]
        hour_counts = Counter(hours)
        
        # Find preferred time ranges
        preferred_hours = [hour for hour, count in hour_counts.most_common(3)]
        
        # Determine time preference categories
        morning_hours = [h for h in preferred_hours if 6 <= h <= 11]
        afternoon_hours = [h for h in preferred_hours if 12 <= h <= 17]
        evening_hours = [h for h in preferred_hours if 18 <= h <= 23]
        
        preferred_times = []
        if morning_hours:
            preferred_times.append("morning")
        if afternoon_hours:
            preferred_times.append("afternoon")
        if evening_hours:
            preferred_times.append("evening")
        
        return {
            "preferred_times": preferred_times,
            "most_active_hours": preferred_hours,
            "confidence": 0.6 if len(interactions) > 10 else 0.3
        }
    
    def _update_preferences(
        self,
        current_preferences: UserPreferences,
        style_prefs: Dict[str, Any],
        length_prefs: Dict[str, Any],
        topic_prefs: Dict[str, Any],
        timing_prefs: Dict[str, Any]
    ) -> Tuple[UserPreferences, Dict[str, Any]]:
        """Update user preferences with learned data."""
        changes = {}
        
        # Update communication style
        if style_prefs.get("confidence", 0) > 0.5:
            formality = style_prefs.get("preferred_formality")
            if formality == "formal":
                new_style = CommunicationStyle.PROFESSIONAL
            else:
                new_style = CommunicationStyle.CASUAL
            
            if current_preferences.communication_style != new_style:
                changes["communication_style"] = {
                    "old": current_preferences.communication_style,
                    "new": new_style
                }
                current_preferences.communication_style = new_style
        
        # Update response length preference
        if length_prefs.get("confidence", 0) > 0.5:
            new_length = length_prefs.get("preferred_length")
            if current_preferences.preferred_response_length != new_length:
                changes["preferred_response_length"] = {
                    "old": current_preferences.preferred_response_length,
                    "new": new_length
                }
                current_preferences.preferred_response_length = new_length
        
        # Update topic interests
        if topic_prefs.get("confidence", 0) > 0.4:
            new_topics = topic_prefs.get("topic_interests", [])
            # Merge with existing topics, keeping unique ones
            all_topics = list(set(current_preferences.topics_of_interest + new_topics))
            # Keep only top 15 topics
            if len(all_topics) > 15:
                all_topics = all_topics[:15]
            
            if set(current_preferences.topics_of_interest) != set(all_topics):
                changes["topics_of_interest"] = {
                    "old": current_preferences.topics_of_interest,
                    "new": all_topics
                }
                current_preferences.topics_of_interest = all_topics
        
        # Update personal context with timing preferences
        if timing_prefs.get("confidence", 0) > 0.4:
            timing_data = {
                "preferred_times": timing_prefs.get("preferred_times", []),
                "most_active_hours": timing_prefs.get("most_active_hours", [])
            }
            current_preferences.personal_context["timing_preferences"] = timing_data
            changes["timing_preferences"] = timing_data
        
        return current_preferences, changes
    
    def _calculate_learning_confidence(
        self,
        user_id: str,
        patterns: List[InteractionPattern],
        feedback: Optional[float]
    ) -> float:
        """Calculate confidence in preference learning."""
        confidence_factors = []
        
        # Interaction count factor
        interaction_count = len(self.user_interactions[user_id])
        interaction_confidence = min(1.0, interaction_count / 20.0)
        confidence_factors.append(interaction_confidence)
        
        # Pattern strength factor
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            confidence_factors.append(avg_pattern_confidence)
        else:
            confidence_factors.append(0.3)
        
        # Feedback factor
        if feedback is not None:
            feedback_confidence = feedback
            confidence_factors.append(feedback_confidence)
        else:
            confidence_factors.append(0.5)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _extract_topic_patterns(self, text: str) -> List[InteractionPattern]:
        """Extract topic patterns from text."""
        # Simple topic extraction - in a real implementation,
        # this would use more sophisticated NLP techniques
        topic_keywords = {
            "technology": ["tech", "computer", "software", "programming", "AI", "machine learning"],
            "health": ["health", "fitness", "exercise", "diet", "wellness", "medical"],
            "work": ["work", "job", "career", "business", "professional", "office"],
            "personal": ["family", "friends", "relationship", "personal", "life"],
            "learning": ["learn", "study", "education", "course", "skill", "knowledge"]
        }
        
        patterns = []
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 2:
                patterns.append(InteractionPattern(
                    pattern_type=f"topic_interest_{topic}",
                    pattern_description=f"User shows interest in {topic}",
                    frequency=matches,
                    confidence=min(1.0, matches / 10.0),
                    examples=[]
                ))
        
        return patterns
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword matching."""
        topics = []
        
        # Define topic categories with keywords
        topic_categories = {
            "technology": ["technology", "tech", "computer", "software", "programming", "AI", "artificial intelligence", "machine learning", "data science"],
            "health": ["health", "fitness", "exercise", "diet", "nutrition", "wellness", "medical", "doctor", "medicine"],
            "work": ["work", "job", "career", "business", "professional", "office", "meeting", "project", "team"],
            "education": ["education", "learning", "study", "course", "school", "university", "knowledge", "skill"],
            "relationships": ["relationship", "family", "friends", "love", "dating", "marriage", "social"],
            "hobbies": ["hobby", "music", "art", "sports", "reading", "cooking", "travel", "photography"],
            "finance": ["money", "finance", "investment", "budget", "savings", "economy", "business"],
            "entertainment": ["movie", "film", "book", "game", "entertainment", "fun", "leisure"]
        }
        
        text_lower = text.lower()
        
        for topic, keywords in topic_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _cache_preferences(self, user_id: str, preferences: UserPreferences) -> None:
        """Cache user preferences."""
        cache_key = f"preferences_{user_id}"
        preferences_dict = {
            "topics_of_interest": preferences.topics_of_interest,
            "communication_style": preferences.communication_style.value,
            "preferred_response_length": preferences.preferred_response_length.value,
            "learning_goals": preferences.learning_goals,
            "personal_context": preferences.personal_context
        }
        self.cache.set(cache_key, preferences_dict)
    
    def _get_cached_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get cached user preferences."""
        cache_key = f"preferences_{user_id}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return UserPreferences(
                topics_of_interest=cached_data.get("topics_of_interest", []),
                communication_style=CommunicationStyle(cached_data.get("communication_style", "friendly")),
                preferred_response_length=ResponseLength(cached_data.get("preferred_response_length", "moderate")),
                learning_goals=cached_data.get("learning_goals", []),
                personal_context=cached_data.get("personal_context", {})
            )
        
        return None
    
    def _analyze_preference_stability(
        self,
        learning_results: List[PreferenceLearningResult]
    ) -> Dict[str, Any]:
        """Analyze stability of learned preferences."""
        if len(learning_results) < 2:
            return {"stability": "insufficient_data"}
        
        # Check communication style stability
        styles = [result.learned_preferences.communication_style for result in learning_results]
        style_changes = sum(1 for i in range(1, len(styles)) if styles[i] != styles[i-1])
        
        # Check response length stability
        lengths = [result.learned_preferences.preferred_response_length for result in learning_results]
        length_changes = sum(1 for i in range(1, len(lengths)) if lengths[i] != lengths[i-1])
        
        total_changes = style_changes + length_changes
        stability_score = 1.0 - (total_changes / (len(learning_results) * 2))
        
        return {
            "stability_score": stability_score,
            "style_changes": style_changes,
            "length_changes": length_changes,
            "total_sessions": len(learning_results)
        }
    
    def _analyze_confidence_trends(
        self,
        learning_results: List[PreferenceLearningResult]
    ) -> Dict[str, Any]:
        """Analyze confidence trends in learning."""
        if not learning_results:
            return {"trend": "no_data"}
        
        confidences = [result.confidence_score for result in learning_results]
        
        if len(confidences) < 2:
            return {"trend": "insufficient_data", "current_confidence": confidences[0]}
        
        # Calculate trend
        recent_avg = sum(confidences[-3:]) / min(3, len(confidences))
        older_avg = sum(confidences[:-3]) / max(1, len(confidences) - 3)
        
        if recent_avg > older_avg + 0.1:
            trend = "improving"
        elif recent_avg < older_avg - 0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_confidence": confidences[-1],
            "average_confidence": sum(confidences) / len(confidences),
            "confidence_range": [min(confidences), max(confidences)]
        }
    
    def _identify_preference_drift(
        self,
        learning_results: List[PreferenceLearningResult]
    ) -> Dict[str, Any]:
        """Identify preference drift over time."""
        if len(learning_results) < 3:
            return {"drift": "insufficient_data"}
        
        # Compare first third with last third
        first_third = learning_results[:len(learning_results)//3]
        last_third = learning_results[-len(learning_results)//3:]
        
        # Check for style drift
        first_styles = [r.learned_preferences.communication_style for r in first_third]
        last_styles = [r.learned_preferences.communication_style for r in last_third]
        
        style_drift = len(set(first_styles).symmetric_difference(set(last_styles))) > 0
        
        # Check for topic drift
        first_topics = set()
        for r in first_third:
            first_topics.update(r.learned_preferences.topics_of_interest)
        
        last_topics = set()
        for r in last_third:
            last_topics.update(r.learned_preferences.topics_of_interest)
        
        topic_overlap = len(first_topics.intersection(last_topics)) / max(1, len(first_topics.union(last_topics)))
        topic_drift = topic_overlap < 0.5
        
        return {
            "style_drift": style_drift,
            "topic_drift": topic_drift,
            "topic_overlap_ratio": topic_overlap,
            "drift_detected": style_drift or topic_drift
        }
    
    def _predict_communication_style_needs(
        self,
        preferences: UserPreferences,
        context: ConversationContext,
        emotional_state: EmotionalState
    ) -> Optional[Dict[str, Any]]:
        """Predict communication style needs."""
        # Adjust style based on emotional state
        if emotional_state.primary_emotion.value in ["sadness", "fear"]:
            needed_style = "supportive"
            confidence = 0.8
        elif emotional_state.primary_emotion.value == "anger":
            needed_style = "calm"
            confidence = 0.9
        else:
            needed_style = preferences.communication_style.value
            confidence = 0.6
        
        return {
            "need_type": "communication_style",
            "predicted_need": needed_style,
            "confidence": confidence,
            "reasoning": f"Based on {emotional_state.primary_emotion.value} emotion"
        }
    
    def _predict_response_length_needs(
        self,
        preferences: UserPreferences,
        context: ConversationContext
    ) -> Optional[Dict[str, Any]]:
        """Predict response length needs."""
        user_message_length = len(context.message_text)
        
        # Match user's communication length
        if user_message_length > 200:
            needed_length = "detailed"
            confidence = 0.8
        elif user_message_length < 50:
            needed_length = "brief"
            confidence = 0.8
        else:
            needed_length = preferences.preferred_response_length.value
            confidence = 0.6
        
        return {
            "need_type": "response_length",
            "predicted_need": needed_length,
            "confidence": confidence,
            "reasoning": f"User message length: {user_message_length} characters"
        }
    
    def _predict_topic_interest_alignment(
        self,
        preferences: UserPreferences,
        context: ConversationContext
    ) -> Optional[Dict[str, Any]]:
        """Predict topic interest alignment."""
        message_topics = self._extract_topics_from_text(context.message_text)
        
        if not message_topics:
            return None
        
        # Check alignment with user interests
        aligned_topics = [
            topic for topic in message_topics
            if topic in preferences.topics_of_interest
        ]
        
        alignment_score = len(aligned_topics) / len(message_topics) if message_topics else 0
        
        return {
            "need_type": "topic_alignment",
            "predicted_need": "high_interest" if alignment_score > 0.5 else "moderate_interest",
            "confidence": alignment_score,
            "reasoning": f"Topic alignment: {alignment_score:.2f}"
        }
    
    def _predict_emotional_support_needs(
        self,
        preferences: UserPreferences,
        emotional_state: EmotionalState
    ) -> Optional[Dict[str, Any]]:
        """Predict emotional support needs."""
        if emotional_state.intensity < 0.3:
            return None
        
        support_level = "high" if emotional_state.intensity > 0.7 else "moderate"
        
        return {
            "need_type": "emotional_support",
            "predicted_need": support_level,
            "confidence": emotional_state.confidence,
            "reasoning": f"Emotional intensity: {emotional_state.intensity:.2f}"
        }


# Singleton instance
_user_preference_learner_instance = None
_user_preference_learner_lock = threading.Lock()


def get_user_preference_learner() -> UserPreferenceLearner:
    """
    Get singleton user preference learner instance.
    
    Returns:
        Shared UserPreferenceLearner instance
    """
    global _user_preference_learner_instance
    
    if _user_preference_learner_instance is None:
        with _user_preference_learner_lock:
            if _user_preference_learner_instance is None:
                _user_preference_learner_instance = UserPreferenceLearner()
    
    return _user_preference_learner_instance