"""
Topic preference learning module.

Learns user topic preferences through conversation analysis, engagement tracking,
and behavioral pattern recognition to improve conversation relevance and personalization.
"""

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    ConversationTopic,
    EmotionalState,
    UserPreferences,
)
from morgan.services.llm import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TopicEngagement:
    """User engagement with a specific topic."""

    topic: str
    engagement_score: float  # 0.0 to 1.0
    interaction_count: int
    total_time_spent: timedelta
    emotional_responses: List[str] = field(default_factory=list)
    last_discussed: datetime = field(default_factory=datetime.utcnow)
    preference_strength: float = 0.5  # 0.0 to 1.0


@dataclass
class TopicCluster:
    """Cluster of related topics."""

    cluster_name: str
    topics: List[str]
    cluster_score: float
    representative_topic: str
    user_affinity: float = 0.0


@dataclass
class TopicLearningResult:
    """Result of topic preference learning."""

    learned_preferences: List[TopicEngagement]
    topic_clusters: List[TopicCluster]
    preference_changes: Dict[str, Any]
    confidence_score: float
    learning_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TopicPrediction:
    """Prediction of user interest in a topic."""

    topic: str
    predicted_interest: float  # 0.0 to 1.0
    confidence: float
    reasoning: str
    related_topics: List[str] = field(default_factory=list)


class TopicPreferenceLearner:
    """
    Topic preference learning system.

    Features:
    - Topic extraction from conversations
    - Engagement tracking per topic
    - Topic clustering and relationship mapping
    - Preference evolution tracking
    - Interest prediction for new topics
    - Seasonal and temporal preference patterns
    """

    def __init__(self):
        """Initialize topic preference learner."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for topic data
        cache_dir = self.settings.morgan_data_dir / "cache" / "topics"
        self.cache = FileCache(cache_dir)

        # User topic data
        self.user_topic_engagements: Dict[str, Dict[str, TopicEngagement]] = defaultdict(
            dict
        )
        self.user_topic_clusters: Dict[str, List[TopicCluster]] = defaultdict(list)
        self.learning_history: Dict[str, List[TopicLearningResult]] = defaultdict(list)

        # Topic taxonomy and relationships
        self.topic_taxonomy = self._initialize_topic_taxonomy()
        self.topic_relationships = self._initialize_topic_relationships()

        logger.info("Topic Preference Learner initialized")

    def learn_topic_preferences(
        self,
        user_id: str,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
        engagement_indicators: Optional[Dict[str, Any]] = None,
    ) -> TopicLearningResult:
        """
        Learn topic preferences from conversation.

        Args:
            user_id: User identifier
            conversation_context: Current conversation context
            emotional_state: User's emotional state
            engagement_indicators: Optional engagement metrics

        Returns:
            Topic learning result
        """
        # Extract topics from conversation
        extracted_topics = self._extract_topics_from_conversation(
            conversation_context.message_text
        )

        # Calculate engagement scores
        engagement_scores = self._calculate_topic_engagement(
            extracted_topics, emotional_state, engagement_indicators
        )

        # Update user topic engagements
        self._update_topic_engagements(
            user_id, extracted_topics, engagement_scores, conversation_context
        )

        # Perform topic clustering
        topic_clusters = self._cluster_user_topics(user_id)

        # Analyze preference changes
        preference_changes = self._analyze_preference_changes(user_id)

        # Calculate learning confidence
        confidence_score = self._calculate_learning_confidence(
            user_id, extracted_topics, emotional_state
        )

        # Create learning result
        current_engagements = list(self.user_topic_engagements[user_id].values())
        result = TopicLearningResult(
            learned_preferences=current_engagements,
            topic_clusters=topic_clusters,
            preference_changes=preference_changes,
            confidence_score=confidence_score,
        )

        # Store learning result
        self.learning_history[user_id].append(result)
        self.user_topic_clusters[user_id] = topic_clusters

        # Cache updated preferences
        self._cache_topic_preferences(user_id)

        logger.debug(
            f"Learned topic preferences for user {user_id}: "
            f"{len(extracted_topics)} topics, confidence={confidence_score:.2f}"
        )

        return result

    def predict_topic_interest(
        self, user_id: str, topic: str, context: Optional[str] = None
    ) -> TopicPrediction:
        """
        Predict user interest in a specific topic.

        Args:
            user_id: User identifier
            topic: Topic to predict interest for
            context: Optional context for prediction

        Returns:
            Topic interest prediction
        """
        # Get user's topic engagements
        user_engagements = self.user_topic_engagements.get(user_id, {})

        # Direct topic match
        if topic in user_engagements:
            engagement = user_engagements[topic]
            return TopicPrediction(
                topic=topic,
                predicted_interest=engagement.preference_strength,
                confidence=0.9,
                reasoning="Direct topic engagement history available",
                related_topics=self._get_related_topics(topic),
            )

        # Cluster-based prediction
        cluster_prediction = self._predict_from_clusters(user_id, topic)
        if cluster_prediction:
            return cluster_prediction

        # Similarity-based prediction
        similarity_prediction = self._predict_from_similarity(user_id, topic)
        if similarity_prediction:
            return similarity_prediction

        # Default prediction
        return TopicPrediction(
            topic=topic,
            predicted_interest=0.5,
            confidence=0.3,
            reasoning="No specific data available, using neutral prediction",
        )

    def get_recommended_topics(
        self,
        user_id: str,
        current_context: Optional[str] = None,
        emotional_state: Optional[EmotionalState] = None,
        count: int = 5,
    ) -> List[ConversationTopic]:
        """
        Get recommended topics for conversation.

        Args:
            user_id: User identifier
            current_context: Current conversation context
            emotional_state: Current emotional state
            count: Number of recommendations

        Returns:
            List of recommended conversation topics
        """
        recommendations = []

        # Get user's top engaged topics
        user_engagements = self.user_topic_engagements.get(user_id, {})
        top_topics = sorted(
            user_engagements.items(),
            key=lambda x: x[1].preference_strength,
            reverse=True,
        )

        # Add top engaged topics
        for topic_name, engagement in top_topics[:count // 2]:
            recommendations.append(
                ConversationTopic(
                    topic=topic_name,
                    relevance_score=engagement.preference_strength,
                    category="high_engagement",
                )
            )

        # Add contextually relevant topics
        if current_context:
            contextual_topics = self._get_contextual_topics(
                current_context, user_engagements
            )
            for topic in contextual_topics[: count - len(recommendations)]:
                recommendations.append(topic)

        # Add emotionally appropriate topics
        if emotional_state and len(recommendations) < count:
            emotional_topics = self._get_emotional_topics(
                emotional_state, user_engagements
            )
            for topic in emotional_topics[: count - len(recommendations)]:
                recommendations.append(topic)

        # Fill remaining slots with discovery topics
        while len(recommendations) < count:
            discovery_topic = self._get_discovery_topic(user_id, recommendations)
            if discovery_topic:
                recommendations.append(discovery_topic)
            else:
                break

        return recommendations[:count]

    def analyze_topic_evolution(
        self, user_id: str, timeframe_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze how user's topic preferences have evolved.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            Topic evolution analysis
        """
        # Get learning history
        learning_history = self.learning_history.get(user_id, [])
        if not learning_history:
            return {"error": "No learning history available"}

        # Filter by timeframe
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=timeframe_days)
        recent_learning = [
            result
            for result in learning_history
            if result.learning_timestamp >= cutoff_date
        ]

        if not recent_learning:
            return {"error": "No recent learning data available"}

        # Analyze evolution
        evolution_analysis = {
            "timeframe_days": timeframe_days,
            "learning_sessions": len(recent_learning),
            "topic_stability": self._analyze_topic_stability(recent_learning),
            "emerging_interests": self._identify_emerging_interests(recent_learning),
            "declining_interests": self._identify_declining_interests(recent_learning),
            "preference_volatility": self._calculate_preference_volatility(
                recent_learning
            ),
            "cluster_evolution": self._analyze_cluster_evolution(recent_learning),
        }

        return evolution_analysis

    def get_topic_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about user's topic preferences.

        Args:
            user_id: User identifier

        Returns:
            Topic insights and analytics
        """
        user_engagements = self.user_topic_engagements.get(user_id, {})
        if not user_engagements:
            return {"error": "No topic data available"}

        insights = {
            "total_topics_engaged": len(user_engagements),
            "top_interests": self._get_top_interests(user_engagements),
            "engagement_distribution": self._analyze_engagement_distribution(
                user_engagements
            ),
            "topic_diversity_score": self._calculate_topic_diversity(user_engagements),
            "recent_activity": self._analyze_recent_topic_activity(user_engagements),
            "emotional_topic_associations": self._analyze_emotional_associations(
                user_engagements
            ),
            "temporal_patterns": self._analyze_temporal_patterns(user_engagements),
        }

        return insights

    def _extract_topics_from_conversation(self, message_text: str) -> List[str]:
        """Extract topics from conversation text."""
        topics = []

        # Use topic taxonomy for extraction
        message_lower = message_text.lower()

        for category, keywords in self.topic_taxonomy.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(category)

        # Extract named entities as potential topics
        entities = self._extract_named_entities(message_text)
        topics.extend(entities)

        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics

    def _calculate_topic_engagement(
        self,
        topics: List[str],
        emotional_state: EmotionalState,
        engagement_indicators: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate engagement scores for topics."""
        engagement_scores = {}

        base_engagement = 0.5

        # Adjust based on emotional state
        emotional_multiplier = 1.0
        if emotional_state.intensity > 0.7:
            emotional_multiplier = 1.5  # High emotional intensity indicates engagement
        elif emotional_state.intensity < 0.3:
            emotional_multiplier = 0.7  # Low intensity indicates less engagement

        # Adjust based on emotional valence
        if emotional_state.primary_emotion.value in ["joy", "excitement", "interest"]:
            emotional_multiplier *= 1.2
        elif emotional_state.primary_emotion.value in ["sadness", "anger", "fear"]:
            emotional_multiplier *= 0.8

        # Apply engagement indicators
        indicator_multiplier = 1.0
        if engagement_indicators:
            message_length = engagement_indicators.get("message_length", 100)
            response_time = engagement_indicators.get("response_time_seconds", 30)

            # Longer messages indicate higher engagement
            if message_length > 200:
                indicator_multiplier *= 1.3
            elif message_length < 50:
                indicator_multiplier *= 0.8

            # Faster responses indicate higher engagement
            if response_time < 10:
                indicator_multiplier *= 1.2
            elif response_time > 60:
                indicator_multiplier *= 0.9

        # Calculate final engagement scores
        for topic in topics:
            final_engagement = (
                base_engagement * emotional_multiplier * indicator_multiplier
            )
            engagement_scores[topic] = min(1.0, final_engagement)

        return engagement_scores

    def _update_topic_engagements(
        self,
        user_id: str,
        topics: List[str],
        engagement_scores: Dict[str, float],
        context: ConversationContext,
    ) -> None:
        """Update user's topic engagements."""
        current_time = datetime.now(timezone.utc)

        for topic in topics:
            engagement_score = engagement_scores.get(topic, 0.5)

            if topic in self.user_topic_engagements[user_id]:
                # Update existing engagement
                engagement = self.user_topic_engagements[user_id][topic]
                engagement.interaction_count += 1
                engagement.last_discussed = current_time

                # Update engagement score with weighted average
                weight = 0.3  # Weight for new observation
                engagement.engagement_score = (
                    engagement.engagement_score * (1 - weight)
                    + engagement_score * weight
                )

                # Update preference strength based on consistency
                consistency_bonus = 0.1 if engagement_score > 0.6 else -0.05
                engagement.preference_strength = min(
                    1.0, max(0.0, engagement.preference_strength + consistency_bonus)
                )

                # Add emotional response
                if hasattr(context, "emotional_state") and context.emotional_state:
                    emotion = context.emotional_state.primary_emotion.value
                    engagement.emotional_responses.append(emotion)
                    # Keep only recent emotional responses
                    if len(engagement.emotional_responses) > 10:
                        engagement.emotional_responses = engagement.emotional_responses[
                            -10:
                        ]

            else:
                # Create new engagement
                self.user_topic_engagements[user_id][topic] = TopicEngagement(
                    topic=topic,
                    engagement_score=engagement_score,
                    interaction_count=1,
                    total_time_spent=timedelta(minutes=1),  # Estimated
                    preference_strength=engagement_score,
                    last_discussed=current_time,
                )

    def _cluster_user_topics(self, user_id: str) -> List[TopicCluster]:
        """Cluster user's topics based on relationships and engagement."""
        user_engagements = self.user_topic_engagements.get(user_id, {})
        if len(user_engagements) < 3:
            return []

        clusters = []
        topics = list(user_engagements.keys())
        clustered_topics = set()

        # Create clusters based on topic relationships
        for topic in topics:
            if topic in clustered_topics:
                continue

            related_topics = self._get_related_topics(topic)
            cluster_topics = [topic]

            # Find related topics that user has engaged with
            for related_topic in related_topics:
                if (
                    related_topic in user_engagements
                    and related_topic not in clustered_topics
                ):
                    cluster_topics.append(related_topic)

            if len(cluster_topics) >= 2:
                # Calculate cluster score
                cluster_score = sum(
                    user_engagements[t].preference_strength for t in cluster_topics
                ) / len(cluster_topics)

                # Find representative topic (highest engagement)
                representative = max(
                    cluster_topics,
                    key=lambda t: user_engagements[t].preference_strength,
                )

                cluster = TopicCluster(
                    cluster_name=f"{representative}_cluster",
                    topics=cluster_topics,
                    cluster_score=cluster_score,
                    representative_topic=representative,
                    user_affinity=cluster_score,
                )

                clusters.append(cluster)
                clustered_topics.update(cluster_topics)

        return clusters

    def _analyze_preference_changes(self, user_id: str) -> Dict[str, Any]:
        """Analyze changes in topic preferences."""
        learning_history = self.learning_history.get(user_id, [])
        if len(learning_history) < 2:
            return {}

        current_prefs = learning_history[-1].learned_preferences
        previous_prefs = learning_history[-2].learned_preferences

        changes = {
            "new_topics": [],
            "strengthened_topics": [],
            "weakened_topics": [],
            "lost_topics": [],
        }

        # Create lookup dictionaries
        current_dict = {pref.topic: pref for pref in current_prefs}
        previous_dict = {pref.topic: pref for pref in previous_prefs}

        # Find new topics
        for topic in current_dict:
            if topic not in previous_dict:
                changes["new_topics"].append(topic)

        # Find lost topics
        for topic in previous_dict:
            if topic not in current_dict:
                changes["lost_topics"].append(topic)

        # Find strengthened/weakened topics
        for topic in current_dict:
            if topic in previous_dict:
                current_strength = current_dict[topic].preference_strength
                previous_strength = previous_dict[topic].preference_strength
                change = current_strength - previous_strength

                if change > 0.1:
                    changes["strengthened_topics"].append((topic, change))
                elif change < -0.1:
                    changes["weakened_topics"].append((topic, abs(change)))

        return changes

    def _calculate_learning_confidence(
        self, user_id: str, topics: List[str], emotional_state: EmotionalState
    ) -> float:
        """Calculate confidence in topic learning."""
        confidence_factors = []

        # Topic count factor
        topic_confidence = min(1.0, len(topics) / 5.0)
        confidence_factors.append(topic_confidence)

        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)

        # User history factor
        user_engagements = self.user_topic_engagements.get(user_id, {})
        history_confidence = min(1.0, len(user_engagements) / 10.0)
        confidence_factors.append(history_confidence)

        # Learning history factor
        learning_history = self.learning_history.get(user_id, [])
        learning_confidence = min(1.0, len(learning_history) / 5.0)
        confidence_factors.append(learning_confidence)

        return sum(confidence_factors) / len(confidence_factors)

    def _predict_from_clusters(
        self, user_id: str, topic: str
    ) -> Optional[TopicPrediction]:
        """Predict interest based on topic clusters."""
        user_clusters = self.user_topic_clusters.get(user_id, [])

        for cluster in user_clusters:
            related_topics = self._get_related_topics(topic)
            cluster_overlap = len(set(cluster.topics).intersection(set(related_topics)))

            if cluster_overlap > 0:
                # Predict based on cluster affinity
                predicted_interest = cluster.user_affinity * (cluster_overlap / len(related_topics))
                
                return TopicPrediction(
                    topic=topic,
                    predicted_interest=predicted_interest,
                    confidence=0.7,
                    reasoning=f"Related to {cluster.cluster_name} with {cluster_overlap} overlapping topics",
                    related_topics=cluster.topics,
                )

        return None

    def _predict_from_similarity(
        self, user_id: str, topic: str
    ) -> Optional[TopicPrediction]:
        """Predict interest based on topic similarity."""
        user_engagements = self.user_topic_engagements.get(user_id, {})
        related_topics = self._get_related_topics(topic)

        # Find similar topics user has engaged with
        similar_engagements = []
        for related_topic in related_topics:
            if related_topic in user_engagements:
                similar_engagements.append(user_engagements[related_topic])

        if similar_engagements:
            # Average engagement of similar topics
            avg_engagement = sum(
                eng.preference_strength for eng in similar_engagements
            ) / len(similar_engagements)

            return TopicPrediction(
                topic=topic,
                predicted_interest=avg_engagement * 0.8,  # Discount for uncertainty
                confidence=0.6,
                reasoning=f"Similar to {len(similar_engagements)} topics user has engaged with",
                related_topics=related_topics,
            )

        return None

    def _get_contextual_topics(
        self, context: str, user_engagements: Dict[str, TopicEngagement]
    ) -> List[ConversationTopic]:
        """Get topics relevant to current context."""
        contextual_topics = []
        context_topics = self._extract_topics_from_conversation(context)

        for topic in context_topics:
            if topic in user_engagements:
                engagement = user_engagements[topic]
                contextual_topics.append(
                    ConversationTopic(
                        topic=topic,
                        relevance_score=engagement.preference_strength,
                        category="contextual",
                    )
                )

        return contextual_topics

    def _get_emotional_topics(
        self, emotional_state: EmotionalState, user_engagements: Dict[str, TopicEngagement]
    ) -> List[ConversationTopic]:
        """Get topics appropriate for current emotional state."""
        emotional_topics = []

        # Define emotion-topic mappings
        emotion_topic_map = {
            "joy": ["entertainment", "hobbies", "achievements", "relationships"],
            "sadness": ["support", "comfort", "understanding", "healing"],
            "anger": ["problem_solving", "stress_relief", "communication"],
            "fear": ["safety", "reassurance", "planning", "support"],
            "surprise": ["discovery", "learning", "exploration"],
            "neutral": ["general", "interests", "daily_life"],
        }

        emotion = emotional_state.primary_emotion.value
        relevant_topic_categories = emotion_topic_map.get(emotion, ["general"])

        for topic_name, engagement in user_engagements.items():
            # Check if topic matches emotional needs
            if any(cat in topic_name.lower() for cat in relevant_topic_categories):
                emotional_topics.append(
                    ConversationTopic(
                        topic=topic_name,
                        relevance_score=engagement.preference_strength,
                        category="emotional_match",
                    )
                )

        return emotional_topics

    def _get_discovery_topic(
        self, user_id: str, existing_recommendations: List[ConversationTopic]
    ) -> Optional[ConversationTopic]:
        """Get a discovery topic to expand user's interests."""
        user_engagements = self.user_topic_engagements.get(user_id, {})
        existing_topics = {rec.topic for rec in existing_recommendations}

        # Find topics related to user's interests but not yet explored
        for topic_name, engagement in user_engagements.items():
            related_topics = self._get_related_topics(topic_name)
            for related_topic in related_topics:
                if (
                    related_topic not in user_engagements
                    and related_topic not in existing_topics
                ):
                    return ConversationTopic(
                        topic=related_topic,
                        relevance_score=0.4,  # Lower score for discovery
                        category="discovery",
                    )

        return None

    def _initialize_topic_taxonomy(self) -> Dict[str, List[str]]:
        """Initialize topic taxonomy with keywords."""
        return {
            "technology": [
                "technology", "tech", "computer", "software", "programming", "AI",
                "artificial intelligence", "machine learning", "data science", "coding",
                "development", "app", "website", "digital", "internet", "cyber"
            ],
            "health": [
                "health", "fitness", "exercise", "diet", "nutrition", "wellness",
                "medical", "doctor", "medicine", "mental health", "therapy",
                "workout", "gym", "running", "yoga", "meditation"
            ],
            "work": [
                "work", "job", "career", "business", "professional", "office",
                "meeting", "project", "team", "management", "leadership",
                "productivity", "skills", "training", "interview"
            ],
            "education": [
                "education", "learning", "study", "course", "school", "university",
                "knowledge", "skill", "training", "teaching", "research",
                "academic", "degree", "certification", "book", "reading"
            ],
            "relationships": [
                "relationship", "family", "friends", "love", "dating", "marriage",
                "social", "communication", "trust", "support", "connection",
                "friendship", "romance", "partnership", "community"
            ],
            "hobbies": [
                "hobby", "music", "art", "sports", "reading", "cooking", "travel",
                "photography", "gaming", "crafts", "collecting", "gardening",
                "painting", "writing", "dancing", "singing"
            ],
            "finance": [
                "money", "finance", "investment", "budget", "savings", "economy",
                "business", "banking", "credit", "debt", "retirement", "taxes",
                "insurance", "stocks", "cryptocurrency", "wealth"
            ],
            "entertainment": [
                "movie", "film", "book", "game", "entertainment", "fun", "leisure",
                "television", "streaming", "music", "concert", "theater",
                "comedy", "drama", "adventure", "mystery"
            ],
            "lifestyle": [
                "lifestyle", "home", "fashion", "beauty", "style", "design",
                "decoration", "organization", "minimalism", "sustainability",
                "environment", "nature", "outdoor", "adventure"
            ],
            "food": [
                "food", "cooking", "recipe", "restaurant", "meal", "cuisine",
                "nutrition", "diet", "healthy eating", "baking", "grilling",
                "vegetarian", "vegan", "organic", "local food"
            ]
        }

    def _initialize_topic_relationships(self) -> Dict[str, List[str]]:
        """Initialize topic relationships."""
        return {
            "technology": ["work", "education", "entertainment", "finance"],
            "health": ["lifestyle", "food", "hobbies", "work"],
            "work": ["technology", "education", "finance", "relationships"],
            "education": ["technology", "work", "hobbies", "relationships"],
            "relationships": ["lifestyle", "health", "entertainment", "hobbies"],
            "hobbies": ["entertainment", "lifestyle", "health", "relationships"],
            "finance": ["work", "technology", "education", "lifestyle"],
            "entertainment": ["hobbies", "relationships", "lifestyle", "technology"],
            "lifestyle": ["health", "hobbies", "relationships", "food"],
            "food": ["health", "lifestyle", "hobbies", "entertainment"],
        }

    def _get_related_topics(self, topic: str) -> List[str]:
        """Get topics related to the given topic."""
        return self.topic_relationships.get(topic, [])

    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities as potential topics."""
        # Simple implementation - in production, use NLP library
        entities = []
        
        # Look for capitalized words that might be entities
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 3 and word.isalpha():
                entities.append(word.lower())
        
        return entities[:3]  # Limit to avoid noise

    def _cache_topic_preferences(self, user_id: str) -> None:
        """Cache user's topic preferences."""
        cache_key = f"topic_preferences_{user_id}"
        
        # Convert engagements to cacheable format
        engagements_data = {}
        for topic, engagement in self.user_topic_engagements[user_id].items():
            engagements_data[topic] = {
                "engagement_score": engagement.engagement_score,
                "interaction_count": engagement.interaction_count,
                "preference_strength": engagement.preference_strength,
                "last_discussed": engagement.last_discussed.isoformat(),
                "emotional_responses": engagement.emotional_responses,
            }
        
        self.cache.set(cache_key, engagements_data)

    def _analyze_topic_stability(self, learning_results: List[TopicLearningResult]) -> Dict[str, Any]:
        """Analyze stability of topic preferences."""
        if len(learning_results) < 2:
            return {"stability": "insufficient_data"}

        # Track topic presence across sessions
        topic_presence = defaultdict(int)
        total_sessions = len(learning_results)

        for result in learning_results:
            topics_in_session = {pref.topic for pref in result.learned_preferences}
            for topic in topics_in_session:
                topic_presence[topic] += 1

        # Calculate stability metrics
        stable_topics = [
            topic for topic, count in topic_presence.items()
            if count >= total_sessions * 0.7
        ]
        
        volatile_topics = [
            topic for topic, count in topic_presence.items()
            if count <= total_sessions * 0.3
        ]

        stability_score = len(stable_topics) / max(1, len(topic_presence))

        return {
            "stability_score": stability_score,
            "stable_topics": stable_topics,
            "volatile_topics": volatile_topics,
            "total_unique_topics": len(topic_presence),
        }

    def _identify_emerging_interests(self, learning_results: List[TopicLearningResult]) -> List[str]:
        """Identify emerging topic interests."""
        if len(learning_results) < 3:
            return []

        # Compare recent vs older sessions
        recent_sessions = learning_results[-3:]
        older_sessions = learning_results[:-3]

        recent_topics = set()
        for result in recent_sessions:
            recent_topics.update(pref.topic for pref in result.learned_preferences)

        older_topics = set()
        for result in older_sessions:
            older_topics.update(pref.topic for pref in result.learned_preferences)

        # Emerging topics appear in recent but not older sessions
        emerging = list(recent_topics - older_topics)
        return emerging[:5]  # Limit to top 5

    def _identify_declining_interests(self, learning_results: List[TopicLearningResult]) -> List[str]:
        """Identify declining topic interests."""
        if len(learning_results) < 3:
            return []

        # Compare recent vs older sessions
        recent_sessions = learning_results[-3:]
        older_sessions = learning_results[:-3]

        recent_topics = set()
        for result in recent_sessions:
            recent_topics.update(pref.topic for pref in result.learned_preferences)

        older_topics = set()
        for result in older_sessions:
            older_topics.update(pref.topic for pref in result.learned_preferences)

        # Declining topics appear in older but not recent sessions
        declining = list(older_topics - recent_topics)
        return declining[:5]  # Limit to top 5

    def _calculate_preference_volatility(self, learning_results: List[TopicLearningResult]) -> float:
        """Calculate volatility of topic preferences."""
        if len(learning_results) < 2:
            return 0.0

        volatility_scores = []
        
        for i in range(1, len(learning_results)):
            current_topics = {pref.topic for pref in learning_results[i].learned_preferences}
            previous_topics = {pref.topic for pref in learning_results[i-1].learned_preferences}
            
            # Calculate Jaccard distance (1 - Jaccard similarity)
            intersection = len(current_topics.intersection(previous_topics))
            union = len(current_topics.union(previous_topics))
            
            if union > 0:
                jaccard_similarity = intersection / union
                volatility = 1 - jaccard_similarity
                volatility_scores.append(volatility)

        return sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0.0

    def _analyze_cluster_evolution(self, learning_results: List[TopicLearningResult]) -> Dict[str, Any]:
        """Analyze evolution of topic clusters."""
        if len(learning_results) < 2:
            return {"evolution": "insufficient_data"}

        # Compare cluster structures
        first_clusters = learning_results[0].topic_clusters
        last_clusters = learning_results[-1].topic_clusters

        evolution = {
            "initial_clusters": len(first_clusters),
            "final_clusters": len(last_clusters),
            "cluster_growth": len(last_clusters) - len(first_clusters),
            "stable_clusters": 0,
            "new_clusters": 0,
            "dissolved_clusters": 0,
        }

        # Analyze cluster stability (simplified)
        first_cluster_names = {cluster.cluster_name for cluster in first_clusters}
        last_cluster_names = {cluster.cluster_name for cluster in last_clusters}

        evolution["stable_clusters"] = len(first_cluster_names.intersection(last_cluster_names))
        evolution["new_clusters"] = len(last_cluster_names - first_cluster_names)
        evolution["dissolved_clusters"] = len(first_cluster_names - last_cluster_names)

        return evolution

    def _get_top_interests(self, user_engagements: Dict[str, TopicEngagement]) -> List[Dict[str, Any]]:
        """Get user's top interests."""
        sorted_engagements = sorted(
            user_engagements.items(),
            key=lambda x: x[1].preference_strength,
            reverse=True
        )

        top_interests = []
        for topic, engagement in sorted_engagements[:10]:
            top_interests.append({
                "topic": topic,
                "preference_strength": engagement.preference_strength,
                "engagement_score": engagement.engagement_score,
                "interaction_count": engagement.interaction_count,
                "last_discussed": engagement.last_discussed.isoformat(),
            })

        return top_interests

    def _analyze_engagement_distribution(self, user_engagements: Dict[str, TopicEngagement]) -> Dict[str, Any]:
        """Analyze distribution of engagement scores."""
        if not user_engagements:
            return {}

        engagement_scores = [eng.engagement_score for eng in user_engagements.values()]
        preference_strengths = [eng.preference_strength for eng in user_engagements.values()]

        return {
            "engagement_mean": sum(engagement_scores) / len(engagement_scores),
            "engagement_std": self._calculate_std(engagement_scores),
            "preference_mean": sum(preference_strengths) / len(preference_strengths),
            "preference_std": self._calculate_std(preference_strengths),
            "high_engagement_topics": len([s for s in engagement_scores if s > 0.7]),
            "low_engagement_topics": len([s for s in engagement_scores if s < 0.3]),
        }

    def _calculate_topic_diversity(self, user_engagements: Dict[str, TopicEngagement]) -> float:
        """Calculate topic diversity score."""
        if not user_engagements:
            return 0.0

        # Count topics per category
        category_counts = defaultdict(int)
        for topic in user_engagements.keys():
            # Find which category this topic belongs to
            for category, keywords in self.topic_taxonomy.items():
                if any(keyword in topic.lower() for keyword in keywords):
                    category_counts[category] += 1
                    break

        # Calculate diversity using Shannon entropy
        total_topics = sum(category_counts.values())
        if total_topics == 0:
            return 0.0

        entropy = 0.0
        for count in category_counts.values():
            if count > 0:
                p = count / total_topics
                entropy -= p * (p.bit_length() - 1)  # Approximation of log2

        # Normalize by maximum possible entropy
        max_entropy = len(category_counts).bit_length() - 1 if category_counts else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _analyze_recent_topic_activity(self, user_engagements: Dict[str, TopicEngagement]) -> Dict[str, Any]:
        """Analyze recent topic activity."""
        if not user_engagements:
            return {}

        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=7)
        very_recent_cutoff = now - timedelta(days=1)

        recent_topics = [
            topic for topic, eng in user_engagements.items()
            if eng.last_discussed >= recent_cutoff
        ]

        very_recent_topics = [
            topic for topic, eng in user_engagements.items()
            if eng.last_discussed >= very_recent_cutoff
        ]

        return {
            "topics_discussed_last_week": len(recent_topics),
            "topics_discussed_yesterday": len(very_recent_topics),
            "most_recent_topics": recent_topics[:5],
            "activity_level": "high" if len(recent_topics) > 5 else "moderate" if len(recent_topics) > 2 else "low",
        }

    def _analyze_emotional_associations(self, user_engagements: Dict[str, TopicEngagement]) -> Dict[str, Any]:
        """Analyze emotional associations with topics."""
        emotional_associations = defaultdict(list)

        for topic, engagement in user_engagements.items():
            if engagement.emotional_responses:
                # Count emotional responses for this topic
                emotion_counts = Counter(engagement.emotional_responses)
                most_common_emotion = emotion_counts.most_common(1)[0][0]
                emotional_associations[most_common_emotion].append(topic)

        return dict(emotional_associations)

    def _analyze_temporal_patterns(self, user_engagements: Dict[str, TopicEngagement]) -> Dict[str, Any]:
        """Analyze temporal patterns in topic engagement."""
        if not user_engagements:
            return {}

        # Analyze by day of week and hour (simplified)
        discussion_times = [eng.last_discussed for eng in user_engagements.values()]
        
        # Group by hour of day
        hour_counts = defaultdict(int)
        for dt in discussion_times:
            hour_counts[dt.hour] += 1

        most_active_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 12

        return {
            "most_active_hour": most_active_hour,
            "total_discussions": len(discussion_times),
            "discussion_span_days": (max(discussion_times) - min(discussion_times)).days if len(discussion_times) > 1 else 0,
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# Singleton instance
_topic_preference_learner_instance = None
_topic_preference_learner_lock = threading.Lock()


def get_topic_preference_learner() -> TopicPreferenceLearner:
    """
    Get singleton topic preference learner instance.

    Returns:
        Shared TopicPreferenceLearner instance
    """
    global _topic_preference_learner_instance

    if _topic_preference_learner_instance is None:
        with _topic_preference_learner_lock:
            if _topic_preference_learner_instance is None:
                _topic_preference_learner_instance = TopicPreferenceLearner()

    return _topic_preference_learner_instance