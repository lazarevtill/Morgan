"""
Enhanced Memory Processor with Emotional Awareness

Automatically extracts and stores conversation memories with emotional context,
importance scoring with emotional weighting, and personal preference detection.
"""

import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.core.memory import ConversationTurn
from morgan.emotional.intelligence_engine import get_emotional_intelligence_engine
from morgan.emotional.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
    UserPreferences,
)
from morgan.services.embedding_service import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient

logger = get_logger(__name__)


@dataclass
class Memory:
    """Enhanced memory with emotional context."""

    memory_id: str
    content: str
    importance_score: float
    entities: List[str]
    concepts: List[str]
    conversation_id: str
    turn_id: str
    timestamp: datetime
    feedback_rating: Optional[int] = None
    emotional_context: Optional[Dict[str, Any]] = None
    user_mood: Optional[str] = None
    relationship_significance: float = 0.0
    personal_preferences: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.personal_preferences is None:
            self.personal_preferences = []


@dataclass
class MemoryExtractionResult:
    """Result of memory extraction process."""

    memories: List[Memory]
    emotional_insights: Dict[str, Any]
    preference_updates: List[str]
    relationship_indicators: List[str]


class MemoryProcessor:
    """
    Enhanced Memory Processor with Emotional Awareness

    Features:
    - Automatic memory extraction with emotional context
    - Importance scoring with emotional weighting
    - Personal preference and relationship context detection
    - Memory storage with emotional and relationship metadata
    """

    # Patterns for identifying important content
    IMPORTANCE_PATTERNS = {
        "high_importance": [
            r"\b(important|crucial|critical|essential|vital|key)\b",
            r"\b(remember|don\'t forget|keep in mind|note that)\b",
            r"\b(breakthrough|insight|realization|discovery)\b",
            r"\b(goal|objective|target|achievement)\b",
            r"\b(problem|issue|challenge|difficulty)\b",
        ],
        "personal_context": [
            r"\b(I am|I\'m|my|mine|personally)\b",
            r"\b(family|friend|colleague|partner|spouse)\b",
            r"\b(work|job|career|project|company)\b",
            r"\b(hobby|interest|passion|love|enjoy)\b",
            r"\b(feel|feeling|emotion|mood)\b",
        ],
        "learning_indicators": [
            r"\b(learn|understand|grasp|comprehend|figure out)\b",
            r"\b(skill|knowledge|technique|method|approach)\b",
            r"\b(practice|exercise|training|study)\b",
            r"\b(improve|better|enhance|develop)\b",
        ],
        "relationship_signals": [
            r"\b(trust|comfortable|safe|open|honest)\b",
            r"\b(share|tell|confide|reveal|admit)\b",
            r"\b(appreciate|grateful|thankful|helpful)\b",
            r"\b(understand me|get me|relate to)\b",
        ],
    }

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "person": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        "organization": r"\b([A-Z][a-z]+(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Organization))?)\b",
        "technology": r"\b(Python|JavaScript|Docker|Kubernetes|AWS|React|Node\.js|SQL|API)\b",
        "location": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country))?)\b",
    }

    def __init__(self):
        """Initialize memory processor."""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBClient()
        self.emotional_engine = get_emotional_intelligence_engine()

        # Memory collections
        self.memory_collection = "morgan_memories"

        # Processing settings
        self.min_importance_threshold = 0.3
        self.max_memories_per_turn = 5
        self.emotional_weight_multiplier = 1.5

        # Ensure collections exist
        self._ensure_collections()

        logger.info("Enhanced Memory Processor initialized")

    def extract_memories(
        self,
        conversation_turn: ConversationTurn,
        feedback_rating: Optional[int] = None,
        emotional_context: Optional[EmotionalState] = None,
    ) -> MemoryExtractionResult:
        """
        Extract memories from conversation turn with emotional awareness.

        Args:
            conversation_turn: The conversation turn to analyze
            feedback_rating: Optional user feedback rating
            emotional_context: Optional emotional state context

        Returns:
            Memory extraction result with emotional insights
        """
        try:
            # Analyze emotional context if not provided
            if emotional_context is None:
                context = ConversationContext(
                    user_id="default",  # Will be updated with actual user ID
                    conversation_id=conversation_turn.conversation_id,
                    message_text=conversation_turn.question,
                    timestamp=datetime.fromisoformat(conversation_turn.timestamp),
                )
                emotional_context = self.emotional_engine.analyze_emotion(
                    conversation_turn.question, context
                )

            # Extract potential memories from both question and answer
            question_memories = self._extract_memories_from_text(
                conversation_turn.question,
                conversation_turn,
                emotional_context,
                feedback_rating,
                is_user_message=True,
            )

            answer_memories = self._extract_memories_from_text(
                conversation_turn.answer,
                conversation_turn,
                emotional_context,
                feedback_rating,
                is_user_message=False,
            )

            # Combine and deduplicate memories
            all_memories = question_memories + answer_memories
            unique_memories = self._deduplicate_memories(all_memories)

            # Filter by importance threshold
            important_memories = [
                memory
                for memory in unique_memories
                if memory.importance_score >= self.min_importance_threshold
            ]

            # Limit number of memories per turn
            if len(important_memories) > self.max_memories_per_turn:
                important_memories.sort(key=lambda m: m.importance_score, reverse=True)
                important_memories = important_memories[: self.max_memories_per_turn]

            # Extract emotional insights
            emotional_insights = self._extract_emotional_insights(
                conversation_turn, emotional_context
            )

            # Extract preference updates
            preference_updates = self._extract_preference_updates(
                conversation_turn, emotional_context
            )

            # Extract relationship indicators
            relationship_indicators = self._extract_relationship_indicators(
                conversation_turn, emotional_context
            )

            logger.debug(
                f"Extracted {len(important_memories)} memories from conversation turn "
                f"{conversation_turn.turn_id}"
            )

            return MemoryExtractionResult(
                memories=important_memories,
                emotional_insights=emotional_insights,
                preference_updates=preference_updates,
                relationship_indicators=relationship_indicators,
            )

        except Exception as e:
            logger.error(f"Failed to extract memories: {e}")
            return MemoryExtractionResult(
                memories=[],
                emotional_insights={},
                preference_updates=[],
                relationship_indicators=[],
            )

    def score_importance(
        self, content: str, context: Dict[str, Any], emotional_weight: float = 1.0
    ) -> float:
        """
        Score importance of content with emotional weighting.

        Args:
            content: Text content to score
            context: Additional context for scoring
            emotional_weight: Emotional weighting factor

        Returns:
            Importance score (0.0 to 1.0)
        """
        base_score = 0.0
        content_lower = content.lower()

        # Pattern-based scoring
        for category, patterns in self.IMPORTANCE_PATTERNS.items():
            category_score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                category_score += matches * 0.1

            # Weight different categories
            if category == "high_importance":
                base_score += category_score * 1.0
            elif category == "personal_context":
                base_score += category_score * 0.8
            elif category == "learning_indicators":
                base_score += category_score * 0.7
            elif category == "relationship_signals":
                base_score += category_score * 0.9

        # Content length factor (longer content often more important)
        length_factor = min(1.0, len(content) / 200.0)
        base_score += length_factor * 0.2

        # Feedback rating factor
        feedback_rating = context.get("feedback_rating")
        if feedback_rating:
            # Convert 1-5 to -1 to 1
            feedback_factor = (feedback_rating - 3) / 2.0
            base_score += feedback_factor * 0.3

        # Apply emotional weighting
        final_score = base_score * emotional_weight

        # Normalize to 0-1 range
        return min(1.0, max(0.0, final_score))

    def detect_entities(self, content: str) -> List[str]:
        """
        Detect entities in content.

        Args:
            content: Text content to analyze

        Returns:
            List of detected entities
        """
        entities = []

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Take first group if tuple
                if len(match) > 2:  # Filter out very short matches
                    entities.append(f"{entity_type}:{match}")

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def extract_personal_preferences(
        self, conversation_history: List[ConversationTurn]
    ) -> UserPreferences:
        """
        Extract personal preferences from conversation history.

        Args:
            conversation_history: List of conversation turns

        Returns:
            Extracted user preferences
        """
        topics_of_interest = []
        learning_goals = []
        personal_context = {}

        # Analyze all conversations for patterns
        for turn in conversation_history:
            # Extract topics from questions
            question_topics = self._extract_topics_from_text(turn.question)
            topics_of_interest.extend(question_topics)

            # Extract learning goals
            learning_patterns = [
                r"want to learn (\w+)",
                r"interested in (\w+)",
                r"studying (\w+)",
                r"working on (\w+)",
            ]

            for pattern in learning_patterns:
                matches = re.findall(pattern, turn.question.lower())
                learning_goals.extend(matches)

        # Count frequency and keep most common
        topic_counts = Counter(topics_of_interest)
        top_topics = [topic for topic, _ in topic_counts.most_common(10)]

        goal_counts = Counter(learning_goals)
        top_goals = [goal for goal, _ in goal_counts.most_common(5)]

        # Determine communication style based on message patterns
        communication_style = self._determine_communication_style(conversation_history)

        # Determine preferred response length
        response_length = self._determine_response_length_preference(
            conversation_history
        )

        return UserPreferences(
            topics_of_interest=top_topics,
            communication_style=communication_style,
            preferred_response_length=response_length,
            learning_goals=top_goals,
            personal_context=personal_context,
        )

    def store_memory(self, memory: Memory) -> bool:
        """
        Store memory in vector database.

        Args:
            memory: Memory to store

        Returns:
            True if stored successfully
        """
        try:
            # Create embedding for memory content
            memory_embedding = self.embedding_service.encode(
                text=memory.content, instruction="document"
            )

            # Prepare payload with emotional and relationship metadata
            payload = asdict(memory)

            # Convert datetime to ISO string
            if isinstance(payload["timestamp"], datetime):
                payload["timestamp"] = payload["timestamp"].isoformat()

            # Create point for vector database
            point = {
                "id": memory.memory_id,
                "vector": memory_embedding,
                "payload": payload,
            }

            # Store in vector database
            self.vector_db.upsert_points(self.memory_collection, [point])

            logger.debug(f"Stored memory: {memory.memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory {memory.memory_id}: {e}")
            return False

    def _extract_memories_from_text(
        self,
        text: str,
        conversation_turn: ConversationTurn,
        emotional_context: EmotionalState,
        feedback_rating: Optional[int],
        is_user_message: bool,
    ) -> List[Memory]:
        """Extract memories from text content."""
        memories = []

        # Split text into sentences for granular memory extraction
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue

            # Calculate emotional weighting
            emotional_weight = self._calculate_emotional_weight(
                emotional_context, is_user_message
            )

            # Score importance
            context = {
                "feedback_rating": feedback_rating,
                "is_user_message": is_user_message,
                "emotional_context": emotional_context,
            }

            importance_score = self.score_importance(
                sentence, context, emotional_weight
            )

            # Only create memory if above threshold
            if importance_score >= self.min_importance_threshold:
                # Extract entities and concepts
                entities = self.detect_entities(sentence)
                concepts = self._extract_concepts(sentence)

                # Extract personal preferences
                personal_prefs = self._extract_personal_preferences_from_text(sentence)

                # Calculate relationship significance
                relationship_significance = self._calculate_relationship_significance(
                    sentence, emotional_context
                )

                memory = Memory(
                    memory_id=str(uuid.uuid4()),
                    content=sentence.strip(),
                    importance_score=importance_score,
                    entities=entities,
                    concepts=concepts,
                    conversation_id=conversation_turn.conversation_id,
                    turn_id=conversation_turn.turn_id,
                    timestamp=datetime.fromisoformat(conversation_turn.timestamp),
                    feedback_rating=feedback_rating,
                    emotional_context={
                        "primary_emotion": emotional_context.primary_emotion.value,
                        "intensity": emotional_context.intensity,
                        "confidence": emotional_context.confidence,
                    },
                    user_mood=emotional_context.primary_emotion.value,
                    relationship_significance=relationship_significance,
                    personal_preferences=personal_prefs,
                )

                memories.append(memory)

        return memories

    def _calculate_emotional_weight(
        self, emotional_context: EmotionalState, is_user_message: bool
    ) -> float:
        """Calculate emotional weighting factor."""
        base_weight = 1.0

        # Higher weight for emotional messages
        emotional_intensity_weight = 1.0 + (emotional_context.intensity * 0.5)

        # Higher weight for user messages (more personal)
        user_message_weight = 1.2 if is_user_message else 1.0

        # Higher weight for certain emotions
        emotion_weights = {
            EmotionType.JOY: 1.3,
            EmotionType.SADNESS: 1.4,
            EmotionType.ANGER: 1.2,
            EmotionType.FEAR: 1.3,
            EmotionType.SURPRISE: 1.1,
            EmotionType.DISGUST: 1.0,
            EmotionType.NEUTRAL: 1.0,
        }

        emotion_weight = emotion_weights.get(emotional_context.primary_emotion, 1.0)

        return (
            base_weight
            * emotional_intensity_weight
            * user_message_weight
            * emotion_weight
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLP library)
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple concept extraction based on important words
        concept_patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Proper nouns
            r"\b(\w+ing)\b",  # Gerunds (actions)
            r"\b(\w+tion)\b",  # Abstract concepts ending in -tion
            r"\b(\w+ness)\b",  # Abstract concepts ending in -ness
        ]

        concepts = []
        text.lower()

        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)

        # Filter out common words and short concepts
        filtered_concepts = [
            concept
            for concept in concepts
            if len(concept) > 3
            and concept.lower() not in ["this", "that", "with", "from"]
        ]

        return list(set(filtered_concepts))  # Remove duplicates

    def _extract_personal_preferences_from_text(self, text: str) -> List[str]:
        """Extract personal preferences from text."""
        preferences = []
        text_lower = text.lower()

        # Preference patterns
        preference_patterns = [
            r"i like (\w+)",
            r"i prefer (\w+)",
            r"i enjoy (\w+)",
            r"i love (\w+)",
            r"i hate (\w+)",
            r"i dislike (\w+)",
        ]

        for pattern in preference_patterns:
            matches = re.findall(pattern, text_lower)
            preferences.extend(matches)

        return preferences

    def _calculate_relationship_significance(
        self, text: str, emotional_context: EmotionalState
    ) -> float:
        """Calculate relationship significance of content."""
        significance = 0.0
        text_lower = text.lower()

        # Relationship building indicators
        relationship_patterns = [
            r"\b(trust|comfortable|safe|open)\b",
            r"\b(personal|private|share|tell)\b",
            r"\b(understand|relate|connect)\b",
            r"\b(appreciate|grateful|thankful)\b",
        ]

        for pattern in relationship_patterns:
            matches = len(re.findall(pattern, text_lower))
            significance += matches * 0.2

        # Emotional intensity adds to significance
        significance += emotional_context.intensity * 0.3

        # Certain emotions are more significant for relationships
        emotion_significance = {
            EmotionType.JOY: 0.3,
            EmotionType.SADNESS: 0.4,
            EmotionType.FEAR: 0.3,
            EmotionType.SURPRISE: 0.2,
            EmotionType.ANGER: 0.1,
            EmotionType.DISGUST: 0.1,
            EmotionType.NEUTRAL: 0.1,
        }

        significance += emotion_significance.get(emotional_context.primary_emotion, 0.1)

        return min(1.0, significance)

    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories based on content similarity."""
        if not memories:
            return memories

        unique_memories = []
        seen_content = set()

        for memory in memories:
            # Simple deduplication based on content similarity
            content_key = memory.content.lower().strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_memories.append(memory)

        return unique_memories

    def _extract_emotional_insights(
        self, conversation_turn: ConversationTurn, emotional_context: EmotionalState
    ) -> Dict[str, Any]:
        """Extract emotional insights from conversation."""
        return {
            "primary_emotion": emotional_context.primary_emotion.value,
            "emotional_intensity": emotional_context.intensity,
            "emotional_confidence": emotional_context.confidence,
            "emotional_indicators": emotional_context.emotional_indicators,
            "conversation_emotional_tone": self._analyze_conversation_tone(
                conversation_turn
            ),
        }

    def _extract_preference_updates(
        self, conversation_turn: ConversationTurn, emotional_context: EmotionalState
    ) -> List[str]:
        """Extract preference updates from conversation."""
        updates = []

        # Extract preferences from question
        question_prefs = self._extract_personal_preferences_from_text(
            conversation_turn.question
        )
        updates.extend(question_prefs)

        # Extract topics of interest
        topics = self._extract_topics_from_text(conversation_turn.question)
        updates.extend([f"topic:{topic}" for topic in topics])

        return updates

    def _extract_relationship_indicators(
        self, conversation_turn: ConversationTurn, emotional_context: EmotionalState
    ) -> List[str]:
        """Extract relationship building indicators."""
        indicators = []

        # Check for trust indicators
        if any(
            word in conversation_turn.question.lower()
            for word in ["trust", "comfortable", "safe", "personal"]
        ):
            indicators.append("trust_building")

        # Check for emotional sharing
        if emotional_context.intensity > 0.6:
            indicators.append("emotional_sharing")

        # Check for positive feedback
        if conversation_turn.feedback_rating and conversation_turn.feedback_rating >= 4:
            indicators.append("positive_feedback")

        # Check for length (longer messages indicate engagement)
        if len(conversation_turn.question) > 100:
            indicators.append("high_engagement")

        return indicators

    def _analyze_conversation_tone(self, conversation_turn: ConversationTurn) -> str:
        """Analyze overall tone of conversation."""
        question_lower = conversation_turn.question.lower()

        # Simple tone analysis
        if any(word in question_lower for word in ["please", "thank", "appreciate"]):
            return "polite"
        elif any(word in question_lower for word in ["urgent", "quickly", "asap"]):
            return "urgent"
        elif any(word in question_lower for word in ["confused", "help", "stuck"]):
            return "seeking_help"
        elif "?" in conversation_turn.question:
            return "inquisitive"
        else:
            return "neutral"

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple topic extraction based on nouns and technical terms
        topic_patterns = [
            r"\b(Python|JavaScript|Docker|Kubernetes|AWS|React|Node\.js|SQL|API)\b",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Proper nouns
        ]

        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, text)
            topics.extend(matches)

        # Filter and clean topics
        filtered_topics = [
            topic.lower()
            for topic in topics
            if (len(topic) > 2 and topic.lower() not in ["the", "and", "for", "with"])
        ]

        return list(set(filtered_topics))  # Remove duplicates

    def _determine_communication_style(
        self, conversation_history: List[ConversationTurn]
    ) -> str:
        """Determine user's communication style from history."""
        if not conversation_history:
            return "friendly"

        # Analyze patterns in user messages
        formal_indicators = 0
        casual_indicators = 0
        technical_indicators = 0

        for turn in conversation_history:
            question_lower = turn.question.lower()

            # Formal indicators
            if any(
                word in question_lower for word in ["please", "would you", "could you"]
            ):
                formal_indicators += 1

            # Casual indicators
            if any(word in question_lower for word in ["hey", "hi", "thanks", "cool"]):
                casual_indicators += 1

            # Technical indicators
            if any(
                word in question_lower
                for word in ["function", "method", "class", "api"]
            ):
                technical_indicators += 1

        # Determine dominant style
        if technical_indicators > max(formal_indicators, casual_indicators):
            return "technical"
        elif formal_indicators > casual_indicators:
            return "formal"
        else:
            return "casual"

    def _determine_response_length_preference(
        self, conversation_history: List[ConversationTurn]
    ) -> str:
        """Determine user's preferred response length."""
        if not conversation_history:
            return "detailed"

        # Analyze user question lengths as indicator of preference
        question_lengths = [len(turn.question) for turn in conversation_history]
        avg_length = sum(question_lengths) / len(question_lengths)

        if avg_length > 150:
            return "comprehensive"
        elif avg_length > 50:
            return "detailed"
        else:
            return "brief"

    def _ensure_collections(self):
        """Ensure required collections exist."""
        try:
            # Get embedding dimension
            embedding_dim = self.embedding_service.get_embedding_dimension()

            # Create memory collection
            if not self.vector_db.collection_exists(self.memory_collection):
                self.vector_db.create_collection(
                    name=self.memory_collection,
                    vector_size=embedding_dim,
                    distance="cosine",
                )
                logger.info(f"Created memory collection: {self.memory_collection}")

        except Exception as e:
            logger.error(f"Failed to ensure collections: {e}")
            raise


# Singleton instance for global access
_memory_processor_instance = None


def get_memory_processor() -> MemoryProcessor:
    """
    Get singleton memory processor instance.

    Returns:
        Shared MemoryProcessor instance
    """
    global _memory_processor_instance

    if _memory_processor_instance is None:
        _memory_processor_instance = MemoryProcessor()

    return _memory_processor_instance
