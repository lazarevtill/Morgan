"""
Conversation flow management module.

Manages conversation flow, turn-taking, topic transitions, and maintains
conversational coherence for natural and engaging interactions.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    ConversationTopic,
    EmotionalState,
    UserPreferences,
)
from morgan.services.llm import get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationState(Enum):
    """Current state of the conversation."""

    OPENING = "opening"
    ACTIVE = "active"
    TOPIC_TRANSITION = "topic_transition"
    CLOSING = "closing"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"


class FlowDirection(Enum):
    """Direction of conversation flow."""

    FORWARD = "forward"
    BACKWARD = "backward"
    LATERAL = "lateral"
    CIRCULAR = "circular"


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""

    speaker: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    emotional_state: Optional[EmotionalState] = None
    topic: Optional[str] = None
    turn_type: str = "normal"  # normal, question, answer, clarification
    confidence: float = 1.0


@dataclass
class TopicTransition:
    """Represents a topic transition in conversation."""

    from_topic: str
    to_topic: str
    transition_type: str  # natural, forced, user_initiated, assistant_initiated
    timestamp: datetime
    success_score: float = 0.0
    user_engagement: float = 0.0


@dataclass
class ConversationFlow:
    """Complete conversation flow analysis."""

    conversation_id: str
    user_id: str
    state: ConversationState
    current_topic: Optional[str]
    turns: List[ConversationTurn] = field(default_factory=list)
    topic_transitions: List[TopicTransition] = field(default_factory=list)
    flow_direction: FlowDirection = FlowDirection.FORWARD
    coherence_score: float = 1.0
    engagement_level: float = 0.5
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FlowManagementResult:
    """Result of flow management analysis."""

    recommended_action: str
    confidence: float
    reasoning: str
    suggested_response_type: str
    topic_suggestions: List[ConversationTopic] = field(default_factory=list)
    flow_adjustments: Dict[str, Any] = field(default_factory=dict)


class ConversationFlowManager:
    """
    Conversation flow management system.

    Features:
    - Conversation state tracking and management
    - Topic transition detection and optimization
    - Turn-taking coordination
    - Flow coherence maintenance
    - Engagement level monitoring
    - Natural conversation pacing
    """

    def __init__(self):
        """Initialize conversation flow manager."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Active conversation flows
        self.active_flows: Dict[str, ConversationFlow] = {}

        # Flow management history
        self.flow_history: Dict[str, List[ConversationFlow]] = {}

        logger.info("Conversation Flow Manager initialized")

    def manage_conversation_flow(
        self,
        conversation_id: str,
        user_id: str,
        context: ConversationContext,
        emotional_state: EmotionalState,
        user_preferences: Optional[UserPreferences] = None,
    ) -> FlowManagementResult:
        """
        Manage conversation flow for current interaction.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            context: Current conversation context
            emotional_state: User's emotional state
            user_preferences: User's preferences

        Returns:
            Flow management result with recommendations
        """
        # Get or create conversation flow
        flow = self._get_or_create_flow(conversation_id, user_id)

        # Add current turn to flow
        current_turn = self._create_conversation_turn(
            "user", context.message_text, emotional_state
        )
        flow.turns.append(current_turn)

        # Update flow state
        self._update_flow_state(flow, context, emotional_state)

        # Analyze topic transitions
        topic_transition = self._analyze_topic_transition(flow, context)
        if topic_transition:
            flow.topic_transitions.append(topic_transition)

        # Calculate flow metrics
        self._calculate_flow_metrics(flow)

        # Generate flow management recommendations
        result = self._generate_flow_recommendations(
            flow, context, emotional_state, user_preferences
        )

        # Update flow with assistant response planning
        self._plan_assistant_response(flow, result)

        # Store updated flow
        self.active_flows[conversation_id] = flow
        flow.last_activity = datetime.utcnow()

        logger.debug(
            f"Managed conversation flow for {conversation_id}: "
            f"state={flow.state.value}, coherence={flow.coherence_score:.2f}"
        )

        return result

    def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get current conversation state."""
        flow = self.active_flows.get(conversation_id)
        return flow.state if flow else None

    def transition_topic(
        self,
        conversation_id: str,
        new_topic: str,
        transition_type: str = "natural",
    ) -> bool:
        """
        Transition to a new topic in conversation.

        Args:
            conversation_id: Conversation identifier
            new_topic: New topic to transition to
            transition_type: Type of transition

        Returns:
            Success of topic transition
        """
        flow = self.active_flows.get(conversation_id)
        if not flow:
            return False

        # Create topic transition
        transition = TopicTransition(
            from_topic=flow.current_topic or "unknown",
            to_topic=new_topic,
            transition_type=transition_type,
            timestamp=datetime.utcnow(),
        )

        # Update flow
        flow.current_topic = new_topic
        flow.topic_transitions.append(transition)
        flow.state = ConversationState.TOPIC_TRANSITION

        logger.debug(
            f"Topic transition in {conversation_id}: "
            f"{transition.from_topic} -> {transition.to_topic}"
        )

        return True

    def pause_conversation(self, conversation_id: str) -> bool:
        """Pause conversation flow."""
        flow = self.active_flows.get(conversation_id)
        if not flow:
            return False

        flow.state = ConversationState.PAUSED
        logger.debug(f"Paused conversation {conversation_id}")
        return True

    def resume_conversation(self, conversation_id: str) -> bool:
        """Resume paused conversation flow."""
        flow = self.active_flows.get(conversation_id)
        if not flow or flow.state != ConversationState.PAUSED:
            return False

        flow.state = ConversationState.ACTIVE
        flow.last_activity = datetime.utcnow()
        logger.debug(f"Resumed conversation {conversation_id}")
        return True

    def end_conversation(self, conversation_id: str) -> bool:
        """End conversation and archive flow."""
        flow = self.active_flows.get(conversation_id)
        if not flow:
            return False

        # Update state and archive
        flow.state = ConversationState.CLOSING
        self._archive_conversation_flow(conversation_id, flow)

        # Remove from active flows
        del self.active_flows[conversation_id]

        logger.debug(f"Ended conversation {conversation_id}")
        return True

    def get_flow_analytics(
        self, user_id: str, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get conversation flow analytics for user.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            Flow analytics data
        """
        # Get user's conversation history
        user_flows = []
        for flows in self.flow_history.values():
            user_flows.extend([f for f in flows if f.user_id == user_id])

        # Filter by timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_flows = [f for f in user_flows if f.started_at >= cutoff_date]

        if not recent_flows:
            return {"error": "No conversation data available"}

        # Calculate analytics
        analytics = {
            "total_conversations": len(recent_flows),
            "average_turns_per_conversation": self._calculate_average_turns(
                recent_flows
            ),
            "average_coherence_score": self._calculate_average_coherence(recent_flows),
            "average_engagement_level": self._calculate_average_engagement(
                recent_flows
            ),
            "topic_transition_success_rate": self._calculate_transition_success_rate(
                recent_flows
            ),
            "most_common_topics": self._get_most_common_topics(recent_flows),
            "conversation_patterns": self._analyze_conversation_patterns(recent_flows),
            "flow_direction_distribution": self._analyze_flow_directions(recent_flows),
        }

        return analytics

    def _get_or_create_flow(self, conversation_id: str, user_id: str) -> ConversationFlow:
        """Get existing flow or create new one."""
        if conversation_id in self.active_flows:
            return self.active_flows[conversation_id]

        # Create new flow
        flow = ConversationFlow(
            conversation_id=conversation_id,
            user_id=user_id,
            state=ConversationState.OPENING,
            current_topic=None,
        )

        return flow

    def _create_conversation_turn(
        self, speaker: str, content: str, emotional_state: Optional[EmotionalState]
    ) -> ConversationTurn:
        """Create a conversation turn."""
        # Determine turn type
        turn_type = "normal"
        if content.strip().endswith("?"):
            turn_type = "question"
        elif any(
            word in content.lower()
            for word in ["what", "how", "why", "when", "where", "who"]
        ):
            turn_type = "question"

        # Extract topic (simplified)
        topic = self._extract_topic_from_content(content)

        return ConversationTurn(
            speaker=speaker,
            content=content,
            timestamp=datetime.utcnow(),
            emotional_state=emotional_state,
            topic=topic,
            turn_type=turn_type,
        )

    def _update_flow_state(
        self,
        flow: ConversationFlow,
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> None:
        """Update conversation flow state."""
        # Determine new state based on context
        if len(flow.turns) == 1:
            flow.state = ConversationState.OPENING
        elif self._is_closing_signal(context.message_text):
            flow.state = ConversationState.CLOSING
        elif emotional_state.primary_emotion.value in ["anger", "frustration"]:
            flow.state = ConversationState.INTERRUPTED
        else:
            flow.state = ConversationState.ACTIVE

        # Update current topic if detected
        current_topic = self._extract_topic_from_content(context.message_text)
        if current_topic and current_topic != flow.current_topic:
            flow.current_topic = current_topic

    def _analyze_topic_transition(
        self, flow: ConversationFlow, context: ConversationContext
    ) -> Optional[TopicTransition]:
        """Analyze if a topic transition occurred."""
        if len(flow.turns) < 2:
            return None

        current_topic = self._extract_topic_from_content(context.message_text)
        previous_topic = flow.current_topic

        if current_topic and previous_topic and current_topic != previous_topic:
            # Determine transition type
            transition_type = "natural"
            if any(
                phrase in context.message_text.lower()
                for phrase in ["by the way", "speaking of", "that reminds me"]
            ):
                transition_type = "natural"
            elif any(
                phrase in context.message_text.lower()
                for phrase in ["let's talk about", "i want to discuss", "can we"]
            ):
                transition_type = "user_initiated"

            return TopicTransition(
                from_topic=previous_topic,
                to_topic=current_topic,
                transition_type=transition_type,
                timestamp=datetime.utcnow(),
            )

        return None

    def _calculate_flow_metrics(self, flow: ConversationFlow) -> None:
        """Calculate flow metrics like coherence and engagement."""
        if len(flow.turns) < 2:
            return

        # Calculate coherence score based on topic consistency
        coherence_factors = []

        # Topic coherence
        topics = [turn.topic for turn in flow.turns if turn.topic]
        if topics:
            unique_topics = len(set(topics))
            topic_coherence = 1.0 - (unique_topics - 1) / max(1, len(topics))
            coherence_factors.append(topic_coherence)

        # Turn type coherence (questions followed by answers)
        turn_coherence = self._calculate_turn_coherence(flow.turns)
        coherence_factors.append(turn_coherence)

        # Calculate overall coherence
        if coherence_factors:
            flow.coherence_score = sum(coherence_factors) / len(coherence_factors)

        # Calculate engagement level based on emotional intensity and turn length
        engagement_factors = []
        for turn in flow.turns:
            if turn.emotional_state:
                engagement_factors.append(turn.emotional_state.intensity)
            # Longer messages indicate higher engagement
            length_engagement = min(1.0, len(turn.content) / 200.0)
            engagement_factors.append(length_engagement)

        if engagement_factors:
            flow.engagement_level = sum(engagement_factors) / len(engagement_factors)

    def _generate_flow_recommendations(
        self,
        flow: ConversationFlow,
        context: ConversationContext,
        emotional_state: EmotionalState,
        user_preferences: Optional[UserPreferences],
    ) -> FlowManagementResult:
        """Generate flow management recommendations."""
        # Determine recommended action based on flow state
        if flow.state == ConversationState.OPENING:
            action = "engage_warmly"
            response_type = "welcoming"
        elif flow.state == ConversationState.CLOSING:
            action = "close_gracefully"
            response_type = "farewell"
        elif flow.coherence_score < 0.5:
            action = "clarify_topic"
            response_type = "clarification"
        elif flow.engagement_level < 0.3:
            action = "increase_engagement"
            response_type = "engaging"
        else:
            action = "continue_naturally"
            response_type = "conversational"

        # Generate reasoning
        reasoning = self._generate_flow_reasoning(flow, emotional_state)

        # Calculate confidence
        confidence = self._calculate_recommendation_confidence(flow, emotional_state)

        # Generate topic suggestions
        topic_suggestions = self._generate_topic_suggestions(
            flow, user_preferences, emotional_state
        )

        # Generate flow adjustments
        flow_adjustments = self._generate_flow_adjustments(flow)

        return FlowManagementResult(
            recommended_action=action,
            confidence=confidence,
            reasoning=reasoning,
            suggested_response_type=response_type,
            topic_suggestions=topic_suggestions,
            flow_adjustments=flow_adjustments,
        )

    def _plan_assistant_response(
        self, flow: ConversationFlow, result: FlowManagementResult
    ) -> None:
        """Plan assistant response based on flow management."""
        # Create planned assistant turn
        planned_turn = ConversationTurn(
            speaker="assistant",
            content="",  # Will be filled by response generator
            timestamp=datetime.utcnow(),
            turn_type=result.suggested_response_type,
        )

        # Add to flow for planning purposes
        flow.turns.append(planned_turn)

    def _extract_topic_from_content(self, content: str) -> Optional[str]:
        """Extract topic from content using simple keyword matching."""
        # Define topic keywords
        topic_keywords = {
            "technology": ["tech", "computer", "software", "AI", "programming"],
            "health": ["health", "fitness", "exercise", "diet", "wellness"],
            "work": ["work", "job", "career", "business", "professional"],
            "relationships": ["family", "friends", "relationship", "love"],
            "education": ["learn", "study", "education", "course", "school"],
            "entertainment": ["movie", "music", "book", "game", "fun"],
            "travel": ["travel", "trip", "vacation", "journey", "visit"],
            "food": ["food", "cooking", "recipe", "restaurant", "meal"],
        }

        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return topic

        return None

    def _is_closing_signal(self, content: str) -> bool:
        """Check if content contains closing signals."""
        closing_signals = [
            "goodbye",
            "bye",
            "see you",
            "talk later",
            "gotta go",
            "have to go",
            "thanks for",
            "that's all",
        ]

        content_lower = content.lower()
        return any(signal in content_lower for signal in closing_signals)

    def _calculate_turn_coherence(self, turns: List[ConversationTurn]) -> float:
        """Calculate coherence based on turn types."""
        if len(turns) < 2:
            return 1.0

        coherent_pairs = 0
        total_pairs = len(turns) - 1

        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]

            # Questions should be followed by answers
            if (
                current_turn.turn_type == "question"
                and next_turn.speaker != current_turn.speaker
            ):
                coherent_pairs += 1
            # Normal turns should flow naturally
            elif current_turn.turn_type == "normal":
                coherent_pairs += 1

        return coherent_pairs / total_pairs if total_pairs > 0 else 1.0

    def _generate_flow_reasoning(
        self, flow: ConversationFlow, emotional_state: EmotionalState
    ) -> str:
        """Generate reasoning for flow recommendations."""
        reasoning_parts = []

        reasoning_parts.append(f"Conversation state: {flow.state.value}")
        reasoning_parts.append(f"Coherence score: {flow.coherence_score:.2f}")
        reasoning_parts.append(f"Engagement level: {flow.engagement_level:.2f}")

        if emotional_state.intensity > 0.7:
            reasoning_parts.append("High emotional intensity detected")

        if len(flow.turns) > 10:
            reasoning_parts.append("Extended conversation detected")

        return "; ".join(reasoning_parts)

    def _calculate_recommendation_confidence(
        self, flow: ConversationFlow, emotional_state: EmotionalState
    ) -> float:
        """Calculate confidence in flow recommendations."""
        confidence_factors = []

        # Flow history factor
        history_confidence = min(1.0, len(flow.turns) / 5.0)
        confidence_factors.append(history_confidence)

        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)

        # Coherence factor
        confidence_factors.append(flow.coherence_score)

        return sum(confidence_factors) / len(confidence_factors)

    def _generate_topic_suggestions(
        self,
        flow: ConversationFlow,
        user_preferences: Optional[UserPreferences],
        emotional_state: EmotionalState,
    ) -> List[ConversationTopic]:
        """Generate topic suggestions for conversation."""
        suggestions = []

        # Suggest based on user interests
        if user_preferences and user_preferences.topics_of_interest:
            for topic in user_preferences.topics_of_interest[:3]:
                suggestions.append(
                    ConversationTopic(
                        topic=topic, relevance_score=0.8, category="user_interest"
                    )
                )

        # Suggest based on emotional state
        if emotional_state.primary_emotion.value == "joy":
            suggestions.append(
                ConversationTopic(
                    topic="positive experiences",
                    relevance_score=0.7,
                    category="emotional_alignment",
                )
            )
        elif emotional_state.primary_emotion.value in ["sadness", "fear"]:
            suggestions.append(
                ConversationTopic(
                    topic="supportive discussion",
                    relevance_score=0.9,
                    category="emotional_support",
                )
            )

        # Suggest based on conversation history
        if flow.current_topic:
            related_topics = self._get_related_topics(flow.current_topic)
            for topic in related_topics[:2]:
                suggestions.append(
                    ConversationTopic(
                        topic=topic, relevance_score=0.6, category="related_topic"
                    )
                )

        return suggestions

    def _generate_flow_adjustments(self, flow: ConversationFlow) -> Dict[str, Any]:
        """Generate flow adjustments based on current state."""
        adjustments = {}

        if flow.coherence_score < 0.5:
            adjustments["increase_coherence"] = True
            adjustments["clarify_topic"] = True

        if flow.engagement_level < 0.3:
            adjustments["increase_engagement"] = True
            adjustments["ask_engaging_question"] = True

        if len(flow.turns) > 20:
            adjustments["consider_topic_change"] = True

        return adjustments

    def _get_related_topics(self, current_topic: str) -> List[str]:
        """Get topics related to current topic."""
        topic_relations = {
            "technology": ["programming", "AI", "software", "innovation"],
            "health": ["fitness", "nutrition", "wellness", "mental health"],
            "work": ["career", "productivity", "skills", "goals"],
            "relationships": ["communication", "trust", "support", "growth"],
            "education": ["skills", "knowledge", "growth", "learning"],
        }

        return topic_relations.get(current_topic, [])

    def _archive_conversation_flow(
        self, conversation_id: str, flow: ConversationFlow
    ) -> None:
        """Archive completed conversation flow."""
        if flow.user_id not in self.flow_history:
            self.flow_history[flow.user_id] = []

        self.flow_history[flow.user_id].append(flow)

        # Keep only recent flows (last 50 per user)
        if len(self.flow_history[flow.user_id]) > 50:
            self.flow_history[flow.user_id] = self.flow_history[flow.user_id][-50:]

    def _calculate_average_turns(self, flows: List[ConversationFlow]) -> float:
        """Calculate average turns per conversation."""
        if not flows:
            return 0.0

        total_turns = sum(len(flow.turns) for flow in flows)
        return total_turns / len(flows)

    def _calculate_average_coherence(self, flows: List[ConversationFlow]) -> float:
        """Calculate average coherence score."""
        if not flows:
            return 0.0

        total_coherence = sum(flow.coherence_score for flow in flows)
        return total_coherence / len(flows)

    def _calculate_average_engagement(self, flows: List[ConversationFlow]) -> float:
        """Calculate average engagement level."""
        if not flows:
            return 0.0

        total_engagement = sum(flow.engagement_level for flow in flows)
        return total_engagement / len(flows)

    def _calculate_transition_success_rate(self, flows: List[ConversationFlow]) -> float:
        """Calculate topic transition success rate."""
        total_transitions = 0
        successful_transitions = 0

        for flow in flows:
            total_transitions += len(flow.topic_transitions)
            successful_transitions += sum(
                1 for t in flow.topic_transitions if t.success_score > 0.5
            )

        return (
            successful_transitions / total_transitions if total_transitions > 0 else 0.0
        )

    def _get_most_common_topics(self, flows: List[ConversationFlow]) -> List[str]:
        """Get most common topics from flows."""
        topic_counts = {}

        for flow in flows:
            for turn in flow.turns:
                if turn.topic:
                    topic_counts[turn.topic] = topic_counts.get(turn.topic, 0) + 1

        # Return top 5 topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]

    def _analyze_conversation_patterns(
        self, flows: List[ConversationFlow]
    ) -> Dict[str, Any]:
        """Analyze conversation patterns."""
        patterns = {
            "average_conversation_duration": self._calculate_average_duration(flows),
            "most_common_opening_state": self._get_most_common_opening_state(flows),
            "most_common_closing_state": self._get_most_common_closing_state(flows),
            "topic_diversity": self._calculate_topic_diversity(flows),
        }

        return patterns

    def _analyze_flow_directions(self, flows: List[ConversationFlow]) -> Dict[str, int]:
        """Analyze flow direction distribution."""
        direction_counts = {}

        for flow in flows:
            direction = flow.flow_direction.value
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        return direction_counts

    def _calculate_average_duration(self, flows: List[ConversationFlow]) -> float:
        """Calculate average conversation duration in minutes."""
        if not flows:
            return 0.0

        total_duration = 0
        for flow in flows:
            if flow.turns:
                duration = (flow.last_activity - flow.started_at).total_seconds() / 60
                total_duration += duration

        return total_duration / len(flows)

    def _get_most_common_opening_state(self, flows: List[ConversationFlow]) -> str:
        """Get most common opening state."""
        # For simplicity, assume all start with OPENING
        return ConversationState.OPENING.value

    def _get_most_common_closing_state(self, flows: List[ConversationFlow]) -> str:
        """Get most common closing state."""
        # For simplicity, assume all end with CLOSING
        return ConversationState.CLOSING.value

    def _calculate_topic_diversity(self, flows: List[ConversationFlow]) -> float:
        """Calculate topic diversity across conversations."""
        all_topics = set()
        total_topic_mentions = 0

        for flow in flows:
            for turn in flow.turns:
                if turn.topic:
                    all_topics.add(turn.topic)
                    total_topic_mentions += 1

        return (
            len(all_topics) / total_topic_mentions if total_topic_mentions > 0 else 0.0
        )


# Singleton instance
_conversation_flow_manager_instance = None
_conversation_flow_manager_lock = threading.Lock()


def get_conversation_flow_manager() -> ConversationFlowManager:
    """
    Get singleton conversation flow manager instance.

    Returns:
        Shared ConversationFlowManager instance
    """
    global _conversation_flow_manager_instance

    if _conversation_flow_manager_instance is None:
        with _conversation_flow_manager_lock:
            if _conversation_flow_manager_instance is None:
                _conversation_flow_manager_instance = ConversationFlowManager()

    return _conversation_flow_manager_instance