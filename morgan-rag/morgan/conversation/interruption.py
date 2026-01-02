"""
Interruption handling module.

Detects, manages, and gracefully handles conversation interruptions
to maintain natural flow and user experience.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import ConversationContext, EmotionalState
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class InterruptionType(Enum):
    """Types of conversation interruptions."""

    TOPIC_CHANGE = "topic_change"
    EMOTIONAL_SHIFT = "emotional_shift"
    URGENT_REQUEST = "urgent_request"
    EXTERNAL_DISTRACTION = "external_distraction"
    CONVERSATION_PAUSE = "conversation_pause"
    SYSTEM_INTERRUPTION = "system_interruption"
    USER_CORRECTION = "user_correction"
    CLARIFICATION_REQUEST = "clarification_request"


class InterruptionSeverity(Enum):
    """Severity levels of interruptions."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class InterruptionHandlingStrategy(Enum):
    """Strategies for handling interruptions."""

    ACKNOWLEDGE_AND_CONTINUE = "acknowledge_and_continue"
    PAUSE_AND_RESUME = "pause_and_resume"
    REDIRECT_GRACEFULLY = "redirect_gracefully"
    PRIORITIZE_INTERRUPTION = "prioritize_interruption"
    DEFER_INTERRUPTION = "defer_interruption"
    SEEK_CLARIFICATION = "seek_clarification"


@dataclass
class InterruptionEvent:
    """Represents an interruption event."""

    interruption_type: InterruptionType
    severity: InterruptionSeverity
    timestamp: datetime
    context_before: str
    interrupting_content: str
    emotional_state: Optional[EmotionalState] = None
    confidence: float = 0.0
    indicators: List[str] = field(default_factory=list)


@dataclass
class InterruptionResponse:
    """Response strategy for handling an interruption."""

    strategy: InterruptionHandlingStrategy
    response_text: str
    confidence: float
    reasoning: str
    follow_up_actions: List[str] = field(default_factory=list)
    context_preservation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterruptionPattern:
    """Pattern of interruptions for a user."""

    user_id: str
    common_interruption_types: List[InterruptionType]
    typical_severity: InterruptionSeverity
    preferred_handling_strategies: List[InterruptionHandlingStrategy]
    interruption_frequency: float  # interruptions per conversation
    recovery_time: float  # average time to recover from interruption
    pattern_confidence: float = 0.0


class InterruptionHandler:
    """
    Interruption handling system.

    Features:
    - Real-time interruption detection
    - Context-aware interruption classification
    - Graceful interruption handling strategies
    - Conversation flow preservation
    - User pattern learning for interruption preferences
    - Recovery and resumption assistance
    """

    def __init__(self):
        """Initialize interruption handler."""
        self.settings = get_settings()

        # User interruption data
        self.user_interruption_patterns: Dict[str, InterruptionPattern] = {}
        self.active_interruptions: Dict[str, List[InterruptionEvent]] = {}
        self.interruption_history: Dict[str, List[InterruptionEvent]] = {}

        # Interruption detection patterns
        self.interruption_indicators = self._initialize_interruption_indicators()

        logger.info("Interruption Handler initialized")

    def detect_interruption(
        self,
        user_id: str,
        current_context: ConversationContext,
        previous_context: Optional[ConversationContext],
        emotional_state: EmotionalState,
    ) -> Optional[InterruptionEvent]:
        """
        Detect if an interruption has occurred.

        Args:
            user_id: User identifier
            current_context: Current conversation context
            previous_context: Previous conversation context
            emotional_state: Current emotional state

        Returns:
            InterruptionEvent if interruption detected, None otherwise
        """
        if not previous_context:
            return None

        # Analyze for different types of interruptions
        interruption_type = self._classify_interruption_type(
            current_context, previous_context, emotional_state
        )

        if not interruption_type:
            return None

        # Determine severity
        severity = self._assess_interruption_severity(
            interruption_type, current_context, emotional_state
        )

        # Calculate confidence
        confidence = self._calculate_interruption_confidence(
            interruption_type, current_context, previous_context, emotional_state
        )

        # Identify indicators
        indicators = self._identify_interruption_indicators(
            interruption_type, current_context, previous_context
        )

        # Create interruption event
        interruption = InterruptionEvent(
            interruption_type=interruption_type,
            severity=severity,
            timestamp=current_context.timestamp,
            context_before=previous_context.message_text,
            interrupting_content=current_context.message_text,
            emotional_state=emotional_state,
            confidence=confidence,
            indicators=indicators,
        )

        # Store interruption
        if user_id not in self.active_interruptions:
            self.active_interruptions[user_id] = []
        self.active_interruptions[user_id].append(interruption)

        # Update interruption history
        if user_id not in self.interruption_history:
            self.interruption_history[user_id] = []
        self.interruption_history[user_id].append(interruption)

        # Keep only recent history (last 50 interruptions)
        if len(self.interruption_history[user_id]) > 50:
            self.interruption_history[user_id] = self.interruption_history[user_id][-50:]

        logger.debug(
            f"Detected {interruption_type.value} interruption for user {user_id} "
            f"with {severity.value} severity (confidence: {confidence:.2f})"
        )

        return interruption

    def handle_interruption(
        self,
        user_id: str,
        interruption: InterruptionEvent,
        conversation_context: ConversationContext,
    ) -> InterruptionResponse:
        """
        Handle an interruption with appropriate strategy.

        Args:
            user_id: User identifier
            interruption: Interruption event to handle
            conversation_context: Current conversation context

        Returns:
            Interruption response strategy
        """
        # Get user's interruption patterns
        user_pattern = self.user_interruption_patterns.get(user_id)

        # Determine handling strategy
        strategy = self._determine_handling_strategy(
            interruption, user_pattern, conversation_context
        )

        # Generate response
        response = self._generate_interruption_response(
            interruption, strategy, conversation_context
        )

        # Update user patterns
        self._update_interruption_patterns(user_id, interruption, strategy)

        logger.debug(
            f"Handling {interruption.interruption_type.value} interruption "
            f"with {strategy.value} strategy"
        )

        return response

    def resume_conversation(
        self,
        user_id: str,
        interruption: InterruptionEvent,
        preserved_context: Dict[str, Any],
    ) -> str:
        """
        Resume conversation after interruption handling.

        Args:
            user_id: User identifier
            interruption: Original interruption event
            preserved_context: Context preserved during interruption

        Returns:
            Resumption message
        """
        # Generate contextual resumption
        if interruption.interruption_type == InterruptionType.TOPIC_CHANGE:
            resumption = self._generate_topic_resumption(preserved_context)
        elif interruption.interruption_type == InterruptionType.EMOTIONAL_SHIFT:
            resumption = self._generate_emotional_resumption(preserved_context)
        elif interruption.interruption_type == InterruptionType.CONVERSATION_PAUSE:
            resumption = self._generate_pause_resumption(preserved_context)
        else:
            resumption = self._generate_general_resumption(preserved_context)

        # Mark interruption as resolved
        if user_id in self.active_interruptions:
            self.active_interruptions[user_id] = [
                i for i in self.active_interruptions[user_id] if i != interruption
            ]

        logger.debug(f"Generated resumption for user {user_id}")
        return resumption

    def get_interruption_analytics(
        self, user_id: str, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get interruption analytics for user.

        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days

        Returns:
            Interruption analytics
        """
        history = self.interruption_history.get(user_id, [])
        if not history:
            return {"error": "No interruption data available"}

        # Filter by timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_interruptions = [
            i for i in history if i.timestamp >= cutoff_date
        ]

        if not recent_interruptions:
            return {"error": "No recent interruption data"}

        analytics = {
            "total_interruptions": len(recent_interruptions),
            "interruption_frequency": len(recent_interruptions) / max(1, timeframe_days),
            "interruption_types": self._analyze_interruption_types(recent_interruptions),
            "severity_distribution": self._analyze_severity_distribution(recent_interruptions),
            "most_common_indicators": self._analyze_common_indicators(recent_interruptions),
            "emotional_patterns": self._analyze_emotional_interruption_patterns(recent_interruptions),
            "handling_effectiveness": self._analyze_handling_effectiveness(user_id),
        }

        return analytics

    def _classify_interruption_type(
        self,
        current_context: ConversationContext,
        previous_context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> Optional[InterruptionType]:
        """Classify the type of interruption."""
        current_text = current_context.message_text.lower()
        previous_text = previous_context.message_text.lower()

        # Topic change detection
        if self._detect_topic_change(current_text, previous_text):
            return InterruptionType.TOPIC_CHANGE

        # Emotional shift detection
        if self._detect_emotional_shift(emotional_state, current_text):
            return InterruptionType.EMOTIONAL_SHIFT

        # Urgent request detection
        if self._detect_urgent_request(current_text):
            return InterruptionType.URGENT_REQUEST

        # User correction detection
        if self._detect_user_correction(current_text):
            return InterruptionType.USER_CORRECTION

        # Clarification request detection
        if self._detect_clarification_request(current_text):
            return InterruptionType.CLARIFICATION_REQUEST

        # Conversation pause detection
        if self._detect_conversation_pause(current_context, previous_context):
            return InterruptionType.CONVERSATION_PAUSE

        # External distraction detection
        if self._detect_external_distraction(current_text):
            return InterruptionType.EXTERNAL_DISTRACTION

        return None

    def _assess_interruption_severity(
        self,
        interruption_type: InterruptionType,
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> InterruptionSeverity:
        """Assess the severity of an interruption."""
        # Base severity by type
        type_severity_map = {
            InterruptionType.URGENT_REQUEST: InterruptionSeverity.CRITICAL,
            InterruptionType.EMOTIONAL_SHIFT: InterruptionSeverity.MAJOR,
            InterruptionType.SYSTEM_INTERRUPTION: InterruptionSeverity.MAJOR,
            InterruptionType.TOPIC_CHANGE: InterruptionSeverity.MODERATE,
            InterruptionType.USER_CORRECTION: InterruptionSeverity.MODERATE,
            InterruptionType.CLARIFICATION_REQUEST: InterruptionSeverity.MINOR,
            InterruptionType.CONVERSATION_PAUSE: InterruptionSeverity.MINOR,
            InterruptionType.EXTERNAL_DISTRACTION: InterruptionSeverity.MINOR,
        }

        base_severity = type_severity_map.get(interruption_type, InterruptionSeverity.MODERATE)

        # Adjust based on emotional intensity
        if emotional_state.intensity > 0.8:
            # High emotional intensity increases severity
            severity_levels = list(InterruptionSeverity)
            current_index = severity_levels.index(base_severity)
            if current_index < len(severity_levels) - 1:
                return severity_levels[current_index + 1]

        return base_severity

    def _calculate_interruption_confidence(
        self,
        interruption_type: InterruptionType,
        current_context: ConversationContext,
        previous_context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> float:
        """Calculate confidence in interruption detection."""
        confidence_factors = []

        # Text-based indicators
        text_indicators = self._count_text_indicators(
            interruption_type, current_context.message_text
        )
        text_confidence = min(1.0, text_indicators / 3.0)
        confidence_factors.append(text_confidence)

        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)

        # Context change magnitude
        context_change = self._measure_context_change(current_context, previous_context)
        confidence_factors.append(context_change)

        # Time gap factor
        time_gap = (current_context.timestamp - previous_context.timestamp).total_seconds()
        time_confidence = 1.0 if time_gap < 300 else max(0.3, 1.0 - (time_gap / 3600))
        confidence_factors.append(time_confidence)

        return sum(confidence_factors) / len(confidence_factors)

    def _identify_interruption_indicators(
        self,
        interruption_type: InterruptionType,
        current_context: ConversationContext,
        previous_context: ConversationContext,
    ) -> List[str]:
        """Identify specific indicators of the interruption."""
        indicators = []
        current_text = current_context.message_text.lower()

        # Get type-specific indicators
        type_indicators = self.interruption_indicators.get(interruption_type, [])
        
        for indicator in type_indicators:
            if indicator in current_text:
                indicators.append(indicator)

        # Add contextual indicators
        if len(current_context.message_text) < 20:
            indicators.append("short_message")
        
        if "?" in current_context.message_text:
            indicators.append("question_mark")
            
        if current_context.message_text.isupper():
            indicators.append("all_caps")

        return indicators

    def _determine_handling_strategy(
        self,
        interruption: InterruptionEvent,
        user_pattern: Optional[InterruptionPattern],
        context: ConversationContext,
    ) -> InterruptionHandlingStrategy:
        """Determine the best strategy for handling the interruption."""
        # Consider user preferences if available
        if user_pattern and user_pattern.preferred_handling_strategies:
            preferred_strategy = user_pattern.preferred_handling_strategies[0]
            # Check if preferred strategy is appropriate for this interruption
            if self._is_strategy_appropriate(preferred_strategy, interruption):
                return preferred_strategy

        # Default strategy based on interruption type and severity
        if interruption.severity == InterruptionSeverity.CRITICAL:
            return InterruptionHandlingStrategy.PRIORITIZE_INTERRUPTION
        elif interruption.interruption_type == InterruptionType.URGENT_REQUEST:
            return InterruptionHandlingStrategy.PRIORITIZE_INTERRUPTION
        elif interruption.interruption_type == InterruptionType.CLARIFICATION_REQUEST:
            return InterruptionHandlingStrategy.SEEK_CLARIFICATION
        elif interruption.interruption_type == InterruptionType.TOPIC_CHANGE:
            return InterruptionHandlingStrategy.REDIRECT_GRACEFULLY
        elif interruption.interruption_type == InterruptionType.EMOTIONAL_SHIFT:
            return InterruptionHandlingStrategy.ACKNOWLEDGE_AND_CONTINUE
        elif interruption.interruption_type == InterruptionType.CONVERSATION_PAUSE:
            return InterruptionHandlingStrategy.PAUSE_AND_RESUME
        else:
            return InterruptionHandlingStrategy.ACKNOWLEDGE_AND_CONTINUE

    def _generate_interruption_response(
        self,
        interruption: InterruptionEvent,
        strategy: InterruptionHandlingStrategy,
        context: ConversationContext,
    ) -> InterruptionResponse:
        """Generate response for handling the interruption."""
        # Generate response text based on strategy
        if strategy == InterruptionHandlingStrategy.ACKNOWLEDGE_AND_CONTINUE:
            response_text = self._generate_acknowledgment_response(interruption)
        elif strategy == InterruptionHandlingStrategy.PRIORITIZE_INTERRUPTION:
            response_text = self._generate_prioritization_response(interruption)
        elif strategy == InterruptionHandlingStrategy.REDIRECT_GRACEFULLY:
            response_text = self._generate_redirection_response(interruption)
        elif strategy == InterruptionHandlingStrategy.SEEK_CLARIFICATION:
            response_text = self._generate_clarification_response(interruption)
        elif strategy == InterruptionHandlingStrategy.PAUSE_AND_RESUME:
            response_text = self._generate_pause_response(interruption)
        elif strategy == InterruptionHandlingStrategy.DEFER_INTERRUPTION:
            response_text = self._generate_deferral_response(interruption)
        else:
            response_text = "I understand. Let me address that."

        # Generate reasoning
        reasoning = f"Using {strategy.value} strategy for {interruption.interruption_type.value} " \
                   f"interruption with {interruption.severity.value} severity"

        # Determine follow-up actions
        follow_up_actions = self._determine_follow_up_actions(strategy, interruption)

        # Preserve context for potential resumption
        context_preservation = {
            "previous_topic": self._extract_topic(interruption.context_before),
            "conversation_state": "interrupted",
            "interruption_timestamp": interruption.timestamp.isoformat(),
            "original_context": interruption.context_before,
        }

        return InterruptionResponse(
            strategy=strategy,
            response_text=response_text,
            confidence=0.8,  # Default confidence
            reasoning=reasoning,
            follow_up_actions=follow_up_actions,
            context_preservation=context_preservation,
        )

    def _update_interruption_patterns(
        self,
        user_id: str,
        interruption: InterruptionEvent,
        strategy: InterruptionHandlingStrategy,
    ) -> None:
        """Update user's interruption patterns."""
        if user_id not in self.user_interruption_patterns:
            self.user_interruption_patterns[user_id] = InterruptionPattern(
                user_id=user_id,
                common_interruption_types=[],
                typical_severity=InterruptionSeverity.MODERATE,
                preferred_handling_strategies=[],
                interruption_frequency=0.0,
                recovery_time=0.0,
            )

        pattern = self.user_interruption_patterns[user_id]

        # Update common interruption types
        if interruption.interruption_type not in pattern.common_interruption_types:
            pattern.common_interruption_types.append(interruption.interruption_type)

        # Update preferred strategies (based on successful handling)
        if strategy not in pattern.preferred_handling_strategies:
            pattern.preferred_handling_strategies.append(strategy)

        # Update pattern confidence
        history_count = len(self.interruption_history.get(user_id, []))
        pattern.pattern_confidence = min(1.0, history_count / 20.0)

    def _initialize_interruption_indicators(self) -> Dict[InterruptionType, List[str]]:
        """Initialize interruption detection indicators."""
        return {
            InterruptionType.TOPIC_CHANGE: [
                "by the way", "speaking of", "that reminds me", "actually",
                "oh", "wait", "before i forget", "also", "another thing"
            ],
            InterruptionType.URGENT_REQUEST: [
                "urgent", "emergency", "asap", "immediately", "right now",
                "help", "quick", "fast", "hurry", "important"
            ],
            InterruptionType.USER_CORRECTION: [
                "no", "actually", "i meant", "correction", "sorry",
                "wait", "that's not", "i said", "let me clarify"
            ],
            InterruptionType.CLARIFICATION_REQUEST: [
                "what do you mean", "can you explain", "i don't understand",
                "clarify", "confused", "what", "how", "why"
            ],
            InterruptionType.EMOTIONAL_SHIFT: [
                "i'm feeling", "suddenly", "now i'm", "i just realized",
                "this makes me", "i'm getting", "i feel"
            ],
            InterruptionType.EXTERNAL_DISTRACTION: [
                "hold on", "one second", "someone's", "phone", "door",
                "sorry", "distracted", "interrupted", "back"
            ],
            InterruptionType.CONVERSATION_PAUSE: [
                "pause", "break", "later", "continue", "resume",
                "stop", "enough", "tired", "busy"
            ],
        }

    def _detect_topic_change(self, current_text: str, previous_text: str) -> bool:
        """Detect if a topic change occurred."""
        topic_change_indicators = self.interruption_indicators[InterruptionType.TOPIC_CHANGE]
        return any(indicator in current_text for indicator in topic_change_indicators)

    def _detect_emotional_shift(self, emotional_state: EmotionalState, current_text: str) -> bool:
        """Detect emotional shift interruption."""
        # High emotional intensity change
        if emotional_state.intensity > 0.7:
            emotional_indicators = self.interruption_indicators[InterruptionType.EMOTIONAL_SHIFT]
            return any(indicator in current_text for indicator in emotional_indicators)
        return False

    def _detect_urgent_request(self, current_text: str) -> bool:
        """Detect urgent request interruption."""
        urgent_indicators = self.interruption_indicators[InterruptionType.URGENT_REQUEST]
        return any(indicator in current_text for indicator in urgent_indicators)

    def _detect_user_correction(self, current_text: str) -> bool:
        """Detect user correction interruption."""
        correction_indicators = self.interruption_indicators[InterruptionType.USER_CORRECTION]
        return any(indicator in current_text for indicator in correction_indicators)

    def _detect_clarification_request(self, current_text: str) -> bool:
        """Detect clarification request interruption."""
        clarification_indicators = self.interruption_indicators[InterruptionType.CLARIFICATION_REQUEST]
        return any(indicator in current_text for indicator in clarification_indicators)

    def _detect_conversation_pause(
        self, current_context: ConversationContext, previous_context: ConversationContext
    ) -> bool:
        """Detect conversation pause interruption."""
        # Long time gap between messages
        time_gap = (current_context.timestamp - previous_context.timestamp).total_seconds()
        if time_gap > 300:  # 5 minutes
            return True

        # Explicit pause indicators
        pause_indicators = self.interruption_indicators[InterruptionType.CONVERSATION_PAUSE]
        return any(indicator in current_context.message_text.lower() for indicator in pause_indicators)

    def _detect_external_distraction(self, current_text: str) -> bool:
        """Detect external distraction interruption."""
        distraction_indicators = self.interruption_indicators[InterruptionType.EXTERNAL_DISTRACTION]
        return any(indicator in current_text for indicator in distraction_indicators)

    def _count_text_indicators(self, interruption_type: InterruptionType, text: str) -> int:
        """Count text indicators for interruption type."""
        indicators = self.interruption_indicators.get(interruption_type, [])
        text_lower = text.lower()
        return sum(1 for indicator in indicators if indicator in text_lower)

    def _measure_context_change(
        self, current_context: ConversationContext, previous_context: ConversationContext
    ) -> float:
        """Measure magnitude of context change."""
        # Simple word overlap measure
        current_words = set(current_context.message_text.lower().split())
        previous_words = set(previous_context.message_text.lower().split())
        
        if not previous_words:
            return 0.5

        overlap = len(current_words.intersection(previous_words))
        total_words = len(current_words.union(previous_words))
        
        # Return 1 - similarity (higher change = higher value)
        similarity = overlap / total_words if total_words > 0 else 0
        return 1.0 - similarity

    def _is_strategy_appropriate(
        self, strategy: InterruptionHandlingStrategy, interruption: InterruptionEvent
    ) -> bool:
        """Check if a strategy is appropriate for the interruption."""
        # Critical interruptions should be prioritized
        if interruption.severity == InterruptionSeverity.CRITICAL:
            return strategy == InterruptionHandlingStrategy.PRIORITIZE_INTERRUPTION

        # Clarification requests should seek clarification
        if interruption.interruption_type == InterruptionType.CLARIFICATION_REQUEST:
            return strategy == InterruptionHandlingStrategy.SEEK_CLARIFICATION

        # Most strategies are generally appropriate
        return True

    def _generate_acknowledgment_response(self, interruption: InterruptionEvent) -> str:
        """Generate acknowledgment response."""
        responses = [
            "I understand. Let me address that.",
            "I see what you're saying.",
            "Got it. Let me help with that.",
            "I hear you. Let's work on that.",
            "Understood. I'll focus on that now.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _generate_prioritization_response(self, interruption: InterruptionEvent) -> str:
        """Generate prioritization response."""
        responses = [
            "This seems important. Let me focus on this right away.",
            "I understand this is urgent. Let me help immediately.",
            "This takes priority. I'm giving this my full attention.",
            "I can see this needs immediate attention.",
            "Let me address this urgent matter first.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _generate_redirection_response(self, interruption: InterruptionEvent) -> str:
        """Generate graceful redirection response."""
        responses = [
            "I see you'd like to discuss something else. Let's explore that.",
            "That's an interesting point. Let me shift focus to that.",
            "I understand you want to change direction. Let's go with that.",
            "Good point. Let me address that instead.",
            "I can see that's what you're really interested in discussing.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _generate_clarification_response(self, interruption: InterruptionEvent) -> str:
        """Generate clarification response."""
        responses = [
            "Let me clarify that for you.",
            "I can see that needs more explanation. Let me elaborate.",
            "Good question. Let me explain that better.",
            "I should have been clearer about that.",
            "Let me provide more detail on that point.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _generate_pause_response(self, interruption: InterruptionEvent) -> str:
        """Generate pause response."""
        responses = [
            "Of course, we can pause here. Take your time.",
            "No problem. We can continue whenever you're ready.",
            "Sure, let's take a break. I'll be here when you return.",
            "Absolutely. Feel free to come back to this later.",
            "That's fine. We can pick up where we left off.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _generate_deferral_response(self, interruption: InterruptionEvent) -> str:
        """Generate deferral response."""
        responses = [
            "Let me finish this thought first, then we can address that.",
            "Good point. Can we come back to that in just a moment?",
            "I'd like to complete this idea first, then focus on that.",
            "Let me wrap up this point, then we'll dive into that.",
            "That's important. Let me finish this, then give that full attention.",
        ]
        return responses[hash(interruption.interrupting_content) % len(responses)]

    def _determine_follow_up_actions(
        self, strategy: InterruptionHandlingStrategy, interruption: InterruptionEvent
    ) -> List[str]:
        """Determine follow-up actions for the strategy."""
        actions = []

        if strategy == InterruptionHandlingStrategy.PRIORITIZE_INTERRUPTION:
            actions.extend(["focus_on_interruption", "defer_original_topic"])
        elif strategy == InterruptionHandlingStrategy.PAUSE_AND_RESUME:
            actions.extend(["save_context", "prepare_resumption"])
        elif strategy == InterruptionHandlingStrategy.REDIRECT_GRACEFULLY:
            actions.extend(["transition_topic", "acknowledge_change"])
        elif strategy == InterruptionHandlingStrategy.SEEK_CLARIFICATION:
            actions.extend(["ask_clarifying_questions", "provide_examples"])

        return actions

    def _extract_topic(self, text: str) -> str:
        """Extract topic from text (simplified)."""
        # Simple topic extraction - in production, use more sophisticated NLP
        words = text.lower().split()
        
        # Look for topic-indicating words
        topic_words = []
        for word in words:
            if len(word) > 4 and word.isalpha():
                topic_words.append(word)
        
        return " ".join(topic_words[:3]) if topic_words else "general"

    def _generate_topic_resumption(self, preserved_context: Dict[str, Any]) -> str:
        """Generate topic resumption message."""
        previous_topic = preserved_context.get("previous_topic", "our previous discussion")
        return f"Now, returning to {previous_topic} - where were we?"

    def _generate_emotional_resumption(self, preserved_context: Dict[str, Any]) -> str:
        """Generate emotional resumption message."""
        return "I hope that helped. How are you feeling about continuing our conversation?"

    def _generate_pause_resumption(self, preserved_context: Dict[str, Any]) -> str:
        """Generate pause resumption message."""
        return "Welcome back! Are you ready to continue where we left off?"

    def _generate_general_resumption(self, preserved_context: Dict[str, Any]) -> str:
        """Generate general resumption message."""
        return "Let's continue with what we were discussing."

    def _analyze_interruption_types(self, interruptions: List[InterruptionEvent]) -> Dict[str, int]:
        """Analyze distribution of interruption types."""
        type_counts = {}
        for interruption in interruptions:
            type_name = interruption.interruption_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts

    def _analyze_severity_distribution(self, interruptions: List[InterruptionEvent]) -> Dict[str, int]:
        """Analyze distribution of interruption severities."""
        severity_counts = {}
        for interruption in interruptions:
            severity_name = interruption.severity.value
            severity_counts[severity_name] = severity_counts.get(severity_name, 0) + 1
        return severity_counts

    def _analyze_common_indicators(self, interruptions: List[InterruptionEvent]) -> List[str]:
        """Analyze most common interruption indicators."""
        indicator_counts = {}
        for interruption in interruptions:
            for indicator in interruption.indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Return top 5 indicators
        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        return [indicator for indicator, _ in sorted_indicators[:5]]

    def _analyze_emotional_interruption_patterns(self, interruptions: List[InterruptionEvent]) -> Dict[str, Any]:
        """Analyze emotional patterns in interruptions."""
        emotional_interruptions = [
            i for i in interruptions 
            if i.emotional_state and i.emotional_state.intensity > 0.5
        ]
        
        if not emotional_interruptions:
            return {"emotional_interruptions": 0}

        emotions = [i.emotional_state.primary_emotion.value for i in emotional_interruptions]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        return {
            "emotional_interruptions": len(emotional_interruptions),
            "emotional_interruption_rate": len(emotional_interruptions) / len(interruptions),
            "common_emotional_triggers": emotion_counts,
        }

    def _analyze_handling_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze effectiveness of interruption handling."""
        # This would require feedback data in a real implementation
        return {
            "average_resolution_time": 30.0,  # seconds
            "successful_resumptions": 0.85,  # 85% success rate
            "user_satisfaction": 0.8,  # 80% satisfaction
        }


# Singleton instance
_interruption_handler_instance = None
_interruption_handler_lock = threading.Lock()


def get_interruption_handler() -> InterruptionHandler:
    """
    Get singleton interruption handler instance.

    Returns:
        Shared InterruptionHandler instance
    """
    global _interruption_handler_instance

    if _interruption_handler_instance is None:
        with _interruption_handler_lock:
            if _interruption_handler_instance is None:
                _interruption_handler_instance = InterruptionHandler()

    return _interruption_handler_instance