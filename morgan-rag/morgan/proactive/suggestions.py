"""
Suggestion Engine for Proactive Assistance.

Generates contextual suggestions based on:
- User context and patterns
- Conversation history
- Time and activity patterns
- Goals and interests
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.llm_service import LLMService, get_llm_service
from morgan.proactive.monitor import ContextMonitor, UserContext, get_context_monitor
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class SuggestionType(str, Enum):
    """Types of suggestions."""

    TOPIC_EXPLORATION = "topic_exploration"  # Explore related topics
    TASK_REMINDER = "task_reminder"  # Reminder about pending tasks
    KNOWLEDGE_SHARE = "knowledge_share"  # Proactive knowledge sharing
    CHECK_IN = "check_in"  # Emotional/progress check-in
    TIP = "tip"  # Helpful tip based on context
    FOLLOW_UP = "follow_up"  # Follow up on previous conversation
    GOAL_PROGRESS = "goal_progress"  # Update on goal progress
    LEARNING_OPPORTUNITY = "learning_opportunity"  # Learning suggestion


class SuggestionPriority(str, Enum):
    """Priority of suggestions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    """A proactive suggestion."""

    suggestion_id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    content: str
    user_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    action: Optional[str] = None  # Suggested action
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    shown: bool = False
    dismissed: bool = False
    acted_upon: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.suggestion_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "content": self.content,
            "action": self.action,
            "context": self.context,
        }


class SuggestionEngine:
    """
    Generates proactive suggestions for users.

    Capabilities:
    - Context-aware suggestion generation
    - Priority-based suggestion ordering
    - Suggestion tracking and analytics
    - Integration with context monitor

    Example:
        >>> engine = SuggestionEngine()
        >>>
        >>> # Generate suggestions for a user
        >>> suggestions = await engine.generate_suggestions("user123")
        >>>
        >>> # Get top suggestion
        >>> top = suggestions[0] if suggestions else None
        >>> print(f"Suggestion: {top.title}")
        >>>
        >>> # Mark as shown
        >>> await engine.mark_shown(top.suggestion_id)
    """

    SUGGESTION_PROMPT = """Based on this user context, generate a helpful proactive suggestion.

User Context:
- Current topic: {topic}
- Emotional state: {emotional_state}
- Recent queries: {recent_queries}
- Active goals: {goals}
- Last activity: {last_activity}
- Session duration: {session_duration}

Generate a brief, helpful suggestion that:
1. Is relevant to their current context
2. Provides value without being intrusive
3. Encourages engagement or helps achieve their goals

Respond in JSON format:
{{
    "type": "topic_exploration|task_reminder|knowledge_share|check_in|tip|follow_up|goal_progress|learning_opportunity",
    "priority": "high|medium|low",
    "title": "Brief title (max 50 chars)",
    "content": "The suggestion content (2-3 sentences)",
    "action": "Optional suggested action"
}}"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        context_monitor: Optional[ContextMonitor] = None,
        max_suggestions_per_session: int = 5,
    ):
        """
        Initialize suggestion engine.

        Args:
            llm_service: LLM for generating suggestions
            context_monitor: Context monitor for user data
            max_suggestions_per_session: Limit suggestions per session
        """
        self.settings = get_settings()
        self.llm = llm_service or get_llm_service()
        self.monitor = context_monitor or get_context_monitor()
        self.max_per_session = max_suggestions_per_session

        # Suggestion storage
        self._suggestions: Dict[str, List[Suggestion]] = {}  # user_id -> suggestions
        self._suggestion_counts: Dict[str, int] = {}  # user_id -> count this session

        logger.info(
            f"SuggestionEngine initialized: max_per_session={max_suggestions_per_session}"
        )

    async def generate_suggestions(
        self,
        user_id: str,
        count: int = 3,
        types: Optional[List[SuggestionType]] = None,
    ) -> List[Suggestion]:
        """
        Generate suggestions for a user.

        Args:
            user_id: User to generate suggestions for
            count: Number of suggestions to generate
            types: Optional filter by suggestion types

        Returns:
            List of suggestions sorted by priority
        """
        # Check session limit
        session_count = self._suggestion_counts.get(user_id, 0)
        if session_count >= self.max_per_session:
            logger.debug(f"Session limit reached for {user_id}")
            return []

        # Get user context
        context = await self.monitor.get_context(user_id)
        if not context:
            return await self._generate_default_suggestions(user_id, count)

        suggestions = []

        # Generate rule-based suggestions
        rule_suggestions = await self._generate_rule_based_suggestions(context)
        suggestions.extend(rule_suggestions)

        # Generate LLM-based suggestions if needed
        remaining = count - len(suggestions)
        if remaining > 0:
            llm_suggestions = await self._generate_llm_suggestions(context, remaining)
            suggestions.extend(llm_suggestions)

        # Filter by types if specified
        if types:
            suggestions = [s for s in suggestions if s.suggestion_type in types]

        # Sort by priority
        priority_order = {
            SuggestionPriority.HIGH: 0,
            SuggestionPriority.MEDIUM: 1,
            SuggestionPriority.LOW: 2,
        }
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 1))

        # Store and return
        if user_id not in self._suggestions:
            self._suggestions[user_id] = []
        self._suggestions[user_id].extend(suggestions[:count])

        return suggestions[:count]

    async def _generate_rule_based_suggestions(
        self,
        context: UserContext,
    ) -> List[Suggestion]:
        """Generate suggestions based on rules."""
        suggestions = []

        # Return visit suggestion
        if context.last_activity:
            gap = datetime.utcnow() - context.last_activity
            if gap.total_seconds() > 3600:  # More than 1 hour
                suggestions.append(
                    Suggestion(
                        suggestion_id=str(uuid.uuid4()),
                        suggestion_type=SuggestionType.FOLLOW_UP,
                        priority=SuggestionPriority.MEDIUM,
                        title="Welcome back!",
                        content=f"Last time we discussed {context.current_topic or 'various topics'}. Would you like to continue or explore something new?",
                        user_id=context.user_id,
                        context={"gap_hours": gap.total_seconds() / 3600},
                    )
                )

        # Goal reminder
        if context.active_goals:
            suggestions.append(
                Suggestion(
                    suggestion_id=str(uuid.uuid4()),
                    suggestion_type=SuggestionType.GOAL_PROGRESS,
                    priority=SuggestionPriority.HIGH,
                    title="Goal Check-in",
                    content=f"You're working on: {context.active_goals[0]}. How's it going? I can help if you're stuck.",
                    user_id=context.user_id,
                    context={"goals": context.active_goals},
                    action="Share your progress",
                )
            )

        # Topic exploration (if detected pattern)
        if context.patterns.get("common_topic"):
            topic = context.patterns["common_topic"]
            suggestions.append(
                Suggestion(
                    suggestion_id=str(uuid.uuid4()),
                    suggestion_type=SuggestionType.TOPIC_EXPLORATION,
                    priority=SuggestionPriority.MEDIUM,
                    title=f"Explore more about {topic}?",
                    content=f"I noticed you've been interested in {topic}. Want me to share some related topics or deeper insights?",
                    user_id=context.user_id,
                    context={"detected_topic": topic},
                    action="Show related topics",
                )
            )

        # Learning opportunity (many interactions)
        if context.interaction_count > 10:
            suggestions.append(
                Suggestion(
                    suggestion_id=str(uuid.uuid4()),
                    suggestion_type=SuggestionType.LEARNING_OPPORTUNITY,
                    priority=SuggestionPriority.LOW,
                    title="Learning Summary",
                    content="We've had great conversations! Would you like me to summarize the key things we've discussed?",
                    user_id=context.user_id,
                    context={"interaction_count": context.interaction_count},
                    action="Generate summary",
                )
            )

        return suggestions

    async def _generate_llm_suggestions(
        self,
        context: UserContext,
        count: int,
    ) -> List[Suggestion]:
        """Generate suggestions using LLM."""
        suggestions = []

        try:
            # Calculate session duration
            session_duration = "Unknown"
            if context.session_start:
                duration = datetime.utcnow() - context.session_start
                session_duration = f"{duration.total_seconds() / 60:.0f} minutes"

            prompt = self.SUGGESTION_PROMPT.format(
                topic=context.current_topic or "Not set",
                emotional_state=context.emotional_state or "Unknown",
                recent_queries=(
                    ", ".join(context.recent_queries[-5:])
                    if context.recent_queries
                    else "None"
                ),
                goals=(
                    ", ".join(context.active_goals) if context.active_goals else "None"
                ),
                last_activity=(
                    context.last_activity.isoformat()
                    if context.last_activity
                    else "Unknown"
                ),
                session_duration=session_duration,
            )

            for _ in range(min(count, 2)):  # Generate up to 2 LLM suggestions
                response = self.llm.generate(
                    prompt=prompt,
                    temperature=0.8,  # Higher temperature for variety
                    max_tokens=300,
                )

                # Parse response
                import json
                import re

                content = response.content
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    data = json.loads(json_match.group(0))

                    suggestion = Suggestion(
                        suggestion_id=str(uuid.uuid4()),
                        suggestion_type=SuggestionType(data.get("type", "tip")),
                        priority=SuggestionPriority(data.get("priority", "medium")),
                        title=data.get("title", "Suggestion")[:50],
                        content=data.get("content", ""),
                        user_id=context.user_id,
                        action=data.get("action"),
                        context={"source": "llm"},
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.warning(f"LLM suggestion generation failed: {e}")

        return suggestions

    async def _generate_default_suggestions(
        self,
        user_id: str,
        count: int,
    ) -> List[Suggestion]:
        """Generate default suggestions for new users."""
        return [
            Suggestion(
                suggestion_id=str(uuid.uuid4()),
                suggestion_type=SuggestionType.TIP,
                priority=SuggestionPriority.LOW,
                title="Getting Started",
                content="I'm Morgan, your AI assistant. I can help with research, analysis, and answering questions. What would you like to explore today?",
                user_id=user_id,
                action="Start a conversation",
            ),
        ][:count]

    async def mark_shown(self, suggestion_id: str):
        """Mark a suggestion as shown."""
        for user_suggestions in self._suggestions.values():
            for suggestion in user_suggestions:
                if suggestion.suggestion_id == suggestion_id:
                    suggestion.shown = True
                    return

    async def mark_dismissed(self, suggestion_id: str):
        """Mark a suggestion as dismissed."""
        for user_suggestions in self._suggestions.values():
            for suggestion in user_suggestions:
                if suggestion.suggestion_id == suggestion_id:
                    suggestion.dismissed = True
                    return

    async def mark_acted_upon(self, suggestion_id: str):
        """Mark a suggestion as acted upon."""
        for user_suggestions in self._suggestions.values():
            for suggestion in user_suggestions:
                if suggestion.suggestion_id == suggestion_id:
                    suggestion.acted_upon = True
                    return

    def get_user_suggestions(self, user_id: str) -> List[Suggestion]:
        """Get all suggestions for a user."""
        return self._suggestions.get(user_id, [])

    def clear_user_suggestions(self, user_id: str):
        """Clear suggestions for a user."""
        if user_id in self._suggestions:
            del self._suggestions[user_id]
        if user_id in self._suggestion_counts:
            del self._suggestion_counts[user_id]

    def reset_session(self, user_id: str):
        """Reset suggestion count for a new session."""
        self._suggestion_counts[user_id] = 0


# Singleton instance
_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    """Get singleton suggestion engine instance."""
    global _engine
    if _engine is None:
        _engine = SuggestionEngine()
    return _engine
