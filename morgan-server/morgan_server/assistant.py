"""
Morgan Assistant - Core orchestration of all engines.

This module integrates the shared Morgan Core library to provide
a complete personal assistant experience.

Refactored to use `morgan-rag` shared domain models.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from morgan.core.assistant import MorganAssistant as CoreMorganAssistant
from morgan.intelligence.core.models import EmotionalState

# from morgan.services.llm_service import LLMClient, LLMMessage # Removed as classes don't exist in Core

# Re-exporting Response models for compatibility if needed,
# though we should rely on API models in api/models.py
# But the internal wrapper needs its own response type or reuse Core's.
# To minimize disruption to api/routes/chat.py, we keep AssistantResponse strictly compatible.


@dataclass
class AssistantContext:
    """
    Context for an assistant interaction.
    Kept for backward compatibility if other modules import it,
    though largely internal now.
    """

    user_id: str
    conversation_id: Optional[str] = None
    message: str = ""
    # Other fields deprecated/unused in new implementation but kept for shape
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    """Response from the assistant - internal server model."""

    answer: str
    conversation_id: str
    emotional_tone: Optional[str] = None
    empathy_level: Optional[float] = None
    personalization_elements: List[str] = field(default_factory=list)
    milestone_celebration: Optional[str] = None
    confidence: float = 1.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Integrated module fields
    flow_state: Optional[str] = None
    quality_score: Optional[float] = None
    proactive_suggestions: List[str] = field(default_factory=list)
    reasoning_summary: Optional[str] = None
    cultural_adaptations: List[str] = field(default_factory=list)
    habit_adaptations: List[str] = field(default_factory=list)
    nonverbal_cues_detected: List[str] = field(default_factory=list)


class MorganAssistant:
    """
    Morgan Assistant - Wrapper around morgan-rag CoreMorganAssistant.

    Delegates orchestration to the shared domain logic in `morgan` package.
    """

    def __init__(
        self,
        llm_client: Any = None,  # Legacy arg, ignored as Core handles its own LLM
        # Legacy args ignored/deprecated
        emotional_intelligence: Any = None,
        personality_system: Any = None,
        roleplay_system: Any = None,
        rag_system: Any = None,
        search_system: Any = None,
        profile_manager: Any = None,
        preference_manager: Any = None,
        memory_system: Any = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the Morgan Assistant Wrapper.

        Args:
            llm_client: Deprecated. Core uses internal LLM service.
            ...: Other args are deprecated.
        """
        # Initialize the actual core assistant
        self.core = CoreMorganAssistant(config_path=config_path)

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        use_knowledge: bool = True,
        use_memory: bool = True,
        # API allows these flags, Core defaults to True usually.
        # We can pass them if Core supports, otherwise we consume them.
    ) -> AssistantResponse:
        """
        Process a chat message and generate a response.
        """
        # Delegate to Core
        # Core 'ask' is async? 'ask' in assistant.py is async def ask(...)

        core_response = await self.core.ask(
            question=message,
            conversation_id=conversation_id,
            user_id=user_id,
            include_sources=use_knowledge,
        )

        # Map core_response to AssistantResponse
        sources_list = []
        if core_response.sources:
            for src in core_response.sources:
                sources_list.append(
                    {
                        "content": src,
                        "score": 1.0,  # Default as core doesn't return score in simple list
                        "document_id": "unknown",
                        "chunk_id": "unknown",
                    }
                )

        # Handle milestone celebration (Core returns object or None)
        milestone_msg = None
        if core_response.milestone_celebration:
            # If it's an object, get description, else string
            if hasattr(core_response.milestone_celebration, "description"):
                milestone_msg = core_response.milestone_celebration.description
            else:
                milestone_msg = str(core_response.milestone_celebration)

        return AssistantResponse(
            answer=core_response.answer,
            conversation_id=core_response.conversation_id or conversation_id or "",
            emotional_tone=core_response.emotional_tone,
            empathy_level=core_response.empathy_level,
            personalization_elements=core_response.personalization_elements or [],
            milestone_celebration=milestone_msg,
            confidence=core_response.confidence,
            sources=sources_list,
            metadata={
                "suggestions": core_response.suggestions,
                "thinking": core_response.thinking,
            },
            # Integrated module fields
            flow_state=getattr(core_response, "flow_state", None),
            quality_score=getattr(core_response, "quality_score", None),
            proactive_suggestions=getattr(core_response, "proactive_suggestions", None) or [],
            reasoning_summary=getattr(core_response, "reasoning_summary", None),
            cultural_adaptations=getattr(core_response, "cultural_adaptations", None) or [],
            habit_adaptations=getattr(core_response, "habit_adaptations", None) or [],
            nonverbal_cues_detected=getattr(core_response, "nonverbal_cues_detected", None) or [],
        )

    # Legacy method support - mapped to Core functionality where possible

    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history."""
        # Core has get_conversation_history(conversation_id)
        if conversation_id:
            history = self.core.get_conversation_history(conversation_id)
            # Map history format if needed
            # Core returns List[Dict] usually
            return history[-limit:]
        return []

    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        # Core has knowledge.search_knowledge
        # Assuming synchronous or async? Core assistant.py uses synchronous validation wrapper?
        # self.knowledge.search_knowledge is likely async or sync?
        # In orchestrator it was: search_results = self.knowledge.search_knowledge(...)
        # Let's check if it's awaitable. Orchestrator didn't await it.
        results = self.core.knowledge.search_knowledge(query, max_results=limit)
        return results

    def get_user_profile(self, user_id: str) -> Optional[Any]:
        """Get user profile."""
        if hasattr(self.core, "emotional_processor"):
            return self.core.emotional_processor.get_or_create_user_profile(user_id)
        return None

    def update_user_profile(self, user_id: str, **updates) -> Any:
        """Update user profile."""
        profile = self.get_user_profile(user_id)
        if not profile:
            return None

        # Apply updates
        if "preferred_name" in updates:
            profile.preferred_name = updates["preferred_name"]

        if "communication_style" in updates:
            # Check if it's already an enum or string
            val = updates["communication_style"]
            # If it's an enum, use it. If string, leave it (model might expect enum)
            # Core CompanionProfile uses communication_preferences which is UserPreferences object
            # We might need to update communication_preferences
            if not profile.communication_preferences:
                from morgan.intelligence.core.models import (
                    UserPreferences,
                    CommunicationStyle,
                    ResponseLength,
                )

                profile.communication_preferences = UserPreferences(
                    topics_of_interest=[],
                    communication_style=CommunicationStyle.FRIENDLY,
                    preferred_response_length=ResponseLength.MODERATE,
                )

            profile.communication_preferences.communication_style = val

        if "response_length" in updates:
            if not profile.communication_preferences:
                from morgan.intelligence.core.models import (
                    UserPreferences,
                    CommunicationStyle,
                    ResponseLength,
                )

                profile.communication_preferences = UserPreferences(
                    topics_of_interest=[],
                    communication_style=CommunicationStyle.FRIENDLY,
                    preferred_response_length=ResponseLength.MODERATE,
                )
            profile.communication_preferences.preferred_response_length = updates[
                "response_length"
            ]

        if "topics_of_interest" in updates:
            # This is on profile directly or preferences?
            # CompanionProfile has topics_of_interest? No, UserPreferences has it.
            if not profile.communication_preferences:
                from morgan.intelligence.core.models import (
                    UserPreferences,
                    CommunicationStyle,
                    ResponseLength,
                )

                profile.communication_preferences = UserPreferences(
                    topics_of_interest=[],
                    communication_style=CommunicationStyle.FRIENDLY,
                    preferred_response_length=ResponseLength.MODERATE,
                )
            profile.communication_preferences.topics_of_interest = updates[
                "topics_of_interest"
            ]

        return profile

    async def analyze_emotional_state(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's emotional state over time."""
        if hasattr(self.core, "emotional_engine"):
            return self.core.emotional_engine.track_mood_patterns(user_id)
        return {}

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the assistant's components.

        Returns:
            Dict with 'status' and 'details' keys.
        """
        status_val = "up"
        details = {}

        try:
            # Check knowledge engine
            if hasattr(self.core, "knowledge"):
                # Simple liveness check via statistics
                self.core.knowledge.get_statistics()
                details["knowledge"] = "up"
            else:
                details["knowledge"] = "unknown"

            # Check memory engine
            if hasattr(self.core, "memory"):
                # Simple check
                details["memory"] = "up"
            else:
                details["memory"] = "unknown"

        except Exception as e:
            status_val = "degraded"
            details["error"] = str(e)

        return {"status": status_val, "details": details}

    async def get_wellness_insights(self, user_id: str) -> Dict[str, Any]:
        """Get wellness insights for a user."""
        if hasattr(self.core, "orchestrator"):
            return await self.core.orchestrator.get_wellness_insights(user_id)
        return {"message": "Wellness tracking not available"}

    async def get_habit_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get detected habit patterns for a user."""
        if hasattr(self.core, "orchestrator"):
            return await self.core.orchestrator.get_habit_patterns(user_id)
        return {"habits": [], "total_interactions": 0}

    async def get_conversation_quality(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation quality assessment."""
        if hasattr(self.core, "orchestrator"):
            return await self.core.orchestrator.get_conversation_quality(conversation_id)
        return {"message": "Quality assessment not available"}

    async def get_proactive_suggestions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get proactive suggestions for a user."""
        if hasattr(self.core, "orchestrator"):
            return await self.core.orchestrator.get_suggestions(user_id)
        return []

    async def shutdown(self):
        """Shutdown the assistant."""
        # Perform any cleanup needed for Core
        pass
