"""
Conversation management for Morgan Assistant.

Handles conversation flow, context building, and session management
following KISS principles - focused on conversation logic only.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.intelligence.core.models import CompanionProfile, ConversationContext, EmotionalState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages conversation flow and context.

    KISS: Single responsibility - handle conversation logic and context.
    """

    def __init__(self):
        """Initialize conversation manager."""
        self.active_conversations: Dict[str, List[ConversationContext]] = {}
        logger.info("Conversation manager initialized")

    def create_conversation_context(
        self, question: str, conversation_id: Optional[str], user_id: Optional[str]
    ) -> ConversationContext:
        """Create conversation context for emotional analysis."""
        # Get previous messages from conversation history
        previous_messages = []
        if conversation_id and conversation_id in self.active_conversations:
            previous_messages = [
                ctx.message_text
                for ctx in self.active_conversations[conversation_id][-3:]
            ]

        context = ConversationContext(
            user_id=user_id or "anonymous",
            conversation_id=conversation_id or str(uuid.uuid4()),
            message_text=question,
            timestamp=datetime.utcnow(),
            previous_messages=previous_messages,
        )

        # Track conversation context
        if conversation_id:
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = []
            self.active_conversations[conversation_id].append(context)

            # Keep only recent contexts (last 10)
            if len(self.active_conversations[conversation_id]) > 10:
                self.active_conversations[conversation_id] = self.active_conversations[
                    conversation_id
                ][-10:]

        return context

    def build_emotional_context(
        self,
        question: str,
        search_results: List[Dict],
        memory_context: str,
        emotional_state: EmotionalState,
        conversation_style: Optional[Any],
        user_profile: Optional[CompanionProfile],
        max_context: int = 8192,
    ) -> str:
        """Build emotionally aware context for LLM generation."""
        context_parts = []

        # Enhanced system prompt with emotional awareness
        emotional_guidance = f"""
You are Morgan, a helpful, knowledgeable, conversational, and emotionally aware AI assistant.

Current user emotional state: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.1f})
Your response should be {self._get_emotional_response_guidance(emotional_state)}.
"""

        if conversation_style:
            emotional_guidance += f"""
Conversation style adaptation:
- Formality level: {conversation_style.formality_level:.1f} (0=casual, 1=formal)
- Technical depth: {conversation_style.technical_depth:.1f} (0=simple, 1=technical)
- Empathy emphasis: {conversation_style.empathy_emphasis:.1f} (0=minimal, 1=high)
- Response length: {conversation_style.response_length_target.value}
- Personality traits: {', '.join(conversation_style.personality_traits)}
"""

        if user_profile:
            emotional_guidance += f"""
User relationship context:
- Preferred name: {user_profile.preferred_name}
- Relationship age: {user_profile.get_relationship_age_days()} days
- Trust level: {user_profile.trust_level:.1f}
- Communication style: {user_profile.communication_preferences.communication_style.value}
- Topics of interest: {', '.join(user_profile.communication_preferences.topics_of_interest[:3])}
"""

        context_parts.append(emotional_guidance)

        # Memory context for continuity
        if memory_context:
            context_parts.append(f"\nConversation History:\n{memory_context}")

        # Relevant knowledge from search
        if search_results:
            context_parts.append("\nRelevant Knowledge:")
            for i, result in enumerate(search_results[:5], 1):
                # Handle both dict and SearchResult object
                if hasattr(result, "source"):
                    source = result.source
                    content = result.content[:500]
                else:
                    source = result.get("source", "Unknown")
                    content = result.get("content", "")[:500]
                context_parts.append(f"\n{i}. Source: {source}\n{content}")

        # The current question with emotional context
        context_parts.append(
            f"\nUser Question (emotional state: {emotional_state.primary_emotion.value}): {question}"
        )
        context_parts.append(
            "\nPlease provide a helpful, emotionally aware response as Morgan:"
        )

        # Join and truncate if needed
        full_context = "\n".join(context_parts)

        if len(full_context) > max_context:
            # Truncate intelligently - keep emotional guidance and question
            emotional_part = context_parts[0]
            question_part = context_parts[-2] + context_parts[-1]

            available_space = (
                max_context - len(emotional_part) - len(question_part) - 100
            )

            # Truncate middle content
            middle_content = "\n".join(context_parts[1:-2])
            if len(middle_content) > available_space:
                middle_content = (
                    middle_content[:available_space]
                    + "\n[... truncated for length ...]"
                )

            full_context = emotional_part + "\n" + middle_content + "\n" + question_part

        return full_context

    def build_basic_context(
        self,
        question: str,
        search_results: List[Dict],
        conversation_context: str,
        max_context: int = 8192,
    ) -> str:
        """
        Build basic context for LLM generation (non-emotional mode).

        Human-first approach: Include relevant information in a clear,
        structured way that helps Morgan give better answers.
        """
        context_parts = []

        # System prompt - define Morgan's personality
        context_parts.append(
            """You are Morgan, a helpful, knowledgeable, conversational, and emotionally aware AI assistant.

Your goal is to provide helpful, accurate, and conversational responses.
Always cite your sources when possible and be transparent about your reasoning.
If you're not sure about something, say so honestly.
Provide follow-up suggestions when appropriate."""
        )

        # Conversation context for continuity
        if conversation_context:
            context_parts.append(f"\nConversation Context:\n{conversation_context}")

        # Relevant knowledge from search
        if search_results:
            context_parts.append("\nRelevant Knowledge:")
            for i, result in enumerate(search_results[:5], 1):
                # Handle both dict and SearchResult object
                if hasattr(result, "source"):
                    source = result.source
                    content = result.content[:500]
                else:
                    source = result.get("source", "Unknown")
                    content = result.get("content", "")[:500]  # Limit length
                context_parts.append(f"\n{i}. Source: {source}\n{content}")

        # The current question
        context_parts.append(f"\nHuman Question: {question}")
        context_parts.append("\nPlease provide a helpful response as Morgan:")

        # Join and truncate if needed
        full_context = "\n".join(context_parts)

        if len(full_context) > max_context:
            # Truncate intelligently - keep system prompt and question
            system_prompt = context_parts[0]
            question_part = context_parts[-2] + context_parts[-1]

            available_space = (
                max_context - len(system_prompt) - len(question_part) - 100
            )

            # Truncate middle parts
            middle_content = "\n".join(context_parts[1:-2])
            if len(middle_content) > available_space:
                middle_content = (
                    middle_content[:available_space]
                    + "\n[... truncated for length ...]"
                )

            full_context = system_prompt + "\n" + middle_content + "\n" + question_part

        return full_context

    def extract_topics_from_question(self, question: str) -> List[str]:
        """Extract topics from user question."""
        # Simple topic extraction - could be enhanced
        import re

        # Look for technical terms and proper nouns
        topics = []

        # Technical terms
        tech_terms = re.findall(
            r"\b(Python|JavaScript|Docker|Kubernetes|AWS|React|Node\.js|SQL|API|Git|Linux)\b",
            question,
            re.IGNORECASE,
        )
        topics.extend([term.lower() for term in tech_terms])

        # Proper nouns (potential topics)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", question)
        topics.extend([noun.lower() for noun in proper_nouns if len(noun) > 3])

        return list(set(topics))  # Remove duplicates

    def _get_emotional_response_guidance(self, emotional_state: EmotionalState) -> str:
        """Get guidance for emotional response tone."""
        guidance_map = {
            "joy": "warm, celebratory, and enthusiastic",
            "sadness": "gentle, supportive, and comforting",
            "anger": "calm, understanding, and de-escalating",
            "fear": "reassuring, confident, and supportive",
            "surprise": "curious, engaging, and exploratory",
            "disgust": "respectful, neutral, and understanding",
            "neutral": "friendly, helpful, and engaging",
        }

        base_guidance = guidance_map.get(
            emotional_state.primary_emotion.value, "empathetic and supportive"
        )

        # Adjust based on intensity
        if emotional_state.intensity > 0.7:
            return f"especially {base_guidance} due to high emotional intensity"
        elif emotional_state.intensity < 0.3:
            return f"gently {base_guidance} with subtle emotional awareness"
        else:
            return base_guidance
