"""
Response handling and processing for Morgan Assistant.

Handles response generation, formatting, and metadata processing
following KISS principles - one clear responsibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..emotional.models import EmotionalState, RelationshipMilestone
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Response:
    """
    Human-friendly response from Morgan with emotional awareness.

    Enhanced structure that includes emotional context and personalization.
    """

    answer: str
    sources: List[str] = None
    confidence: float = 0.0
    thinking: str = None  # Morgan's reasoning process
    suggestions: List[str] = None  # Follow-up suggestions
    conversation_id: str = None
    # Enhanced companion features
    emotional_tone: str = None  # Emotional tone of response
    empathy_level: float = 0.0  # Level of empathy in response
    personalization_elements: List[str] = None  # Personalization applied
    relationship_context: str = None  # Relationship context used
    milestone_celebration: Optional[RelationshipMilestone] = None

    def __post_init__(self):
        """Ensure lists are always initialized."""
        if self.sources is None:
            self.sources = []
        if self.suggestions is None:
            self.suggestions = []
        if self.personalization_elements is None:
            self.personalization_elements = []


class ResponseHandler:
    """
    Handles response processing and formatting.

    KISS: Single responsibility - process and format responses.
    """

    def __init__(self):
        """Initialize response handler."""
        self.response_cache = {}
        logger.info("Response handler initialized")

    def create_response(
        self,
        answer: str,
        sources: List[str] = None,
        confidence: float = 0.0,
        thinking: str = None,
        suggestions: List[str] = None,
        conversation_id: str = None,
        emotional_tone: str = None,
        empathy_level: float = 0.0,
        personalization_elements: List[str] = None,
        relationship_context: str = None,
        milestone_celebration: Optional[RelationshipMilestone] = None,
    ) -> Response:
        """Create a properly formatted response."""
        return Response(
            answer=answer,
            sources=sources or [],
            confidence=confidence,
            thinking=thinking,
            suggestions=suggestions or [],
            conversation_id=conversation_id,
            emotional_tone=emotional_tone,
            empathy_level=empathy_level,
            personalization_elements=personalization_elements or [],
            relationship_context=relationship_context,
            milestone_celebration=milestone_celebration,
        )

    def enhance_response_with_emotion(
        self,
        base_response: str,
        emotional_state: EmotionalState,
        empathy_level: float = 0.0,
    ) -> Dict[str, Any]:
        """Enhance response with emotional context."""
        try:
            # Simple emotional enhancement
            enhanced_data = {
                "answer": base_response,
                "emotional_tone": self._get_emotional_tone(emotional_state),
                "empathy_level": empathy_level,
                "personalization_elements": [],
            }

            # Add emotional indicators if high intensity
            if emotional_state.intensity > 0.7:
                enhanced_data["personalization_elements"].append(
                    f"High {emotional_state.primary_emotion.value} detected"
                )

            return enhanced_data

        except Exception as e:
            logger.error(f"Failed to enhance response with emotion: {e}")
            return {"answer": base_response, "emotional_tone": "neutral"}

    def extract_sources(self, search_results: List[Dict]) -> List[str]:
        """Extract source references from search results."""
        sources = []
        for result in search_results:
            source = result.get("source", "")
            if source and source not in sources:
                sources.append(source)
        return sources[:5]  # Limit to top 5 sources

    def generate_suggestions(
        self, question: str, context: str, user_interests: List[str] = None
    ) -> List[str]:
        """Generate follow-up suggestions."""
        suggestions = []

        # Simple suggestion generation based on question content
        question_lower = question.lower()

        if "docker" in question_lower:
            suggestions.extend(
                [
                    "How do I optimize Docker performance?",
                    "What are Docker best practices?",
                    "How do I troubleshoot Docker issues?",
                ]
            )
        elif "python" in question_lower:
            suggestions.extend(
                [
                    "What are Python best practices?",
                    "How do I optimize Python code?",
                    "What Python libraries should I know?",
                ]
            )
        elif "api" in question_lower:
            suggestions.extend(
                [
                    "How do I design RESTful APIs?",
                    "What are API security best practices?",
                    "How do I test APIs effectively?",
                ]
            )

        # Add interest-based suggestions
        if user_interests:
            for interest in user_interests[:2]:
                suggestions.append(f"Tell me more about {interest}")

        return suggestions[:3]  # Limit to 3 suggestions

    def _get_emotional_tone(self, emotional_state: EmotionalState) -> str:
        """Get emotional tone description."""
        emotion = emotional_state.primary_emotion.value
        intensity = emotional_state.intensity

        tone_map = {
            "joy": "warm and enthusiastic",
            "sadness": "gentle and supportive",
            "anger": "calm and understanding",
            "fear": "reassuring and confident",
            "surprise": "curious and engaging",
            "disgust": "respectful and neutral",
            "neutral": "friendly and helpful",
        }

        base_tone = tone_map.get(emotion, "empathetic")

        if intensity > 0.7:
            return f"especially {base_tone}"
        elif intensity < 0.3:
            return f"subtly {base_tone}"
        else:
            return base_tone

    def format_error_response(self, error: Exception) -> Response:
        """Generate empathetic error response."""
        empathetic_responses = [
            "I apologize, but I encountered a technical issue while trying to help you. "
            "I understand this might be frustrating, and I'm here to try again when you're ready.",
            "I'm sorry, something went wrong on my end. I know it's disappointing when technology "
            "doesn't work as expected. Please feel free to try asking your question again.",
            "I encountered an unexpected error and I'm truly sorry about that. "
            "Your question is important to me, so please don't hesitate to rephrase it or try again.",
        ]

        import random

        error_message = random.choice(empathetic_responses)

        return Response(
            answer=error_message,
            confidence=0.0,
            emotional_tone="supportive and understanding",
            empathy_level=0.8,
        )
