"""
Feature Module API Routes

Endpoints for accessing integrated feature modules:
- GET /api/suggestions/{user_id} - Proactive suggestions
- GET /api/wellness/{user_id} - Wellness insights
- GET /api/habits/{user_id} - Habit patterns
- GET /api/quality/{conversation_id} - Conversation quality
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status

from morgan_server.api.models import (
    WellnessResponse,
    WellnessInsight,
    HabitsResponse,
    HabitPatternItem,
    ConversationQualityResponse,
    QualityDimensionScore,
    SuggestionsResponse,
    ErrorResponse,
)
from morgan_server.assistant import MorganAssistant


router = APIRouter(prefix="/api", tags=["features"])

# Global assistant instance (shared with chat routes)
_assistant: Optional[MorganAssistant] = None


def set_assistant(assistant: MorganAssistant) -> None:
    """Set the global assistant instance."""
    global _assistant
    _assistant = assistant


def get_assistant() -> MorganAssistant:
    """Get the global assistant instance."""
    if _assistant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant not initialized",
        )
    return _assistant


@router.get(
    "/suggestions/{user_id}",
    response_model=SuggestionsResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get proactive suggestions",
    description="Returns proactive suggestions for a user based on context and patterns.",
)
async def get_suggestions(user_id: str) -> SuggestionsResponse:
    """Get proactive suggestions for a user."""
    assistant = get_assistant()
    try:
        suggestions = await assistant.get_proactive_suggestions(user_id)
        return SuggestionsResponse(
            user_id=user_id,
            suggestions=suggestions if isinstance(suggestions, list) else [],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}",
        )


@router.get(
    "/wellness/{user_id}",
    response_model=WellnessResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get wellness insights",
    description="Returns wellness tracking insights for a user.",
)
async def get_wellness(user_id: str) -> WellnessResponse:
    """Get wellness insights for a user."""
    assistant = get_assistant()
    try:
        data = await assistant.get_wellness_insights(user_id)
        insights = []
        if isinstance(data, dict) and "insights" in data:
            for item in data["insights"]:
                insights.append(WellnessInsight(
                    category=item.get("category", "unknown"),
                    score=item.get("score"),
                    trend=item.get("trend"),
                    recommendations=item.get("recommendations", []),
                ))
        return WellnessResponse(
            user_id=user_id,
            insights=insights,
            overall_score=data.get("overall_score") if isinstance(data, dict) else None,
            metadata=data if isinstance(data, dict) else {},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get wellness insights: {str(e)}",
        )


@router.get(
    "/habits/{user_id}",
    response_model=HabitsResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get habit patterns",
    description="Returns detected habit patterns for a user.",
)
async def get_habits(user_id: str) -> HabitsResponse:
    """Get habit patterns for a user."""
    assistant = get_assistant()
    try:
        data = await assistant.get_habit_patterns(user_id)
        habits = []
        if isinstance(data, dict) and "habits" in data:
            for item in data["habits"]:
                habits.append(HabitPatternItem(
                    name=item.get("name", "unknown"),
                    habit_type=item.get("type", "unknown"),
                    frequency=item.get("frequency", "unknown"),
                    confidence=item.get("confidence", "low"),
                    consistency=item.get("consistency", 0.0),
                ))
        return HabitsResponse(
            user_id=user_id,
            habits=habits,
            total_interactions=data.get("total_interactions", 0) if isinstance(data, dict) else 0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get habit patterns: {str(e)}",
        )


@router.get(
    "/quality/{conversation_id}",
    response_model=ConversationQualityResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get conversation quality",
    description="Returns quality assessment for a conversation.",
)
async def get_quality(conversation_id: str) -> ConversationQualityResponse:
    """Get conversation quality assessment."""
    assistant = get_assistant()
    try:
        data = await assistant.get_conversation_quality(conversation_id)
        dimensions = []
        if isinstance(data, dict) and "dimensions" in data:
            for item in data["dimensions"]:
                dimensions.append(QualityDimensionScore(
                    dimension=item.get("dimension", "unknown"),
                    score=item.get("score", 0.0),
                    level=item.get("level", "unknown"),
                ))
        return ConversationQualityResponse(
            conversation_id=conversation_id,
            overall_score=data.get("overall_score") if isinstance(data, dict) else None,
            overall_level=data.get("overall_level") if isinstance(data, dict) else None,
            dimensions=dimensions,
            strengths=data.get("strengths", []) if isinstance(data, dict) else [],
            improvements=data.get("improvements", []) if isinstance(data, dict) else [],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation quality: {str(e)}",
        )
