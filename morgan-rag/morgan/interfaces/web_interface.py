"""
Web-based interface for Morgan RAG with emotional intelligence.

Provides a FastAPI-based web interface with real-time chat, relationship
timeline visualization, preference management, and companion feedback systems.

KISS: Clean web interface that uses modular components.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..core.assistant import MorganAssistant
from ..utils.logger import get_logger
from .feedback_system import CompanionFeedbackSystem, FeedbackType
from .websocket_handler import WebSocketManager, websocket_endpoint

logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message model for API."""

    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class PreferenceUpdate(BaseModel):
    """User preference update model."""

    user_id: str
    communication_style: Optional[str] = None
    response_length: Optional[str] = None
    topics_of_interest: Optional[List[str]] = None
    preferred_name: Optional[str] = None


class FeedbackSubmission(BaseModel):
    """Feedback submission model."""

    conversation_id: str
    user_id: Optional[str] = None
    rating: int
    comment: Optional[str] = None
    emotional_rating: Optional[int] = None


class MorganWebInterface:
    """
    Web interface for Morgan RAG with emotional intelligence and companion features.

    Provides:
    - Real-time chat with emotional awareness
    - Relationship timeline visualization
    - User preference management
    - Companion feedback and satisfaction tracking

    KISS: Clean interface that orchestrates web components.
    """

    def __init__(self, morgan_assistant: Optional[MorganAssistant] = None):
        """Initialize the web interface."""
        self.app = FastAPI(
            title="Morgan RAG - Emotionally Intelligent AI Companion",
            description="Human-first AI assistant with emotional intelligence and companion features",
            version="1.0.0",
        )

        self.morgan = morgan_assistant or MorganAssistant()

        # Initialize modular components
        self.ws_manager = WebSocketManager(self.morgan)
        self.feedback_system = CompanionFeedbackSystem()

        # Setup templates and static files
        self.templates_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"

        # Create directories if they don't exist
        self.templates_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)

        self.templates = Jinja2Templates(directory=str(self.templates_dir))

        # Setup routes
        self._setup_routes()

        logger.info("Morgan web interface initialized")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        # Mount static files
        self.app.mount(
            "/static", StaticFiles(directory=str(self.static_dir)), name="static"
        )

        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Main chat interface."""
            return self.templates.TemplateResponse(
                "chat.html", {"request": request, "title": "Morgan - Your AI Companion"}
            )

        @self.app.get("/preferences/{user_id}", response_class=HTMLResponse)
        async def preferences_page(request: Request, user_id: str):
            """User preferences management page."""
            profile = self.morgan.relationship_manager.profiles.get(user_id)
            return self.templates.TemplateResponse(
                "preferences.html",
                {
                    "request": request,
                    "user_id": user_id,
                    "profile": profile,
                    "title": "Preferences - Morgan",
                },
            )

        @self.app.get("/timeline/{user_id}", response_class=HTMLResponse)
        async def timeline_page(request: Request, user_id: str):
            """Relationship timeline visualization page."""
            profile = self.morgan.relationship_manager.profiles.get(user_id)
            if not profile:
                raise HTTPException(status_code=404, detail="User profile not found")

            return self.templates.TemplateResponse(
                "timeline.html",
                {
                    "request": request,
                    "user_id": user_id,
                    "profile": profile,
                    "title": "Relationship Timeline - Morgan",
                },
            )

        @self.app.websocket("/ws/{user_id}")
        async def websocket_route(websocket: WebSocket, user_id: str):
            """WebSocket endpoint for real-time chat."""
            await websocket_endpoint(websocket, user_id, self.ws_manager)

        @self.app.post("/api/chat")
        async def chat_api(message_data: ChatMessage):
            """REST API endpoint for chat."""
            try:
                response = self.morgan.ask(
                    question=message_data.message,
                    conversation_id=message_data.conversation_id,
                    user_id=message_data.user_id,
                )

                return {
                    "answer": response.answer,
                    "emotional_tone": response.emotional_tone,
                    "empathy_level": response.empathy_level,
                    "personalization_elements": response.personalization_elements,
                    "milestone_celebration": (
                        {
                            "type": response.milestone_celebration.milestone_type.value,
                            "description": response.milestone_celebration.description,
                        }
                        if response.milestone_celebration
                        else None
                    ),
                    "conversation_id": response.conversation_id,
                    "confidence": response.confidence,
                    "sources": response.sources,
                }
            except Exception as e:
                logger.error(f"Chat API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/preferences")
        async def update_preferences(preferences: PreferenceUpdate):
            """Update user preferences."""
            try:
                profile = self.morgan.relationship_manager.profiles.get(
                    preferences.user_id
                )
                if not profile:
                    # Create new profile if it doesn't exist
                    profile = (
                        self.morgan.emotional_processor.get_or_create_user_profile(
                            preferences.user_id
                        )
                    )

                # Update preferences
                if preferences.communication_style:
                    from morgan.intelligence.core.models import CommunicationStyle

                    profile.communication_preferences.communication_style = (
                        CommunicationStyle(preferences.communication_style)
                    )

                if preferences.response_length:
                    from morgan.intelligence.core.models import ResponseLength

                    profile.communication_preferences.preferred_response_length = (
                        ResponseLength(preferences.response_length)
                    )

                if preferences.topics_of_interest:
                    profile.communication_preferences.topics_of_interest = (
                        preferences.topics_of_interest
                    )

                if preferences.preferred_name:
                    profile.preferred_name = preferences.preferred_name

                profile.communication_preferences.last_updated = datetime.now(timezone.utc)

                return {
                    "status": "success",
                    "message": "Preferences updated successfully",
                }

            except Exception as e:
                logger.error(f"Preferences update error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/feedback")
        async def submit_feedback(feedback: FeedbackSubmission):
            """Submit companion feedback using the feedback system."""
            try:
                feedback_id = self.feedback_system.collect_feedback(
                    user_id=feedback.user_id or "anonymous",
                    conversation_id=feedback.conversation_id,
                    feedback_type=FeedbackType.CONVERSATION_RATING,
                    rating=feedback.rating,
                    comment=feedback.comment,
                )

                # Also use Morgan's built-in feedback system
                success = self.morgan.provide_feedback(
                    conversation_id=feedback.conversation_id,
                    rating=feedback.rating,
                    comment=feedback.comment,
                    user_id=feedback.user_id,
                )

                if success and feedback_id:
                    return {
                        "status": "success",
                        "message": "Feedback submitted successfully",
                        "feedback_id": feedback_id,
                    }
                else:
                    raise HTTPException(
                        status_code=400, detail="Failed to submit feedback"
                    )

            except Exception as e:
                logger.error(f"Feedback submission error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/profile/{user_id}")
        async def get_user_profile(user_id: str):
            """Get user profile and relationship insights."""
            try:
                insights = self.morgan.get_relationship_insights(user_id)
                profile = self.morgan.relationship_manager.profiles.get(user_id)

                if profile:
                    return {
                        "profile": {
                            "user_id": profile.user_id,
                            "preferred_name": profile.preferred_name,
                            "relationship_age_days": profile.get_relationship_age_days(),
                            "interaction_count": profile.interaction_count,
                            "trust_level": profile.trust_level,
                            "engagement_score": profile.engagement_score,
                            "communication_style": profile.communication_preferences.communication_style.value,
                            "response_length": profile.communication_preferences.preferred_response_length.value,
                            "topics_of_interest": profile.communication_preferences.topics_of_interest,
                        },
                        "insights": insights,
                    }
                else:
                    return {"profile": None, "insights": insights}

            except Exception as e:
                logger.error(f"Profile retrieval error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/timeline/{user_id}")
        async def get_timeline_data(user_id: str):
            """Get relationship timeline data."""
            try:
                profile = self.morgan.relationship_manager.profiles.get(user_id)
                if not profile:
                    raise HTTPException(
                        status_code=404, detail="User profile not found"
                    )

                timeline_data = []
                for milestone in profile.relationship_milestones:
                    timeline_data.append(
                        {
                            "id": milestone.milestone_id,
                            "type": milestone.milestone_type.value,
                            "description": milestone.description,
                            "timestamp": milestone.timestamp.isoformat(),
                            "emotional_significance": milestone.emotional_significance,
                            "celebration_acknowledged": milestone.celebration_acknowledged,
                        }
                    )

                # Sort by timestamp
                timeline_data.sort(key=lambda x: x["timestamp"])

                return {
                    "timeline": timeline_data,
                    "profile_summary": {
                        "relationship_age_days": profile.get_relationship_age_days(),
                        "interaction_count": profile.interaction_count,
                        "trust_level": profile.trust_level,
                        "engagement_score": profile.engagement_score,
                    },
                }

            except Exception as e:
                logger.error(f"Timeline data error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/suggestions/{user_id}")
        async def get_conversation_suggestions(user_id: str):
            """Get conversation topic suggestions."""
            try:
                suggestions = self.morgan.suggest_conversation_topics(user_id)
                return {"suggestions": suggestions}

            except Exception as e:
                logger.error(f"Suggestions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/feedback/{user_id}")
        async def get_feedback_summary(user_id: str):
            """Get feedback summary for user."""
            try:
                summary = self.feedback_system.get_user_feedback_summary(user_id)
                return summary

            except Exception as e:
                logger.error(f"Feedback summary error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/milestones/{user_id}")
        async def get_milestone_statistics(user_id: str):
            """Get milestone statistics for user."""
            try:
                stats = self.morgan.get_milestone_statistics(user_id)
                return stats

            except Exception as e:
                logger.error(f"Milestone statistics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the web interface."""
        import uvicorn

        logger.info(f"Starting Morgan web interface on {host}:{port}")
        uvicorn.run(
            self.app, host=host, port=port, log_level="info" if debug else "warning"
        )


# Convenience function to start the web interface
def start_web_interface(morgan_assistant: Optional[MorganAssistant] = None, **kwargs):
    """Start the Morgan web interface."""
    interface = MorganWebInterface(morgan_assistant)
    interface.run(**kwargs)


if __name__ == "__main__":
    # Demo the web interface
    start_web_interface(debug=True)
