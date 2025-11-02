"""Simple feedback system for Morgan."""

from enum import Enum
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    CONVERSATION_RATING = "conversation_rating"
    HELPFULNESS = "helpfulness"
    EMPATHY = "empathy"


class CompanionFeedbackSystem:
    """Simple feedback collection system."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize feedback system."""
        self.feedback_data = {}
        logger.info("Feedback system initialized")
    
    def collect_feedback(
        self,
        user_id: str,
        conversation_id: str,
        feedback_type: FeedbackType,
        rating: int,
        comment: Optional[str] = None,
        emotional_state: Optional[Any] = None
    ) -> str:
        """Collect user feedback."""
        feedback_id = f"{user_id}_{conversation_id}_{rating}"
        
        if user_id not in self.feedback_data:
            self.feedback_data[user_id] = []
        
        self.feedback_data[user_id].append({
            "id": feedback_id,
            "rating": rating,
            "comment": comment,
            "type": feedback_type.value
        })
        
        return feedback_id
    
    def get_user_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """Get feedback summary for user."""
        if user_id not in self.feedback_data:
            return {"total_feedback": 0, "average_rating": 0.0}
        
        feedback_list = self.feedback_data[user_id]
        total = len(feedback_list)
        avg_rating = sum(f["rating"] for f in feedback_list) / total if total > 0 else 0.0
        
        return {
            "total_feedback": total,
            "average_rating": avg_rating,
            "recent_feedback": feedback_list[-5:]
        }