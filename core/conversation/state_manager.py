"""
Conversation state management for Morgan Core
"""
from typing import Dict, Optional
import uuid
import time

from .context import ConversationContext


class ConversationStateManager:
    """Manages conversation state and context for users"""

    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.last_activity: Dict[str, float] = {}
        self.context_timeout = 1800  # 30 minutes in seconds

    def get_context(self, user_id: str = "default") -> ConversationContext:
        """Get conversation context for a user"""
        # Update last activity time
        self.last_activity[user_id] = time.time()

        # Create context if it doesn't exist
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(user_id)

        return self.contexts[user_id]

    def create_context(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation context"""
        # Generate user ID if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())

        # Create new context
        self.contexts[user_id] = ConversationContext(user_id)
        self.last_activity[user_id] = time.time()

        return user_id

    def reset_context(self, user_id: str) -> bool:
        """Reset conversation context for a user"""
        if user_id in self.contexts:
            self.contexts[user_id] = ConversationContext(user_id)
            self.last_activity[user_id] = time.time()
            return True
        return False

    def clear_expired_contexts(self):
        """Clear expired conversation contexts"""
        current_time = time.time()
        expired_users = []

        for user_id, last_time in self.last_activity.items():
            if current_time - last_time > self.context_timeout:
                expired_users.append(user_id)

        for user_id in expired_users:
            del self.contexts[user_id]
            del self.last_activity[user_id]