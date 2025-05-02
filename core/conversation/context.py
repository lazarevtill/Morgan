"""
Conversation context management for Morgan Core
"""
from typing import List, Dict, Any
import time


class ConversationContext:
    """Manages context for a single conversation"""

    def __init__(self, user_id: str, max_history: int = 10):
        self.user_id = user_id
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.variables: Dict[str, Any] = {}
        self.active_intents: List[str] = []

    def add_user_message(self, text: str):
        """Add a user message to the conversation history"""
        message = {
            "role": "user",
            "content": text,
            "timestamp": time.time()
        }
        self._add_to_history(message)

    def add_assistant_message(self, text: str):
        """Add an assistant message to the conversation history"""
        message = {
            "role": "assistant",
            "content": text,
            "timestamp": time.time()
        }
        self._add_to_history(message)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.history.copy()

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get the last N messages from conversation history"""
        return self.history[-n:] if n < len(self.history) else self.history.copy()

    def set_variable(self, key: str, value: Any):
        """Set a context variable"""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.variables.get(key, default)

    def push_active_intent(self, intent: str):
        """Push an active intent to the stack"""
        self.active_intents.append(intent)

    def pop_active_intent(self) -> str:
        """Pop the most recent active intent from the stack"""
        if self.active_intents:
            return self.active_intents.pop()
        return None

    def get_active_intent(self) -> str:
        """Get the current active intent"""
        if self.active_intents:
            return self.active_intents[-1]
        return None

    def clear_history(self):
        """Clear the conversation history"""
        self.history = []

    def _add_to_history(self, message: Dict[str, Any]):
        """Add a message to history and maintain max size"""
        self.history.append(message)

        # Trim history if it exceeds the maximum size
        if len(self.history) > self.max_history:
            self.history = self.history[1:]