"""
Enhanced conversation context management for Morgan Core
"""
from typing import List, Dict, Any, Optional, Set
import time
import json
import logging
import uuid
import os
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages context for a single conversation"""

    def __init__(self, user_id: str, max_history: int = 20, persistent: bool = True,
                 data_dir: Optional[str] = None):
        self.user_id = user_id
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.last_updated_at = time.time()
        self.variables: Dict[str, Any] = {}
        self.active_intents: List[str] = []
        self.entity_mentions: Dict[str, float] = {}  # Entity name -> last mention time
        self.persistent = persistent
        self.data_dir = data_dir
        self.unsaved_changes = False

        # Try to load existing history if this is a persistent context
        if persistent and data_dir:
            self._load_from_disk()

    def add_user_message(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a user message to the conversation history"""
        message = {
            "role": "user",
            "content": text,
            "timestamp": time.time()
        }

        if metadata:
            message["metadata"] = metadata

        self._add_to_history(message)
        self._update_last_active()

        # Extract potential entity mentions from user message
        self._extract_entity_mentions(text)

    def add_assistant_message(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an assistant message to the conversation history"""
        message = {
            "role": "assistant",
            "content": text,
            "timestamp": time.time()
        }

        if metadata:
            message["metadata"] = metadata

        self._add_to_history(message)
        self._update_last_active()

    def add_system_message(self, text: str):
        """Add a system message to the conversation history"""
        message = {
            "role": "system",
            "content": text,
            "timestamp": time.time()
        }

        self._add_to_history(message)
        self._update_last_active()

    def get_history(self, max_age: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history, optionally filtered by age

        Args:
            max_age: Optional maximum age in seconds

        Returns:
            List of messages
        """
        if max_age is None:
            return self.history.copy()

        cutoff_time = time.time() - max_age
        return [msg for msg in self.history if msg.get("timestamp", 0) >= cutoff_time]

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get the last N messages from conversation history"""
        return self.history[-n:] if n < len(self.history) else self.history.copy()

    def get_recent_interactions(self, duration: float = 300.0) -> List[Dict[str, Any]]:
        """
        Get interactions from the last N seconds

        Args:
            duration: Time window in seconds

        Returns:
            List of recent messages
        """
        cutoff_time = time.time() - duration
        return [msg for msg in self.history if msg.get("timestamp", 0) >= cutoff_time]

    def get_formatted_history(self, n: Optional[int] = None) -> str:
        """
        Get formatted conversation history for LLM context

        Args:
            n: Optional number of messages to include

        Returns:
            Formatted string representation of history
        """
        messages = self.history[-n:] if n is not None and n < len(self.history) else self.history

        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def set_variable(self, key: str, value: Any):
        """Set a context variable"""
        self.variables[key] = value
        self.unsaved_changes = True

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.variables.get(key, default)

    def push_active_intent(self, intent: str):
        """Push an active intent to the stack"""
        self.active_intents.append(intent)
        self.unsaved_changes = True

    def pop_active_intent(self) -> Optional[str]:
        """Pop the most recent active intent from the stack"""
        if self.active_intents:
            self.unsaved_changes = True
            return self.active_intents.pop()
        return None

    def get_active_intent(self) -> Optional[str]:
        """Get the current active intent"""
        if self.active_intents:
            return self.active_intents[-1]
        return None

    def record_entity_mention(self, entity_name: str):
        """
        Record a mention of an entity

        Args:
            entity_name: Name of the entity
        """
        self.entity_mentions[entity_name] = time.time()
        self.unsaved_changes = True

    def get_recently_mentioned_entities(self, max_age: float = 300.0) -> List[str]:
        """
        Get recently mentioned entities

        Args:
            max_age: Maximum age in seconds

        Returns:
            List of recently mentioned entity names
        """
        now = time.time()
        return [
            entity for entity, mention_time in self.entity_mentions.items()
            if now - mention_time <= max_age
        ]

    def clear_history(self):
        """Clear the conversation history"""
        self.history = []
        self.unsaved_changes = True

    def save(self):
        """Save the conversation context to disk"""
        if not self.persistent or not self.data_dir or not self.unsaved_changes:
            return

        try:
            # Create data directory if it doesn't exist
            context_dir = Path(self.data_dir) / "contexts"
            context_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            context_file = context_dir / f"{self.user_id}.json"

            context_data = {
                "user_id": self.user_id,
                "created_at": self.created_at,
                "last_updated_at": self.last_updated_at,
                "history": self.history,
                "variables": self.variables,
                "active_intents": self.active_intents,
                "entity_mentions": self.entity_mentions
            }

            with open(context_file, 'w') as f:
                json.dump(context_data, f, indent=2)

            self.unsaved_changes = False
            logger.debug(f"Saved conversation context for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving conversation context for user {self.user_id}: {e}")

    def _load_from_disk(self):
        """Load the conversation context from disk"""
        if not self.data_dir:
            return

        context_file = Path(self.data_dir) / "contexts" / f"{self.user_id}.json"

        if not context_file.exists():
            return

        try:
            with open(context_file, 'r') as f:
                context_data = json.load(f)

            self.created_at = context_data.get("created_at", self.created_at)
            self.last_updated_at = context_data.get("last_updated_at", self.last_updated_at)
            self.history = context_data.get("history", [])
            self.variables = context_data.get("variables", {})
            self.active_intents = context_data.get("active_intents", [])
            self.entity_mentions = context_data.get("entity_mentions", {})

            logger.debug(f"Loaded conversation context for user {self.user_id} with {len(self.history)} messages")
        except Exception as e:
            logger.error(f"Error loading conversation context for user {self.user_id}: {e}")

    def _add_to_history(self, message: Dict[str, Any]):
        """Add a message to history and maintain max size"""
        self.history.append(message)
        self.unsaved_changes = True

        # Trim history if it exceeds the maximum size
        if len(self.history) > self.max_history:
            self.history = self.history[len(self.history) - self.max_history:]

    def _update_last_active(self):
        """Update the last active timestamp"""
        self.last_updated_at = time.time()
        self.unsaved_changes = True

    def _extract_entity_mentions(self, text: str):
        """
        Extract potential entity mentions from text

        This is a simple implementation that looks for common patterns.
        A more sophisticated implementation might use NER.
        """
        # Look for patterns like "turn on the X", "status of Y", etc.
        patterns = [
            r'turn on (?:the )?(\w+(?:\s\w+)?)',
            r'turn off (?:the )?(\w+(?:\s\w+)?)',
            r'(?:what\'s|what is|how is|status of) (?:the )?(\w+(?:\s\w+)?)',
            r'set (?:the )?(\w+(?:\s\w+)?) to'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if match and len(match) > 2:  # Avoid very short matches
                    self.record_entity_mention(match)

import re  # Add missing import


class ConversationStateManager:
    """Manages conversation state and context for users"""

    def __init__(self, data_dir: Optional[str] = None, max_history: int = 20,
                 context_timeout: int = 1800, save_interval: int = 60):
        self.contexts: Dict[str, ConversationContext] = {}
        self.last_activity: Dict[str, float] = {}
        self.context_timeout = context_timeout  # 30 minutes in seconds
        self.data_dir = data_dir
        self.max_history = max_history
        self.save_interval = save_interval
        self.last_save_time = 0
        self.save_task = None

    def get_context(self, user_id: str = "default") -> ConversationContext:
        """Get conversation context for a user"""
        # Update last activity time
        self.last_activity[user_id] = time.time()

        # Create context if it doesn't exist
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(
                user_id,
                max_history=self.max_history,
                persistent=True,
                data_dir=self.data_dir
            )

        return self.contexts[user_id]

    def create_context(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation context"""
        # Generate user ID if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())

        # Create new context
        self.contexts[user_id] = ConversationContext(
            user_id,
            max_history=self.max_history,
            persistent=True,
            data_dir=self.data_dir
        )
        self.last_activity[user_id] = time.time()

        return user_id

    def reset_context(self, user_id: str) -> bool:
        """Reset conversation context for a user"""
        if user_id in self.contexts:
            # Archive the old context if it's persistent
            if self.contexts[user_id].persistent and self.data_dir:
                try:
                    self._archive_context(user_id)
                except Exception as e:
                    logger.error(f"Error archiving context for user {user_id}: {e}")

            # Create a new context
            self.contexts[user_id] = ConversationContext(
                user_id,
                max_history=self.max_history,
                persistent=True,
                data_dir=self.data_dir
            )
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
            # Save context before removing
            if user_id in self.contexts and self.contexts[user_id].persistent:
                self.contexts[user_id].save()

            del self.contexts[user_id]
            del self.last_activity[user_id]

            logger.debug(f"Cleared expired context for user {user_id}")

    def save_all_contexts(self):
        """Save all contexts to disk"""
        for user_id, context in self.contexts.items():
            if context.persistent:
                context.save()

        self.last_save_time = time.time()
        logger.debug(f"Saved {len(self.contexts)} conversation contexts")

    def start_save_task(self):
        """Start the periodic context saving task"""
        if self.save_task is None:
            self.save_task = asyncio.create_task(self._save_periodically())
            logger.info("Started conversation context save task")

    def stop_save_task(self):
        """Stop the periodic context saving task"""
        if self.save_task:
            self.save_task.cancel()
            self.save_task = None

            # Save all contexts one last time
            self.save_all_contexts()
            logger.info("Stopped conversation context save task")

    async def _save_periodically(self):
        """Periodically save contexts to disk"""
        try:
            while True:
                await asyncio.sleep(self.save_interval)
                self.save_all_contexts()
        except asyncio.CancelledError:
            # Task was cancelled, do any cleanup if needed
            pass
        except Exception as e:
            logger.error(f"Error in context save task: {e}")

    def _archive_context(self, user_id: str):
        """Archive a context before resetting it"""
        if not self.data_dir:
            return

        context = self.contexts.get(user_id)
        if not context:
            return

        try:
            # Create archive directory if it doesn't exist
            archive_dir = Path(self.data_dir) / "contexts" / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Create archive filename with timestamp
            timestamp = int(time.time())
            archive_file = archive_dir / f"{user_id}_{timestamp}.json"

            # Save context data
            context_data = {
                "user_id": user_id,
                "created_at": context.created_at,
                "last_updated_at": context.last_updated_at,
                "history": context.history,
                "variables": context.variables,
                "active_intents": context.active_intents,
                "entity_mentions": context.entity_mentions,
                "archived_at": timestamp
            }

            with open(archive_file, 'w') as f:
                json.dump(context_data, f, indent=2)

            logger.debug(f"Archived conversation context for user {user_id}")
        except Exception as e:
            logger.error(f"Error archiving conversation context for user {user_id}: {e}")
            raise