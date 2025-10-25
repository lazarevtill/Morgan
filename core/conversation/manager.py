"""
Conversation manager for Morgan Core Service
"""
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from shared.config.base import ServiceConfig
from shared.models.base import Message, ConversationContext
from shared.utils.logging import setup_logging


class ConversationManager:
    """Manages conversation contexts and state"""

    def __init__(self, config: ServiceConfig, max_history: int = 50, timeout: int = 1800):
        self.config = config
        self.max_history = max_history
        self.timeout = timeout  # seconds
        self.logger = setup_logging("conversation_manager", "INFO", "logs/conversation.log")

        # In-memory conversation storage (can be replaced with Redis/DB)
        self.conversations: Dict[str, ConversationContext] = {}
        self.last_accessed: Dict[str, float] = {}

        # Background task management
        self.cleanup_task: Optional[asyncio.Task] = None

        self.logger.info(f"Conversation manager initialized (max_history={max_history}, timeout={timeout}s)")

    def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for a user"""
        current_time = time.time()

        if user_id in self.conversations:
            # Update access time
            self.last_accessed[user_id] = current_time
            return self.conversations[user_id]
        else:
            # Create new conversation context
            context = ConversationContext(
                conversation_id=f"conv_{user_id}_{int(current_time)}",
                user_id=user_id,
                messages=[]
            )

            self.conversations[user_id] = context
            self.last_accessed[user_id] = current_time

            self.logger.info(f"Created new conversation context for user: {user_id}")
            return context

    def reset_context(self, user_id: str):
        """Reset conversation context for a user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            del self.last_accessed[user_id]
            self.logger.info(f"Reset conversation context for user: {user_id}")
        else:
            self.logger.warning(f"No conversation context found for user: {user_id}")

    def add_message(self, user_id: str, message: Message):
        """Add a message to conversation context"""
        context = self.get_context(user_id)
        context.add_message(message)

        # Trim history if too long
        if len(context.messages) > self.max_history:
            context.messages = context.messages[-self.max_history:]

        self.logger.debug(f"Added message to conversation {user_id}: {message.role}")

    def get_recent_messages(self, user_id: str, count: int = 10) -> list:
        """Get recent messages from conversation"""
        context = self.get_context(user_id)
        return context.get_last_n_messages(count)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())

        # Count active conversations (accessed within timeout)
        current_time = time.time()
        active_conversations = sum(
            1 for last_access in self.last_accessed.values()
            if current_time - last_access < self.timeout
        )

        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages,
            "average_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0,
            "oldest_conversation": min(self.last_accessed.values()) if self.last_accessed else None,
            "newest_conversation": max(self.last_accessed.values()) if self.last_accessed else None
        }

    def cleanup_expired(self):
        """Clean up expired conversation contexts"""
        current_time = time.time()
        expired_users = []

        for user_id, last_access in self.last_accessed.items():
            if current_time - last_access > self.timeout:
                expired_users.append(user_id)

        for user_id in expired_users:
            if user_id in self.conversations:
                del self.conversations[user_id]
            del self.last_accessed[user_id]

        if expired_users:
            self.logger.info(f"Cleaned up {len(expired_users)} expired conversations: {expired_users}")

    def save_conversation(self, user_id: str, file_path: Optional[str] = None):
        """Save conversation to file (for debugging/backup)"""
        if user_id not in self.conversations:
            self.logger.warning(f"No conversation found for user: {user_id}")
            return

        context = self.conversations[user_id]

        if file_path is None:
            file_path = f"data/conversations/{user_id}_{int(time.time())}.json"

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save conversation
        conversation_data = {
            "conversation_id": context.conversation_id,
            "user_id": context.user_id,
            "created_at": context.created_at.isoformat() if context.created_at else None,
            "updated_at": context.updated_at.isoformat() if context.updated_at else None,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata
                }
                for msg in context.messages
            ],
            "metadata": context.metadata
        }

        import json
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        self.logger.info(f"Saved conversation for user {user_id} to {file_path}")

    def load_conversation(self, user_id: str, file_path: str) -> bool:
        """Load conversation from file"""
        try:
            import json

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Create conversation context
            context = ConversationContext(
                conversation_id=data["conversation_id"],
                user_id=data["user_id"],
                messages=[
                    Message(
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=datetime.fromisoformat(msg["timestamp"]) if msg["timestamp"] else None,
                        metadata=msg.get("metadata")
                    )
                    for msg in data["messages"]
                ],
                created_at=datetime.fromisoformat(data["created_at"]) if data["created_at"] else None,
                updated_at=datetime.fromisoformat(data["updated_at"]) if data["updated_at"] else None,
                metadata=data.get("metadata")
            )

            self.conversations[user_id] = context
            self.last_accessed[user_id] = time.time()

            self.logger.info(f"Loaded conversation for user {user_id} from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load conversation from {file_path}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get conversation manager status"""
        return {
            "total_conversations": len(self.conversations),
            "max_history": self.max_history,
            "timeout_seconds": self.timeout,
            "stats": self.get_conversation_stats()
        }

    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                self.cleanup_expired()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)
