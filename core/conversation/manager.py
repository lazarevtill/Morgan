"""
Conversation manager for Morgan Core Service with PostgreSQL and Redis integration
"""
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from uuid import uuid4

from shared.config.base import ServiceConfig
from shared.models.base import Message, ConversationContext
from shared.models.database import ConversationModel, MessageModel, UserPreferencesModel
from shared.utils.logging import setup_logging
from shared.utils.database import DatabaseManager, get_db_manager
from shared.utils.redis_client import RedisManager, get_redis_manager


class ConversationManager:
    """Manages conversation contexts and state with PostgreSQL and Redis"""

    def __init__(self, config: ServiceConfig, max_history: int = 50, timeout: int = 1800):
        self.config = config
        self.max_history = max_history
        self.timeout = timeout  # seconds
        self.logger = setup_logging("conversation_manager", "INFO", "logs/conversation.log")

        # Database and cache managers
        self.db: Optional[DatabaseManager] = None
        self.redis: Optional[RedisManager] = None

        # In-memory fallback storage
        self.conversations: Dict[str, ConversationContext] = {}
        self.last_accessed: Dict[str, float] = {}

        # Background task management
        self.cleanup_task: Optional[asyncio.Task] = None

        self.logger.info(f"Conversation manager initialized (max_history={max_history}, timeout={timeout}s)")

    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            self.db = await get_db_manager()
            self.redis = await get_redis_manager()
            self.logger.info("Conversation manager connected to PostgreSQL and Redis")
        except Exception as e:
            self.logger.warning(f"Failed to connect to DB/Redis, using in-memory fallback: {e}")

    async def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for a user"""
        current_time = time.time()

        # Try Redis cache first
        if self.redis:
            try:
                cached_context = await self.redis.get_conversation_context(user_id)
                if cached_context:
                    self.last_accessed[user_id] = current_time
                    return ConversationContext(**cached_context)
            except Exception as e:
                self.logger.warning(f"Redis get context error: {e}")

        # Try in-memory cache
        if user_id in self.conversations:
            self.last_accessed[user_id] = current_time
            return self.conversations[user_id]

        # Load from database
        if self.db:
            try:
                conversations = await self.db.get_user_conversations(user_id, limit=1)
                if conversations:
                    conv = conversations[0]
                    messages = await self.db.get_recent_messages(conv.conversation_id, self.max_history)
                    
                    context = ConversationContext(
                        conversation_id=conv.conversation_id,
                        user_id=conv.user_id,
                        messages=[Message(
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.created_at,
                            metadata=msg.metadata
                        ) for msg in messages]
                    )
                    
                    # Cache in Redis and memory
                    self.conversations[user_id] = context
                    self.last_accessed[user_id] = current_time
                    if self.redis:
                        await self.redis.cache_conversation_context(user_id, context.to_dict(), ttl=self.timeout)
                    
                    self.logger.info(f"Loaded conversation context for user: {user_id} from DB")
                    return context
            except Exception as e:
                self.logger.error(f"Database get context error: {e}")

        # Create new conversation context
        context = ConversationContext(
            conversation_id=f"conv_{user_id}_{int(current_time)}",
            user_id=user_id,
            messages=[]
        )

        # Save to database
        if self.db:
            try:
                conv_model = ConversationModel(
                    conversation_id=context.conversation_id,
                    user_id=user_id,
                    title=None,
                    metadata={}
                )
                await self.db.create_conversation(conv_model)
            except Exception as e:
                self.logger.error(f"Database create conversation error: {e}")

        self.conversations[user_id] = context
        self.last_accessed[user_id] = current_time

        # Cache in Redis
        if self.redis:
            try:
                await self.redis.cache_conversation_context(user_id, context.to_dict(), ttl=self.timeout)
            except Exception as e:
                self.logger.warning(f"Redis cache context error: {e}")

        self.logger.info(f"Created new conversation context for user: {user_id}")
        return context

    async def reset_context(self, user_id: str):
        """Reset conversation context for a user"""
        # Remove from memory
        if user_id in self.conversations:
            del self.conversations[user_id]
            del self.last_accessed[user_id]

        # Remove from Redis
        if self.redis:
            try:
                await self.redis.delete(f"conversation:{user_id}")
            except Exception as e:
                self.logger.warning(f"Redis delete context error: {e}")

        # Mark as inactive in database (keep history)
        if self.db:
            try:
                conversations = await self.db.get_user_conversations(user_id, limit=1)
                if conversations:
                    await self.db.update_conversation(
                        conversations[0].conversation_id,
                        is_active=False
                    )
            except Exception as e:
                self.logger.error(f"Database reset context error: {e}")

        self.logger.info(f"Reset conversation context for user: {user_id}")

    async def add_message(self, user_id: str, message: Message):
        """Add a message to conversation context"""
        context = await self.get_context(user_id)
        context.add_message(message)

        # Trim history if too long
        if len(context.messages) > self.max_history:
            context.messages = context.messages[-self.max_history:]

        # Save to database
        if self.db:
            try:
                # Get conversation UUID from DB
                conv = await self.db.get_conversation(context.conversation_id)
                if conv:
                    msg_model = MessageModel(
                        conversation_id=conv.id,
                        role=message.role,
                        content=message.content,
                        sequence_number=len(context.messages),
                        metadata=message.metadata or {}
                    )
                    await self.db.add_message(msg_model)
            except Exception as e:
                self.logger.error(f"Database add message error: {e}")

        # Update Redis cache
        if self.redis:
            try:
                await self.redis.cache_conversation_context(user_id, context.to_dict(), ttl=self.timeout)
            except Exception as e:
                self.logger.warning(f"Redis update context error: {e}")

        self.logger.debug(f"Added message to conversation {user_id}: {message.role}")

    async def get_recent_messages(self, user_id: str, count: int = 10) -> list:
        """Get recent messages from conversation"""
        context = await self.get_context(user_id)
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

    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferencesModel]:
        """Get user preferences from cache or database"""
        # Try Redis cache first
        if self.redis:
            try:
                cached_prefs = await self.redis.get_user_preferences(user_id)
                if cached_prefs:
                    return UserPreferencesModel(**cached_prefs)
            except Exception as e:
                self.logger.warning(f"Redis get preferences error: {e}")

        # Load from database
        if self.db:
            try:
                prefs = await self.db.get_user_preferences(user_id)
                if prefs:
                    # Cache in Redis
                    if self.redis:
                        await self.redis.cache_user_preferences(user_id, prefs.dict())
                    return prefs
            except Exception as e:
                self.logger.error(f"Database get preferences error: {e}")

        return None

    async def update_user_preferences(self, preferences: UserPreferencesModel):
        """Update user preferences"""
        # Save to database
        if self.db:
            try:
                prefs = await self.db.upsert_user_preferences(preferences)
                # Update Redis cache
                if self.redis:
                    await self.redis.cache_user_preferences(preferences.user_id, prefs.dict())
                return prefs
            except Exception as e:
                self.logger.error(f"Database update preferences error: {e}")
        
        return None

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
                # Sync in-memory cache with Redis periodically
                if self.redis:
                    await self._sync_with_redis()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)

    async def _sync_with_redis(self):
        """Sync in-memory cache with Redis"""
        try:
            for user_id, context in self.conversations.items():
                await self.redis.cache_conversation_context(user_id, context.to_dict(), ttl=self.timeout)
        except Exception as e:
            self.logger.error(f"Redis sync error: {e}")
