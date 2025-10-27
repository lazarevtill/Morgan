"""
Conversation manager for Morgan Core Service with PostgreSQL and Redis
"""
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from uuid import UUID

from shared.config.base import ServiceConfig
from shared.models.base import Message, ConversationContext
from shared.models.database import ConversationModel, MessageModel
from shared.utils.logging import setup_logging
from shared.utils.database import DatabaseClient
from shared.utils.redis_client import RedisClient


class ConversationManager:
    """Manages conversation contexts and state with PostgreSQL and Redis"""

    def __init__(
        self,
        config: ServiceConfig,
        db_client: Optional[DatabaseClient] = None,
        redis_client: Optional[RedisClient] = None,
        max_history: int = 50,
        timeout: int = 1800
    ):
        self.config = config
        self.max_history = max_history
        self.timeout = timeout  # seconds
        self.logger = setup_logging("conversation_manager", "INFO", "logs/conversation.log")

        # Database clients
        self.db = db_client
        self.redis = redis_client

        # In-memory fallback cache
        self.conversations: Dict[str, ConversationContext] = {}
        self.last_accessed: Dict[str, float] = {}

        # Background task management
        self.cleanup_task: Optional[asyncio.Task] = None

        self.logger.info(
            f"Conversation manager initialized (max_history={max_history}, timeout={timeout}s, "
            f"db={'enabled' if db_client else 'disabled'}, redis={'enabled' if redis_client else 'disabled'})"
        )

    async def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for a user"""
        current_time = time.time()

        # Try Redis cache first
        if self.redis:
            try:
                cached_context = await self.redis.get_json(f"conv:context:{user_id}")
                if cached_context:
                    self.last_accessed[user_id] = current_time
                    # Convert to ConversationContext
                    messages = [Message(**msg) for msg in cached_context.get("messages", [])]
                    context = ConversationContext(
                        conversation_id=cached_context["conversation_id"],
                        user_id=user_id,
                        messages=messages
                    )
                    self.conversations[user_id] = context
                    return context
            except Exception as e:
                self.logger.warning(f"Redis cache miss for user {user_id}: {e}")

        # Check memory cache
        if user_id in self.conversations:
            self.last_accessed[user_id] = current_time
            return self.conversations[user_id]

        # Try to load from database
        if self.db:
            try:
                # Get or create conversation in DB
                conversations = await self.db.get_user_conversations(user_id, limit=1)
                if conversations:
                    conv = conversations[0]
                    # Load messages
                    messages_db = await self.db.get_recent_messages(conv.id, self.max_history)
                    messages = [
                        Message(
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.created_at,
                            metadata=msg.metadata
                        )
                        for msg in messages_db
                    ]
                    
                    context = ConversationContext(
                        conversation_id=conv.conversation_id,
                        user_id=user_id,
                        messages=messages
                    )
                else:
                    # Create new conversation in DB
                    conversation_id = f"conv_{user_id}_{int(current_time)}"
                    conv_model = ConversationModel(
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    await self.db.create_conversation(conv_model)
                    
                    context = ConversationContext(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        messages=[]
                    )
                
                # Cache in memory and Redis
                self.conversations[user_id] = context
                self.last_accessed[user_id] = current_time
                
                if self.redis:
                    await self._cache_context_to_redis(user_id, context)
                
                self.logger.info(f"Loaded/created conversation from DB for user: {user_id}")
                return context
                
            except Exception as e:
                self.logger.error(f"Database error for user {user_id}: {e}")

        # Fallback: create in-memory only
        context = ConversationContext(
            conversation_id=f"conv_{user_id}_{int(current_time)}",
            user_id=user_id,
            messages=[]
        )
        self.conversations[user_id] = context
        self.last_accessed[user_id] = current_time
        self.logger.info(f"Created in-memory conversation for user: {user_id}")
        return context

    async def reset_context(self, user_id: str):
        """Reset conversation context for a user"""
        # Clear from memory
        if user_id in self.conversations:
            del self.conversations[user_id]
            del self.last_accessed[user_id]
        
        # Clear from Redis
        if self.redis:
            try:
                await self.redis.delete(f"conv:context:{user_id}")
            except Exception as e:
                self.logger.warning(f"Failed to clear Redis cache for user {user_id}: {e}")
        
        # Mark as inactive in database
        if self.db:
            try:
                conversations = await self.db.get_user_conversations(user_id, limit=1)
                if conversations:
                    await self.db.update_conversation(
                        conversations[0].conversation_id,
                        is_active=False
                    )
            except Exception as e:
                self.logger.warning(f"Failed to update DB for user {user_id}: {e}")
        
        self.logger.info(f"Reset conversation context for user: {user_id}")

    async def add_message(self, user_id: str, message: Message):
        """Add a message to conversation context"""
        context = await self.get_context(user_id)
        context.add_message(message)

        # Trim history if too long
        if len(context.messages) > self.max_history:
            context.messages = context.messages[-self.max_history:]

        # Persist to database
        if self.db:
            try:
                conv = await self.db.get_conversation(context.conversation_id)
                if conv:
                    message_model = MessageModel(
                        conversation_id=conv.id,
                        role=message.role,
                        content=message.content,
                        sequence_number=len(context.messages),
                        metadata=message.metadata
                    )
                    await self.db.add_message(message_model)
            except Exception as e:
                self.logger.error(f"Failed to persist message to DB: {e}")

        # Update Redis cache
        if self.redis:
            try:
                await self._cache_context_to_redis(user_id, context)
            except Exception as e:
                self.logger.warning(f"Failed to update Redis cache: {e}")

        self.logger.debug(f"Added message to conversation {user_id}: {message.role}")

    async def get_recent_messages(self, user_id: str, count: int = 10) -> list:
        """Get recent messages from conversation"""
        context = await self.get_context(user_id)
        return context.get_last_n_messages(count)

    async def _cache_context_to_redis(self, user_id: str, context: ConversationContext):
        """Cache conversation context to Redis"""
        if not self.redis:
            return
        
        try:
            cache_data = {
                "conversation_id": context.conversation_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                        "metadata": msg.metadata
                    }
                    for msg in context.messages[-self.max_history:]  # Only cache recent messages
                ]
            }
            await self.redis.set_json(
                f"conv:context:{user_id}",
                cache_data,
                expire=self.timeout
            )
        except Exception as e:
            self.logger.error(f"Failed to cache context to Redis: {e}")

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

    def get_status(self) -> Dict[str, Any]:
        """Get conversation manager status"""
        return {
            "total_conversations": len(self.conversations),
            "max_history": self.max_history,
            "timeout_seconds": self.timeout,
            "stats": self.get_conversation_stats(),
            "db_connected": self.db is not None,
            "redis_connected": self.redis is not None
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

