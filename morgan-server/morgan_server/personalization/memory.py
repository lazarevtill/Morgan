"""
Memory System module for the Personalization Layer.

This module provides conversation memory management including:
- Conversation history storage
- Long-term memory across sessions
- Context retrieval from memory
- Memory summarization
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict


class MessageRole(str, Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    A single message in a conversation.
    
    Attributes:
        role: Role of the message sender
        content: Message content
        timestamp: When the message was sent
        metadata: Additional message metadata
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """
    A conversation containing multiple messages.
    
    Attributes:
        conversation_id: Unique conversation identifier
        user_id: User identifier
        messages: List of messages in the conversation
        created_at: When the conversation was created
        last_updated: When the conversation was last updated
        summary: Optional conversation summary
        metadata: Additional conversation metadata
    """
    conversation_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "summary": self.summary,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            messages=[
                Message.from_dict(msg) for msg in data.get("messages", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            summary=data.get("summary"),
            metadata=data.get("metadata", {})
        )
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: Role of the message sender
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Created Message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_updated = datetime.now()
        return message
    
    def get_messages(
        self,
        role: Optional[MessageRole] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages from the conversation.
        
        Args:
            role: Optional role filter
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        messages = self.messages
        
        # Filter by role if specified
        if role is not None:
            messages = [msg for msg in messages if msg.role == role]
        
        # Apply limit if specified (most recent messages)
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def get_context_window(
        self, max_messages: int = 10
    ) -> List[Message]:
        """
        Get recent messages for context window.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        return self.messages[-max_messages:]
    
    def message_count(self) -> int:
        """Get total number of messages in conversation."""
        return len(self.messages)


class MemoryManager:
    """
    Memory management system for conversation history.
    
    This class provides:
    - Conversation history storage
    - Long-term memory across sessions
    - Context retrieval from memory
    - Memory summarization
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_context_messages: int = 10
    ):
        """
        Initialize the memory manager.
        
        Args:
            storage_dir: Directory for memory storage
                (default: ./data/memory)
            max_context_messages: Maximum messages in context window
                (default: 10)
        """
        self.storage_dir = Path(storage_dir or "./data/memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_context_messages = max_context_messages
        self.conversations: Dict[str, Conversation] = {}
        self.user_conversations: Dict[str, List[str]] = defaultdict(list)
        self._load_all_conversations()
    
    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            metadata: Optional conversation metadata
            
        Returns:
            Created Conversation
            
        Raises:
            ValueError: If conversation already exists
        """
        if conversation_id in self.conversations:
            raise ValueError(
                f"Conversation already exists: {conversation_id}"
            )
        
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id] = conversation
        self.user_conversations[user_id].append(conversation_id)
        self._save_conversation(conversation)
        
        return conversation
    
    def get_conversation(
        self, conversation_id: str
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation or None if not found
        """
        return self.conversations.get(conversation_id)
    
    def get_or_create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Get existing conversation or create new one.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            metadata: Optional conversation metadata
            
        Returns:
            Conversation
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            conversation = self.create_conversation(
                conversation_id, user_id, metadata
            )
        return conversation
    
    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            role: Role of the message sender
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Created Message
            
        Raises:
            ValueError: If conversation doesn't exist
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(
                f"Conversation not found: {conversation_id}"
            )
        
        message = conversation.add_message(role, content, metadata)
        self._save_conversation(conversation)
        
        return message
    
    def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """
        Get recent messages for context.
        
        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum messages to return
                (default: uses max_context_messages)
            
        Returns:
            List of recent messages
            
        Raises:
            ValueError: If conversation doesn't exist
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(
                f"Conversation not found: {conversation_id}"
            )
        
        limit = max_messages or self.max_context_messages
        return conversation.get_context_window(limit)
    
    def get_user_conversations(
        self, user_id: str
    ) -> List[Conversation]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversations
        """
        conversation_ids = self.user_conversations.get(user_id, [])
        conversations = []
        
        for conv_id in conversation_ids:
            conversation = self.get_conversation(conv_id)
            if conversation is not None:
                conversations.append(conversation)
        
        return conversations
    
    def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Tuple[Conversation, Message, float]]:
        """
        Search conversations for matching messages.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of (conversation, message, relevance_score) tuples
        """
        results = []
        query_lower = query.lower()
        
        # Get all user conversations
        conversations = self.get_user_conversations(user_id)
        
        # Search through messages
        for conversation in conversations:
            for message in conversation.messages:
                # Simple relevance scoring based on query presence
                content_lower = message.content.lower()
                if query_lower in content_lower:
                    # Calculate simple relevance score
                    # (could be enhanced with more sophisticated scoring)
                    score = content_lower.count(query_lower) / len(
                        content_lower.split()
                    )
                    results.append((conversation, message, score))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top results
        return results[:limit]
    
    def summarize_conversation(
        self, conversation_id: str
    ) -> str:
        """
        Generate a summary of a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation summary
            
        Raises:
            ValueError: If conversation doesn't exist
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(
                f"Conversation not found: {conversation_id}"
            )
        
        # If summary already exists, return it
        if conversation.summary:
            return conversation.summary
        
        # Generate simple summary
        # (In production, this would use an LLM for better summaries)
        message_count = conversation.message_count()
        user_messages = conversation.get_messages(role=MessageRole.USER)
        assistant_messages = conversation.get_messages(
            role=MessageRole.ASSISTANT
        )
        
        # Extract key topics (simple word frequency)
        all_content = " ".join(
            msg.content for msg in conversation.messages
        )
        words = all_content.lower().split()
        # Filter out common words (simple approach)
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "was",
            "are", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "i", "you", "he", "she", "it", "we",
            "they", "this", "that", "these", "those"
        }
        filtered_words = [w for w in words if w not in common_words]
        
        # Count word frequencies
        word_freq: Dict[str, int] = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 3 words
        top_words = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )[:3]
        topics = [word for word, _ in top_words]
        
        summary = (
            f"Conversation with {message_count} messages "
            f"({len(user_messages)} from user, "
            f"{len(assistant_messages)} from assistant). "
        )
        
        if topics:
            summary += f"Topics discussed: {', '.join(topics)}."
        
        # Save summary
        conversation.summary = summary
        self._save_conversation(conversation)
        
        return summary
    
    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Args:
            user_id: Optional user ID to filter stats
            
        Returns:
            Dictionary with memory statistics
        """
        if user_id is not None:
            conversations = self.get_user_conversations(user_id)
        else:
            conversations = list(self.conversations.values())
        
        if not conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "oldest_conversation": None,
                "newest_conversation": None
            }
        
        total_messages = sum(
            conv.message_count() for conv in conversations
        )
        oldest = min(conversations, key=lambda c: c.created_at)
        newest = max(conversations, key=lambda c: c.created_at)
        
        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "oldest_conversation": oldest.created_at.isoformat(),
            "newest_conversation": newest.created_at.isoformat()
        }
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            True if deleted, False if not found
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return False
        
        # Remove from user conversations list
        user_id = conversation.user_id
        if user_id in self.user_conversations:
            if conversation_id in self.user_conversations[user_id]:
                self.user_conversations[user_id].remove(conversation_id)
        
        # Remove from memory
        del self.conversations[conversation_id]
        
        # Remove from disk
        conv_path = self._get_conversation_path(conversation_id)
        if conv_path.exists():
            conv_path.unlink()
        
        return True
    
    def cleanup_old_conversations(
        self,
        user_id: str,
        keep_recent: int = 10
    ) -> int:
        """
        Clean up old conversations, keeping only recent ones.
        
        Args:
            user_id: User identifier
            keep_recent: Number of recent conversations to keep
            
        Returns:
            Number of conversations deleted
        """
        conversations = self.get_user_conversations(user_id)
        
        if len(conversations) <= keep_recent:
            return 0
        
        # Sort by last_updated (oldest first)
        conversations.sort(key=lambda c: c.last_updated)
        
        # Delete old conversations
        to_delete = conversations[:-keep_recent]
        deleted_count = 0
        
        for conversation in to_delete:
            if self.delete_conversation(conversation.conversation_id):
                deleted_count += 1
        
        return deleted_count
    
    def _get_conversation_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation."""
        # Sanitize conversation_id for filename
        safe_id = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in conversation_id
        )
        return self.storage_dir / f"{safe_id}.json"
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to disk."""
        conv_path = self._get_conversation_path(conversation.conversation_id)
        with open(conv_path, 'w', encoding='utf-8') as f:
            json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_conversation(
        self, conversation_id: str
    ) -> Optional[Conversation]:
        """Load conversation from disk."""
        conv_path = self._get_conversation_path(conversation_id)
        if not conv_path.exists():
            return None
        
        try:
            with open(conv_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log error but don't crash
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def _load_all_conversations(self) -> None:
        """Load all conversations from disk."""
        if not self.storage_dir.exists():
            return
        
        for conv_file in self.storage_dir.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                conversation = Conversation.from_dict(data)
                self.conversations[conversation.conversation_id] = conversation
                self.user_conversations[conversation.user_id].append(
                    conversation.conversation_id
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error but continue loading other conversations
                print(f"Error loading conversation from {conv_file}: {e}")
                continue
