"""
Memory Storage - Conversation and emotional memory storage

Provides storage for conversation memories and emotional context.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, List, Optional
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """Simple conversation memory representation."""
    id: str
    content: str
    importance_score: float
    entities: List[str]
    concepts: List[str]
    conversation_id: str
    user_id: str
    timestamp: datetime
    feedback_rating: Optional[int] = None
    emotional_context: Optional[Dict[str, Any]] = None


@dataclass
class EmotionalState:
    """User's emotional state representation."""
    user_id: str
    primary_emotion: str
    intensity: float
    confidence: float
    secondary_emotions: List[str]
    emotional_indicators: List[str]
    timestamp: datetime
    context: Optional[str] = None


@dataclass
class UserProfile:
    """User profile with preferences and patterns."""
    user_id: str
    preferred_name: str
    communication_style: str
    topics_of_interest: List[str]
    learning_goals: List[str]
    interaction_patterns: Dict[str, Any]
    emotional_patterns: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MemoryStorage:
    """
    Memory storage following KISS principles.
    
    Single responsibility: Manage conversation and emotional memory storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_dir = Path(config.get('storage_dir', './data/memory'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.memories_file = self.storage_dir / 'memories.jsonl'
        self.emotions_file = self.storage_dir / 'emotions.jsonl'
        self.profiles_file = self.storage_dir / 'profiles.json'
        
        # In-memory caches for performance
        self._memory_cache: Dict[str, ConversationMemory] = {}
        self._profile_cache: Dict[str, UserProfile] = {}
        
        # Load existing data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load existing data from storage files."""
        try:
            # Load profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    for user_id, profile_data in profiles_data.items():
                        profile_data['created_at'] = datetime.fromisoformat(
                            profile_data['created_at'])
                        profile_data['updated_at'] = datetime.fromisoformat(
                            profile_data['updated_at'])
                        self._profile_cache[user_id] = UserProfile(
                            **profile_data)
                        
            # Load recent memories into cache (last 1000)
            if self.memories_file.exists():
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Load last 1000 memories
                    for line in lines[-1000:]:
                        memory_data = json.loads(line.strip())
                        memory_data['timestamp'] = datetime.fromisoformat(
                            memory_data['timestamp'])
                        memory = ConversationMemory(**memory_data)
                        self._memory_cache[memory.id] = memory
                        
            logger.info("Loaded %d profiles and %d memories",
                       len(self._profile_cache), len(self._memory_cache))
            
        except Exception as e:
            logger.error("Error loading memory data: %s", e)
            
    def store_memory(self, memory: ConversationMemory) -> bool:
        """
        Store a conversation memory.
        
        Args:
            memory: ConversationMemory to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Add to cache
            self._memory_cache[memory.id] = memory
            
            # Append to file
            memory_data = asdict(memory)
            memory_data['timestamp'] = memory.timestamp.isoformat()
            
            with open(self.memories_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory_data) + '\n')
                
            logger.debug("Stored memory: %s", memory.id)
            return True
            
        except Exception as e:
            logger.error("Error storing memory %s: %s", memory.id, e)
            return False
            
    def get_memory(self, memory_id: str) -> Optional[ConversationMemory]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            ConversationMemory if found, None otherwise
        """
        try:
            # Check cache first
            if memory_id in self._memory_cache:
                return self._memory_cache[memory_id]
                
            # Search in file if not in cache
            if self.memories_file.exists():
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        memory_data = json.loads(line.strip())
                        if memory_data['id'] == memory_id:
                            memory_data['timestamp'] = datetime.fromisoformat(
                                memory_data['timestamp'])
                            memory = ConversationMemory(**memory_data)
                            # Add to cache
                            self._memory_cache[memory_id] = memory
                            return memory
                            
            return None
            
        except Exception as e:
            logger.error("Error retrieving memory %s: %s", memory_id, e)
            return None
            
    def search_memories(self, user_id: str, query: Optional[str] = None,
                       limit: int = 10,
                       min_importance: float = 0.0) -> List[ConversationMemory]:
        """
        Search memories for a user.
        
        Args:
            user_id: User ID to search for
            query: Optional text query
            limit: Maximum number of results
            min_importance: Minimum importance score
            
        Returns:
            List of matching memories
        """
        try:
            results = []
            
            # Search in cache first
            for memory in self._memory_cache.values():
                if (memory.user_id == user_id and
                    memory.importance_score >= min_importance):
                    
                    if (query is None or
                        query.lower() in memory.content.lower()):
                        results.append(memory)
                        
            # If we need more results, search in file
            if len(results) < limit and self.memories_file.exists():
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                            
                        memory_data = json.loads(line.strip())
                        if (memory_data['user_id'] == user_id and
                            memory_data['importance_score'] >= min_importance and
                            memory_data['id'] not in self._memory_cache):
                            
                            if (query is None or
                                query.lower() in memory_data['content'].lower()):
                                memory_data['timestamp'] = datetime.fromisoformat(
                                    memory_data['timestamp'])
                                memory = ConversationMemory(**memory_data)
                                results.append(memory)
                                
            # Sort by importance and timestamp
            results.sort(key=lambda x: (x.importance_score, x.timestamp),
                        reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error("Error searching memories for user %s: %s", user_id, e)
            return []
            
    def store_emotional_state(self, emotional_state: EmotionalState) -> bool:
        """
        Store an emotional state.
        
        Args:
            emotional_state: EmotionalState to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            emotion_data = asdict(emotional_state)
            emotion_data['timestamp'] = emotional_state.timestamp.isoformat()
            
            with open(self.emotions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(emotion_data) + '\n')
                
            logger.debug("Stored emotional state for user: %s",
                        emotional_state.user_id)
            return True
            
        except Exception as e:
            logger.error("Error storing emotional state: %s", e)
            return False
            
    def get_recent_emotions(self, user_id: str, limit: int = 10) -> List[EmotionalState]:
        """
        Get recent emotional states for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            
        Returns:
            List of recent emotional states
        """
        try:
            emotions = []
            
            if self.emotions_file.exists():
                with open(self.emotions_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Search from end of file (most recent)
                    for line in reversed(lines):
                        if len(emotions) >= limit:
                            break
                            
                        emotion_data = json.loads(line.strip())
                        if emotion_data['user_id'] == user_id:
                            emotion_data['timestamp'] = datetime.fromisoformat(
                                emotion_data['timestamp'])
                            emotion = EmotionalState(**emotion_data)
                            emotions.append(emotion)
                            
            return emotions
            
        except Exception as e:
            logger.error("Error getting emotions for user %s: %s", user_id, e)
            return []
            
    def store_user_profile(self, profile: UserProfile) -> bool:
        """
        Store or update a user profile.
        
        Args:
            profile: UserProfile to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Update cache
            self._profile_cache[profile.user_id] = profile
            
            # Save all profiles to file
            profiles_data = {}
            for user_id, user_profile in self._profile_cache.items():
                profile_data = asdict(user_profile)
                profile_data['created_at'] = user_profile.created_at.isoformat()
                profile_data['updated_at'] = user_profile.updated_at.isoformat()
                profiles_data[user_id] = profile_data
                
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
                
            logger.debug("Stored profile for user: %s", profile.user_id)
            return True
            
        except Exception as e:
            logger.error("Error storing profile for user %s: %s",
                        profile.user_id, e)
            return False
            
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile if found, None otherwise
        """
        return self._profile_cache.get(user_id)
        
    def list_users(self) -> List[str]:
        """List all user IDs with profiles."""
        return list(self._profile_cache.keys())
        
    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory storage statistics.
        
        Args:
            user_id: Optional user ID to get stats for specific user
            
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                'total_profiles': len(self._profile_cache),
                'cached_memories': len(self._memory_cache)
            }
            
            if user_id:
                user_memories = [m for m in self._memory_cache.values()
                               if m.user_id == user_id]
                stats['user_memories'] = len(user_memories)
                if user_memories:
                    stats['avg_importance'] = (
                        sum(m.importance_score for m in user_memories) /
                        len(user_memories))
                else:
                    stats['avg_importance'] = 0
                
            # Count total memories in file
            if self.memories_file.exists():
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    stats['total_memories'] = sum(1 for _ in f)
            else:
                stats['total_memories'] = 0
                
            return stats
            
        except Exception as e:
            logger.error("Error getting memory stats: %s", e)
            return {'error': str(e)}