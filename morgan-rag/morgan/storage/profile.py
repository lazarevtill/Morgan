"""
Profile Storage - User profiles and preferences storage

Provides storage for user profiles, preferences, and relationship data.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, List, Optional
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User's learned preferences and interests."""
    topics_of_interest: List[str]
    communication_style: str  # formal, casual, technical, friendly
    preferred_response_length: str  # brief, detailed, comprehensive
    learning_goals: List[str]
    personal_context: Dict[str, Any]
    language_preference: str = "en"
    timezone: Optional[str] = None


@dataclass
class RelationshipMilestone:
    """Significant moments in the user-Morgan relationship."""
    milestone_type: str  # first_conversation, breakthrough_moment, etc.
    description: str
    timestamp: datetime
    emotional_significance: float
    related_memories: List[str]  # memory IDs


@dataclass
class CompanionProfile:
    """Complete companion relationship profile."""
    user_id: str
    relationship_duration: timedelta
    interaction_count: int
    preferred_name: str  # what user likes to be called
    communication_preferences: UserPreferences
    emotional_patterns: Dict[str, Any]
    shared_memories: List[str]  # memory IDs
    relationship_milestones: List[RelationshipMilestone]
    last_interaction: datetime
    created_at: datetime
    updated_at: datetime


class ProfileStorage:
    """
    Profile storage following KISS principles.
    
    Single responsibility: Manage user profiles and preferences storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_dir = Path(config.get('storage_dir', './data/profiles'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.profiles_file = self.storage_dir / 'companion_profiles.json'
        self.preferences_file = self.storage_dir / 'user_preferences.json'
        self.milestones_file = self.storage_dir / 'milestones.jsonl'
        
        # In-memory caches for performance
        self._profile_cache: Dict[str, CompanionProfile] = {}
        self._preferences_cache: Dict[str, UserPreferences] = {}
        
        # Load existing data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load existing data from storage files."""
        try:
            # Load companion profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    for user_id, profile_data in profiles_data.items():
                        # Convert datetime strings back to datetime objects
                        profile_data['relationship_duration'] = timedelta(
                            seconds=profile_data['relationship_duration'])
                        profile_data['last_interaction'] = (
                            datetime.fromisoformat(
                                profile_data['last_interaction']))
                        profile_data['created_at'] = datetime.fromisoformat(
                            profile_data['created_at'])
                        profile_data['updated_at'] = datetime.fromisoformat(
                            profile_data['updated_at'])
                        
                        # Convert preferences
                        if 'communication_preferences' in profile_data:
                            profile_data['communication_preferences'] = (
                                UserPreferences(
                                    **profile_data['communication_preferences']))
                            
                        # Convert milestones
                        milestones = []
                        for milestone_data in profile_data.get(
                                'relationship_milestones', []):
                            milestone_data['timestamp'] = (
                                datetime.fromisoformat(
                                    milestone_data['timestamp']))
                            milestones.append(
                                RelationshipMilestone(**milestone_data))
                        profile_data['relationship_milestones'] = milestones
                        
                        self._profile_cache[user_id] = CompanionProfile(**profile_data)
                        
            # Load user preferences
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    preferences_data = json.load(f)
                    for user_id, pref_data in preferences_data.items():
                        self._preferences_cache[user_id] = UserPreferences(
                            **pref_data)
                        
            logger.info("Loaded %d companion profiles and %d preference sets",
                       len(self._profile_cache), len(self._preferences_cache))
            
        except Exception as e:
            logger.error("Error loading profile data: %s", e)
            
    def store_companion_profile(self, profile: CompanionProfile) -> bool:
        """
        Store or update a companion profile.
        
        Args:
            profile: CompanionProfile to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Update cache
            self._profile_cache[profile.user_id] = profile
            
            # Prepare data for JSON serialization
            profiles_data = {}
            for user_id, user_profile in self._profile_cache.items():
                profile_data = asdict(user_profile)
                
                # Convert datetime objects to ISO strings
                profile_data['relationship_duration'] = (
                    user_profile.relationship_duration.total_seconds())
                profile_data['last_interaction'] = (
                    user_profile.last_interaction.isoformat())
                profile_data['created_at'] = (
                    user_profile.created_at.isoformat())
                profile_data['updated_at'] = (
                    user_profile.updated_at.isoformat())
                
                # Convert milestones
                milestones_data = []
                for milestone in user_profile.relationship_milestones:
                    milestone_data = asdict(milestone)
                    milestone_data['timestamp'] = milestone.timestamp.isoformat()
                    milestones_data.append(milestone_data)
                profile_data['relationship_milestones'] = milestones_data
                
                profiles_data[user_id] = profile_data
                
            # Save to file
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
                
            logger.debug("Stored companion profile for user: %s",
                        profile.user_id)
            return True
            
        except Exception as e:
            logger.error("Error storing companion profile for user %s: %s",
                        profile.user_id, e)
            return False
            
    def get_companion_profile(self, user_id: str) -> Optional[CompanionProfile]:
        """
        Get a companion profile.
        
        Args:
            user_id: User ID
            
        Returns:
            CompanionProfile if found, None otherwise
        """
        return self._profile_cache.get(user_id)
        
    def store_user_preferences(self, user_id: str, preferences: UserPreferences) -> bool:
        """
        Store or update user preferences.
        
        Args:
            user_id: User ID
            preferences: UserPreferences to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Update cache
            self._preferences_cache[user_id] = preferences
            
            # Prepare data for JSON serialization
            preferences_data = {}
            for uid, prefs in self._preferences_cache.items():
                preferences_data[uid] = asdict(prefs)
                
            # Save to file
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(preferences_data, f, indent=2, ensure_ascii=False)
                
            logger.debug("Stored preferences for user: %s", user_id)
            return True
            
        except Exception as e:
            logger.error("Error storing preferences for user %s: %s", user_id, e)
            return False
            
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            UserPreferences if found, None otherwise
        """
        return self._preferences_cache.get(user_id)
        
    def add_relationship_milestone(self, user_id: str, milestone: RelationshipMilestone) -> bool:
        """
        Add a relationship milestone for a user.
        
        Args:
            user_id: User ID
            milestone: RelationshipMilestone to add
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Get or create companion profile
            profile = self.get_companion_profile(user_id)
            if not profile:
                # Create new profile
                profile = CompanionProfile(
                    user_id=user_id,
                    relationship_duration=timedelta(0),
                    interaction_count=0,
                    preferred_name=user_id,
                    communication_preferences=UserPreferences(
                        topics_of_interest=[],
                        communication_style="friendly",
                        preferred_response_length="detailed",
                        learning_goals=[],
                        personal_context={}
                    ),
                    emotional_patterns={},
                    shared_memories=[],
                    relationship_milestones=[],
                    last_interaction=datetime.now(),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
            # Add milestone
            profile.relationship_milestones.append(milestone)
            profile.updated_at = datetime.now()
            
            # Store updated profile
            success = self.store_companion_profile(profile)
            
            if success:
                # Also log milestone to separate file
                milestone_data = asdict(milestone)
                milestone_data['timestamp'] = milestone.timestamp.isoformat()
                milestone_data['user_id'] = user_id
                
                with open(self.milestones_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(milestone_data) + '\n')
                    
            return success
            
        except Exception as e:
            logger.error("Error adding milestone for user %s: %s", user_id, e)
            return False
            
    def get_relationship_milestones(self, user_id: str, limit: int = 10) -> List[RelationshipMilestone]:
        """
        Get relationship milestones for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of milestones to return
            
        Returns:
            List of relationship milestones
        """
        try:
            profile = self.get_companion_profile(user_id)
            if profile:
                # Sort by timestamp (most recent first)
                milestones = sorted(
                    profile.relationship_milestones,
                    key=lambda x: x.timestamp,
                    reverse=True
                )
                return milestones[:limit]
            
            return []
                
        except Exception as e:
            logger.error("Error getting milestones for user %s: %s", user_id, e)
            return []
            
    def update_interaction_stats(self, user_id: str) -> bool:
        """
        Update interaction statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            profile = self.get_companion_profile(user_id)
            if profile:
                # Update stats
                profile.interaction_count += 1
                current_time = datetime.now()
                profile.relationship_duration = (
                    current_time - profile.created_at)
                profile.last_interaction = current_time
                profile.updated_at = current_time
                
                return self.store_companion_profile(profile)
            
            # Create new profile for first interaction
            profile = CompanionProfile(
                user_id=user_id,
                relationship_duration=timedelta(0),
                interaction_count=1,
                preferred_name=user_id,
                communication_preferences=UserPreferences(
                    topics_of_interest=[],
                    communication_style="friendly",
                    preferred_response_length="detailed",
                    learning_goals=[],
                    personal_context={}
                ),
                emotional_patterns={},
                shared_memories=[],
                relationship_milestones=[],
                last_interaction=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return self.store_companion_profile(profile)
                
        except Exception as e:
            logger.error("Error updating interaction stats for user %s: %s",
                        user_id, e)
            return False
            
    def list_users(self) -> List[str]:
        """List all user IDs with profiles."""
        return list(self._profile_cache.keys())
        
    def get_profile_stats(self) -> Dict[str, Any]:
        """
        Get profile storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                'total_companion_profiles': len(self._profile_cache),
                'total_preference_sets': len(self._preferences_cache),
                'users_with_both': len(
                    set(self._profile_cache.keys()) &
                    set(self._preferences_cache.keys()))
            }
            
            # Calculate average relationship duration
            if self._profile_cache:
                avg_duration = sum(
                    profile.relationship_duration.total_seconds()
                    for profile in self._profile_cache.values()
                ) / len(self._profile_cache)
                stats['avg_relationship_duration_days'] = (
                    avg_duration / (24 * 3600))
                
            # Count total milestones
            total_milestones = sum(
                len(profile.relationship_milestones)
                for profile in self._profile_cache.values()
            )
            stats['total_milestones'] = total_milestones
            
            return stats
            
        except Exception as e:
            logger.error("Error getting profile stats: %s", e)
            return {'error': str(e)}