"""
User Profile module for the Personalization Layer.

This module provides user profile management including:
- User profile model (name, preferences, metrics)
- Profile persistence
- Trust and engagement metrics
- Communication preferences
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class CommunicationStyle(str, Enum):
    """Communication style preferences."""
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    PLAYFUL = "playful"


class ResponseLength(str, Enum):
    """Response length preferences."""
    BRIEF = "brief"
    MODERATE = "moderate"
    DETAILED = "detailed"


@dataclass
class UserProfile:
    """
    User profile containing preferences and metrics.
    
    Attributes:
        user_id: Unique user identifier
        preferred_name: User's preferred name
        created_at: Profile creation timestamp
        last_updated: Last profile update timestamp
        communication_style: Preferred communication style
        response_length: Preferred response length
        topics_of_interest: List of topics the user is interested in
        trust_level: Trust level (0.0 to 1.0)
        engagement_score: Engagement score (0.0 to 1.0)
        relationship_age_days: Days since first interaction
        interaction_count: Total number of interactions
        metadata: Additional profile metadata
    """
    user_id: str
    preferred_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    communication_style: CommunicationStyle = CommunicationStyle.FRIENDLY
    response_length: ResponseLength = ResponseLength.MODERATE
    topics_of_interest: List[str] = field(default_factory=list)
    trust_level: float = 0.0
    engagement_score: float = 0.0
    relationship_age_days: int = 0
    interaction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "preferred_name": self.preferred_name,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "communication_style": self.communication_style.value,
            "response_length": self.response_length.value,
            "topics_of_interest": self.topics_of_interest,
            "trust_level": self.trust_level,
            "engagement_score": self.engagement_score,
            "relationship_age_days": self.relationship_age_days,
            "interaction_count": self.interaction_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create profile from dictionary."""
        return cls(
            user_id=data["user_id"],
            preferred_name=data.get("preferred_name"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            communication_style=CommunicationStyle(
                data.get("communication_style", "friendly")
            ),
            response_length=ResponseLength(
                data.get("response_length", "moderate")
            ),
            topics_of_interest=data.get("topics_of_interest", []),
            trust_level=data.get("trust_level", 0.0),
            engagement_score=data.get("engagement_score", 0.0),
            relationship_age_days=data.get("relationship_age_days", 0),
            interaction_count=data.get("interaction_count", 0),
            metadata=data.get("metadata", {})
        )


class ProfileManager:
    """
    Profile management system for user profiles.
    
    This class provides:
    - Profile creation and retrieval
    - Profile updates
    - Profile persistence
    - Metrics calculation and updates
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the profile manager.

        Args:
            storage_dir: Directory for profile storage
                (default: ./data/profiles)
        """
        self.storage_dir = Path(storage_dir or "./data/profiles")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, UserProfile] = {}
        self._load_all_profiles()
    
    def create_profile(
        self,
        user_id: str,
        preferred_name: Optional[str] = None,
        communication_style: Optional[CommunicationStyle] = None,
        response_length: Optional[ResponseLength] = None,
        topics_of_interest: Optional[List[str]] = None
    ) -> UserProfile:
        """
        Create a new user profile.
        
        Args:
            user_id: Unique user identifier
            preferred_name: User's preferred name
            communication_style: Preferred communication style
            response_length: Preferred response length
            topics_of_interest: List of topics of interest
            
        Returns:
            Created UserProfile
            
        Raises:
            ValueError: If profile already exists
        """
        if user_id in self.profiles:
            raise ValueError(f"Profile already exists for user: {user_id}")
        
        profile = UserProfile(
            user_id=user_id,
            preferred_name=preferred_name,
            communication_style=(
                communication_style or CommunicationStyle.FRIENDLY
            ),
            response_length=(
                response_length or ResponseLength.MODERATE
            ),
            topics_of_interest=topics_of_interest or []
        )
        
        self.profiles[user_id] = profile
        self._save_profile(profile)
        
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile or None if not found
        """
        return self.profiles.get(user_id)
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        Get existing profile or create new one.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile
        """
        profile = self.get_profile(user_id)
        if profile is None:
            profile = self.create_profile(user_id)
        return profile
    
    def update_profile(
        self,
        user_id: str,
        preferred_name: Optional[str] = None,
        communication_style: Optional[CommunicationStyle] = None,
        response_length: Optional[ResponseLength] = None,
        topics_of_interest: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """
        Update a user profile.
        
        Args:
            user_id: User identifier
            preferred_name: New preferred name
            communication_style: New communication style
            response_length: New response length
            topics_of_interest: New topics of interest
            metadata: Additional metadata to update
            
        Returns:
            Updated UserProfile
            
        Raises:
            ValueError: If profile doesn't exist
        """
        profile = self.get_profile(user_id)
        if profile is None:
            raise ValueError(f"Profile not found for user: {user_id}")
        
        # Update fields if provided
        if preferred_name is not None:
            profile.preferred_name = preferred_name
        if communication_style is not None:
            profile.communication_style = communication_style
        if response_length is not None:
            profile.response_length = response_length
        if topics_of_interest is not None:
            profile.topics_of_interest = topics_of_interest
        if metadata is not None:
            profile.metadata.update(metadata)
        
        profile.last_updated = datetime.now()
        self._save_profile(profile)
        
        return profile
    
    def update_metrics(
        self,
        user_id: str,
        trust_level: Optional[float] = None,
        engagement_score: Optional[float] = None,
        relationship_age_days: Optional[int] = None,
        interaction_count: Optional[int] = None
    ) -> UserProfile:
        """
        Update profile metrics.
        
        Args:
            user_id: User identifier
            trust_level: New trust level (0.0 to 1.0)
            engagement_score: New engagement score (0.0 to 1.0)
            relationship_age_days: New relationship age in days
            interaction_count: New interaction count
            
        Returns:
            Updated UserProfile
            
        Raises:
            ValueError: If profile doesn't exist or metrics are invalid
        """
        profile = self.get_profile(user_id)
        if profile is None:
            raise ValueError(f"Profile not found for user: {user_id}")
        
        # Validate and update metrics
        if trust_level is not None:
            if not 0.0 <= trust_level <= 1.0:
                raise ValueError("Trust level must be between 0.0 and 1.0")
            profile.trust_level = trust_level
        
        if engagement_score is not None:
            if not 0.0 <= engagement_score <= 1.0:
                raise ValueError(
                    "Engagement score must be between 0.0 and 1.0"
                )
            profile.engagement_score = engagement_score
        
        if relationship_age_days is not None:
            if relationship_age_days < 0:
                raise ValueError("Relationship age days must be non-negative")
            profile.relationship_age_days = relationship_age_days
        
        if interaction_count is not None:
            if interaction_count < 0:
                raise ValueError("Interaction count must be non-negative")
            profile.interaction_count = interaction_count
        
        profile.last_updated = datetime.now()
        self._save_profile(profile)
        
        return profile
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deleted, False if not found
        """
        if user_id not in self.profiles:
            return False
        
        # Remove from memory
        del self.profiles[user_id]
        
        # Remove from disk
        profile_path = self._get_profile_path(user_id)
        if profile_path.exists():
            profile_path.unlink()
        
        return True
    
    def list_profiles(self) -> List[str]:
        """
        List all user IDs with profiles.
        
        Returns:
            List of user IDs
        """
        return list(self.profiles.keys())
    
    def calculate_relationship_age(self, user_id: str) -> int:
        """
        Calculate relationship age in days.
        
        Args:
            user_id: User identifier
            
        Returns:
            Age in days, or 0 if profile not found
        """
        profile = self.get_profile(user_id)
        if profile is None:
            return 0
        
        age = (datetime.now() - profile.created_at).days
        return max(0, age)
    
    def _get_profile_path(self, user_id: str) -> Path:
        """Get file path for a profile."""
        # Sanitize user_id for filename
        safe_id = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in user_id
        )
        return self.storage_dir / f"{safe_id}.json"
    
    def _save_profile(self, profile: UserProfile) -> None:
        """Save profile to disk."""
        profile_path = self._get_profile_path(profile.user_id)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from disk."""
        profile_path = self._get_profile_path(user_id)
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return UserProfile.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log error but don't crash
            print(f"Error loading profile {user_id}: {e}")
            return None
    
    def _load_all_profiles(self) -> None:
        """Load all profiles from disk."""
        if not self.storage_dir.exists():
            return
        
        for profile_file in self.storage_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                self.profiles[profile.user_id] = profile
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error but continue loading other profiles
                print(f"Error loading profile from {profile_file}: {e}")
                continue
