"""
Preferences module for the Personalization Layer.

This module provides preference management including:
- Communication style preferences
- Response length preferences
- Topic interest tracking
- Preference learning from interactions
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import Counter


class PreferenceType(str, Enum):
    """Types of preferences that can be tracked."""
    COMMUNICATION_STYLE = "communication_style"
    RESPONSE_LENGTH = "response_length"
    TOPIC_INTEREST = "topic_interest"
    FORMALITY_LEVEL = "formality_level"


@dataclass
class InteractionFeedback:
    """
    Feedback from a single interaction.

    Attributes:
        timestamp: When the interaction occurred
        message_length: Length of user's message
        response_length: Length of assistant's response
        topics: Topics discussed in the interaction
        user_satisfaction: Optional satisfaction score (0.0 to 1.0)
        communication_style_used: Style used in the response
        metadata: Additional interaction metadata
    """
    timestamp: datetime
    message_length: int
    response_length: int
    topics: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None
    communication_style_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceScore:
    """
    Score for a specific preference value.

    Attributes:
        value: The preference value (e.g., "casual", "brief")
        score: Confidence score (0.0 to 1.0)
        count: Number of interactions supporting this preference
        last_updated: When this score was last updated
    """
    value: str
    score: float = 0.0
    count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "score": self.score,
            "count": self.count,
            "last_updated": self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceScore":
        """Create from dictionary."""
        return cls(
            value=data["value"],
            score=data["score"],
            count=data["count"],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )


@dataclass
class UserPreferences:
    """
    User preferences learned from interactions.

    Attributes:
        user_id: Unique user identifier
        communication_style_scores: Scores for different communication styles
        response_length_scores: Scores for different response lengths
        topic_interests: Topics and their interest scores
        interaction_history: Recent interaction feedback
        created_at: When preferences were first created
        last_updated: When preferences were last updated
        metadata: Additional preference metadata
    """
    user_id: str
    communication_style_scores: Dict[str, PreferenceScore] = field(
        default_factory=dict
    )
    response_length_scores: Dict[str, PreferenceScore] = field(
        default_factory=dict
    )
    topic_interests: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[InteractionFeedback] = field(
        default_factory=list
    )
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "communication_style_scores": {
                k: v.to_dict()
                for k, v in self.communication_style_scores.items()
            },
            "response_length_scores": {
                k: v.to_dict()
                for k, v in self.response_length_scores.items()
            },
            "topic_interests": self.topic_interests,
            "interaction_history": [
                {
                    "timestamp": ih.timestamp.isoformat(),
                    "message_length": ih.message_length,
                    "response_length": ih.response_length,
                    "topics": ih.topics,
                    "user_satisfaction": ih.user_satisfaction,
                    "communication_style_used": ih.communication_style_used,
                    "metadata": ih.metadata
                }
                for ih in self.interaction_history
            ],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create preferences from dictionary."""
        return cls(
            user_id=data["user_id"],
            communication_style_scores={
                k: PreferenceScore.from_dict(v)
                for k, v in data.get("communication_style_scores", {}).items()
            },
            response_length_scores={
                k: PreferenceScore.from_dict(v)
                for k, v in data.get("response_length_scores", {}).items()
            },
            topic_interests=data.get("topic_interests", {}),
            interaction_history=[
                InteractionFeedback(
                    timestamp=datetime.fromisoformat(ih["timestamp"]),
                    message_length=ih["message_length"],
                    response_length=ih["response_length"],
                    topics=ih.get("topics", []),
                    user_satisfaction=ih.get("user_satisfaction"),
                    communication_style_used=ih.get(
                        "communication_style_used"
                    ),
                    metadata=ih.get("metadata", {})
                )
                for ih in data.get("interaction_history", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {})
        )


class PreferenceManager:
    """
    Preference management system for learning user preferences.

    This class provides:
    - Preference storage and retrieval
    - Preference learning from interactions
    - Preference application and recommendations
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_history_size: int = 100
    ):
        """
        Initialize the preference manager.

        Args:
            storage_dir: Directory for preference storage
                (default: ./data/preferences)
            max_history_size: Maximum number of interactions to keep
                in history (default: 100)
        """
        self.storage_dir = Path(storage_dir or "./data/preferences")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_size = max_history_size
        self.preferences: Dict[str, UserPreferences] = {}
        self._load_all_preferences()

    def create_preferences(self, user_id: str) -> UserPreferences:
        """
        Create new user preferences.

        Args:
            user_id: Unique user identifier

        Returns:
            Created UserPreferences

        Raises:
            ValueError: If preferences already exist
        """
        if user_id in self.preferences:
            raise ValueError(
                f"Preferences already exist for user: {user_id}"
            )

        prefs = UserPreferences(user_id=user_id)
        self.preferences[user_id] = prefs
        self._save_preferences(prefs)

        return prefs

    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences or None if not found
        """
        return self.preferences.get(user_id)

    def get_or_create_preferences(self, user_id: str) -> UserPreferences:
        """
        Get existing preferences or create new ones.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences
        """
        prefs = self.get_preferences(user_id)
        if prefs is None:
            prefs = self.create_preferences(user_id)
        return prefs

    def record_interaction(
        self,
        user_id: str,
        message_length: int,
        response_length: int,
        topics: Optional[List[str]] = None,
        user_satisfaction: Optional[float] = None,
        communication_style_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserPreferences:
        """
        Record an interaction for preference learning.

        Args:
            user_id: User identifier
            message_length: Length of user's message
            response_length: Length of assistant's response
            topics: Topics discussed in the interaction
            user_satisfaction: Optional satisfaction score (0.0 to 1.0)
            communication_style_used: Style used in the response
            metadata: Additional interaction metadata

        Returns:
            Updated UserPreferences

        Raises:
            ValueError: If user_satisfaction is out of range
        """
        if user_satisfaction is not None:
            if not 0.0 <= user_satisfaction <= 1.0:
                raise ValueError(
                    "user_satisfaction must be between 0.0 and 1.0"
                )

        prefs = self.get_or_create_preferences(user_id)

        # Create interaction feedback
        feedback = InteractionFeedback(
            timestamp=datetime.now(),
            message_length=message_length,
            response_length=response_length,
            topics=topics or [],
            user_satisfaction=user_satisfaction,
            communication_style_used=communication_style_used,
            metadata=metadata or {}
        )

        # Add to history (maintain max size)
        prefs.interaction_history.append(feedback)
        if len(prefs.interaction_history) > self.max_history_size:
            prefs.interaction_history = prefs.interaction_history[
                -self.max_history_size:
            ]

        # Update preferences based on interaction
        self._learn_from_interaction(prefs, feedback)

        prefs.last_updated = datetime.now()
        self._save_preferences(prefs)

        return prefs

    def get_preferred_communication_style(
        self, user_id: str
    ) -> Optional[str]:
        """
        Get the user's preferred communication style.

        Args:
            user_id: User identifier

        Returns:
            Preferred communication style or None if no preference
        """
        prefs = self.get_preferences(user_id)
        if prefs is None or not prefs.communication_style_scores:
            return None

        # Return style with highest score
        best_style = max(
            prefs.communication_style_scores.values(),
            key=lambda x: x.score
        )
        return best_style.value if best_style.score > 0.0 else None

    def get_preferred_response_length(
        self, user_id: str
    ) -> Optional[str]:
        """
        Get the user's preferred response length.

        Args:
            user_id: User identifier

        Returns:
            Preferred response length or None if no preference
        """
        prefs = self.get_preferences(user_id)
        if prefs is None or not prefs.response_length_scores:
            return None

        # Return length with highest score
        best_length = max(
            prefs.response_length_scores.values(),
            key=lambda x: x.score
        )
        return best_length.value if best_length.score > 0.0 else None

    def get_topic_interests(
        self, user_id: str, min_score: float = 0.0
    ) -> List[str]:
        """
        Get topics the user is interested in.

        Args:
            user_id: User identifier
            min_score: Minimum interest score to include (default: 0.0)

        Returns:
            List of topics sorted by interest score (descending)
        """
        prefs = self.get_preferences(user_id)
        if prefs is None:
            return []

        # Filter by min_score and sort by score
        topics = [
            topic for topic, score in prefs.topic_interests.items()
            if score >= min_score
        ]
        topics.sort(
            key=lambda t: prefs.topic_interests[t],
            reverse=True
        )

        return topics

    def add_topic_interest(
        self,
        user_id: str,
        topic: str,
        score: float = 1.0
    ) -> UserPreferences:
        """
        Add or update a topic interest.

        Args:
            user_id: User identifier
            topic: Topic name
            score: Interest score (0.0 to 1.0)

        Returns:
            Updated UserPreferences

        Raises:
            ValueError: If score is out of range
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

        prefs = self.get_or_create_preferences(user_id)

        # Update or add topic interest
        prefs.topic_interests[topic] = score
        prefs.last_updated = datetime.now()
        self._save_preferences(prefs)

        return prefs

    def remove_topic_interest(
        self, user_id: str, topic: str
    ) -> UserPreferences:
        """
        Remove a topic interest.

        Args:
            user_id: User identifier
            topic: Topic name

        Returns:
            Updated UserPreferences

        Raises:
            ValueError: If preferences don't exist
        """
        prefs = self.get_preferences(user_id)
        if prefs is None:
            raise ValueError(f"Preferences not found for user: {user_id}")

        # Remove topic if it exists
        if topic in prefs.topic_interests:
            del prefs.topic_interests[topic]
            prefs.last_updated = datetime.now()
            self._save_preferences(prefs)

        return prefs

    def get_preference_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user preferences.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with preference summary
        """
        prefs = self.get_preferences(user_id)
        if prefs is None:
            return {
                "user_id": user_id,
                "exists": False
            }

        return {
            "user_id": user_id,
            "exists": True,
            "preferred_communication_style": (
                self.get_preferred_communication_style(user_id)
            ),
            "preferred_response_length": (
                self.get_preferred_response_length(user_id)
            ),
            "topic_interests": self.get_topic_interests(user_id),
            "interaction_count": len(prefs.interaction_history),
            "created_at": prefs.created_at.isoformat(),
            "last_updated": prefs.last_updated.isoformat()
        }

    def delete_preferences(self, user_id: str) -> bool:
        """
        Delete user preferences.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        if user_id not in self.preferences:
            return False

        # Remove from memory
        del self.preferences[user_id]

        # Remove from disk
        prefs_path = self._get_preferences_path(user_id)
        if prefs_path.exists():
            prefs_path.unlink()

        return True

    def _learn_from_interaction(
        self,
        prefs: UserPreferences,
        feedback: InteractionFeedback
    ) -> None:
        """
        Learn preferences from an interaction.

        Args:
            prefs: User preferences to update
            feedback: Interaction feedback
        """
        # Learn communication style preference
        if feedback.communication_style_used:
            self._update_preference_score(
                prefs.communication_style_scores,
                feedback.communication_style_used,
                feedback.user_satisfaction or 0.5
            )

        # Learn response length preference
        response_length_category = self._categorize_response_length(
            feedback.response_length
        )
        self._update_preference_score(
            prefs.response_length_scores,
            response_length_category,
            feedback.user_satisfaction or 0.5
        )

        # Learn topic interests
        for topic in feedback.topics:
            current_score = prefs.topic_interests.get(topic, 0.0)
            # Increase score based on satisfaction
            satisfaction = feedback.user_satisfaction or 0.5
            new_score = min(1.0, current_score + (satisfaction * 0.1))
            prefs.topic_interests[topic] = new_score

    def _update_preference_score(
        self,
        scores: Dict[str, PreferenceScore],
        value: str,
        satisfaction: float
    ) -> None:
        """
        Update a preference score based on satisfaction.

        Args:
            scores: Dictionary of preference scores
            value: Preference value to update
            satisfaction: Satisfaction score (0.0 to 1.0)
        """
        if value not in scores:
            scores[value] = PreferenceScore(value=value)

        score = scores[value]
        score.count += 1

        # Update score using exponential moving average
        alpha = 0.3  # Learning rate
        score.score = (alpha * satisfaction) + ((1 - alpha) * score.score)
        score.last_updated = datetime.now()

    def _categorize_response_length(self, length: int) -> str:
        """
        Categorize response length into brief/moderate/detailed.

        Args:
            length: Response length in characters

        Returns:
            Category: "brief", "moderate", or "detailed"
        """
        if length < 200:
            return "brief"
        elif length < 800:
            return "moderate"
        else:
            return "detailed"

    def _get_preferences_path(self, user_id: str) -> Path:
        """Get file path for preferences."""
        # Sanitize user_id for filename
        safe_id = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in user_id
        )
        return self.storage_dir / f"{safe_id}_preferences.json"

    def _save_preferences(self, prefs: UserPreferences) -> None:
        """Save preferences to disk."""
        prefs_path = self._get_preferences_path(prefs.user_id)
        with open(prefs_path, 'w', encoding='utf-8') as f:
            json.dump(prefs.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Load preferences from disk."""
        prefs_path = self._get_preferences_path(user_id)
        if not prefs_path.exists():
            return None

        try:
            with open(prefs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return UserPreferences.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log error but don't crash
            print(f"Error loading preferences {user_id}: {e}")
            return None

    def _load_all_preferences(self) -> None:
        """Load all preferences from disk."""
        if not self.storage_dir.exists():
            return

        for prefs_file in self.storage_dir.glob("*_preferences.json"):
            try:
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                prefs = UserPreferences.from_dict(data)
                self.preferences[prefs.user_id] = prefs
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error but continue loading other preferences
                print(f"Error loading preferences from {prefs_file}: {e}")
                continue
