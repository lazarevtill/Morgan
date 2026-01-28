"""
User preference management system.

Provides comprehensive preference management with validation, persistence,
and intelligent defaults based on user behavior patterns.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..intelligence.core.models import CommunicationStyle, ResponseLength, UserPreferences
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreferenceChange:
    """Represents a preference change event."""

    change_id: str
    user_id: str
    preference_type: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    reason: Optional[str] = None


@dataclass
class PreferenceValidationResult:
    """Result of preference validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class UserPreferenceManager:
    """
    Manages user preferences with intelligent defaults and validation.

    Features:
    - Comprehensive preference validation
    - Automatic preference learning from behavior
    - Preference change history and rollback
    - Export/import functionality
    - Intelligent suggestions based on usage patterns
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the preference manager."""
        self.storage_path = (
            Path(storage_path) if storage_path else Path("data/preferences")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.preference_history: Dict[str, List[PreferenceChange]] = {}
        self.validation_rules = self._setup_validation_rules()

        logger.info(
            f"User preference manager initialized with storage: {self.storage_path}"
        )

    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences with intelligent defaults."""
        try:
            pref_file = self.storage_path / f"{user_id}_preferences.json"

            if pref_file.exists():
                with open(pref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert back to UserPreferences object
                preferences = UserPreferences(
                    topics_of_interest=data.get("topics_of_interest", []),
                    communication_style=CommunicationStyle(
                        data.get("communication_style", "friendly")
                    ),
                    preferred_response_length=ResponseLength(
                        data.get("preferred_response_length", "detailed")
                    ),
                    learning_goals=data.get("learning_goals", []),
                    personal_context=data.get("personal_context", {}),
                    interaction_preferences=data.get("interaction_preferences", {}),
                    last_updated=datetime.fromisoformat(
                        data.get("last_updated", datetime.now(timezone.utc).isoformat())
                    ),
                )

                return preferences
            else:
                # Return intelligent defaults
                return self._create_default_preferences(user_id)

        except Exception as e:
            logger.error(f"Failed to load preferences for user {user_id}: {e}")
            return self._create_default_preferences(user_id)

    def update_user_preferences(
        self,
        user_id: str,
        preferences: Union[UserPreferences, Dict[str, Any]],
        reason: Optional[str] = None,
    ) -> PreferenceValidationResult:
        """Update user preferences with validation."""
        try:
            # Convert dict to UserPreferences if needed
            if isinstance(preferences, dict):
                current_prefs = self.get_user_preferences(user_id)
                preferences = self._merge_preference_updates(current_prefs, preferences)

            # Validate preferences
            validation_result = self.validate_preferences(preferences)

            if not validation_result.is_valid:
                logger.warning(
                    f"Invalid preferences for user {user_id}: {validation_result.errors}"
                )
                return validation_result

            # Get current preferences for change tracking
            current_prefs = self.get_user_preferences(user_id)

            # Track changes
            if current_prefs:
                self._track_preference_changes(
                    user_id, current_prefs, preferences, reason
                )

            # Save preferences
            self._save_preferences(user_id, preferences)

            logger.info(f"Updated preferences for user {user_id}")
            return validation_result

        except Exception as e:
            logger.error(f"Failed to update preferences for user {user_id}: {e}")
            return PreferenceValidationResult(
                is_valid=False,
                errors=[f"Failed to update preferences: {str(e)}"],
                warnings=[],
                suggestions=[],
            )

    def validate_preferences(
        self, preferences: UserPreferences
    ) -> PreferenceValidationResult:
        """Validate user preferences against rules."""
        errors = []
        warnings = []
        suggestions = []

        # Validate topics of interest
        if len(preferences.topics_of_interest) > 20:
            warnings.append(
                "You have many topics of interest. Consider focusing on your top priorities."
            )

        # Check for duplicate topics
        unique_topics = {topic.lower() for topic in preferences.topics_of_interest}
        if len(unique_topics) < len(preferences.topics_of_interest):
            warnings.append("Some topics of interest are duplicated.")
            suggestions.append(
                "Remove duplicate topics to keep your interests organized."
            )

        # Validate communication style consistency
        if (
            preferences.communication_style == CommunicationStyle.FORMAL
            and preferences.preferred_response_length == ResponseLength.BRIEF
        ):
            suggestions.append(
                "Formal communication often benefits from detailed responses."
            )

        # Validate learning goals
        if len(preferences.learning_goals) > 10:
            warnings.append(
                "You have many learning goals. Consider prioritizing the most important ones."
            )

        # Check for empty or very short goals
        short_goals = [
            goal for goal in preferences.learning_goals if len(goal.strip()) < 10
        ]
        if short_goals:
            suggestions.append(
                "Consider adding more detail to your learning goals for better assistance."
            )

        # Validate personal context
        if preferences.personal_context:
            sensitive_keys = ["password", "ssn", "credit_card", "bank_account"]
            for key in preferences.personal_context.keys():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    errors.append(
                        f"Personal context should not contain sensitive information like '{key}'."
                    )

        is_valid = len(errors) == 0

        return PreferenceValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, suggestions=suggestions
        )

    def learn_preferences_from_behavior(
        self, user_id: str, interaction_data: Dict[str, Any]
    ) -> Optional[UserPreferences]:
        """Learn and update preferences from user behavior."""
        try:
            current_prefs = self.get_user_preferences(user_id)
            if not current_prefs:
                return None

            updated = False

            # Learn topics from conversation content
            if "topics_discussed" in interaction_data:
                new_topics = interaction_data["topics_discussed"]
                for topic in new_topics:
                    if (
                        topic not in current_prefs.topics_of_interest
                        and len(current_prefs.topics_of_interest) < 15
                    ):
                        current_prefs.topics_of_interest.append(topic)
                        updated = True
                        logger.debug(f"Learned new topic '{topic}' for user {user_id}")

            # Learn communication style from message patterns
            if "message_style_indicators" in interaction_data:
                indicators = interaction_data["message_style_indicators"]

                # Detect formal language
                if indicators.get("formal_language_ratio", 0) > 0.7:
                    if current_prefs.communication_style != CommunicationStyle.FORMAL:
                        current_prefs.communication_style = CommunicationStyle.FORMAL
                        updated = True
                        logger.debug(
                            f"Learned formal communication style for user {user_id}"
                        )

                # Detect technical language
                elif indicators.get("technical_terms_ratio", 0) > 0.5:
                    if (
                        current_prefs.communication_style
                        != CommunicationStyle.TECHNICAL
                    ):
                        current_prefs.communication_style = CommunicationStyle.TECHNICAL
                        updated = True
                        logger.debug(
                            f"Learned technical communication style for user {user_id}"
                        )

            # Learn response length preference from feedback
            if "response_length_feedback" in interaction_data:
                feedback = interaction_data["response_length_feedback"]

                if (
                    feedback == "too_long"
                    and current_prefs.preferred_response_length != ResponseLength.BRIEF
                ):
                    current_prefs.preferred_response_length = ResponseLength.BRIEF
                    updated = True
                elif (
                    feedback == "too_short"
                    and current_prefs.preferred_response_length
                    != ResponseLength.COMPREHENSIVE
                ):
                    current_prefs.preferred_response_length = (
                        ResponseLength.COMPREHENSIVE
                    )
                    updated = True

            # Learn from emotional patterns
            if "emotional_preferences" in interaction_data:
                emotional_prefs = interaction_data["emotional_preferences"]

                # Update interaction preferences based on emotional responses
                if "empathy_preference" in emotional_prefs:
                    current_prefs.interaction_preferences["empathy_level"] = (
                        emotional_prefs["empathy_preference"]
                    )
                    updated = True

                if "formality_preference" in emotional_prefs:
                    current_prefs.interaction_preferences["formality_level"] = (
                        emotional_prefs["formality_preference"]
                    )
                    updated = True

            if updated:
                current_prefs.last_updated = datetime.now(timezone.utc)
                self._save_preferences(user_id, current_prefs)
                logger.info(f"Learned and updated preferences for user {user_id}")
                return current_prefs

            return None

        except Exception as e:
            logger.error(
                f"Failed to learn preferences from behavior for user {user_id}: {e}"
            )
            return None

    def get_preference_suggestions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get intelligent preference suggestions based on usage patterns."""
        suggestions = []

        try:
            preferences = self.get_user_preferences(user_id)
            if not preferences:
                return suggestions

            # Suggest topics based on interaction history
            if len(preferences.topics_of_interest) < 3:
                suggestions.append(
                    {
                        "type": "topic_suggestion",
                        "title": "Add More Interests",
                        "description": "Adding more topics of interest helps me provide more relevant responses.",
                        "action": "add_topics",
                        "priority": "medium",
                    }
                )

            # Suggest learning goals if none exist
            if not preferences.learning_goals:
                suggestions.append(
                    {
                        "type": "learning_goal_suggestion",
                        "title": "Set Learning Goals",
                        "description": "Setting learning goals helps me track your progress and provide targeted assistance.",
                        "action": "add_learning_goals",
                        "priority": "high",
                    }
                )

            # Suggest communication style optimization
            history = self.preference_history.get(user_id, [])
            style_changes = [
                change
                for change in history
                if change.preference_type == "communication_style"
            ]

            if len(style_changes) > 3:
                suggestions.append(
                    {
                        "type": "communication_optimization",
                        "title": "Optimize Communication Style",
                        "description": "You've changed your communication style several times. Let me help you find the perfect fit.",
                        "action": "optimize_communication",
                        "priority": "medium",
                    }
                )

            # Suggest preference review if not updated recently
            days_since_update = (datetime.now(timezone.utc) - preferences.last_updated).days
            if days_since_update > 30:
                suggestions.append(
                    {
                        "type": "preference_review",
                        "title": "Review Your Preferences",
                        "description": f"It's been {days_since_update} days since your last preference update. Your needs might have evolved.",
                        "action": "review_preferences",
                        "priority": "low",
                    }
                )

        except Exception as e:
            logger.error(
                f"Failed to generate preference suggestions for user {user_id}: {e}"
            )

        return suggestions

    def export_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user preferences for backup or transfer."""
        try:
            preferences = self.get_user_preferences(user_id)
            if not preferences:
                return None

            export_data = {
                "user_id": user_id,
                "preferences": asdict(preferences),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
            }

            # Include preference history if available
            if user_id in self.preference_history:
                export_data["preference_history"] = [
                    asdict(change) for change in self.preference_history[user_id]
                ]

            logger.info(f"Exported preferences for user {user_id}")
            return export_data

        except Exception as e:
            logger.error(f"Failed to export preferences for user {user_id}: {e}")
            return None

    def import_preferences(
        self,
        user_id: str,
        import_data: Dict[str, Any],
        merge_with_existing: bool = True,
    ) -> bool:
        """Import user preferences from backup or transfer."""
        try:
            if "preferences" not in import_data:
                logger.error("Invalid import data: missing preferences")
                return False

            pref_data = import_data["preferences"]

            # Create UserPreferences object
            imported_prefs = UserPreferences(
                topics_of_interest=pref_data.get("topics_of_interest", []),
                communication_style=CommunicationStyle(
                    pref_data.get("communication_style", "friendly")
                ),
                preferred_response_length=ResponseLength(
                    pref_data.get("preferred_response_length", "detailed")
                ),
                learning_goals=pref_data.get("learning_goals", []),
                personal_context=pref_data.get("personal_context", {}),
                interaction_preferences=pref_data.get("interaction_preferences", {}),
                last_updated=datetime.now(timezone.utc),
            )

            # Merge with existing preferences if requested
            if merge_with_existing:
                existing_prefs = self.get_user_preferences(user_id)
                if existing_prefs:
                    imported_prefs = self._merge_preferences(
                        existing_prefs, imported_prefs
                    )

            # Validate imported preferences
            validation_result = self.validate_preferences(imported_prefs)
            if not validation_result.is_valid:
                logger.error(
                    f"Invalid imported preferences: {validation_result.errors}"
                )
                return False

            # Save imported preferences
            self._save_preferences(user_id, imported_prefs)

            # Import preference history if available
            if "preference_history" in import_data:
                history_data = import_data["preference_history"]
                imported_history = []

                for change_data in history_data:
                    change = PreferenceChange(
                        change_id=change_data["change_id"],
                        user_id=change_data["user_id"],
                        preference_type=change_data["preference_type"],
                        old_value=change_data["old_value"],
                        new_value=change_data["new_value"],
                        timestamp=datetime.fromisoformat(change_data["timestamp"]),
                        reason=change_data.get("reason"),
                    )
                    imported_history.append(change)

                self.preference_history[user_id] = imported_history

            logger.info(f"Successfully imported preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to import preferences for user {user_id}: {e}")
            return False

    def rollback_preferences(self, user_id: str, steps: int = 1) -> bool:
        """Rollback user preferences to a previous state."""
        try:
            if user_id not in self.preference_history:
                logger.warning(f"No preference history found for user {user_id}")
                return False

            history = self.preference_history[user_id]
            if len(history) < steps:
                logger.warning(
                    f"Not enough history to rollback {steps} steps for user {user_id}"
                )
                return False

            # Get the target state
            target_changes = history[-steps:]

            # Reconstruct preferences by reversing changes
            current_prefs = self.get_user_preferences(user_id)
            if not current_prefs:
                return False

            for change in reversed(target_changes):
                if hasattr(current_prefs, change.preference_type):
                    setattr(current_prefs, change.preference_type, change.old_value)

            current_prefs.last_updated = datetime.now(timezone.utc)

            # Save rolled back preferences
            self._save_preferences(user_id, current_prefs)

            # Remove rolled back changes from history
            self.preference_history[user_id] = history[:-steps]

            logger.info(f"Rolled back {steps} preference changes for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback preferences for user {user_id}: {e}")
            return False

    def _create_default_preferences(self, user_id: str) -> UserPreferences:
        """Create intelligent default preferences."""
        return UserPreferences(
            topics_of_interest=[],
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length=ResponseLength.DETAILED,
            learning_goals=[],
            personal_context={},
            interaction_preferences={
                "empathy_level": 0.7,
                "formality_level": 0.3,
                "technical_depth": 0.5,
            },
            last_updated=datetime.now(timezone.utc),
        )

    def _merge_preference_updates(
        self, current: UserPreferences, updates: Dict[str, Any]
    ) -> UserPreferences:
        """Merge preference updates with current preferences."""
        # Create a copy of current preferences
        merged = UserPreferences(
            topics_of_interest=current.topics_of_interest.copy(),
            communication_style=current.communication_style,
            preferred_response_length=current.preferred_response_length,
            learning_goals=current.learning_goals.copy(),
            personal_context=current.personal_context.copy(),
            interaction_preferences=current.interaction_preferences.copy(),
            last_updated=datetime.now(timezone.utc),
        )

        # Apply updates
        for key, value in updates.items():
            if hasattr(merged, key):
                if key == "communication_style" and isinstance(value, str):
                    merged.communication_style = CommunicationStyle(value)
                elif key == "preferred_response_length" and isinstance(value, str):
                    merged.preferred_response_length = ResponseLength(value)
                else:
                    setattr(merged, key, value)

        return merged

    def _merge_preferences(
        self, existing: UserPreferences, imported: UserPreferences
    ) -> UserPreferences:
        """Merge existing and imported preferences intelligently."""
        # Merge topics of interest (union)
        merged_topics = list(
            set(existing.topics_of_interest + imported.topics_of_interest)
        )

        # Merge learning goals (union)
        merged_goals = list(set(existing.learning_goals + imported.learning_goals))

        # Merge personal context (imported takes precedence)
        merged_context = existing.personal_context.copy()
        merged_context.update(imported.personal_context)

        # Merge interaction preferences (imported takes precedence)
        merged_interaction_prefs = existing.interaction_preferences.copy()
        merged_interaction_prefs.update(imported.interaction_preferences)

        return UserPreferences(
            topics_of_interest=merged_topics,
            communication_style=imported.communication_style,  # Use imported
            preferred_response_length=imported.preferred_response_length,  # Use imported
            learning_goals=merged_goals,
            personal_context=merged_context,
            interaction_preferences=merged_interaction_prefs,
            last_updated=datetime.now(timezone.utc),
        )

    def _track_preference_changes(
        self,
        user_id: str,
        old_prefs: UserPreferences,
        new_prefs: UserPreferences,
        reason: Optional[str] = None,
    ):
        """Track changes between preference states."""
        if user_id not in self.preference_history:
            self.preference_history[user_id] = []

        # Compare preferences and track changes
        changes = []

        if old_prefs.communication_style != new_prefs.communication_style:
            changes.append(
                PreferenceChange(
                    change_id=str(uuid.uuid4()),
                    user_id=user_id,
                    preference_type="communication_style",
                    old_value=old_prefs.communication_style.value,
                    new_value=new_prefs.communication_style.value,
                    timestamp=datetime.now(timezone.utc),
                    reason=reason,
                )
            )

        if old_prefs.preferred_response_length != new_prefs.preferred_response_length:
            changes.append(
                PreferenceChange(
                    change_id=str(uuid.uuid4()),
                    user_id=user_id,
                    preference_type="preferred_response_length",
                    old_value=old_prefs.preferred_response_length.value,
                    new_value=new_prefs.preferred_response_length.value,
                    timestamp=datetime.now(timezone.utc),
                    reason=reason,
                )
            )

        if old_prefs.topics_of_interest != new_prefs.topics_of_interest:
            changes.append(
                PreferenceChange(
                    change_id=str(uuid.uuid4()),
                    user_id=user_id,
                    preference_type="topics_of_interest",
                    old_value=old_prefs.topics_of_interest,
                    new_value=new_prefs.topics_of_interest,
                    timestamp=datetime.now(timezone.utc),
                    reason=reason,
                )
            )

        # Add changes to history
        self.preference_history[user_id].extend(changes)

        # Keep only last 50 changes per user
        if len(self.preference_history[user_id]) > 50:
            self.preference_history[user_id] = self.preference_history[user_id][-50:]

    def _save_preferences(self, user_id: str, preferences: UserPreferences):
        """Save preferences to storage."""
        pref_file = self.storage_path / f"{user_id}_preferences.json"

        # Convert to dict for JSON serialization
        pref_data = {
            "topics_of_interest": preferences.topics_of_interest,
            "communication_style": preferences.communication_style.value,
            "preferred_response_length": preferences.preferred_response_length.value,
            "learning_goals": preferences.learning_goals,
            "personal_context": preferences.personal_context,
            "interaction_preferences": preferences.interaction_preferences,
            "last_updated": preferences.last_updated.isoformat(),
        }

        with open(pref_file, "w", encoding="utf-8") as f:
            json.dump(pref_data, f, indent=2, ensure_ascii=False)

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for preferences."""
        return {
            "max_topics": 20,
            "max_learning_goals": 10,
            "min_goal_length": 10,
            "forbidden_context_keys": [
                "password",
                "ssn",
                "credit_card",
                "bank_account",
            ],
            "valid_communication_styles": [style.value for style in CommunicationStyle],
            "valid_response_lengths": [length.value for length in ResponseLength],
        }
