"""
Tests for feature flag toggling behavior.

Validates that the Settings model includes all 7 feature flag fields,
that they default to True, and that orchestrator service initialization
respects the flag values.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestFeatureFlags:
    """Test that feature flags properly control module initialization."""

    def test_settings_has_all_feature_flags(self):
        """Settings should include all 7 feature flag fields."""
        from morgan.config.settings import Settings

        field_names = [
            "morgan_enable_conversation_flow",
            "morgan_enable_communication_adapter",
            "morgan_enable_habits",
            "morgan_enable_learning",
            "morgan_enable_quality_assessment",
            "morgan_enable_cultural_awareness",
            "morgan_enable_nonverbal_detection",
        ]
        # Pydantic v2 uses model_fields; Pydantic v1 uses __fields__
        if hasattr(Settings, "model_fields"):
            fields = Settings.model_fields
        else:
            fields = Settings.__fields__

        for field_name in field_names:
            assert field_name in fields, (
                f"Settings is missing feature flag field: {field_name}"
            )

    def test_feature_flags_default_to_true(self):
        """All feature flags should default to True."""
        from morgan.config.settings import Settings

        flag_names = [
            "morgan_enable_conversation_flow",
            "morgan_enable_communication_adapter",
            "morgan_enable_habits",
            "morgan_enable_learning",
            "morgan_enable_quality_assessment",
            "morgan_enable_cultural_awareness",
            "morgan_enable_nonverbal_detection",
        ]
        # Pydantic v2 uses model_fields with FieldInfo.default
        if hasattr(Settings, "model_fields"):
            fields = Settings.model_fields
            for flag_name in flag_names:
                assert flag_name in fields, f"Missing field: {flag_name}"
                assert fields[flag_name].default is True, (
                    f"{flag_name} should default to True, got {fields[flag_name].default}"
                )
        else:
            # Pydantic v1 fallback
            fields = Settings.__fields__
            for flag_name in flag_names:
                assert flag_name in fields, f"Missing field: {flag_name}"
                assert fields[flag_name].default is True, (
                    f"{flag_name} should default to True"
                )

    def test_conversation_services_respect_flags(self):
        """Conversation services should only init when their flag is True."""
        mock_settings = MagicMock()
        mock_settings.morgan_enable_conversation_flow = False
        mock_settings.morgan_enable_quality_assessment = False
        mock_settings.morgan_max_search_results = 5
        mock_settings.morgan_enable_reasoning = False
        mock_settings.morgan_enable_proactive = False

        with patch("morgan.core.application.orchestrators.get_settings", return_value=mock_settings):
            # Patch all init methods except conversation to isolate the test
            with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):
                # Mock the conversation module imports
                mock_flow_mgr = MagicMock()
                mock_quality = MagicMock()
                mock_topics = MagicMock()
                mock_interruption = MagicMock()

                with patch.dict("sys.modules", {}):
                    with patch("morgan.conversation.flow.get_conversation_flow_manager", return_value=mock_flow_mgr), \
                         patch("morgan.conversation.quality.get_conversation_quality_assessor", return_value=mock_quality), \
                         patch("morgan.conversation.topics.get_topic_preference_learner", return_value=mock_topics), \
                         patch("morgan.conversation.interruption.get_interruption_handler", return_value=mock_interruption):

                        from morgan.core.application.orchestrators import ConversationOrchestrator
                        orch = ConversationOrchestrator(
                            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                            enable_reasoning=False, enable_proactive=False,
                        )
                        # Flow manager and quality assessor should be None when flags are False
                        assert orch._flow_manager is None, (
                            "Flow manager should not be initialized when flag is False"
                        )
                        assert orch._quality_assessor is None, (
                            "Quality assessor should not be initialized when flag is False"
                        )

    def test_communication_services_respect_flags(self):
        """Communication services should only init when their flag is True."""
        mock_settings = MagicMock()
        mock_settings.morgan_enable_communication_adapter = False
        mock_settings.morgan_enable_nonverbal_detection = False
        mock_settings.morgan_enable_cultural_awareness = False
        mock_settings.morgan_max_search_results = 5
        mock_settings.morgan_enable_reasoning = False
        mock_settings.morgan_enable_proactive = False

        with patch("morgan.core.application.orchestrators.get_settings", return_value=mock_settings):
            with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):
                mock_style = MagicMock()
                mock_nonverbal = MagicMock()
                mock_cultural = MagicMock()
                mock_prefs = MagicMock()

                with patch("morgan.communication.style.get_communication_style_adapter", return_value=mock_style), \
                     patch("morgan.communication.nonverbal.get_nonverbal_cue_detector", return_value=mock_nonverbal), \
                     patch("morgan.communication.cultural.get_cultural_emotional_awareness", return_value=mock_cultural), \
                     patch("morgan.communication.preferences.get_user_preference_learner", return_value=mock_prefs):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch._communication_adapter is None, (
                        "Communication adapter should not be initialized when flag is False"
                    )
                    assert orch._nonverbal_detector is None, (
                        "Nonverbal detector should not be initialized when flag is False"
                    )
                    assert orch._cultural_awareness is None, (
                        "Cultural awareness should not be initialized when flag is False"
                    )

    def test_learning_services_respect_flags(self):
        """Learning services should only init when their flag is True."""
        mock_settings = MagicMock()
        mock_settings.morgan_enable_learning = False
        mock_settings.morgan_max_search_results = 5
        mock_settings.morgan_enable_reasoning = False
        mock_settings.morgan_enable_proactive = False

        with patch("morgan.core.application.orchestrators.get_settings", return_value=mock_settings):
            with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):
                mock_engine = MagicMock()

                with patch("morgan.learning.engine.get_learning_engine", return_value=mock_engine):
                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch._learning_engine is None, (
                        "Learning engine should not be initialized when flag is False"
                    )

    def test_habit_services_respect_flags(self):
        """Habit services should only init when their flag is True."""
        mock_settings = MagicMock()
        mock_settings.morgan_enable_habits = False
        mock_settings.morgan_max_search_results = 5
        mock_settings.morgan_enable_reasoning = False
        mock_settings.morgan_enable_proactive = False

        with patch("morgan.core.application.orchestrators.get_settings", return_value=mock_settings):
            with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                 patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"):
                with patch("morgan.habits.detector.HabitDetector"), \
                     patch("morgan.habits.adaptation.HabitBasedAdaptation"), \
                     patch("morgan.habits.wellness.WellnessHabitTracker"):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch._habit_detector is None, (
                        "Habit detector should not be initialized when flag is False"
                    )
                    assert orch._habit_adaptation is None, (
                        "Habit adaptation should not be initialized when flag is False"
                    )
                    assert orch._wellness_tracker is None, (
                        "Wellness tracker should not be initialized when flag is False"
                    )
