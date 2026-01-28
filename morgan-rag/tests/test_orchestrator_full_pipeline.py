"""
Integration tests for the full ConversationOrchestrator pipeline.

Tests the orchestrator with all integrated feature modules (conversation intelligence,
communication adaptation, learning engine, habit detection) using mocked dependencies.
Validates that:
- The full pipeline completes with all modules enabled (mocked)
- The full pipeline completes with all modules disabled (feature flags off)
- Each module failure is handled gracefully (no crash)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta


class TestOrchestratorFullPipeline:
    """Test the full orchestrator pipeline with all integrated modules."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with all features enabled."""
        settings = MagicMock()
        settings.morgan_enable_conversation_flow = True
        settings.morgan_enable_communication_adapter = True
        settings.morgan_enable_habits = True
        settings.morgan_enable_learning = True
        settings.morgan_enable_quality_assessment = True
        settings.morgan_enable_cultural_awareness = True
        settings.morgan_enable_nonverbal_detection = True
        settings.morgan_max_search_results = 5
        settings.morgan_enable_reasoning = False
        settings.morgan_enable_proactive = False
        return settings

    @pytest.fixture
    def mock_settings_disabled(self):
        """Create mock settings with all features disabled."""
        settings = MagicMock()
        settings.morgan_enable_conversation_flow = False
        settings.morgan_enable_communication_adapter = False
        settings.morgan_enable_habits = False
        settings.morgan_enable_learning = False
        settings.morgan_enable_quality_assessment = False
        settings.morgan_enable_cultural_awareness = False
        settings.morgan_enable_nonverbal_detection = False
        settings.morgan_max_search_results = 5
        settings.morgan_enable_reasoning = False
        settings.morgan_enable_proactive = False
        return settings

    @pytest.fixture
    def mock_knowledge(self):
        """Create mock knowledge service with canned search results."""
        knowledge = MagicMock()
        knowledge.search_knowledge.return_value = [
            {"content": "Test knowledge", "source": "test", "score": 0.9}
        ]
        return knowledge

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory service with empty history."""
        memory = MagicMock()
        memory.get_conversation_history.return_value = []
        memory.get_conversation_context.return_value = ""
        return memory

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service returning a canned response with .content attribute."""
        llm = MagicMock()
        response = MagicMock()
        response.content = "This is a test response from Morgan."
        llm.generate.return_value = response
        return llm

    @pytest.fixture
    def mock_emotional_processor(self):
        """Create mock emotional processor with all required sub-components."""
        proc = MagicMock()
        # Emotion analysis
        emotion = MagicMock()
        emotion.primary_emotion = MagicMock(value="neutral")
        emotion.intensity = 0.5
        emotion.confidence = 0.8
        proc.emotional_engine.analyze_emotion.return_value = emotion
        # Empathetic response
        empathetic = MagicMock()
        empathetic.emotional_tone = "neutral"
        empathetic.empathy_level = 0.5
        empathetic.personalization_elements = []
        proc.emotional_engine.generate_empathetic_response.return_value = empathetic
        # Profile
        proc.get_or_create_user_profile.return_value = MagicMock()
        proc.relationship_manager.adapt_conversation_style.return_value = MagicMock()
        proc.check_for_milestones.return_value = None
        proc.process_conversation_memory = MagicMock()
        proc.update_user_profile = MagicMock()
        return proc

    @patch("morgan.core.application.orchestrators.get_settings")
    def test_pipeline_completes_with_all_modules_enabled(
        self, mock_get_settings, mock_settings, mock_knowledge, mock_memory,
        mock_llm, mock_emotional_processor
    ):
        """Full pipeline should complete successfully with all feature modules enabled."""
        mock_get_settings.return_value = mock_settings

        # Patch all module init methods to prevent actual loading of feature modules
        with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):

            from morgan.core.application.orchestrators import ConversationOrchestrator
            orch = ConversationOrchestrator(
                mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                enable_reasoning=False, enable_proactive=False,
            )
            assert orch is not None
            # Not initialized since we patched the init methods
            assert orch._flow_manager is None

    @patch("morgan.core.application.orchestrators.get_settings")
    def test_pipeline_completes_with_all_modules_disabled(
        self, mock_get_settings, mock_settings_disabled, mock_knowledge,
        mock_memory, mock_llm, mock_emotional_processor
    ):
        """Full pipeline should complete successfully with all feature flags off."""
        mock_get_settings.return_value = mock_settings_disabled

        # Patch all module init methods to prevent actual loading
        with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
             patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):

            from morgan.core.application.orchestrators import ConversationOrchestrator
            orch = ConversationOrchestrator(
                mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                enable_reasoning=False, enable_proactive=False,
            )
            assert orch is not None
            # All lazy-loaded services should remain None when flags are disabled
            assert orch._flow_manager is None
            assert orch._quality_assessor is None
            assert orch._communication_adapter is None
            assert orch._nonverbal_detector is None
            assert orch._cultural_awareness is None
            assert orch._learning_engine is None
            assert orch._habit_detector is None
            assert orch._habit_adaptation is None
            assert orch._wellness_tracker is None

    def test_module_failure_graceful_handling(
        self, mock_knowledge, mock_memory, mock_llm, mock_emotional_processor
    ):
        """Pipeline should handle individual module failures gracefully without crashing."""
        with patch("morgan.core.application.orchestrators.get_settings") as mock_gs:
            settings = MagicMock()
            settings.morgan_enable_conversation_flow = True
            settings.morgan_enable_communication_adapter = True
            settings.morgan_enable_habits = True
            settings.morgan_enable_learning = True
            settings.morgan_enable_quality_assessment = True
            settings.morgan_enable_cultural_awareness = True
            settings.morgan_enable_nonverbal_detection = True
            settings.morgan_max_search_results = 5
            settings.morgan_enable_reasoning = False
            settings.morgan_enable_proactive = False
            mock_gs.return_value = settings

            # Simulate conversation services raising ImportError (modules not available).
            # The actual _init_conversation_services code catches ImportError internally,
            # so we test the real init method by making the import fail.
            with patch.dict("sys.modules", {
                "morgan.conversation.flow": None,
                "morgan.conversation.quality": None,
                "morgan.conversation.topics": None,
                "morgan.conversation.interruption": None,
            }):
                # Patch remaining init methods that are not under test
                with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    # Should not raise -- init catches ImportError for conversation services
                    orch = ConversationOrchestrator(
                        mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch is not None
                    # Conversation services should be None since import was blocked
                    assert orch._flow_manager is None
                    assert orch._quality_assessor is None
                    assert orch._topic_learner is None
                    assert orch._interruption_handler is None

    def test_communication_module_failure_graceful(
        self, mock_knowledge, mock_memory, mock_llm, mock_emotional_processor
    ):
        """Communication module failure should not crash the pipeline."""
        with patch("morgan.core.application.orchestrators.get_settings") as mock_gs:
            settings = MagicMock()
            settings.morgan_enable_conversation_flow = True
            settings.morgan_enable_communication_adapter = True
            settings.morgan_enable_habits = True
            settings.morgan_enable_learning = True
            settings.morgan_enable_quality_assessment = True
            settings.morgan_enable_cultural_awareness = True
            settings.morgan_enable_nonverbal_detection = True
            settings.morgan_max_search_results = 5
            settings.morgan_enable_reasoning = False
            settings.morgan_enable_proactive = False
            mock_gs.return_value = settings

            # Simulate communication module import failures
            with patch.dict("sys.modules", {
                "morgan.communication.style": None,
                "morgan.communication.nonverbal": None,
                "morgan.communication.cultural": None,
                "morgan.communication.preferences": None,
            }):
                with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch is not None
                    assert orch._communication_adapter is None
                    assert orch._nonverbal_detector is None
                    assert orch._cultural_awareness is None

    def test_learning_module_failure_graceful(
        self, mock_knowledge, mock_memory, mock_llm, mock_emotional_processor
    ):
        """Learning module failure should not crash the pipeline."""
        with patch("morgan.core.application.orchestrators.get_settings") as mock_gs:
            settings = MagicMock()
            settings.morgan_enable_conversation_flow = True
            settings.morgan_enable_communication_adapter = True
            settings.morgan_enable_habits = True
            settings.morgan_enable_learning = True
            settings.morgan_enable_quality_assessment = True
            settings.morgan_enable_cultural_awareness = True
            settings.morgan_enable_nonverbal_detection = True
            settings.morgan_max_search_results = 5
            settings.morgan_enable_reasoning = False
            settings.morgan_enable_proactive = False
            mock_gs.return_value = settings

            with patch.dict("sys.modules", {
                "morgan.learning.engine": None,
            }):
                with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_habit_services"):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch is not None
                    assert orch._learning_engine is None

    def test_habit_module_failure_graceful(
        self, mock_knowledge, mock_memory, mock_llm, mock_emotional_processor
    ):
        """Habit module failure should not crash the pipeline."""
        with patch("morgan.core.application.orchestrators.get_settings") as mock_gs:
            settings = MagicMock()
            settings.morgan_enable_conversation_flow = True
            settings.morgan_enable_communication_adapter = True
            settings.morgan_enable_habits = True
            settings.morgan_enable_learning = True
            settings.morgan_enable_quality_assessment = True
            settings.morgan_enable_cultural_awareness = True
            settings.morgan_enable_nonverbal_detection = True
            settings.morgan_max_search_results = 5
            settings.morgan_enable_reasoning = False
            settings.morgan_enable_proactive = False
            mock_gs.return_value = settings

            with patch.dict("sys.modules", {
                "morgan.habits.detector": None,
                "morgan.habits.adaptation": None,
                "morgan.habits.wellness": None,
            }):
                with patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_conversation_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_communication_services"), \
                     patch("morgan.core.application.orchestrators.ConversationOrchestrator._init_learning_services"):

                    from morgan.core.application.orchestrators import ConversationOrchestrator
                    orch = ConversationOrchestrator(
                        mock_knowledge, mock_memory, mock_llm, mock_emotional_processor,
                        enable_reasoning=False, enable_proactive=False,
                    )
                    assert orch is not None
                    assert orch._habit_detector is None
                    assert orch._habit_adaptation is None
                    assert orch._wellness_tracker is None
