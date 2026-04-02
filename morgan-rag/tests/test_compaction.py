"""
Tests for the context compaction module.

Covers:
- token_counter: estimate_tokens, estimate_messages_tokens
- auto_compact: AutoCompactTracker, threshold functions, warning state
- compactor: Compactor.compact with mocked LLM and fallback behaviour
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morgan.compaction.token_counter import (
    estimate_tokens,
    estimate_messages_tokens,
)
from morgan.compaction.auto_compact import (
    AutoCompactTracker,
    get_effective_context_window,
    get_auto_compact_threshold,
    calculate_token_warning_state,
    should_auto_compact,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    AUTOCOMPACT_BUFFER_TOKENS,
    WARNING_THRESHOLD_BUFFER,
    ERROR_THRESHOLD_BUFFER,
    MAX_CONSECUTIVE_FAILURES,
    DEFAULT_CONTEXT_WINDOW,
)
from morgan.compaction.compactor import Compactor


# =========================================================================
# Token counter tests
# =========================================================================


class TestEstimateTokens:
    """Tests for estimate_tokens."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_like_empty(self):
        # Empty string is falsy
        assert estimate_tokens("") == 0

    def test_short_text_positive(self):
        result = estimate_tokens("hello world")
        assert result > 0

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("This is a much longer sentence with more words.")
        assert long > short

    def test_returns_int(self):
        result = estimate_tokens("some text")
        assert isinstance(result, int)


class TestEstimateMessagesTokens:
    """Tests for estimate_messages_tokens."""

    def test_empty_list_returns_zero(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = estimate_messages_tokens(messages)
        # Should be > 0 (content tokens + overhead)
        assert result > 0

    def test_multiple_messages_more_tokens(self):
        one = estimate_messages_tokens([{"role": "user", "content": "Hi"}])
        two = estimate_messages_tokens([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ])
        assert two > one

    def test_per_message_overhead(self):
        """Each message adds per-message overhead tokens."""
        content_only = estimate_tokens("Hi")
        message_total = estimate_messages_tokens([{"role": "user", "content": "Hi"}])
        # message_total should exceed bare content (overhead + priming)
        assert message_total > content_only

    def test_missing_content_key(self):
        """Messages without 'content' should not crash."""
        result = estimate_messages_tokens([{"role": "system"}])
        # Just overhead tokens, no content
        assert result > 0


# =========================================================================
# Constants sanity checks
# =========================================================================


class TestConstants:
    """Verify that constants have sensible values."""

    def test_max_output_tokens(self):
        assert MAX_OUTPUT_TOKENS_FOR_SUMMARY == 20_000

    def test_autocompact_buffer(self):
        assert AUTOCOMPACT_BUFFER_TOKENS == 13_000

    def test_warning_buffer(self):
        assert WARNING_THRESHOLD_BUFFER == 20_000

    def test_error_buffer(self):
        assert ERROR_THRESHOLD_BUFFER == 20_000

    def test_max_consecutive_failures(self):
        assert MAX_CONSECUTIVE_FAILURES == 3

    def test_default_context_window(self):
        assert DEFAULT_CONTEXT_WINDOW == 200_000


# =========================================================================
# AutoCompactTracker tests
# =========================================================================


class TestAutoCompactTracker:
    """Tests for the AutoCompactTracker dataclass."""

    def test_default_state(self):
        tracker = AutoCompactTracker()
        assert tracker.compacted is False
        assert tracker.turn_counter == 0
        assert tracker.turn_id == 0
        assert tracker.consecutive_failures == 0

    def test_circuit_breaker_not_tripped_initially(self):
        tracker = AutoCompactTracker()
        assert tracker.circuit_breaker_tripped is False

    def test_circuit_breaker_trips_after_max_failures(self):
        tracker = AutoCompactTracker()
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            tracker.record_failure()
        assert tracker.circuit_breaker_tripped is True

    def test_circuit_breaker_not_tripped_below_max(self):
        tracker = AutoCompactTracker()
        for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
            tracker.record_failure()
        assert tracker.circuit_breaker_tripped is False

    def test_record_success_resets_failures(self):
        tracker = AutoCompactTracker()
        tracker.record_failure()
        tracker.record_failure()
        tracker.record_success()
        assert tracker.consecutive_failures == 0
        assert tracker.compacted is True

    def test_record_success_sets_compacted(self):
        tracker = AutoCompactTracker()
        assert tracker.compacted is False
        tracker.record_success()
        assert tracker.compacted is True

    def test_new_turn_increments(self):
        tracker = AutoCompactTracker()
        tid = tracker.new_turn()
        assert tid == 1
        assert tracker.turn_counter == 1
        assert tracker.turn_id == 1

        tid2 = tracker.new_turn()
        assert tid2 == 2
        assert tracker.turn_counter == 2
        assert tracker.turn_id == 2

    def test_circuit_breaker_resets_on_success(self):
        tracker = AutoCompactTracker()
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            tracker.record_failure()
        assert tracker.circuit_breaker_tripped is True

        tracker.record_success()
        assert tracker.circuit_breaker_tripped is False
        assert tracker.consecutive_failures == 0


# =========================================================================
# Threshold function tests
# =========================================================================


class TestThresholdFunctions:
    """Tests for get_effective_context_window and get_auto_compact_threshold."""

    def test_effective_context_window_default(self):
        result = get_effective_context_window()
        assert result == DEFAULT_CONTEXT_WINDOW - MAX_OUTPUT_TOKENS_FOR_SUMMARY

    def test_effective_context_window_custom(self):
        result = get_effective_context_window(100_000)
        assert result == 100_000 - MAX_OUTPUT_TOKENS_FOR_SUMMARY

    def test_auto_compact_threshold_default(self):
        expected = (
            DEFAULT_CONTEXT_WINDOW
            - MAX_OUTPUT_TOKENS_FOR_SUMMARY
            - AUTOCOMPACT_BUFFER_TOKENS
        )
        assert get_auto_compact_threshold() == expected

    def test_auto_compact_threshold_custom(self):
        window = 100_000
        expected = window - MAX_OUTPUT_TOKENS_FOR_SUMMARY - AUTOCOMPACT_BUFFER_TOKENS
        assert get_auto_compact_threshold(window) == expected


# =========================================================================
# Token warning state tests
# =========================================================================


class TestCalculateTokenWarningState:
    """Tests for calculate_token_warning_state."""

    def test_empty_conversation_safe(self):
        state = calculate_token_warning_state([])
        assert state["percent_left"] > 90
        assert state["tokens_left"] > 0
        assert state["is_above_warning_threshold"] is False
        assert state["is_above_error_threshold"] is False
        assert state["is_above_auto_compact_threshold"] is False
        assert state["is_at_blocking_limit"] is False

    def test_returns_all_keys(self):
        state = calculate_token_warning_state([])
        expected_keys = {
            "percent_left",
            "tokens_left",
            "is_above_warning_threshold",
            "is_above_error_threshold",
            "is_above_auto_compact_threshold",
            "is_at_blocking_limit",
        }
        assert set(state.keys()) == expected_keys

    def test_percent_left_is_int(self):
        state = calculate_token_warning_state([{"role": "user", "content": "hi"}])
        assert isinstance(state["percent_left"], int)

    def test_large_conversation_triggers_thresholds(self):
        """A conversation large enough should trigger warning/error thresholds."""
        # Create a conversation that is very close to the effective limit.
        # With a tiny context window, even a small conversation overflows.
        small_window = 100  # only 100 tokens total
        big_message = [{"role": "user", "content": "x" * 2000}]
        state = calculate_token_warning_state(big_message, context_window=small_window)
        assert state["is_above_warning_threshold"] is True
        assert state["is_above_error_threshold"] is True
        assert state["is_above_auto_compact_threshold"] is True
        assert state["is_at_blocking_limit"] is True
        assert state["percent_left"] == 0

    def test_custom_context_window(self):
        state = calculate_token_warning_state([], context_window=50_000)
        effective = 50_000 - MAX_OUTPUT_TOKENS_FOR_SUMMARY
        assert state["tokens_left"] == effective


# =========================================================================
# should_auto_compact tests
# =========================================================================


class TestShouldAutoCompact:
    """Tests for should_auto_compact."""

    def test_empty_conversation_no_compact(self):
        tracker = AutoCompactTracker()
        assert should_auto_compact(tracker, []) is False

    def test_circuit_breaker_prevents_compact(self):
        tracker = AutoCompactTracker()
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            tracker.record_failure()

        # Even with a huge conversation and tiny window, circuit breaker blocks it
        big = [{"role": "user", "content": "x" * 5000}]
        assert should_auto_compact(tracker, big, context_window=100) is False

    def test_triggers_when_above_threshold(self):
        tracker = AutoCompactTracker()
        big = [{"role": "user", "content": "x" * 5000}]
        result = should_auto_compact(tracker, big, context_window=100)
        assert result is True

    def test_no_trigger_below_threshold(self):
        tracker = AutoCompactTracker()
        small = [{"role": "user", "content": "hi"}]
        result = should_auto_compact(tracker, small, context_window=DEFAULT_CONTEXT_WINDOW)
        assert result is False


# =========================================================================
# Compactor tests
# =========================================================================


class TestCompactor:
    """Tests for the Compactor class."""

    @pytest.fixture
    def compactor(self):
        return Compactor()

    @pytest.mark.asyncio
    async def test_no_compaction_when_too_few_messages(self, compactor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = await compactor.compact(messages, keep_recent=4)
        assert result["was_compacted"] is False
        assert result["compacted_messages"] == messages
        assert result["summary"] == ""
        assert result["tokens_saved"] == 0

    @pytest.mark.asyncio
    async def test_no_compaction_exactly_keep_recent(self, compactor):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
        result = await compactor.compact(messages, keep_recent=4)
        assert result["was_compacted"] is False

    @pytest.mark.asyncio
    @patch("morgan.compaction.compactor.Compactor._summarise_via_llm")
    async def test_compaction_with_mocked_llm(self, mock_llm, compactor):
        mock_llm.return_value = "Summary of earlier conversation."

        messages = [
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old reply 1"},
            {"role": "user", "content": "Old message 2"},
            {"role": "assistant", "content": "Old reply 2"},
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent 2"},
            {"role": "user", "content": "Recent 3"},
            {"role": "assistant", "content": "Recent 4"},
        ]

        result = await compactor.compact(messages, keep_recent=4)

        assert result["was_compacted"] is True
        assert result["summary"] == "Summary of earlier conversation."
        assert result["tokens_saved"] > 0

        # Compacted messages: 1 summary + 4 recent = 5
        assert len(result["compacted_messages"]) == 5
        assert result["compacted_messages"][0]["role"] == "system"
        assert "[Conversation summary]" in result["compacted_messages"][0]["content"]

        # Recent messages preserved
        assert result["compacted_messages"][1]["content"] == "Recent 1"
        assert result["compacted_messages"][4]["content"] == "Recent 4"

    @pytest.mark.asyncio
    @patch("morgan.compaction.compactor.Compactor._summarise_via_llm")
    async def test_compaction_llm_failure_falls_back(self, mock_llm, compactor):
        mock_llm.side_effect = RuntimeError("LLM unavailable")

        messages = [
            {"role": "user", "content": "Old message"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent reply"},
        ]

        result = await compactor.compact(messages, keep_recent=2)

        assert result["was_compacted"] is True
        # Fallback summary should contain the old content
        assert "Old message" in result["summary"]
        assert len(result["compacted_messages"]) == 3  # 1 summary + 2 recent

    @pytest.mark.asyncio
    @patch("morgan.compaction.compactor.Compactor._summarise_via_llm")
    async def test_compaction_keep_recent_zero(self, mock_llm, compactor):
        """keep_recent=0 means all messages are compacted."""
        mock_llm.return_value = "Everything summarised."

        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]

        result = await compactor.compact(messages, keep_recent=0)
        assert result["was_compacted"] is True
        # Only the summary message remains
        assert len(result["compacted_messages"]) == 1
        assert result["compacted_messages"][0]["role"] == "system"

    def test_messages_to_text(self, compactor):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        text = compactor._messages_to_text(messages)
        assert "user: Hello" in text
        assert "assistant: Hi" in text

    def test_truncation_fallback_short(self, compactor):
        result = compactor._truncation_fallback("short text")
        assert result == "short text"

    def test_truncation_fallback_long(self, compactor):
        long_text = "a" * 10000
        result = compactor._truncation_fallback(long_text, max_chars=100)
        assert len(result) < len(long_text)
        assert "[... earlier context truncated ...]" in result


# =========================================================================
# Integration-style test: tracker + should_auto_compact + compactor
# =========================================================================


class TestIntegration:
    """Integration test combining tracker, decision function, and compactor."""

    @pytest.mark.asyncio
    @patch("morgan.compaction.compactor.Compactor._summarise_via_llm")
    async def test_full_workflow(self, mock_llm):
        mock_llm.return_value = "Conversation summary."

        tracker = AutoCompactTracker()
        tracker.new_turn()

        # Build a conversation that exceeds threshold in a tiny window
        messages = [
            {"role": "user", "content": f"Message {i}" * 50}
            for i in range(20)
        ]

        small_window = 500

        if should_auto_compact(tracker, messages, context_window=small_window):
            compactor = Compactor()
            result = await compactor.compact(
                messages, keep_recent=4, context_window=small_window
            )
            if result["was_compacted"]:
                tracker.record_success()
            else:
                tracker.record_failure()

        assert tracker.compacted is True
        assert tracker.consecutive_failures == 0

    def test_circuit_breaker_workflow(self):
        tracker = AutoCompactTracker()
        messages = [{"role": "user", "content": "x" * 5000}]

        # First few attempts should be allowed
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            assert should_auto_compact(tracker, messages, context_window=100) is True
            tracker.record_failure()

        # After MAX_CONSECUTIVE_FAILURES, blocked
        assert should_auto_compact(tracker, messages, context_window=100) is False

        # Recovery
        tracker.record_success()
        assert should_auto_compact(tracker, messages, context_window=100) is True
