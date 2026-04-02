# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Auto-compact tracking and threshold calculations.

Ported from Claude Code's autoCompact.ts. Provides a circuit-breaker-based
tracker that decides when context compaction should fire, plus functions
to calculate token budget thresholds and warning states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from morgan.compaction.token_counter import estimate_messages_tokens

# ---------------------------------------------------------------------------
# Constants (mirroring Claude Code's autoCompact.ts)
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS_FOR_SUMMARY: int = 20_000
"""Max tokens reserved for the compaction summary output."""

AUTOCOMPACT_BUFFER_TOKENS: int = 13_000
"""Buffer tokens subtracted from the context window to determine the
auto-compact threshold (keeps headroom for ongoing generation)."""

WARNING_THRESHOLD_BUFFER: int = 20_000
"""Token buffer for the warning threshold (yellow zone)."""

ERROR_THRESHOLD_BUFFER: int = 20_000
"""Token buffer for the error threshold (red zone)."""

MAX_CONSECUTIVE_FAILURES: int = 3
"""Number of consecutive compaction failures before the circuit breaker trips."""

DEFAULT_CONTEXT_WINDOW: int = 200_000
"""Default context window size in tokens."""


# ---------------------------------------------------------------------------
# AutoCompactTracker dataclass
# ---------------------------------------------------------------------------


@dataclass
class AutoCompactTracker:
    """
    Tracks auto-compact state with a circuit breaker.

    The circuit breaker prevents infinite retry loops when the LLM
    compaction keeps failing (e.g., service down).

    Attributes:
        compacted: Whether the conversation has been compacted at least once.
        turn_counter: Running count of conversation turns.
        turn_id: Identifier for the current turn (incremented via new_turn).
        consecutive_failures: How many compaction attempts failed in a row.
    """

    compacted: bool = False
    turn_counter: int = 0
    turn_id: int = 0
    consecutive_failures: int = 0

    @property
    def circuit_breaker_tripped(self) -> bool:
        """True when too many consecutive failures have occurred."""
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def record_success(self) -> None:
        """Record a successful compaction, resetting the failure counter."""
        self.consecutive_failures = 0
        self.compacted = True

    def record_failure(self) -> None:
        """Record a failed compaction attempt."""
        self.consecutive_failures += 1

    def new_turn(self) -> int:
        """Advance to the next conversation turn and return its id."""
        self.turn_counter += 1
        self.turn_id += 1
        return self.turn_id


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_effective_context_window(context_window: int = DEFAULT_CONTEXT_WINDOW) -> int:
    """
    Compute the effective context window available for conversation content.

    Subtracts the output-token reservation so we know how many tokens of
    *input* we can actually use.

    Args:
        context_window: Total context window size.

    Returns:
        Effective input token budget.
    """
    return context_window - MAX_OUTPUT_TOKENS_FOR_SUMMARY


def get_auto_compact_threshold(context_window: int = DEFAULT_CONTEXT_WINDOW) -> int:
    """
    Compute the token count at which auto-compaction should trigger.

    This is the effective context window minus the auto-compact buffer.

    Args:
        context_window: Total context window size.

    Returns:
        Token threshold for triggering auto-compaction.
    """
    return get_effective_context_window(context_window) - AUTOCOMPACT_BUFFER_TOKENS


def calculate_token_warning_state(
    messages: List[Dict[str, Any]],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> Dict[str, Any]:
    """
    Calculate the token warning state for the current conversation.

    Returns a dict describing how close the conversation is to the
    various context-window limits.

    Args:
        messages: Current conversation messages.
        context_window: Total context window size in tokens.

    Returns:
        Dict with keys:
            - percent_left (int): Percentage of the context window remaining.
            - tokens_left (int): Absolute tokens remaining.
            - is_above_warning_threshold (bool): In the yellow warning zone.
            - is_above_error_threshold (bool): In the red error zone.
            - is_above_auto_compact_threshold (bool): Should auto-compact.
            - is_at_blocking_limit (bool): No room left at all.
    """
    effective = get_effective_context_window(context_window)
    token_count = estimate_messages_tokens(messages)
    tokens_left = max(effective - token_count, 0)

    percent_left = int((tokens_left / effective) * 100) if effective > 0 else 0

    auto_compact_threshold = get_auto_compact_threshold(context_window)
    warning_threshold = effective - WARNING_THRESHOLD_BUFFER
    error_threshold = effective - ERROR_THRESHOLD_BUFFER

    return {
        "percent_left": percent_left,
        "tokens_left": tokens_left,
        "is_above_warning_threshold": token_count >= warning_threshold,
        "is_above_error_threshold": token_count >= error_threshold,
        "is_above_auto_compact_threshold": token_count >= auto_compact_threshold,
        "is_at_blocking_limit": tokens_left == 0,
    }


def should_auto_compact(
    tracker: AutoCompactTracker,
    messages: List[Dict[str, Any]],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> bool:
    """
    Decide whether auto-compaction should run right now.

    Compaction is skipped if the circuit breaker has tripped
    (too many consecutive failures).

    Args:
        tracker: The current AutoCompactTracker state.
        messages: Current conversation messages.
        context_window: Total context window size.

    Returns:
        True if compaction should be triggered.
    """
    if tracker.circuit_breaker_tripped:
        return False

    state = calculate_token_warning_state(messages, context_window)
    return state["is_above_auto_compact_threshold"]
