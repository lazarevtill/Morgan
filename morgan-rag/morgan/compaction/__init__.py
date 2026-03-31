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
Context Compaction for Morgan AI Assistant.

Ported from Claude Code's autoCompact.ts and compact.ts, this module
provides automatic context window management:

- Token counting with tiktoken (falls back to len//4)
- Auto-compact tracking with circuit breaker
- Token warning state calculation
- LLM-driven conversation compaction

Usage:
    from morgan.compaction import (
        estimate_tokens,
        estimate_messages_tokens,
        AutoCompactTracker,
        get_effective_context_window,
        get_auto_compact_threshold,
        calculate_token_warning_state,
        should_auto_compact,
        Compactor,
    )
"""

from morgan.compaction.token_counter import estimate_tokens, estimate_messages_tokens
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

__all__ = [
    # Token counting
    "estimate_tokens",
    "estimate_messages_tokens",
    # Auto-compact tracker and functions
    "AutoCompactTracker",
    "get_effective_context_window",
    "get_auto_compact_threshold",
    "calculate_token_warning_state",
    "should_auto_compact",
    # Constants
    "MAX_OUTPUT_TOKENS_FOR_SUMMARY",
    "AUTOCOMPACT_BUFFER_TOKENS",
    "WARNING_THRESHOLD_BUFFER",
    "ERROR_THRESHOLD_BUFFER",
    "MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_CONTEXT_WINDOW",
    # Compactor
    "Compactor",
]
