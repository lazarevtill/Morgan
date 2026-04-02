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
Token counting utilities for context compaction.

Uses tiktoken (encoding_for_model gpt-4) when available,
falls back to len(text) // 4 as a rough approximation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Attempt to import tiktoken for accurate token counting
_tiktoken_encoding = None

try:
    import tiktoken

    _tiktoken_encoding = tiktoken.encoding_for_model("gpt-4")
except (ImportError, Exception):
    _tiktoken_encoding = None


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses tiktoken with the gpt-4 encoding if available,
    otherwise falls back to len(text) // 4.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    if _tiktoken_encoding is not None:
        return len(_tiktoken_encoding.encode(text))

    # Fallback: roughly 4 characters per token
    return len(text) // 4


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate the total number of tokens across a list of chat messages.

    Each message is expected to have at least a "content" field.
    Per-message overhead (role, separators) adds ~4 tokens per message,
    matching the OpenAI chat format overhead.

    Args:
        messages: List of message dicts, each with at least "role" and "content".

    Returns:
        Estimated total token count.
    """
    if not messages:
        return 0

    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # Per-message overhead: role name + separators (~4 tokens)
        total += 4

    # Every reply is primed with <|start|>assistant<|message|> (~3 tokens)
    total += 3

    return total
