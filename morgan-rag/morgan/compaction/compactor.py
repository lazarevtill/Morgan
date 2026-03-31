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
Conversation compactor.

Ported from Claude Code's compact.ts.  Splits conversation messages into
old (compactable) and recent (kept verbatim), then uses the LLM to
summarise the old portion.  Falls back to a naive text truncation when
the LLM call fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from morgan.compaction.auto_compact import MAX_OUTPUT_TOKENS_FOR_SUMMARY
from morgan.compaction.token_counter import estimate_messages_tokens

logger = logging.getLogger(__name__)

# System prompt sent to the LLM when generating the compaction summary.
_COMPACTION_SYSTEM_PROMPT = (
    "You are a conversation summariser. You will be given the earlier portion "
    "of a conversation between a user and an AI assistant. Produce a concise "
    "summary that preserves all important context, decisions, action items, "
    "code references, and any facts the assistant would need to continue the "
    "conversation without losing track. Output only the summary."
)


class Compactor:
    """
    Compacts a conversation by summarising older messages via the LLM.

    Usage::

        compactor = Compactor()
        result = await compactor.compact(messages)
        if result["was_compacted"]:
            messages = result["compacted_messages"]
    """

    async def compact(
        self,
        messages: List[Dict[str, Any]],
        keep_recent: int = 4,
        context_window: int = 200_000,
    ) -> Dict[str, Any]:
        """
        Compact conversation messages.

        Splits *messages* into an **old** portion (candidates for
        summarisation) and a **recent** portion (kept verbatim).
        The old portion is sent to the LLM for summarisation.

        If the LLM call fails, a truncated-text fallback is used
        instead so the caller always receives a valid result.

        Args:
            messages: Full conversation message list.
            keep_recent: Number of most-recent messages to keep verbatim.
            context_window: Total context window size in tokens.

        Returns:
            Dict with keys:
                - was_compacted (bool): Whether compaction actually happened.
                - compacted_messages (list): The new, shorter message list.
                - summary (str): The generated (or fallback) summary text.
                - tokens_saved (int): Estimated tokens reclaimed.
        """
        if len(messages) <= keep_recent:
            return {
                "was_compacted": False,
                "compacted_messages": list(messages),
                "summary": "",
                "tokens_saved": 0,
            }

        old_messages = messages[:-keep_recent] if keep_recent > 0 else list(messages)
        recent_messages = messages[-keep_recent:] if keep_recent > 0 else []

        old_tokens = estimate_messages_tokens(old_messages)

        # Build the text blob that the LLM will summarise
        old_text = self._messages_to_text(old_messages)

        # Attempt LLM summarisation; fall back to truncation on failure
        try:
            summary = await self._summarise_via_llm(old_text)
        except Exception as exc:
            logger.warning("LLM compaction failed (%s); using truncation fallback", exc)
            summary = self._truncation_fallback(old_text)

        # Build the summary message that replaces the old portion
        summary_message: Dict[str, Any] = {
            "role": "system",
            "content": f"[Conversation summary]\n{summary}",
        }

        compacted_messages = [summary_message] + list(recent_messages)
        new_tokens = estimate_messages_tokens(compacted_messages)
        tokens_saved = max(
            estimate_messages_tokens(messages) - new_tokens, 0
        )

        return {
            "was_compacted": True,
            "compacted_messages": compacted_messages,
            "summary": summary,
            "tokens_saved": tokens_saved,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
        """Serialise messages into a human-readable text block."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    @staticmethod
    async def _summarise_via_llm(text: str) -> str:
        """Call the Morgan LLM service to produce a summary."""
        from morgan.services import get_llm_service

        llm = get_llm_service()

        messages = [
            {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        response = await llm.achat(
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARY,
            temperature=0.3,
        )
        return response.content

    @staticmethod
    def _truncation_fallback(text: str, max_chars: int = 4000) -> str:
        """
        Produce a naive truncated summary when the LLM is unavailable.

        Keeps the first *max_chars* characters followed by a note.
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        return f"{truncated}\n\n[... earlier context truncated ...]"
