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
Memory Consolidator.

Reads recent daily logs and the current MEMORY.md, then uses an LLM
to produce an updated MEMORY.md that consolidates key information.

Ported from OpenClaw's memory-core and Claude Code's memdir patterns.
"""

import logging
from pathlib import Path
from typing import Optional

from morgan.memory_consolidation.daily_log import DailyLogManager

logger = logging.getLogger(__name__)


_CONSOLIDATION_PROMPT = """\
You are a memory consolidation assistant. Your job is to review recent daily \
log entries and the current MEMORY.md, then produce an updated MEMORY.md that \
captures the most important information.

## Current MEMORY.md
{existing_memory}

## Recent Daily Logs
{daily_logs}

## Instructions
1. Preserve important facts and insights from the existing MEMORY.md.
2. Integrate new information from the daily logs.
3. Remove redundant or outdated entries.
4. Organize information clearly with markdown headings.
5. Keep the document concise but comprehensive.
6. Output ONLY the new MEMORY.md content, starting with a # heading.
"""


class MemoryConsolidator:
    """
    Consolidates daily memory logs into a single MEMORY.md using an LLM.

    Args:
        workspace_dir: Root workspace directory. Daily logs are expected
            in ``workspace_dir/memory/``, and MEMORY.md is written to
            ``workspace_dir/MEMORY.md``.

    Example:
        >>> consolidator = MemoryConsolidator(Path("workspace"))
        >>> new_content = consolidator.consolidate(days_to_review=7)
        >>> if new_content:
        ...     print("Memory updated!")
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.memory_dir = self.workspace_dir / "memory"
        self.memory_md_path = self.workspace_dir / "MEMORY.md"
        self.log_manager = DailyLogManager(self.memory_dir)

    def consolidate(self, days_to_review: int = 7) -> Optional[str]:
        """
        Consolidate recent daily logs into MEMORY.md via LLM.

        Reads the last ``days_to_review`` days of daily logs and the
        current MEMORY.md content, sends them to the LLM for
        consolidation, and writes the result back to MEMORY.md.

        Args:
            days_to_review: Number of days of logs to review (default 7).

        Returns:
            The new MEMORY.md content string, or None if there were no
            logs to consolidate or an error occurred.
        """
        # Gather recent logs
        recent_logs = self.log_manager.list_recent(days=days_to_review)
        if not recent_logs:
            logger.info("No recent logs to consolidate.")
            return None

        # Read existing MEMORY.md
        existing_memory = ""
        if self.memory_md_path.exists():
            existing_memory = self.memory_md_path.read_text(encoding="utf-8")

        if not existing_memory:
            existing_memory = "(No existing MEMORY.md)"

        # Format daily logs for the prompt
        daily_logs_text = self._format_logs(recent_logs)

        # Build prompt
        prompt = _CONSOLIDATION_PROMPT.format(
            existing_memory=existing_memory,
            daily_logs=daily_logs_text,
        )

        # Call LLM
        try:
            from morgan.services import get_llm_service

            llm = get_llm_service()
            response = llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise memory consolidation assistant. "
                    "Output only the updated MEMORY.md content."
                ),
            )
            new_content = response.content
        except Exception as e:
            logger.error("Failed to consolidate memory via LLM: %s", e)
            return None

        # Write updated MEMORY.md
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_md_path.write_text(new_content, encoding="utf-8")

        logger.info("Memory consolidated successfully.")
        return new_content

    def _format_logs(self, logs: list) -> str:
        """
        Format a list of log entries for inclusion in the LLM prompt.

        Args:
            logs: List of dicts with ``date`` and ``content`` keys.

        Returns:
            Formatted string with each day's log under a heading.
        """
        parts = []
        for entry in logs:
            parts.append(f"### {entry['date']}\n{entry['content']}")
        return "\n".join(parts)
