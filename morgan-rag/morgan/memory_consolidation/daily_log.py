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
Daily Log Manager.

Manages daily memory log files in YYYY-MM-DD.md format.
Each entry is timestamped with [HH:MM] prefix.

Ported from OpenClaw's memory-core and Claude Code's memdir patterns.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import re


class DailyLogManager:
    """
    Manages daily memory log files.

    Daily logs live in ``memory_dir/YYYY-MM-DD.md`` files.
    Each appended entry receives a ``[HH:MM]`` timestamp prefix.

    Args:
        memory_dir: Directory where daily log files are stored.

    Example:
        >>> mgr = DailyLogManager(Path("memory"))
        >>> mgr.append("Learned about BM25 scoring")
        >>> mgr.read_today()
        '[14:30] Learned about BM25 scoring\\n'
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = Path(memory_dir)

    def _log_path(self, log_date: date) -> Path:
        """Return the file path for a given date's log."""
        return self.memory_dir / f"{log_date.isoformat()}.md"

    def append(self, entry: str, date: Optional[date] = None) -> None:
        """
        Append a timestamped entry to the daily log.

        Args:
            entry: The text content to append.
            date: The date for the log entry. Defaults to today.
        """
        log_date = date or datetime.now().date()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H:%M")
        line = f"[{timestamp}] {entry}\n"

        log_file = self._log_path(log_date)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def read_today(self) -> str:
        """
        Read today's log content.

        Returns:
            The full text content of today's log, or empty string if none.
        """
        return self.read_date(datetime.now().date())

    def read_date(self, log_date: date) -> str:
        """
        Read the log content for a specific date.

        Args:
            log_date: The date to read.

        Returns:
            The full text content of the log, or empty string if none.
        """
        log_file = self._log_path(log_date)
        if not log_file.exists():
            return ""
        return log_file.read_text(encoding="utf-8")

    def list_recent(self, days: int = 7) -> List[Dict]:
        """
        List recent daily logs within the given number of days.

        Args:
            days: Number of days to look back (default 7).

        Returns:
            List of dicts with ``date``, ``content``, and ``path`` keys,
            sorted newest first.
        """
        if not self.memory_dir.exists():
            return []

        cutoff = datetime.now().date() - timedelta(days=days)
        results = []

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

        for log_file in self.memory_dir.iterdir():
            if not log_file.is_file() or not date_pattern.match(log_file.name):
                continue

            date_str = log_file.stem  # YYYY-MM-DD
            try:
                log_date = date.fromisoformat(date_str)
            except ValueError:
                continue

            if log_date < cutoff:
                continue

            content = log_file.read_text(encoding="utf-8")
            results.append(
                {
                    "date": date_str,
                    "content": content,
                    "path": str(log_file),
                }
            )

        # Sort newest first
        results.sort(key=lambda x: x["date"], reverse=True)
        return results
