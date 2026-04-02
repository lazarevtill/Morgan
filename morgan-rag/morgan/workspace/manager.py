"""
WorkspaceManager — owns the on-disk workspace lifecycle.

Responsibilities:
- Bootstrap: create directory structure and default markdown files.
- Load/update: read and write SOUL.md, USER.md, MEMORY.md, TOOLS.md,
  HEARTBEAT.md with safety limits (line/byte truncation for MEMORY.md).
- Daily logs: append timestamped entries and load by date.
- Session context: assemble the full context dict, enforcing the security
  gate that restricts MEMORY.md to main/dm sessions only.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from morgan.workspace.templates import (
    HEARTBEAT_TEMPLATE,
    MEMORY_TEMPLATE,
    SOUL_TEMPLATE,
    TOOLS_TEMPLATE,
    USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Limits for MEMORY.md content returned by load_memory().
_MEMORY_MAX_LINES = 200
_MEMORY_MAX_BYTES = 25 * 1024  # 25 KB

# Session types that are allowed to see MEMORY.md.
_MEMORY_ALLOWED_SESSIONS = frozenset({"main", "dm"})


class WorkspaceManager:
    """Manage the on-disk workspace for Morgan.

    Parameters
    ----------
    workspace_dir:
        Absolute path to the workspace root.  Created on
        :meth:`bootstrap` if it doesn't exist.
    """

    # Mapping of filename -> default template content.
    _DEFAULT_FILES: dict[str, str] = {
        "SOUL.md": SOUL_TEMPLATE,
        "USER.md": USER_TEMPLATE,
        "MEMORY.md": MEMORY_TEMPLATE,
        "TOOLS.md": TOOLS_TEMPLATE,
        "HEARTBEAT.md": HEARTBEAT_TEMPLATE,
    }

    def __init__(self, workspace_dir: Path) -> None:
        self._dir = workspace_dir

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Create workspace directory structure and default files.

        Existing files are never overwritten — only missing ones are
        created with their template defaults.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "memory").mkdir(exist_ok=True)

        for filename, template in self._DEFAULT_FILES.items():
            path = self._dir / filename
            if not path.exists():
                path.write_text(template, encoding="utf-8")
                logger.info("Created default %s", filename)

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_soul(self) -> str:
        """Return the contents of SOUL.md."""
        return self._read("SOUL.md")

    def load_user(self) -> str:
        """Return the contents of USER.md."""
        return self._read("USER.md")

    def load_tools(self) -> str:
        """Return the contents of TOOLS.md."""
        return self._read("TOOLS.md")

    def load_heartbeat(self) -> str:
        """Return the contents of HEARTBEAT.md."""
        return self._read("HEARTBEAT.md")

    def load_memory(self) -> Optional[str]:
        """Return MEMORY.md contents, truncated to safe limits.

        Returns ``None`` if the file does not exist.  Otherwise the
        content is capped at 200 lines **and** 25 KB to avoid blowing
        up the context window.
        """
        path = self._dir / "MEMORY.md"
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        return self._truncate_memory(content)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_soul(self, content: str) -> None:
        """Overwrite SOUL.md with *content*."""
        self._write("SOUL.md", content)

    def update_user(self, content: str) -> None:
        """Overwrite USER.md with *content*."""
        self._write("USER.md", content)

    def update_memory(self, content: str) -> None:
        """Overwrite MEMORY.md with *content*."""
        self._write("MEMORY.md", content)

    # ------------------------------------------------------------------
    # Daily logs
    # ------------------------------------------------------------------

    def load_daily_log(self, date: Optional[datetime] = None) -> Optional[str]:
        """Load the daily log for *date* (defaults to today).

        Returns ``None`` if no log file exists for that date.
        """
        path = self._daily_log_path(date)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def append_daily_log(self, entry: str, date: Optional[datetime] = None) -> None:
        """Append a timestamped entry to the daily log.

        Each entry is prefixed with ``[HH:MM]``.
        """
        now = date or datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%H:%M")
        line = f"[{timestamp}] {entry}\n"

        path = self._daily_log_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Session context
    # ------------------------------------------------------------------

    def load_session_context(
        self, session_type: str = "main"
    ) -> Dict[str, Any]:
        """Assemble the full context dict for a session.

        Returns a dict with keys:
            soul, user, tools, daily_log_today, daily_log_yesterday, memory

        **Security gate:** ``memory`` is set to ``None`` for any
        session type other than ``"main"`` or ``"dm"``.
        """
        now = datetime.now(tz=timezone.utc)
        yesterday = now - timedelta(days=1)

        # Enforce the security gate: only main/dm sessions see memory.
        if session_type in _MEMORY_ALLOWED_SESSIONS:
            memory = self.load_memory()
        else:
            memory = None

        return {
            "soul": self.load_soul(),
            "user": self.load_user(),
            "tools": self.load_tools(),
            "daily_log_today": self.load_daily_log(date=now),
            "daily_log_yesterday": self.load_daily_log(date=yesterday),
            "memory": memory,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read(self, filename: str) -> str:
        path = self._dir / filename
        return path.read_text(encoding="utf-8")

    def _write(self, filename: str, content: str) -> None:
        path = self._dir / filename
        path.write_text(content, encoding="utf-8")

    def _daily_log_path(self, date: Optional[datetime] = None) -> Path:
        """Return the path to the daily log file for *date*."""
        d = date or datetime.now(tz=timezone.utc)
        filename = d.strftime("%Y-%m-%d") + ".md"
        return self._dir / "memory" / filename

    @staticmethod
    def _truncate_memory(content: str) -> str:
        """Truncate *content* to line and byte limits."""
        lines = content.split("\n")
        if len(lines) > _MEMORY_MAX_LINES:
            lines = lines[:_MEMORY_MAX_LINES]
        truncated = "\n".join(lines)

        encoded = truncated.encode("utf-8")
        if len(encoded) > _MEMORY_MAX_BYTES:
            # Decode back safely after byte-slicing to avoid splitting a
            # multi-byte character.
            truncated = encoded[:_MEMORY_MAX_BYTES].decode("utf-8", errors="ignore")

        return truncated
