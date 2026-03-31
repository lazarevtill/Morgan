"""
Tests for morgan.workspace — SOUL.md + Workspace Pattern.

Covers:
- Path resolution (get_morgan_home, get_workspace_path, validation)
- Template constants (non-empty, contain expected sections)
- WorkspaceManager (bootstrap, load/update, daily logs, session context, security gate)
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from morgan.workspace.paths import (
    get_morgan_home,
    get_workspace_path,
    validate_workspace_path,
)
from morgan.workspace.templates import (
    HEARTBEAT_TEMPLATE,
    MEMORY_TEMPLATE,
    SOUL_TEMPLATE,
    TOOLS_TEMPLATE,
    USER_TEMPLATE,
)
from morgan.workspace.manager import WorkspaceManager


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestGetMorganHome:
    def test_default_is_dot_morgan_in_home(self):
        home = get_morgan_home()
        assert home == Path.home() / ".morgan"

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"MORGAN_HOME": "/tmp/custom_morgan"}):
            assert get_morgan_home() == Path("/tmp/custom_morgan")


class TestGetWorkspacePath:
    def test_default_inside_morgan_home(self):
        path = get_workspace_path()
        assert path == get_morgan_home() / "workspace"

    def test_env_override(self):
        with mock.patch.dict(
            os.environ, {"MORGAN_WORKSPACE_PATH": "/tmp/my_workspace"}
        ):
            assert get_workspace_path() == Path("/tmp/my_workspace")


class TestValidateWorkspacePath:
    def test_rejects_relative_path(self):
        with pytest.raises(ValueError, match="[Aa]bsolute"):
            validate_workspace_path(Path("relative/path"))

    def test_rejects_near_root(self):
        with pytest.raises(ValueError, match="[Rr]oot"):
            validate_workspace_path(Path("/"))

    def test_rejects_slash_etc(self):
        with pytest.raises(ValueError, match="[Rr]oot"):
            validate_workspace_path(Path("/etc"))

    def test_accepts_deep_absolute(self):
        # Should not raise
        validate_workspace_path(Path("/home/user/.morgan/workspace"))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_soul_template_nonempty(self):
        assert len(SOUL_TEMPLATE) > 100
        assert "Morgan" in SOUL_TEMPLATE

    def test_soul_template_sections(self):
        for keyword in ["identity", "boundar", "communicat", "growth"]:
            assert keyword.lower() in SOUL_TEMPLATE.lower(), (
                f"SOUL_TEMPLATE missing section related to '{keyword}'"
            )

    def test_user_template_nonempty(self):
        assert len(USER_TEMPLATE) > 50

    def test_user_template_sections(self):
        lower = USER_TEMPLATE.lower()
        for keyword in ["name", "timezone", "language", "preference"]:
            assert keyword in lower, (
                f"USER_TEMPLATE missing section related to '{keyword}'"
            )

    def test_memory_template_nonempty(self):
        assert len(MEMORY_TEMPLATE) > 50

    def test_memory_template_sections(self):
        lower = MEMORY_TEMPLATE.lower()
        for keyword in ["fact", "preference", "decision"]:
            assert keyword in lower, (
                f"MEMORY_TEMPLATE missing section related to '{keyword}'"
            )

    def test_tools_template_nonempty(self):
        assert len(TOOLS_TEMPLATE) > 50

    def test_heartbeat_template_nonempty(self):
        assert len(HEARTBEAT_TEMPLATE) > 50


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    """Return a WorkspaceManager with a temp directory, bootstrapped."""
    ws = WorkspaceManager(tmp_path / "ws")
    ws.bootstrap()
    return ws


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    """Return a bare temp path (not bootstrapped)."""
    return tmp_path / "ws"


class TestBootstrap:
    def test_creates_directory_structure(self, ws_dir: Path):
        wm = WorkspaceManager(ws_dir)
        wm.bootstrap()
        assert ws_dir.is_dir()
        assert (ws_dir / "memory").is_dir()

    def test_creates_default_files(self, ws_dir: Path):
        wm = WorkspaceManager(ws_dir)
        wm.bootstrap()
        for name in ["SOUL.md", "USER.md", "MEMORY.md", "TOOLS.md", "HEARTBEAT.md"]:
            assert (ws_dir / name).exists(), f"Missing {name}"

    def test_does_not_overwrite_existing(self, ws_dir: Path):
        ws_dir.mkdir(parents=True)
        soul_path = ws_dir / "SOUL.md"
        soul_path.write_text("custom soul")

        wm = WorkspaceManager(ws_dir)
        wm.bootstrap()

        assert soul_path.read_text() == "custom soul"

    def test_idempotent(self, workspace: WorkspaceManager):
        # Running bootstrap again should not fail or overwrite
        workspace.bootstrap()
        assert workspace.load_soul() == SOUL_TEMPLATE


class TestLoadFiles:
    def test_load_soul(self, workspace: WorkspaceManager):
        content = workspace.load_soul()
        assert "Morgan" in content

    def test_load_user(self, workspace: WorkspaceManager):
        content = workspace.load_user()
        assert len(content) > 0

    def test_load_tools(self, workspace: WorkspaceManager):
        content = workspace.load_tools()
        assert len(content) > 0

    def test_load_heartbeat(self, workspace: WorkspaceManager):
        content = workspace.load_heartbeat()
        assert len(content) > 0

    def test_load_memory_default(self, workspace: WorkspaceManager):
        content = workspace.load_memory()
        assert content is not None
        assert len(content) > 0

    def test_load_memory_truncates_lines(self, workspace: WorkspaceManager):
        """MEMORY.md content over 200 lines should be truncated."""
        long_content = "\n".join(f"line {i}" for i in range(300))
        workspace.update_memory(long_content)
        loaded = workspace.load_memory()
        assert loaded is not None
        lines = loaded.strip().split("\n")
        assert len(lines) <= 200

    def test_load_memory_truncates_bytes(self, workspace: WorkspaceManager):
        """MEMORY.md content over 25 KB should be truncated."""
        # Each line ~100 bytes, 300 lines = ~30 KB > 25 KB
        big_content = "\n".join("x" * 100 for _ in range(300))
        workspace.update_memory(big_content)
        loaded = workspace.load_memory()
        assert loaded is not None
        assert len(loaded.encode("utf-8")) <= 25 * 1024 + 100  # small margin for safety

    def test_load_memory_returns_none_if_missing(self, ws_dir: Path):
        wm = WorkspaceManager(ws_dir)
        ws_dir.mkdir(parents=True)
        # No MEMORY.md created
        result = wm.load_memory()
        assert result is None


class TestUpdateFiles:
    def test_update_soul(self, workspace: WorkspaceManager):
        workspace.update_soul("new soul content")
        assert workspace.load_soul() == "new soul content"

    def test_update_user(self, workspace: WorkspaceManager):
        workspace.update_user("new user content")
        assert workspace.load_user() == "new user content"

    def test_update_memory(self, workspace: WorkspaceManager):
        workspace.update_memory("new memory content")
        assert workspace.load_memory() == "new memory content"


# ---------------------------------------------------------------------------
# Daily logs
# ---------------------------------------------------------------------------


class TestDailyLogs:
    def test_load_daily_log_returns_none_when_absent(self, workspace: WorkspaceManager):
        result = workspace.load_daily_log()
        assert result is None

    def test_append_and_load_daily_log(self, workspace: WorkspaceManager):
        workspace.append_daily_log("Started working on feature X")
        log = workspace.load_daily_log()
        assert log is not None
        assert "Started working on feature X" in log
        # Should contain a timestamp in [HH:MM] format
        assert "[" in log and "]" in log

    def test_append_multiple_entries(self, workspace: WorkspaceManager):
        workspace.append_daily_log("entry one")
        workspace.append_daily_log("entry two")
        log = workspace.load_daily_log()
        assert log is not None
        assert "entry one" in log
        assert "entry two" in log

    def test_daily_log_with_specific_date(self, workspace: WorkspaceManager):
        specific_date = datetime(2025, 6, 15, 10, 30)
        workspace.append_daily_log("test entry", date=specific_date)
        log = workspace.load_daily_log(date=specific_date)
        assert log is not None
        assert "test entry" in log

    def test_daily_log_different_dates_isolated(self, workspace: WorkspaceManager):
        d1 = datetime(2025, 6, 15)
        d2 = datetime(2025, 6, 16)
        workspace.append_daily_log("day one", date=d1)
        workspace.append_daily_log("day two", date=d2)
        assert "day one" in workspace.load_daily_log(date=d1)
        assert "day two" in workspace.load_daily_log(date=d2)
        assert "day two" not in workspace.load_daily_log(date=d1)


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------


class TestSessionContext:
    def test_main_session_includes_memory(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context(session_type="main")
        assert "soul" in ctx
        assert "user" in ctx
        assert "tools" in ctx
        assert "memory" in ctx
        assert ctx["memory"] is not None  # bootstrapped, so MEMORY.md has content

    def test_main_session_includes_daily_logs(self, workspace: WorkspaceManager):
        workspace.append_daily_log("hello")
        ctx = workspace.load_session_context(session_type="main")
        assert "daily_log_today" in ctx
        assert ctx["daily_log_today"] is not None

    def test_dm_session_includes_memory(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context(session_type="dm")
        assert ctx["memory"] is not None

    def test_group_session_excludes_memory(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context(session_type="group")
        assert ctx["memory"] is None

    def test_cron_session_excludes_memory(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context(session_type="cron")
        assert ctx["memory"] is None

    def test_unknown_session_excludes_memory(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context(session_type="webhook")
        assert ctx["memory"] is None

    def test_session_context_keys(self, workspace: WorkspaceManager):
        ctx = workspace.load_session_context()
        expected_keys = {
            "soul",
            "user",
            "tools",
            "daily_log_today",
            "daily_log_yesterday",
            "memory",
        }
        assert set(ctx.keys()) == expected_keys


class TestSessionContextSecurity:
    """Dedicated test class for the memory security gate."""

    ALLOWED = ("main", "dm")
    BLOCKED = ("group", "cron", "webhook", "automation", "")

    def test_allowed_sessions_get_memory(self, workspace: WorkspaceManager):
        for stype in self.ALLOWED:
            ctx = workspace.load_session_context(session_type=stype)
            assert ctx["memory"] is not None, (
                f"session_type={stype!r} should include memory"
            )

    def test_blocked_sessions_get_none_memory(self, workspace: WorkspaceManager):
        for stype in self.BLOCKED:
            ctx = workspace.load_session_context(session_type=stype)
            assert ctx["memory"] is None, (
                f"session_type={stype!r} should NOT include memory"
            )
