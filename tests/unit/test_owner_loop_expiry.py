"""Only explicit finite login mode can disable its fixed launchd label after expiry."""

import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from morgan_brain import owner_loop


@pytest.mark.parametrize("reason", ["expired", "finished", "launch_limit", "stopped", "shutdown"])
@pytest.mark.parametrize("login", [False, True])
def test_only_exhausted_explicit_login_mode_disables(tmp_path, monkeypatch, capsys, reason, login):
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixture")
    args = [
        "owner_loop",
        str(tmp_path),
        "--server",
        sys.executable,
        "--model",
        str(weights),
        "--seconds",
        "604800",
    ]
    if login:
        args.append("--login-agent")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sys, "platform", "darwin")

    async def serve(controller):
        return reason

    monkeypatch.setattr(owner_loop, "_serve", serve)
    disable = Mock(return_value=True)
    monkeypatch.setattr(owner_loop, "_disable_login_agent", disable)
    owner_loop.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == reason
    if login and reason in ("expired", "finished", "launch_limit"):
        disable.assert_called_once_with()
        assert payload["autostart_disabled"] is True
    else:
        disable.assert_not_called()
        assert "autostart_disabled" not in payload


def test_disable_command_has_fixed_scope_and_timeout(monkeypatch):
    run = Mock()
    monkeypatch.setattr(owner_loop.subprocess, "run", run)
    monkeypatch.setattr(owner_loop.os, "getuid", lambda: 123)
    assert owner_loop._disable_login_agent() is True
    run.assert_called_once_with(
        ["/bin/launchctl", "disable", "gui/123/org.morgan.owner-pilot"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )


@pytest.mark.parametrize(
    "error", [OSError("unavailable"), subprocess.TimeoutExpired("launchctl", 5)]
)
def test_disable_failure_is_reported_without_retry(monkeypatch, error):
    run = Mock(side_effect=error)
    monkeypatch.setattr(owner_loop.subprocess, "run", run)
    assert owner_loop._disable_login_agent() is False
    assert run.call_count == 1


@pytest.mark.parametrize("platform,seconds", [("linux", 604800), ("darwin", 3600)])
def test_login_mode_rejects_other_platform_or_bound(monkeypatch, platform, seconds):
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError, match="macOS and a seven-day"):
        owner_loop._validate_login_mode(seconds)
