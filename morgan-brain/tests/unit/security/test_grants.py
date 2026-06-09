"""Unit tests for capability-token grants on PermissionGate.

Covers: default-deny, AUTO back-compat, explicit grant, param guard,
        DENY-override, revoke, TTL expiry.  Clock is injected via
        ``time.monotonic`` monkeypatching to keep tests deterministic.
"""

from __future__ import annotations

import time

import pytest

from morgan_brain.security.permissions import Grant, PermissionGate, PermissionMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate(default: PermissionMode = PermissionMode.ASK) -> PermissionGate:
    return PermissionGate(default=default)


# ---------------------------------------------------------------------------
# Default-deny behaviour
# ---------------------------------------------------------------------------


def test_unknown_tool_default_ask_is_denied() -> None:
    """An unknown tool with default ASK and no grant must be denied."""
    gate = _gate(PermissionMode.ASK)
    assert gate.check("unknown_tool") is False


def test_unknown_tool_default_deny_is_denied() -> None:
    """DENY default also blocks unknown tools."""
    gate = _gate(PermissionMode.DENY)
    assert gate.check("unknown_tool") is False


# ---------------------------------------------------------------------------
# AUTO mode authorises (back-compat)
# ---------------------------------------------------------------------------


def test_auto_mode_authorises_without_grant() -> None:
    """AUTO mode must authorise calls even when no explicit grant has been installed."""
    gate = _gate(PermissionMode.AUTO)
    assert gate.check("any_tool") is True


def test_auto_mode_per_tool_authorises() -> None:
    """Per-tool AUTO also authorises."""
    gate = _gate(PermissionMode.ASK)  # default deny-like
    gate.set("calculator", PermissionMode.AUTO)
    assert gate.check("calculator") is True
    assert gate.check("other_tool") is False  # default still denies


# ---------------------------------------------------------------------------
# Explicit grant authorises
# ---------------------------------------------------------------------------


def test_explicit_grant_authorises() -> None:
    gate = _gate()  # default ASK
    gate.grant(Grant(tool="calculator"))
    assert gate.check("calculator") is True


def test_grant_with_matching_scope_authorises() -> None:
    gate = _gate()
    gate.grant(Grant(tool="fetcher", scope="execute"))
    assert gate.check("fetcher", scope="read") is True  # read ≤ execute
    assert gate.check("fetcher", scope="write") is True  # write ≤ execute
    assert gate.check("fetcher", scope="execute") is True


def test_grant_with_narrow_scope_blocks_wider_request() -> None:
    gate = _gate()
    gate.grant(Grant(tool="fetcher", scope="read"))
    assert gate.check("fetcher", scope="read") is True
    assert gate.check("fetcher", scope="write") is False  # write > read
    assert gate.check("fetcher", scope="execute") is False


def test_grant_with_allowed_params_accepts_subset() -> None:
    gate = _gate()
    gate.grant(Grant(tool="calculator", allowed_params=["expression"]))
    assert gate.check("calculator", params=["expression"]) is True
    assert gate.check("calculator", params=[]) is True  # empty is subset


def test_grant_with_allowed_params_rejects_superset() -> None:
    gate = _gate()
    gate.grant(Grant(tool="calculator", allowed_params=["expression"]))
    # "extra_param" is not in allowed_params → denied
    assert gate.check("calculator", params=["expression", "extra_param"]) is False


def test_grant_none_allowed_params_permits_any_params() -> None:
    gate = _gate()
    gate.grant(Grant(tool="calculator", allowed_params=None))
    assert gate.check("calculator", params=["a", "b", "c", "d"]) is True


# ---------------------------------------------------------------------------
# DENY overrides grant
# ---------------------------------------------------------------------------


def test_deny_mode_blocks_even_with_grant() -> None:
    """DENY must win regardless of any installed grant."""
    gate = _gate()
    gate.set("dangerous", PermissionMode.DENY)
    gate.grant(Grant(tool="dangerous"))
    assert gate.check("dangerous") is False


def test_deny_default_blocks_even_with_grant() -> None:
    gate = _gate(PermissionMode.DENY)
    gate.grant(Grant(tool="calculator"))
    # DENY default + no per-tool override → blocked
    assert gate.check("calculator") is False


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


def test_revoke_removes_grant() -> None:
    gate = _gate()
    gate.grant(Grant(tool="calculator"))
    assert gate.check("calculator") is True
    gate.revoke("calculator")
    assert gate.check("calculator") is False


def test_revoke_nonexistent_is_noop() -> None:
    gate = _gate()
    gate.revoke("does_not_exist")  # must not raise


# ---------------------------------------------------------------------------
# TTL expiry (clock-injected via monkeypatch)
# ---------------------------------------------------------------------------


def test_grant_within_ttl_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grant that has not expired must be accepted."""
    now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    gate = _gate()
    gate.grant(Grant(tool="calculator", ttl_seconds=60))

    # Advance time to just before expiry
    monkeypatch.setattr(time, "monotonic", lambda: now + 59.9)
    assert gate.check("calculator") is True


def test_grant_after_ttl_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """An expired grant must be treated as absent (default-deny)."""
    now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    gate = _gate()
    gate.grant(Grant(tool="calculator", ttl_seconds=60))

    # Advance time past expiry
    monkeypatch.setattr(time, "monotonic", lambda: now + 60.1)
    assert gate.check("calculator") is False


# ---------------------------------------------------------------------------
# Legacy API still works (back-compat smoke)
# ---------------------------------------------------------------------------


def test_legacy_allowed_still_works() -> None:
    gate = PermissionGate(default=PermissionMode.ASK)
    gate.set("bash", PermissionMode.DENY)
    assert gate.allowed("bash") is False
    assert gate.allowed("calculator") is True  # ASK ≠ DENY


def test_legacy_mode_for_still_works() -> None:
    gate = PermissionGate()
    assert gate.mode_for("anything") is PermissionMode.ASK  # default
    gate.set("bash", PermissionMode.DENY)
    assert gate.mode_for("bash") is PermissionMode.DENY
