"""Tests for the security module."""

import pytest

from morgan.security.memory_gating import MemoryGate
from morgan.security.allowlist import ChannelAllowlist
from morgan.security.permission_modes import SessionPermissionMode


# --- MemoryGate ---

class TestMemoryGate:
    def test_main_allowed(self):
        assert MemoryGate.should_load_memory("main") is True

    def test_dm_allowed(self):
        assert MemoryGate.should_load_memory("dm") is True

    def test_group_denied(self):
        assert MemoryGate.should_load_memory("group") is False

    def test_empty_denied(self):
        assert MemoryGate.should_load_memory("") is False

    def test_arbitrary_denied(self):
        assert MemoryGate.should_load_memory("api") is False


# --- ChannelAllowlist ---

class TestChannelAllowlistOpen:
    def test_open_allows_everyone(self):
        al = ChannelAllowlist(policy="open")
        assert al.is_allowed("anyone") is True
        assert al.is_allowed("stranger") is True


class TestChannelAllowlistAllowlist:
    def test_only_approved(self):
        al = ChannelAllowlist(policy="allowlist", allowed_ids={"alice"})
        assert al.is_allowed("alice") is True
        assert al.is_allowed("bob") is False

    def test_add_and_remove(self):
        al = ChannelAllowlist(policy="allowlist")
        al.add("bob")
        assert al.is_allowed("bob") is True
        al.remove("bob")
        assert al.is_allowed("bob") is False

    def test_remove_missing_no_error(self):
        al = ChannelAllowlist(policy="allowlist")
        al.remove("ghost")  # should not raise


class TestChannelAllowlistPairing:
    def test_pairing_flow(self):
        al = ChannelAllowlist(policy="pairing")
        assert al.is_allowed("eve") is False
        code = al.request_pairing("eve")
        assert isinstance(code, str) and len(code) == 8
        assert al.approve_pairing(code) is True
        assert al.is_allowed("eve") is True

    def test_bad_code(self):
        al = ChannelAllowlist(policy="pairing")
        assert al.approve_pairing("badcode") is False

    def test_code_single_use(self):
        al = ChannelAllowlist(policy="pairing")
        code = al.request_pairing("frank")
        al.approve_pairing(code)
        assert al.approve_pairing(code) is False  # already consumed


class TestChannelAllowlistValidation:
    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            ChannelAllowlist(policy="bogus")


# --- SessionPermissionMode ---

class TestSessionPermissionMode:
    def test_three_modes(self):
        assert len(SessionPermissionMode) == 3

    def test_values(self):
        assert SessionPermissionMode.INTERACTIVE.value == "interactive"
        assert SessionPermissionMode.AUTONOMOUS.value == "autonomous"
        assert SessionPermissionMode.RESTRICTED.value == "restricted"
