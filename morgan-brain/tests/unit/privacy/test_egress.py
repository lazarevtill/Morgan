"""Unit tests for morgan_brain.privacy.egress and morgan_brain.privacy.crypto.

Coverage:
- Local gate (is_remote=False) passes messages through unchanged.
- Remote gate with a redactor: outbound messages have PII redacted; the inner
  client's ``last_messages`` reflect the redacted content.
- Remote gate: response text is rehydrated (original PII restored in the response).
- Remote gate raises PermissionError on a SECRET message when block_secret=True.
- Remote gate with block_secret=False does NOT raise on SECRET messages.
- Remote gate with redactor=None does not alter messages.
- Crypto seam: new_dek() returns 32 bytes and the module is importable without deps.
- seal/open round-trip (guarded by pytest.importorskip for cryptography).
- derive_kek round-trip (guarded by pytest.importorskip for argon2-cffi).
- Import errors are clear when optional deps are absent.
"""

from __future__ import annotations

import pytest

from morgan_brain.privacy.egress import EgressGate
from morgan_brain.privacy.redaction import EgressRedactor
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.wire import ChatMessage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def user_msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def assistant_msg(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


# ---------------------------------------------------------------------------
# Local gate — transparent pass-through
# ---------------------------------------------------------------------------


class TestLocalGate:
    @pytest.mark.asyncio
    async def test_local_passes_through_unchanged(self) -> None:
        inner = FakeChatClient(reply="hello back")
        gate = EgressGate(inner, is_remote=False, redactor=EgressRedactor())
        messages = [user_msg("my email is alice@example.com")]
        result = await gate.agenerate(messages, model="test-model")

        # The inner client received the ORIGINAL (un-redacted) messages
        assert inner.last_messages[0].content == "my email is alice@example.com"
        assert result.text == "hello back"

    @pytest.mark.asyncio
    async def test_local_secret_does_not_raise(self) -> None:
        """Local gate never blocks even SECRET-tier content."""
        inner = FakeChatClient(reply="ok")
        gate = EgressGate(inner, is_remote=False, redactor=EgressRedactor(), block_secret=True)
        messages = [user_msg("api_key = sk-abcdefghijklmnopqrstuvwxyz123456")]
        result = await gate.agenerate(messages, model="m")
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_local_stream_passes_through(self) -> None:
        inner = FakeChatClient(reply="streamed reply")
        gate = EgressGate(inner, is_remote=False, redactor=EgressRedactor())
        messages = [user_msg("tell me something")]

        deltas = []
        async for delta in gate.astream(messages, model="m"):
            deltas.append(delta)

        texts = [d.text for d in deltas if d.text]
        assert "streamed reply" in "".join(texts)


# ---------------------------------------------------------------------------
# Remote gate — redaction of outbound messages
# ---------------------------------------------------------------------------


class TestRemoteGateRedaction:
    @pytest.mark.asyncio
    async def test_outbound_email_is_redacted(self) -> None:
        inner = FakeChatClient(reply="got your message")
        redactor = EgressRedactor()
        gate = EgressGate(inner, is_remote=True, redactor=redactor)

        messages = [user_msg("contact me at alice@example.com")]
        await gate.agenerate(messages, model="m")

        sent_content = inner.last_messages[0].content
        assert "alice@example.com" not in sent_content, (
            f"Email should have been redacted, but got: {sent_content!r}"
        )
        assert "«" in sent_content  # placeholder present

    @pytest.mark.asyncio
    async def test_outbound_phone_is_redacted(self) -> None:
        inner = FakeChatClient(reply="ack")
        gate = EgressGate(inner, is_remote=True, redactor=EgressRedactor())
        await gate.agenerate([user_msg("call 415-555-0100 now")], model="m")
        assert "415-555-0100" not in inner.last_messages[0].content

    @pytest.mark.asyncio
    async def test_response_is_rehydrated(self) -> None:
        """The provider echoes the placeholder back; the gate must restore the original."""
        redactor = EgressRedactor()
        # Pre-populate the redactor's map by running a redact
        text = "user alice@example.com"
        redacted, _rmap = redactor.redact(text)
        # The placeholder that was created
        placeholder = next(iter(_rmap))

        # Inner client echoes the placeholder in its response
        inner = FakeChatClient(reply=f"thanks {placeholder}")
        gate = EgressGate(inner, is_remote=True, redactor=redactor)
        result = await gate.agenerate([user_msg(text)], model="m")

        assert "alice@example.com" in result.text, (
            f"Expected rehydrated email in response, got: {result.text!r}"
        )

    @pytest.mark.asyncio
    async def test_remote_no_redactor_passes_messages_unchanged(self) -> None:
        inner = FakeChatClient(reply="ok")
        gate = EgressGate(inner, is_remote=True, redactor=None)
        await gate.agenerate([user_msg("email alice@example.com")], model="m")
        # Without redactor, messages pass unchanged
        assert inner.last_messages[0].content == "email alice@example.com"

    @pytest.mark.asyncio
    async def test_remote_stream_redacts_outbound(self) -> None:
        """Verify that the stream path redacts outbound messages.

        FakeChatClient.astream does not record last_messages (it is a pure
        generator), so we verify redaction indirectly: capture the messages
        passed into astream by wrapping with a spy.
        """
        captured: list[list[ChatMessage]] = []

        class SpyFakeClient(FakeChatClient):
            async def _astream_impl(  # type: ignore[override]
                self,
                messages: list[ChatMessage],
                *,
                model: str,
                tools: object = None,
            ) -> object:
                captured.append(list(messages))
                async for delta in super()._astream_impl(messages, model=model, tools=tools):
                    yield delta

        inner = SpyFakeClient(reply="acknowledged")
        gate = EgressGate(inner, is_remote=True, redactor=EgressRedactor())
        messages = [user_msg("reach alice@example.com for updates")]

        # Drain the stream
        async for _ in gate.astream(messages, model="m"):
            pass

        # Outbound messages must have been redacted
        assert captured, "SpyFakeClient._astream_impl was never called"
        sent_content = captured[0][0].content
        assert "alice@example.com" not in sent_content, (
            f"Email should have been redacted, got: {sent_content!r}"
        )

    @pytest.mark.asyncio
    async def test_remote_stream_flushes_placeholder_without_finish_delta(self) -> None:
        """If the provider ends the stream without a finish delta, any placeholder
        that straddles the stream boundary must still be rehydrated and emitted."""
        from typing import AsyncIterator

        from morgan_brain.providers.wire import StreamDelta

        # Build a redactor and pre-populate its map so we know the placeholder.
        redactor = EgressRedactor()
        _redacted, rmap = redactor.redact("contact bob@example.org please")
        placeholder = next(iter(rmap))  # e.g. «EMAIL_1»

        # Split the placeholder across two chunks, no finish delta at the end.
        half = len(placeholder) // 2
        chunk_a = placeholder[:half]
        chunk_b = placeholder[half:]

        class NoFinishClient:
            """Fake client that yields a split placeholder with NO finish delta."""

            async def _astream_impl(
                self,
                messages: list[ChatMessage],
                *,
                model: str,
                tools: object = None,
            ) -> AsyncIterator[StreamDelta]:
                yield StreamDelta(kind="text_delta", text=chunk_a)
                yield StreamDelta(kind="text_delta", text=chunk_b)
                # intentionally no finish delta

            def astream(
                self,
                messages: list[ChatMessage],
                *,
                model: str,
                tools: object = None,
            ) -> AsyncIterator[StreamDelta]:
                return self._astream_impl(messages, model=model, tools=tools)

        gate = EgressGate(NoFinishClient(), is_remote=True, redactor=redactor)  # type: ignore[arg-type]

        collected_texts: list[str] = []
        async for delta in gate.astream([user_msg("contact bob@example.org please")], model="m"):
            if delta.kind == "text_delta" and delta.text:
                collected_texts.append(delta.text)

        output = "".join(collected_texts)
        assert "bob@example.org" in output, f"Expected rehydrated email in output, got: {output!r}"
        assert "«" not in output, f"Unrehydrated placeholder leaked into output: {output!r}"


# ---------------------------------------------------------------------------
# Remote gate — SECRET blocking
# ---------------------------------------------------------------------------


class TestSecretBlocking:
    @pytest.mark.asyncio
    async def test_secret_message_raises_permission_error(self) -> None:
        inner = FakeChatClient(reply="ok")
        gate = EgressGate(inner, is_remote=True, redactor=EgressRedactor(), block_secret=True)
        secret_msg = "api_key = sk-abcdefghijklmnopqrstuvwxyz123456"
        with pytest.raises(PermissionError, match="secret-tier"):
            await gate.agenerate([user_msg(secret_msg)], model="m")

    @pytest.mark.asyncio
    async def test_secret_message_no_raise_when_block_disabled(self) -> None:
        inner = FakeChatClient(reply="ok")
        gate = EgressGate(inner, is_remote=True, redactor=EgressRedactor(), block_secret=False)
        secret_msg = "api_key = sk-abcdefghijklmnopqrstuvwxyz123456"
        # Must not raise
        result = await gate.agenerate([user_msg(secret_msg)], model="m")
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_non_secret_does_not_raise(self) -> None:
        inner = FakeChatClient(reply="fine")
        gate = EgressGate(inner, is_remote=True, redactor=None, block_secret=True)
        await gate.agenerate([user_msg("the weather is nice today")], model="m")

    @pytest.mark.asyncio
    async def test_secret_in_stream_raises(self) -> None:
        inner = FakeChatClient(reply="ok")
        gate = EgressGate(inner, is_remote=True, redactor=EgressRedactor(), block_secret=True)
        secret_msg = "my token is ghp_abcdefghijklmnopqrstuvwxyz1234"
        with pytest.raises(PermissionError, match="secret-tier"):
            async for _ in gate.astream([user_msg(secret_msg)], model="m"):
                pass


# ---------------------------------------------------------------------------
# Crypto seam — module importable without optional deps
# ---------------------------------------------------------------------------


class TestCryptoSeam:
    def test_module_imports_without_optional_deps(self) -> None:
        """The crypto module must be importable even if cryptography/argon2 are absent."""
        import importlib

        # Re-import to verify no top-level optional import
        mod = importlib.import_module("morgan_brain.privacy.crypto")
        assert hasattr(mod, "new_dek")
        assert hasattr(mod, "seal")
        assert hasattr(mod, "open_sealed")
        assert hasattr(mod, "derive_kek")

    def test_new_dek_returns_32_bytes(self) -> None:
        from morgan_brain.privacy.crypto import new_dek

        dek = new_dek()
        assert isinstance(dek, bytes)
        assert len(dek) == 32

    def test_new_dek_is_random(self) -> None:
        from morgan_brain.privacy.crypto import new_dek

        assert new_dek() != new_dek()  # astronomically unlikely to collide

    def test_seal_raises_import_error_without_cryptography(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If cryptography is not installed, seal() must raise a clear ImportError."""
        import sys

        # Simulate absence of cryptography by removing it from sys.modules
        # and making the import fail.
        cryptography_key = "cryptography.hazmat.primitives.ciphers.aead"
        original = sys.modules.pop(cryptography_key, None)
        try:
            import builtins

            real_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if "cryptography" in name:
                    raise ImportError("mocked absence of cryptography")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", mock_import)

            from morgan_brain.privacy import crypto  # noqa: PLC0415

            import importlib

            importlib.reload(crypto)

            with pytest.raises(ImportError, match="cryptography"):
                crypto.seal(b"hello", b"\x00" * 32)
        finally:
            if original is not None:
                sys.modules[cryptography_key] = original


# ---------------------------------------------------------------------------
# Crypto seal/open round-trip — guarded by importorskip
# ---------------------------------------------------------------------------


class TestCryptoRoundTrip:
    @pytest.fixture(autouse=True)
    def require_cryptography(self) -> None:
        pytest.importorskip("cryptography", reason="cryptography package not installed")

    def test_seal_open_round_trip(self) -> None:
        """seal() then open_sealed() returns the exact original plaintext."""
        from morgan_brain.privacy.crypto import new_dek, open_sealed, seal

        dek = new_dek()
        plaintext = b"hello, Morgan!"
        ciphertext = seal(plaintext, dek)
        assert ciphertext != plaintext
        assert open_sealed(ciphertext, dek) == plaintext

    def test_seal_is_non_deterministic(self) -> None:
        """Two seals of the same plaintext produce different ciphertexts (random nonce)."""
        from morgan_brain.privacy.crypto import new_dek, seal

        dek = new_dek()
        ct1 = seal(b"data", dek)
        ct2 = seal(b"data", dek)
        assert ct1 != ct2

    def test_wrong_key_raises(self) -> None:
        from morgan_brain.privacy.crypto import new_dek, open_sealed, seal

        dek1 = new_dek()
        dek2 = new_dek()
        ct = seal(b"secret", dek1)
        with pytest.raises(Exception):  # cryptography.exceptions.InvalidTag
            open_sealed(ct, dek2)

    def test_seal_requires_32_byte_dek(self) -> None:
        from morgan_brain.privacy.crypto import seal

        with pytest.raises(ValueError, match="32 bytes"):
            seal(b"data", b"short")

    def test_open_requires_32_byte_dek(self) -> None:
        from morgan_brain.privacy.crypto import open_sealed

        with pytest.raises(ValueError, match="32 bytes"):
            open_sealed(b"x" * 28, b"short")


# ---------------------------------------------------------------------------
# KEK derivation — guarded by importorskip
# ---------------------------------------------------------------------------


class TestKEKDerivation:
    @pytest.fixture(autouse=True)
    def require_argon2(self) -> None:
        pytest.importorskip("argon2", reason="argon2-cffi package not installed")

    def test_derive_kek_returns_32_bytes(self) -> None:
        import os

        from morgan_brain.privacy.crypto import derive_kek

        salt = os.urandom(16)
        kek = derive_kek("mysecretpassphrase", salt)
        assert isinstance(kek, bytes)
        assert len(kek) == 32

    def test_derive_kek_is_deterministic(self) -> None:
        from morgan_brain.privacy.crypto import derive_kek

        salt = b"\x01" * 16
        kek1 = derive_kek("passphrase", salt)
        kek2 = derive_kek("passphrase", salt)
        assert kek1 == kek2

    def test_derive_kek_different_passphrase_different_key(self) -> None:
        from morgan_brain.privacy.crypto import derive_kek

        salt = b"\x02" * 16
        assert derive_kek("pass1", salt) != derive_kek("pass2", salt)

    def test_derive_kek_different_salt_different_key(self) -> None:
        from morgan_brain.privacy.crypto import derive_kek

        assert derive_kek("passphrase", b"\x01" * 16) != derive_kek("passphrase", b"\x02" * 16)
