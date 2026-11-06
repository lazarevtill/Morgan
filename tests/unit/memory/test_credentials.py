"""Synthetic shapes only: refusal must not rewrite evidence or partially persist turns."""

import json
from datetime import UTC, datetime

import pytest

from morgan_brain.app.chat import Chat
from morgan_brain.app.chatgpt_import import import_chatgpt, split_for_embedding
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.credentials import CredentialRefused, refuse_credentials
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.episodic import EventIdentityConflict
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.models import Memory, MemorySource, Message, Role, TemporalFact
from morgan_brain.providers.wire import ChatResult

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def token():
    return "gh" + "p_" + "A" * 36


@pytest.mark.parametrize(
    ("prefix", "payload", "rule"),
    [("gh" + kind + "_", "A" * 36, "github_token") for kind in "pousr"]
    + [
        ("github" + "_pat_", "A" * 22, "github_token"),
        ("sk-" + "proj-", "A" * 20, "openai_key"),
        ("sk-" + "svcacct-", "A" * 19 + "-", "openai_key"),
        ("sk-" + "ant-api03-", "A" * 19 + "_", "anthropic_key"),
    ],
)
def test_supported_shapes_are_refused_without_retaining_the_value(prefix, payload, rule):
    value = prefix + payload
    for text in (value, f'{{"key":"{value}"}}', f"_{value}_", f"слово{value}."):
        with pytest.raises(CredentialRefused) as caught:
            refuse_credentials(text)
        error = caught.value
        assert error.rule == rule
        assert vars(error) == {"rule": rule}
        assert value not in str(error) and value not in repr(error.args)


@pytest.mark.parametrize("kind", ["", "RSA ", "EC ", "OPENSSH ", "DSA ", "ENCRYPTED ", "PGP "])
def test_armored_private_keys_and_truncated_bodies_are_refused(kind):
    label = kind + "PRIVATE KEY" + (" BLOCK" if kind == "PGP " else "")
    for ending in ("", "\n[redacted:jwt]\n-----END " + label + "-----"):
        value = "-----BEGIN " + label + "-----\r\n" + "A" * 40 + ending
        with pytest.raises(CredentialRefused, match="private_key"):
            refuse_credentials(value)


def test_long_input_is_scanned_whole_without_window_or_placeholder_exemptions():
    value = "[redacted:github_token] " + "x" * 100_000 + token() + "y" * 100_000
    with pytest.raises(CredentialRefused, match="github_token"):
        refuse_credentials(value)


@pytest.mark.parametrize(
    "text",
    [
        "Document the GitHub token prefix ghp_ and rotate it",
        "Use sk-proj-<your-key> in the configured environment",
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN PUBLIC KEY-----\n" + "A" * 64,
        "sha256:" + "a" * 64,
        "password=12345678901234567890",  # explicitly not a generic-password detector
        "sk-" + "A" * 48,  # ambiguous legacy prefix is deliberately excluded
        "eyJ" + "A" * 20 + "." + "B" * 20 + "." + "C" * 20,
        "Meeting at 10:30; call +1 212 555 0100; owner@example.com",
        "gh" + "p_" + "A" * 35,  # below the documented supported shape
        "sk-" + "proj-" + "A" * 19,
    ],
)
def test_ordinary_text_and_out_of_scope_shapes_are_not_rewritten(text):
    assert refuse_credentials(text) is None


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return await super().embed(text)


class ReplyClient:
    def __init__(self, reply="Synthetic answer"):
        self.reply = reply
        self.calls = 0

    async def agenerate(self, messages, *, model):
        self.calls += 1
        return ChatResult(text=self.reply, model=model)


@pytest.fixture
def stack():
    conn = open_db(":memory:")
    embedder = CountingEmbedder()
    gate = MemoryGate(build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW))
    history = SessionHistoryStore(conn, clock=lambda: NOW)
    yield gate, conn, embedder, history
    conn.close()


async def test_new_event_refuses_before_embedding_and_leaves_caller_and_database_unchanged(stack):
    gate, conn, embedder, _ = stack
    memory = Memory(id="new", user_id="owner", content=token(), source=MemorySource.UNKNOWN)
    original, before = memory.model_dump(), conn.serialize()
    with pytest.raises(CredentialRefused):
        await gate.store(memory)
    assert embedder.calls == 0 and conn.serialize() == before
    assert memory.model_dump() == original


async def test_exact_replay_and_changed_identity_preserve_existing_contract(stack):
    gate, conn, embedder, _ = stack
    memory = Memory(id="source", user_id="owner", content="Synthetic exact source", author_id="x")
    await gate.store(memory)
    before = conn.serialize()
    assert await gate.store(memory) == "source"
    assert embedder.calls == 1 and conn.serialize() == before
    with pytest.raises(EventIdentityConflict):
        await gate.store(memory.model_copy(update={"content": token()}))
    assert embedder.calls == 1 and conn.serialize() == before
    saved = await gate.get("source", user_id="owner")
    assert saved.content == memory.content and saved.author_id == "x"
    assert saved.source is MemorySource.UNKNOWN


async def test_historical_exact_replay_is_not_a_rescan_or_a_rewrite(stack):
    gate, conn, embedder, _ = stack
    memory = Memory(id="legacy", user_id="owner", content="old source")
    await gate.store(memory)
    # Simulate a row accepted by an older release, in this synthetic in-memory database only.
    conn.execute("UPDATE memories SET content=? WHERE id='legacy'", (token(),))
    conn.commit()
    old = await gate.get("legacy", user_id="owner")
    before = conn.serialize()
    assert await gate.store(old) == "legacy"
    assert embedder.calls == 1 and conn.serialize() == before


async def test_refused_revision_cannot_close_or_replace_its_source(stack):
    gate, conn, embedder, _ = stack
    source = Memory(
        id="A",
        user_id="owner",
        content="Original source",
        source=MemorySource.USER_STATED,
        author_id="person:owner",
        effective_at=NOW,
    )
    await gate.store(source)
    before = conn.serialize()
    revision = source.model_copy(update={"id": "B", "content": token(), "revises_event_ids": ["A"]})
    with pytest.raises(CredentialRefused):
        await gate.store(revision)
    assert conn.serialize() == before and embedder.calls == 1
    evidence = await gate.evidence(user_id="owner", project="personal", evidence_ids=["A", "B"])
    assert evidence.missing_ids == ["B"]
    assert evidence.records[0].revision_state == "active"


@pytest.mark.parametrize("field", ["subject", "predicate", "object"])
async def test_fact_fields_refuse_without_partial_fact_or_project_write(stack, field):
    gate, conn, _, _ = stack
    fact = TemporalFact(user_id="owner", subject="user", predicate="likes", object="tea")
    fact = fact.model_copy(update={field: token()})
    original, before = fact.model_dump(), conn.serialize()
    with pytest.raises(CredentialRefused):
        await gate.upsert_fact(fact)
    assert conn.serialize() == before and fact.model_dump() == original


async def test_both_atomic_turn_events_are_screened_before_either_embedding(stack):
    gate, conn, embedder, history = stack
    memories = [
        Memory(id="q", user_id="owner", content="Question"),
        Memory(id="a", user_id="owner", content=token()),
    ]
    entries = [
        ("owner:s", "personal", Message(user_id="owner", role=role, content=m.content))
        for m, role in zip(memories, (Role.USER, Role.ASSISTANT), strict=True)
    ]
    before = conn.serialize()
    with pytest.raises(CredentialRefused):
        await gate.store_turn(
            memories,
            history=history,
            history_entries=entries,
            expected_generation=gate.capture_erasure_generation(),
        )
    assert embedder.calls == 0 and conn.serialize() == before


@pytest.mark.parametrize("strict", [False, True])
async def test_ask_refuses_before_strict_counting_recall_and_model(stack, strict, monkeypatch):
    gate, conn, embedder, history = stack
    client = ReplyClient()
    chat = Chat(gate=gate, history=history, client=client, model="fake", clock=lambda: NOW)

    async def unexpected(*args, **kwargs):
        pytest.fail("strict preparation must not run")

    monkeypatch.setattr(chat, "_prepare_strict", unexpected)
    before = conn.serialize()
    with pytest.raises(CredentialRefused):
        await chat.ask(user_id="owner", project="personal", text=token(), strict_context=strict)
    assert embedder.calls == client.calls == 0 and conn.serialize() == before


async def test_model_reply_refusal_leaves_no_question_history_or_event(stack):
    gate, conn, embedder, history = stack
    client = ReplyClient(token())
    chat = Chat(gate=gate, history=history, client=client, model="fake", clock=lambda: NOW)
    before = conn.serialize()
    with pytest.raises(CredentialRefused):
        await chat.ask(user_id="owner", project="personal", text="Synthetic question")
    assert client.calls == 1 and embedder.calls == 1  # recall query only, no turn embedding
    assert conn.serialize() == before


async def test_import_checks_whole_visible_turn_before_any_piece_embedding(stack, tmp_path):
    gate, conn, embedder, _ = stack
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "conversation",
                    "mapping": {
                        "n": {
                            "message": {
                                "id": "message",
                                "author": {"role": "user"},
                                "content": {
                                    "content_type": "text",
                                    "parts": ["x" * 5990 + token()],
                                },
                            }
                        }
                    },
                }
            ]
        )
    )

    # Neither individual piece contains the complete shape; whole-source screening must catch it.
    for piece in split_for_embedding("x" * 5990 + token()):
        assert refuse_credentials(piece) is None
    before = conn.serialize()
    with pytest.raises(CredentialRefused):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert embedder.calls == 0 and conn.serialize() == before


@pytest.mark.parametrize(
    "armor",
    [
        "\n",
        "Version: Synthetic test\n\n",
        "Comment: Synthetic\r\n\r\n",
        "Proc-Type: 4,ENCRYPTED\nDEK-Info: AES-256-CBC,SYNTHETIC\n\n",
    ],
)
@pytest.mark.parametrize("label", ["PGP PRIVATE KEY BLOCK", "RSA PRIVATE KEY"])
def test_standard_private_key_armor_headers_and_blank_lines_refuse(armor, label):
    text = "-----BEGIN " + label + "-----\n" + armor + "A" * 64
    with pytest.raises(CredentialRefused, match="private_key"):
        refuse_credentials(text)


async def test_import_legacy_exact_resume_preserves_provenance_and_does_not_embed(
    stack, tmp_path, monkeypatch
):
    gate, conn, embedder, _ = stack
    path = tmp_path / "legacy-synthetic.json"
    value = token()
    conversation = {
        "id": "legacy-conversation",
        "mapping": {
            "n": {
                "message": {
                    "id": "legacy-message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [value]},
                }
            }
        },
    }
    path.write_text(json.dumps([conversation]))
    # Simulate an import made by an older release using synthetic data only.
    with monkeypatch.context() as older:
        older.setattr("morgan_brain.app.chatgpt_import.refuse_credentials", lambda text: None)
        older.setattr("morgan_brain.memory.module.refuse_credentials", lambda text: None)
        first = await import_chatgpt(path, gate=gate, user_id="owner")
    assert first.memories == 1
    before, calls = conn.serialize(), embedder.calls
    replay = await import_chatgpt(path, gate=gate, user_id="owner")
    assert replay.memories == 0 and replay.skipped_turns == 1
    assert replay.unresolved_user_turns == 1
    assert conn.serialize() == before and embedder.calls == calls
    # A changed source still cannot reuse its immutable identity.
    conversation["mapping"]["n"]["message"]["content"]["parts"] = ["changed source"]
    path.write_text(json.dumps([conversation]))
    with pytest.raises(EventIdentityConflict):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert conn.serialize() == before and embedder.calls == calls


def export_message(path, text):
    conversation = {
        "id": "synthetic-conversation",
        "mapping": {
            "n": {
                "message": {
                    "id": "synthetic-message",
                    "author": {"role": "user"},
                    "channel": "final",
                    "content": {"content_type": "text", "parts": [text]},
                }
            }
        },
    }
    path.write_text(json.dumps([conversation]))
    return conversation


async def test_partial_legacy_import_checks_whole_source_before_new_piece(
    stack, tmp_path, monkeypatch
):
    gate, conn, embedder, _ = stack
    path = tmp_path / "partial.json"
    text = "x" * 5990 + token()
    pieces = split_for_embedding(text)
    assert len(pieces) == 2
    export_message(path, text)
    with monkeypatch.context() as partial:
        partial.setattr("morgan_brain.app.chatgpt_import.refuse_credentials", lambda text: None)
        partial.setattr(
            "morgan_brain.app.chatgpt_import.split_for_embedding", lambda text: pieces[:1]
        )
        first = await import_chatgpt(path, gate=gate, user_id="owner")
    assert first.memories == 1
    before, calls = conn.serialize(), embedder.calls
    with pytest.raises(CredentialRefused):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert conn.serialize() == before and embedder.calls == calls


async def test_import_replay_cannot_change_provenance(stack, tmp_path):
    gate, conn, embedder, _ = stack
    path = tmp_path / "provenance.json"
    source = export_message(path, "Synthetic source")
    await import_chatgpt(path, gate=gate, user_id="owner")
    before, calls = conn.serialize(), embedder.calls
    source["mapping"]["n"]["message"]["author"]["role"] = "assistant"
    path.write_text(json.dumps([source]))
    with pytest.raises(EventIdentityConflict):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert conn.serialize() == before and embedder.calls == calls


async def test_import_replay_cannot_resurrect_after_forget(stack, tmp_path, monkeypatch):
    from morgan_brain.memory.store.erasure import StoreInterruptedByForget

    gate, conn, embedder, _ = stack
    path = tmp_path / "forget.json"
    export_message(path, "Synthetic source")
    await import_chatgpt(path, gate=gate, user_id="owner")
    original_get = gate.get
    after_forget = []

    async def forget_after_read(identity, *, user_id):
        existing = await original_get(identity, user_id=user_id)
        await gate.forget(user_id=user_id, project=existing.project)
        after_forget.append(conn.serialize())
        return existing

    monkeypatch.setattr(gate, "get", forget_after_read)
    calls = embedder.calls
    with pytest.raises(StoreInterruptedByForget):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert conn.serialize() == after_forget[0] and embedder.calls == calls
