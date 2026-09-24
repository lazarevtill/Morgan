"""The gate refuses what the caller can rephrase. A provider token in a `remember`, a fact or
`ask`'s question raises `SecretRefused` by rule and offset: nothing is embedded, nothing is sent,
nothing is stored. A generic hit goes on as its redacted text, everywhere the text goes -- the
embedding request, the prompt, the history row, the stored memory -- and the result names the
rules, never the value. What nobody can rephrase -- a history row, `ask`'s reply, a remote read
from the repository -- is redacted and never refused; a remote also loses its userinfo, and its
label is computed from it as read."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from morgan_brain.app.chat import Chat
from morgan_brain.composition import build_memory_context, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets import Hit, SecretRefused
from morgan_brain.memory.secrets.rules import PROVIDER_RULE_NAMES
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.models import Memory, MemoryQuery, Message, Role, TemporalFact
from morgan_brain.surfaces.cli.commands import cmd_import, cmd_remember
from morgan_brain.surfaces.cli.project import Repository, classify
from morgan_brain.surfaces.cli.render import _render_import, _render_remember
from tests.fakes import FakeChatClient, counting_model_server
from tests.unit.memory.conftest import a_version_two_database, build_memory_module

#: A token in no fixture and no dictionary: the value that must appear nowhere.
MARKER = "zq" + "xj" + "vgatemarker"


def _token() -> str:
    return "gh" + "p_" + MARKER + "A" * (36 - len(MARKER))


def _provider_settings(tmp_path: Path, url: str) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        llm_endpoint=url,
        embedding_backend="provider",
        embedding_dim=8,
        embedding_unreachable_budget_seconds=1.0,
    )


def _rules(hits_json: str) -> list[str]:
    return sorted({hit["rule"] for hit in json.loads(hits_json)})


# --- refuse ---------------------------------------------------------------------------------


async def test_a_memory_with_a_provider_token_is_refused_by_rule_and_offset_and_never_embedded(
    tmp_path,
):
    with counting_model_server(embedding_dim=8) as (url, calls):
        ctx = build_memory_context(_provider_settings(tmp_path, url))
        try:
            token = _token()
            with pytest.raises(SecretRefused) as raised:
                await ctx.gate.store(Memory(user_id="u", project="p", content=f"deploy {token}"))
            assert (raised.value.rule, raised.value.start, raised.value.length) == (
                "github_token",
                7,
                len(token),
            )
            assert MARKER not in str(raised.value)
            assert calls.total == 0
            assert ctx.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
            await ctx.gate.store(Memory(user_id="u", project="p", content="a clean note"))
            assert calls.total >= 1  # the server is the one counting
        finally:
            ctx.conn.close()


async def test_a_fact_with_a_provider_token_in_any_field_is_refused(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    for field in ("subject", "predicate", "object"):
        values = {"subject": "harbor", "predicate": "uses", "object": "tea"}
        values[field] = _token()
        with pytest.raises(SecretRefused) as raised:
            await gate.upsert_fact(TemporalFact(user_id="u", project="p", **values))
        assert raised.value.rule == "github_token"
    assert await gate.current_facts(user_id="u", project="p") == []


# --- a generic hit goes on redacted -----------------------------------------------------------


async def test_a_generic_hit_is_stored_redacted_and_the_object_carries_the_hits(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    memory = Memory(user_id="u", project="p", content=f"DB_PASSWORD={MARKER}99 in compose")
    memory_id = await gate.store(memory)
    assert memory.content == "DB_PASSWORD=[redacted:assignment] in compose"
    assert _rules(memory.redactions) == ["assignment"] and _rules(memory.flags) == ["assignment"]
    stored = await gate.get(memory_id, user_id="u")
    assert stored is not None
    assert (stored.content, stored.redactions, stored.flags) == (
        memory.content,
        memory.redactions,
        memory.flags,
    )
    recalled = await gate.recall(MemoryQuery(user_id="u", project="p", text="DB_PASSWORD compose"))
    assert [m.content for m in recalled.memories] == [memory.content]
    for table, column in (("memories", "content"), ("fts_memories", "content")):
        # `table` and `column` come from this literal list, never from data.
        rows = gate._store._conn.execute(f"SELECT {column} FROM {table}").fetchall()  # noqa: S608
        assert all(MARKER not in str(r[0]) for r in rows)


async def test_a_facts_three_fields_are_scanned_and_the_hits_name_the_field(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    fact = TemporalFact(
        user_id="u", project="p", subject="deploy", predicate="needs", object=f"token={MARKER}abc"
    )
    await gate.upsert_fact(fact)
    assert fact.object == "token=[redacted:assignment]"
    assert _rules(fact.redactions) == ["object:assignment"]
    assert _rules(fact.flags) == ["object:assignment"]
    [current] = await gate.current_facts(user_id="u", project="p")
    assert (current.object, current.redactions) == (fact.object, fact.redactions)


async def test_a_generic_hit_in_the_subject_or_the_predicate_names_that_field(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    for field in ("subject", "predicate"):
        values = {"subject": "deploy", "predicate": "needs", "object": "tea"}
        values[field] = f"token={MARKER}abc"
        fact = TemporalFact(user_id="u", project=f"p-{field}", **values)
        await gate.upsert_fact(fact)
        assert getattr(fact, field) == "token=[redacted:assignment]"
        assert _rules(fact.redactions) == [f"{field}:assignment"]
        assert _rules(fact.flags) == [f"{field}:assignment"]


async def test_a_refused_fact_leaves_the_callers_subject_and_predicate_as_they_were(tmp_path):
    """Every field is scanned before any is rewritten: a provider token in the object refuses
    the fact, and the subject and predicate the caller passed, generic hits and all, are left
    as they were on the object it holds."""
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    subject, predicate = f"token={MARKER}abc", f"secret={MARKER}xyz"
    fact = TemporalFact(
        user_id="u", project="p", subject=subject, predicate=predicate, object=_token()
    )
    with pytest.raises(SecretRefused):
        await gate.upsert_fact(fact)
    assert (fact.subject, fact.predicate) == (subject, predicate)
    assert await gate.current_facts(user_id="u", project="p") == []


async def test_the_remember_command_names_the_rules_and_never_the_value(settings_for_tmp):
    result = await cmd_remember(
        argparse.Namespace(text=f"DB_PASSWORD={MARKER}99 in compose"), settings_for_tmp, "p"
    )
    assert (result["redacted"], result["flagged"]) == (["assignment"], ["assignment"])
    assert result["content"] == "DB_PASSWORD=[redacted:assignment] in compose"
    assert MARKER not in json.dumps(result)
    assert _render_remember(result).endswith(" (redacted: assignment) (flagged: assignment)")
    clean = await cmd_remember(argparse.Namespace(text="a clean note"), settings_for_tmp, "p")
    assert (clean["redacted"], clean["flagged"]) == ([], [])
    assert _render_remember(clean) == f"Stored memory {clean['id']} in project 'p'."
    with pytest.raises(SecretRefused):
        await cmd_remember(argparse.Namespace(text=f"deploy {_token()}"), settings_for_tmp, "p")


async def test_the_import_command_counts_what_it_redacted_and_never_the_value(
    settings_for_tmp, tmp_path
):
    message = {
        "id": "m0",
        "author": {"role": "user"},
        "create_time": 1700000000.0,
        "content": {"content_type": "text", "parts": [f"deploy {_token()} now"]},
    }
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps([{"conversation_id": "c2", "mapping": {"m0": {"message": message}}}]),
        encoding="utf-8",
    )

    result = await cmd_import(argparse.Namespace(path=str(export)), settings_for_tmp, "ignored")

    assert (result["memories"], result["redacted_pieces"], result["provider_hits"]) == (1, 1, 1)
    assert MARKER not in json.dumps(result)
    assert _render_import(result).endswith(" 1 pieces redacted (1 provider tokens).")
    clean = {**result, "redacted_pieces": 0, "provider_hits": 0}
    assert _render_import(clean).endswith(" turns skipped.")


# --- ask --------------------------------------------------------------------------------------


async def test_ask_scans_its_question_first_and_the_redacted_text_goes_everywhere(tmp_path):
    with counting_model_server(embedding_dim=8) as (url, calls):
        ctx = build_memory_context(_provider_settings(tmp_path, url))
        try:
            client = FakeChatClient(reply="still right")
            chat = Chat(
                gate=ctx.gate,
                history=ctx.history,
                client=client,
                model="test-model",
                clock=utcnow,
            )
            question = f"is DB_PASSWORD={MARKER}77 still right?"
            redacted = "is DB_PASSWORD=[redacted:assignment] still right?"

            reply = await chat.ask(user_id="u", project="p", text=question)

            assert reply == "still right"
            inputs = [
                text
                for body in calls.bodies
                for text in (body["input"] if isinstance(body["input"], list) else [body["input"]])
            ]
            assert redacted in inputs and all(MARKER not in text for text in inputs)
            prompt = "\n".join(m.content for m in client.last_messages)
            assert redacted in prompt and MARKER not in prompt
            history = ctx.history.recent("u:default", project="p")
            assert [m.content for m in history] == [redacted, "still right"]
            contents = [str(r[0]) for r in ctx.conn.execute("SELECT content FROM memories")]
            assert redacted in contents and all(MARKER not in c for c in contents)
            # The stored question is scanned again, from its own raw text, by `store` --
            # not carried over from `scan_text`'s output -- and it records the same hit.
            redactions = [
                str(r[0])
                for r in ctx.conn.execute(
                    "SELECT redactions FROM memories WHERE source = 'user_stated'"
                )
            ]
            assert any(_rules(r) == ["assignment"] for r in redactions)

            def written() -> tuple[int, int, int, int]:
                """Model calls, chat calls, history rows and memories, as they stand."""
                memories = ctx.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                rows = len(ctx.history.recent("u:default", project="p"))
                return calls.total, client.calls, rows, memories

            before = written()
            with pytest.raises(SecretRefused):
                await chat.ask(user_id="u", project="p", text=f"use {_token()} now")
            assert written() == before
        finally:
            ctx.conn.close()


# --- the columns are read only where a row has them ------------------------------------------


async def test_reads_on_a_version_two_file_tolerate_the_missing_columns(tmp_path):
    a_version_two_database(str(tmp_path / "two" / "morgan.db"))
    ctx = build_memory_context(Settings(data_dir=str(tmp_path / "two"), embedding_backend="hash"))
    try:
        assert ctx.gate.read_only_reason is not None
        memory = await ctx.gate.get("m-1", user_id="u")
        assert memory is not None and (memory.redactions, memory.flags) == ("[]", "[]")
        [fact] = await ctx.gate.current_facts(user_id="u", project="p")
        assert (fact.object, fact.redactions, fact.flags) == ("k8s", "[]", "[]")
        recalled = await ctx.gate.recall(MemoryQuery(user_id="u", project="p", text="harbor"))
        assert any("harbor" in m.content for m in recalled.memories)
    finally:
        ctx.conn.close()


# --- redact: what nobody can rephrase ---------------------------------------------------------


async def test_a_history_row_is_scanned_and_stored_redacted_never_refused(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    await gate.append_history(
        user_id="u",
        project="p",
        session_key="u:s",
        message=Message(user_id="u", role=Role.USER, content=f"deploy with {_token()} now"),
    )
    [row] = gate._store.history.recent("u:s", project="p")
    assert row.content == "deploy with [redacted:github_token] now"
    stored = gate._store._conn.execute("SELECT content FROM session_history").fetchall()
    assert all(MARKER not in str(r[0]) for r in stored)


async def test_asks_reply_is_redacted_before_history_and_before_the_memory(tmp_path):
    """A reply that echoes a provider token was produced and paid for, so `ask` completes: the
    reply it returns, its history row, its memory and the memory's embedding request all carry
    the placeholder, and the memory records the rule."""
    with counting_model_server(embedding_dim=8) as (url, calls):
        ctx = build_memory_context(_provider_settings(tmp_path, url))
        try:
            client = FakeChatClient(reply=f"use {_token()} for the deploy")
            chat = Chat(gate=ctx.gate, history=ctx.history, client=client, model="m", clock=utcnow)

            reply = await chat.ask(user_id="u", project="p", text="what token do I use?")

            assert reply == "use [redacted:github_token] for the deploy"
            history = ctx.history.recent("u:default", project="p")
            assert [m.content for m in history] == ["what token do I use?", reply]
            rows = ctx.conn.execute("SELECT content, source, redactions FROM memories").fetchall()
            by_source = {r["source"]: r for r in rows}
            assert by_source["agent_inferred"]["content"] == reply
            assert _rules(by_source["agent_inferred"]["redactions"]) == ["github_token"]
            assert all(MARKER not in str(r["content"]) for r in rows)
            stored = ctx.conn.execute("SELECT content FROM session_history").fetchall()
            assert all(MARKER not in str(r[0]) for r in stored)
            inputs = [
                text
                for body in calls.bodies
                for text in (body["input"] if isinstance(body["input"], list) else [body["input"]])
            ]
            assert reply in inputs and all(MARKER not in text for text in inputs)
        finally:
            ctx.conn.close()


async def test_a_remote_is_recorded_without_its_userinfo(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    await gate.store(Memory(user_id="u", project="p", content="a note"))
    assert await gate.record_project(
        user_id="u",
        project="p",
        classification="work",
        remote=f"https://user:{MARKER}@git.example/team/p.git",
        root="/src/p",
    )
    row = projects_store.get(gate._store._conn, "p")
    assert row is not None and row.remote == "https://git.example/team/p.git"
    assert row.root == "/src/p"
    assert MARKER not in json.dumps(row.__dict__)


async def test_a_token_in_a_remotes_query_is_stored_redacted_and_classified_as_read(tmp_path):
    """A remote is read from the repository's config, so nobody can rephrase it: past its
    userinfo it is scanned under redact, and a token in its query is stored as its placeholder.
    The label is computed from the remote as read, before that scan: a glob that matches only
    the text the scan redacts still classifies the project."""
    token = _token()
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="hash",
        work_remote_globs=["*private_token=" + token[:4] + "*"],
    )
    repository = Repository(
        root=tmp_path / "harbor",
        remote=f"https://git.example/team/harbor.git?private_token={token}",
        remote_readable=True,
    )

    await cmd_remember(argparse.Namespace(text="a note"), settings, "harbor", repository=repository)

    ctx = build_memory_context(settings)
    try:
        row = projects_store.get(ctx.conn, "harbor")
    finally:
        ctx.conn.close()
    assert row is not None
    assert (row.classification, row.remote, row.root) == (
        "work",
        "https://git.example/team/harbor.git?private_token=[redacted:github_token]",
        str(tmp_path / "harbor"),
    )
    assert classify(row.remote, settings.work_remote_globs) == "personal"
    assert MARKER not in json.dumps(row.__dict__)


@pytest.mark.parametrize("remote", ["https://git.example/team/p.git", "git@git.example:team/p.git"])
async def test_a_clean_remote_is_recorded_exactly_as_given(tmp_path, remote):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    await gate.store(Memory(user_id="u", project="p", content="a note"))
    assert await gate.record_project(
        user_id="u", project="p", classification="personal", remote=remote, root="/src/p"
    )
    row = projects_store.get(gate._store._conn, "p")
    assert row is not None and (row.remote, row.root) == (remote, "/src/p")


def test_the_provider_rule_names_are_what_an_import_counts():
    assert "github_token" in PROVIDER_RULE_NAMES and "assignment" not in PROVIDER_RULE_NAMES


# --- a caller's hits, from a scan of the longer text a piece was cut from ---------------------


async def test_a_callers_hits_are_recorded_where_they_stand_after_the_pieces_own_scan(tmp_path):
    """The piece is scanned again when it is stored, so a caller's hits never let a secret by:
    here that scan redacts a token ahead of the caller's placeholder, which moves, and the
    caller's hit is recorded where the placeholder now stands, beside the new one."""
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    earlier = gate.scan_result(f"DB_PASSWORD={MARKER}99 in compose", verdict="redact")
    piece = f"use {_token()} then {earlier.text}"
    shift = len(piece) - len(earlier.text)
    hits = [replace(hit, start=hit.start + shift) for hit in (*earlier.redactions, *earlier.flags)]
    memory = Memory(user_id="u", project="p", content=piece)

    await gate.store(memory, verdict="redact", hits=hits)

    assert memory.content == (
        "use [redacted:github_token] then DB_PASSWORD=[redacted:assignment] in compose"
    )
    recorded = json.loads(memory.redactions)
    assert sorted(hit["rule"] for hit in recorded) == ["assignment", "github_token"]
    for hit in [*recorded, *json.loads(memory.flags)]:
        placed = memory.content[hit["start"] : hit["start"] + hit["length"]]
        assert placed == f"[redacted:{hit['rule']}]"
    assert _rules(memory.flags) == ["assignment"]
    stored = await gate.get(memory.id, user_id="u")
    assert stored is not None
    assert (stored.redactions, stored.flags) == (memory.redactions, memory.flags)


async def test_a_hit_whose_position_does_not_hold_its_placeholder_is_refused(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")))
    placeholder = "[redacted:assignment]"
    misplaced = Hit("assignment", "generic", 3, len(placeholder), "redact")
    memory = Memory(user_id="u", project="p", content=f"see {placeholder} here")
    with pytest.raises(ValueError, match="assignment"):
        await gate.store(memory, verdict="redact", hits=[misplaced])
    assert gate._store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
