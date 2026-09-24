"""The gate refuses what the caller can rephrase. A provider token in a `remember`, a fact or
`ask`'s question raises `SecretRefused` by rule and offset: nothing is embedded, nothing is sent,
nothing is stored. A generic hit goes on as its redacted text, everywhere the text goes -- the
embedding request, the prompt, the history row, the stored memory -- and the result names the
rules, never the value. Other verdicts besides `refuse` are exercised elsewhere."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from morgan_brain.app.chat import Chat
from morgan_brain.composition import build_memory_context, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets import SecretRefused
from morgan_brain.models import Memory, MemoryQuery, TemporalFact
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.cli.render import _render_remember
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

            before = (calls.total, client.calls, len(history))
            with pytest.raises(SecretRefused):
                await chat.ask(user_id="u", project="p", text=f"use {_token()} now")
            assert (calls.total, client.calls) == before[:2]
            assert len(ctx.history.recent("u:default", project="p")) == before[2]
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
