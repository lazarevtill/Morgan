"""Strict ask is optional; unsupported capability refuses before opening the database."""

import argparse
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from morgan_brain.app.strict_context import AnswerBudget, AnswerResult, StrictContextError
from morgan_brain.config import Settings
from morgan_brain.surfaces.cli.__main__ import build_parser
from morgan_brain.surfaces.cli.commands import cmd_ask
from morgan_brain.surfaces.mcp_server import build_server


def test_cli_strict_opt_in_default_and_flag():
    assert build_parser().parse_args(["ask", "Tea?"]).strict_context is False
    assert build_parser().parse_args(["ask", "Tea?", "--strict-context"]).strict_context is True


@pytest.mark.parametrize(
    "fields",
    [
        {"strict_context_tokens": 0},
        {"strict_context_output_tokens": 0},
        {"strict_context_safety_tokens": -1},
        {"strict_context_tokens": 288},
        {"strict_context_backend": "estimated"},
    ],
)
def test_invalid_strict_settings_refuse(fields):
    with pytest.raises(ValidationError):
        Settings(**fields)


async def test_disabled_cli_strict_refuses_before_context(settings_for_tmp, monkeypatch):
    def unexpected(settings):
        raise AssertionError("Unsupported strict ask opened a context")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_app_context", unexpected)
    with pytest.raises(StrictContextError, match="token_counter_unavailable"):
        await cmd_ask(
            argparse.Namespace(text="Tea?", strict_context=True), settings_for_tmp, "personal"
        )


async def test_disabled_mcp_strict_refuses_before_context(settings_for_tmp, monkeypatch):
    def unexpected(settings):
        raise AssertionError("Unsupported strict MCP opened a context")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_app_context", unexpected)
    with pytest.raises(StrictContextError, match="token_counter_unavailable"):
        await build_server(settings_for_tmp).call_tool(
            "ask_morgan", {"text": "Tea?", "strict_context": True}
        )


@pytest.mark.parametrize("strict", [False, True])
async def test_ask_forwards_optional_strict_and_attribution(settings_for_tmp, monkeypatch, strict):
    settings = settings_for_tmp.model_copy(update={"strict_context_backend": "llamacpp"})
    captured = {}

    class Chat:
        async def ask(self, **fields):
            captured.update(fields)
            return "Tea"

        async def ask_evidence(self, **fields):
            captured.update(fields, strict_context=True)
            return AnswerResult(
                user_id="owner",
                project="personal",
                model="fake",
                answer="Tea",
                evidence_ids=["source"],
                abstained=False,
                budget=AnswerBudget(
                    input_tokens=100,
                    reported_output_tokens=9,
                    total_tokens=4096,
                    output_reserve_tokens=256,
                    safety_tokens=32,
                    template_id="fake",
                    counter_calls=2,
                ),
            )

    ctx = SimpleNamespace(chat=Chat(), conn=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.commands.build_app_context", lambda settings: ctx
    )
    result = await cmd_ask(
        argparse.Namespace(
            text="Tea?", strict_context=strict, source="user_stated", author_id="person"
        ),
        settings,
        "personal",
    )
    assert captured["strict_context"] is strict
    assert captured["author_id"] == "person" and captured["source"].value == "user_stated"
    assert result["response"] == "Tea"
    if strict:
        assert result["schema_version"] == "morgan.answer.v1"
        assert result["evidence_ids"] == ["source"]
    else:
        assert "schema_version" not in result


@pytest.mark.parametrize("fail", [False, True])
async def test_ask_closes_optional_backend_and_db_on_success_or_failure(
    settings_for_tmp, monkeypatch, fail
):
    closed = []

    class Chat:
        async def ask(self, **fields):
            if fail:
                raise StrictContextError("citation_invalid")
            return "Tea"

        async def aclose(self):
            closed.append("backend")

    ctx = SimpleNamespace(chat=Chat(), conn=SimpleNamespace(close=lambda: closed.append("db")))
    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.commands.build_app_context", lambda settings: ctx
    )
    args = argparse.Namespace(text="Tea?")
    if fail:
        with pytest.raises(StrictContextError):
            await cmd_ask(args, settings_for_tmp, "personal")
    else:
        await cmd_ask(args, settings_for_tmp, "personal")
    assert closed == ["backend", "db"]


async def test_database_refusal_happens_before_any_app_provider_construction(
    settings_for_tmp, monkeypatch
):
    import morgan_brain.composition as composition

    calls = []

    def refused(settings):
        raise ValueError("unsupported_future_schema")

    def unexpected(*args, **kwargs):
        calls.append("provider")
        raise AssertionError("Refused database constructed a provider")

    monkeypatch.setattr(composition, "build_memory_context", refused)
    monkeypatch.setattr(composition, "build_chat_client", unexpected)
    monkeypatch.setattr(composition, "build_strict_chat_backend", unexpected)
    settings = settings_for_tmp.model_copy(update={"strict_context_backend": "llamacpp"})
    with pytest.raises(ValueError, match="unsupported_future_schema"):
        composition.build_app_context(settings)
    assert calls == []


def test_app_provider_constructor_failure_closes_memory_connection(settings_for_tmp, monkeypatch):
    import sqlite3

    import morgan_brain.composition as composition

    memory = composition.build_memory_context(settings_for_tmp)

    def fail(*args, **kwargs):
        raise ValueError("unsupported_model")

    monkeypatch.setattr(composition, "build_memory_context", lambda settings: memory)
    monkeypatch.setattr(composition, "build_strict_chat_backend", fail)
    with pytest.raises(ValueError, match="unsupported_model"):
        composition.build_app_context(settings_for_tmp)
    with pytest.raises(sqlite3.ProgrammingError):
        memory.conn.execute("SELECT 1")


def test_ordinary_client_failure_does_not_allocate_strict_backend(settings_for_tmp, monkeypatch):
    import sqlite3

    import morgan_brain.composition as composition

    memory = composition.build_memory_context(settings_for_tmp)
    calls = []

    def fail(settings):
        raise ValueError("ordinary_client_unavailable")

    def unexpected(*args, **kwargs):
        calls.append("strict")
        raise AssertionError("Strict backend constructed after ordinary failure")

    monkeypatch.setattr(composition, "build_memory_context", lambda settings: memory)
    monkeypatch.setattr(composition, "build_chat_client", fail)
    monkeypatch.setattr(composition, "build_strict_chat_backend", unexpected)
    with pytest.raises(ValueError, match="ordinary_client_unavailable"):
        composition.build_app_context(settings_for_tmp)
    assert calls == []
    with pytest.raises(sqlite3.ProgrammingError):
        memory.conn.execute("SELECT 1")


def test_actual_future_database_refuses_before_strict_factory(settings_for_tmp, monkeypatch):
    import sqlite3
    from pathlib import Path

    import morgan_brain.composition as composition
    from morgan_brain.memory.errors import DatabaseSchemaTooNew

    path = Path(composition.sqlite_path(settings_for_tmp.temporal_db_url))
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 999")
        conn.execute("CREATE TABLE future_lineage (payload TEXT)")
        conn.execute("INSERT INTO future_lineage VALUES ('preserve')")
    before = path.read_bytes()
    calls = []

    def unexpected(*args, **kwargs):
        calls.append("provider")
        raise AssertionError("Future database admitted a provider")

    monkeypatch.setattr(composition, "build_chat_client", unexpected)
    monkeypatch.setattr(composition, "build_strict_chat_backend", unexpected)
    settings = settings_for_tmp.model_copy(update={"strict_context_backend": "llamacpp"})
    with pytest.raises(DatabaseSchemaTooNew):
        composition.build_app_context(settings)
    assert calls == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("fail", [False, True])
async def test_consolidation_closes_optional_backend_and_database(
    settings_for_tmp, monkeypatch, fail
):
    from morgan_brain.surfaces.cli.commands import cmd_consolidate

    closed = []

    async def close():
        closed.append("backend")

    async def consolidate(*args, **kwargs):
        if fail:
            raise ValueError("proposal_refused")
        return []

    ctx = SimpleNamespace(
        gate=SimpleNamespace(require_writable=lambda: None),
        chat=SimpleNamespace(aclose=close),
        consolidator=SimpleNamespace(consolidate=consolidate),
        conn=SimpleNamespace(close=lambda: closed.append("db")),
    )
    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.commands.build_app_context", lambda settings: ctx
    )
    args = argparse.Namespace(all_projects=False)
    if fail:
        with pytest.raises(ValueError, match="proposal_refused"):
            await cmd_consolidate(args, settings_for_tmp, "personal")
    else:
        await cmd_consolidate(args, settings_for_tmp, "personal")
    assert closed == ["backend", "db"]
