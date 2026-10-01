"""Progressive evidence access stays bounded, scoped and independent of model calls."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.models import Memory, MemorySource, MemoryStatus, Scope, TemporalFact
from morgan_brain.surfaces.cli.__main__ import build_parser
from morgan_brain.surfaces.cli.commands import cmd_evidence, cmd_facts
from morgan_brain.surfaces.cli.payloads import memory_to_dict
from morgan_brain.surfaces.cli.render import RENDERERS
from morgan_brain.surfaces.cli.validation import evidence_ids
from morgan_brain.surfaces.mcp_server import build_server


@pytest.mark.parametrize("values", [[], ["x"] * 33, [""], [" "], ["x" * 257], [None], "x"])
async def test_invalid_evidence_request_refuses_before_opening_database(
    settings_for_tmp, monkeypatch, values
):
    def unexpected_context(settings):
        raise AssertionError("Invalid request opened database")

    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.commands.build_evidence_context", unexpected_context
    )
    with pytest.raises(ValueError):
        await cmd_evidence(argparse.Namespace(ids=values), settings_for_tmp, "p")


async def test_evidence_refuses_cross_project_request_before_database(
    settings_for_tmp, monkeypatch
):
    def unexpected_context(settings):
        raise AssertionError("Cross-project request opened database")

    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.commands.build_evidence_context", unexpected_context
    )
    with pytest.raises(ValueError, match="one project"):
        await cmd_evidence(argparse.Namespace(ids=["id"], all_projects=True), settings_for_tmp, "p")


def test_evidence_cli_parser_and_deduplication():
    args = build_parser().parse_args(["evidence", "a", "b", "a", "--project", "personal", "--json"])
    assert args.command == "evidence" and args.project == "personal"
    assert evidence_ids(args.ids) == ["a", "b"]


async def test_facts_surface_preserves_durable_provenance_and_validity(settings_for_tmp):
    ctx = build_memory_context(settings_for_tmp)
    effective = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        await ctx.gate.store(
            Memory(
                id="support",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                content="Synthetic direct statement",
                source=MemorySource.USER_STATED,
            )
        )
        await ctx.gate.upsert_fact(
            TemporalFact(
                id="fact",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                subject="user",
                predicate="prefers",
                object="tea",
                source=MemorySource.AGENT_INFERRED,
                author_id="consolidator",
                scope=Scope.SHARED,
                support_event_ids=["support"],
                confidence=0.75,
                valid_from=effective,
            )
        )
        persisted = (
            await ctx.gate.current_facts(user_id=settings_for_tmp.owner_user_id, project="p")
        )[0]
    finally:
        ctx.conn.close()
    payload = await cmd_facts(
        argparse.Namespace(subject=None, all_projects=False), settings_for_tmp, "p"
    )
    record = payload["facts"][0]
    assert record["id"] == "fact" and record["source"] == "agent_inferred"
    assert record["author_id"] == "consolidator" and record["scope"] == "shared"
    assert record["support_event_ids"] == ["support"]
    assert record["recorded_at"] == persisted.recorded_at.isoformat()
    assert record["valid_from"] == effective.isoformat()
    assert record["valid_to"] is None and record["superseded_by"] is None
    assert record["confidence"] == 0.75


async def test_exact_evidence_exposes_quarantine_and_instruction_metadata(settings_for_tmp):
    ctx = build_memory_context(settings_for_tmp)
    try:
        await ctx.gate.store(
            Memory(
                id="quarantined",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                content="Synthetic instruction-shaped source evidence",
                status=MemoryStatus.QUARANTINED,
                instruction_like=True,
            )
        )
    finally:
        ctx.conn.close()
    payload = await cmd_evidence(argparse.Namespace(ids=["quarantined"]), settings_for_tmp, "p")
    record = payload["results"][0]
    assert record["id"] == "quarantined"
    assert record["status"] == "quarantined"
    assert record["instruction_like"] is True


async def test_evidence_wire_contract_scope_and_model_independence(settings_for_tmp, monkeypatch):
    ctx = build_memory_context(settings_for_tmp)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        await ctx.gate.store(
            Memory(
                id="support",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                content="Synthetic direct statement",
                source=MemorySource.USER_STATED,
                author_id="speaker",
                created_at=now,
            )
        )
        await ctx.gate.store(
            Memory(
                id="other-project",
                user_id=settings_for_tmp.owner_user_id,
                project="other",
                content="Must stay inaccessible",
            )
        )
        await ctx.gate.store(
            Memory(
                id="other-owner",
                user_id="different-owner",
                project="p",
                content="Must also stay inaccessible",
            )
        )
        await ctx.gate.upsert_fact(
            TemporalFact(
                id="fact",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                subject="user",
                predicate="prefers",
                object="tea",
                source=MemorySource.AGENT_INFERRED,
                support_event_ids=["support"],
                valid_from=now,
            )
        )
    finally:
        ctx.conn.close()

    async def unexpected_embedding(*args, **kwargs):
        raise AssertionError("Evidence fetched an embedding")

    monkeypatch.setattr("morgan_brain.memory.embedder.FakeEmbedder.embed", unexpected_embedding)
    server = build_server(settings_for_tmp)
    async with create_connected_server_and_client_session(server.mcp) as client:
        result = await client.call_tool(
            "evidence",
            {
                "ids": ["fact", "other-project", "other-owner", "unknown", "fact"],
                "project": "p",
            },
        )
        assert not result.isError
        data = result.structuredContent
        assert data["version"] == "morgan.evidence.v1"
        assert data["requested_ids"] == ["fact", "other-project", "other-owner", "unknown"]
        assert data["missing_ids"] == ["other-project", "other-owner", "unknown"]
        assert [item["id"] for item in data["results"]] == ["fact"]
        fact = data["results"][0]
        assert fact["support_event_ids"] == ["support"]
        assert fact["valid_from"] == now.isoformat()
        assert "inaccessible" not in json.dumps(data)
        roots = await client.call_tool(
            "evidence", {"ids": fact["support_event_ids"], "project": "p"}
        )
        event = roots.structuredContent["results"][0]
        assert event["id"] == "support" and event["source"] == "user_stated"
        assert event["author_id"] == "speaker"
        assert event["created_at"] == now.isoformat() and event["recorded_at"]
    assert "fact" in RENDERERS["evidence"](data)


def test_recall_payload_metadata_is_additive():
    memory = Memory(id="durable", user_id="u", content="evidence")
    data = memory_to_dict(memory)
    assert data["id"] == "durable" and data["source"] == "unknown"
    assert data["support_event_ids"] == [] and data["valid_to"] is None
    assert data["recorded_at"] is None


async def test_evidence_old_database_stays_byte_unchanged_without_model_construction(
    settings_for_tmp, monkeypatch
):
    ctx = build_memory_context(settings_for_tmp)
    try:
        await ctx.gate.store(
            Memory(
                id="old-event",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                content="Synthetic old evidence",
            )
        )
        await ctx.gate.upsert_fact(
            TemporalFact(
                id="old-fact",
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                subject="user",
                predicate="prefers",
                object="tea",
            )
        )
        ctx.conn.execute("ALTER TABLE facts DROP COLUMN recorded_at")
        ctx.conn.execute("ALTER TABLE facts DROP COLUMN support_event_ids")
        ctx.conn.execute("PRAGMA user_version=8")
        ctx.conn.commit()
        ctx.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        ctx.conn.close()
    db_path = Path(sqlite_path(settings_for_tmp.temporal_db_url))
    before = await asyncio.to_thread(db_path.read_bytes)

    def unexpected_backend(*args, **kwargs):
        raise AssertionError("Evidence constructed a model backend")

    monkeypatch.setattr("morgan_brain.composition.build_embedder", unexpected_backend)
    monkeypatch.setattr("morgan_brain.composition.build_chat_client", unexpected_backend)
    changed = settings_for_tmp.model_copy(
        update={"embedding_backend": "provider", "embedding_dim": 4096}
    )
    data = await cmd_evidence(argparse.Namespace(ids=["old-event", "old-fact"]), changed, "p")
    assert [row["id"] for row in data["results"]] == ["old-event", "old-fact"]
    assert data["results"][1]["support_event_ids"] == []
    assert data["results"][1]["recorded_at"] is None
    assert await asyncio.to_thread(db_path.read_bytes) == before


async def test_evidence_missing_database_is_not_created(settings_for_tmp):
    db_path = Path(sqlite_path(settings_for_tmp.temporal_db_url))
    assert not await asyncio.to_thread(db_path.exists)
    with pytest.raises(FileNotFoundError):
        await cmd_evidence(argparse.Namespace(ids=["unknown"]), settings_for_tmp, "personal")
    assert not await asyncio.to_thread(db_path.exists)


@pytest.mark.parametrize("extra", [[], ["--all-projects"]])
def test_evidence_cli_errors_remain_json_without_creating_database(
    tmp_path, monkeypatch, capsys, extra
):
    from morgan_brain.surfaces.cli.__main__ import main

    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path / "missing-data"))
    result = main(["evidence", "unknown", "--project", "personal", "--json", *extra])
    assert result == 1
    data = json.loads(capsys.readouterr().out)
    assert data["error"]
    assert not (tmp_path / "missing-data").exists()
