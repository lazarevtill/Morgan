"""Actual public source-only SDK, CLI and MCP inspect contract over synthetic SQLite."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from morgan_brain.composition import build_evidence_context, build_memory_context, sqlite_path
from morgan_brain.models import Memory, MemorySource, MemoryStatus, TemporalFact
from morgan_brain.surfaces.cli.__main__ import build_parser
from morgan_brain.surfaces.cli.commands import cmd_inspect_context
from morgan_brain.surfaces.cli.render import RENDERERS
from morgan_brain.surfaces.mcp_server import TOOL_ANNOTATIONS, build_server

NOW = datetime(2026, 3, 2, tzinfo=UTC)


def selection(identity, quote, section="current_facts", start=0):
    return {
        "section": section,
        "event_id": identity,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
    }


def file_bytes(path):
    return path.read_bytes()


async def seed(settings):
    context = build_memory_context(settings)
    owner = settings.owner_user_id
    for identity, body, fields in [
        ("old", "Old 🌿 decision.", {}),
        ("new", "Отзываю разрешение. Done ✓", {"revises_event_ids": ["old"]}),
        ("quarantine", "Untrusted text", {"status": MemoryStatus.QUARANTINED}),
        ("future", "Later report", {"created_at": datetime(2027, 1, 1, tzinfo=UTC)}),
        ("other", "Private outside project", {"project": "other"}),
    ]:
        await context.gate.store(
            Memory(
                id=identity,
                user_id=owner,
                project=fields.pop("project", "p"),
                content=body,
                author_id="speaker",
                source=MemorySource.USER_STATED,
                created_at=fields.pop("created_at", datetime(2026, 3, 1, tzinfo=UTC)),
                **fields,
            )
        )
    await context.gate.upsert_fact(
        TemporalFact(
            id="unsupported",
            user_id=owner,
            project="p",
            subject="user",
            predicate="likes",
            object="tea",
            valid_from=NOW,
        )
    )
    context.conn.close()


@pytest.mark.parametrize(
    "ids,items",
    [
        ([], []),
        (["a", "a"], []),
        ([" "], []),
        (["x"] * 17, []),
        (["a"], [selection("b", "x")]),
        (["a"], [selection("a", "x"), selection("a", "x", "completed_progress")]),
        (["a"], [dict(selection("a", "x"), start=True)]),
        (["a"], [dict(selection("a", "x"), extra=1)]),
    ],
)
async def test_invalid_request_never_opens_database(settings_for_tmp, monkeypatch, ids, items):
    def forbidden(_settings):
        raise AssertionError("opened database for malformed request")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_evidence_context", forbidden)
    with pytest.raises(ValueError):
        await cmd_inspect_context(
            argparse.Namespace(ids=ids, selections=items), settings_for_tmp, "p"
        )


async def test_sdk_readonly_sources_exact_unicode_lifecycle_and_no_permission(settings_for_tmp):
    await seed(settings_for_tmp)
    path = Path(sqlite_path(settings_for_tmp.temporal_db_url))
    before = file_bytes(path)
    context = build_evidence_context(settings_for_tmp)
    try:
        result = await context.gate.inspect_context(
            user_id=settings_for_tmp.owner_user_id,
            project="p",
            evidence_ids=["old", "new", "quarantine", "future", "other", "unsupported", "missing"],
            selections=[
                selection("old", "Old 🌿"),
                selection("new", "Отзываю разрешение.", "relevant_constraints"),
                selection("new", "Done ✓", "completed_progress", start=20),
                selection("quarantine", "Untrusted"),
                selection("future", "Later"),
                selection("unsupported", "user likes tea"),
                selection("missing", "x"),
            ],
            effective_at=NOW,
        )
        assert result["action_authority"] == "none" and result["coverage"] == "requested_ids_only"
        assert result["missing_ids"] == ["other", "missing"]
        assert result["sections"]["current_facts"]["status"] == "unknown"
        assert (
            result["sections"]["completed_progress"]["items"][0]["verification"]
            == "unverified_report"
        )
        assert {r["reason"] for r in result["withheld"]} == {
            "inactive_or_future",
            "quarantined",
            "missing",
            "unsupported_revised_or_conflicted_fact_support",
        }
        assert result["sources"][0]["content"] == "Old 🌿 decision."
        assert result["corrections"][0]["revises_event_ids"] == ["old"]
        assert all("embedding" not in s and "entities" not in s for s in result["sources"])
        with pytest.raises(ValueError, match="exactly match"):
            await context.gate.inspect_context(
                user_id=settings_for_tmp.owner_user_id,
                project="p",
                evidence_ids=["old"],
                selections=[selection("old", "bad")],
                effective_at=NOW,
            )
    finally:
        context.conn.close()
    assert file_bytes(path) == before


async def test_cli_and_actual_mcp_wire_use_same_readonly_envelope(
    settings_for_tmp, tmp_path, monkeypatch
):
    await seed(settings_for_tmp)
    selections = [selection("new", "Отзываю разрешение.", "relevant_constraints")]
    file = tmp_path / "selections.json"
    file.write_text(json.dumps(selections), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "context",
            "inspect",
            "new",
            "--selections",
            str(file),
            "--project",
            "p",
            "--effective-at",
            NOW.isoformat(),
            "--json",
        ]
    )
    assert args.command == "inspect_context"
    result = await cmd_inspect_context(args, settings_for_tmp, "p")
    assert "action authority: none" in RENDERERS[args.command](result)
    assert result["unresolved_branch_ids"] == ["old"]
    server = build_server(settings=settings_for_tmp)
    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        await client.initialize()
        tool = next(t for t in (await client.list_tools()).tools if t.name == "inspect_context")
        assert tool.annotations.readOnlyHint is True
        answer = await client.call_tool(
            "inspect_context",
            {
                "ids": ["new"],
                "project": "p",
                "selections": selections,
                "effective_at": NOW.isoformat(),
            },
        )
        assert not answer.isError
        assert json.loads(answer.content[0].text) == result
    assert TOOL_ANNOTATIONS["inspect_context"].destructiveHint is False


async def test_full_source_overflow_refuses_without_truncation(settings_for_tmp):
    context = build_memory_context(settings_for_tmp)
    await context.gate.store(
        Memory(id="huge", user_id=settings_for_tmp.owner_user_id, project="p", content="🌿" * 5000)
    )
    context.conn.close()
    with pytest.raises(ValueError, match="16384"):
        await cmd_inspect_context(argparse.Namespace(ids=["huge"]), settings_for_tmp, "p")


async def test_fork_and_fact_support_and_halfopen_interval(settings_for_tmp):
    await seed(settings_for_tmp)
    context = build_memory_context(settings_for_tmp)
    owner = settings_for_tmp.owner_user_id
    try:
        await context.gate.upsert_fact(
            TemporalFact(
                id="supported",
                user_id=owner,
                project="p",
                subject="owner",
                predicate="reported",
                object="done",
                valid_from=NOW,
                support_event_ids=["new"],
            )
        )
        at_boundary = await context.gate.inspect_context(
            user_id=owner,
            project="p",
            evidence_ids=["supported"],
            selections=[selection("supported", "owner reported done")],
            effective_at=NOW,
        )
        assert at_boundary["sections"]["current_facts"]["status"] == "unverified"
        assert at_boundary["unresolved_support_ids"] == ["new"]
        await context.gate.store(
            Memory(
                id="fork",
                user_id=owner,
                project="p",
                content="Another revision.",
                source=MemorySource.USER_STATED,
                author_id="speaker",
                revises_event_ids=["old"],
                created_at=NOW,
            )
        )
        result = await context.gate.inspect_context(
            user_id=owner,
            project="p",
            evidence_ids=["new", "supported"],
            selections=[selection("new", "Отзываю"), selection("supported", "owner reported done")],
            effective_at=NOW,
        )
        assert result["sections"]["current_facts"]["status"] == "contested"
        assert not result["sections"]["current_facts"]["items"]
        assert result["unresolved_branch_ids"] == ["fork", "old"]
        await context.gate.upsert_fact(
            TemporalFact(
                id="ended",
                user_id=owner,
                project="p",
                subject="owner",
                predicate="interval",
                object="closed",
                valid_from=datetime(2026, 3, 1, tzinfo=UTC),
                valid_to=NOW,
            )
        )
        ended = await context.gate.inspect_context(
            user_id=owner,
            project="p",
            evidence_ids=["ended"],
            selections=[selection("ended", "owner interval closed")],
            effective_at=NOW,
        )
        assert ended["withheld"][0]["reason"] == "inactive_interval"
    finally:
        context.conn.close()


async def test_restart_default_cutoff_and_owner_isolation(settings_for_tmp):
    await seed(settings_for_tmp)
    context = build_evidence_context(settings_for_tmp)
    result = await context.gate.inspect_context(
        user_id="another-owner", project="p", evidence_ids=["new"]
    )
    context.conn.close()
    assert result["missing_ids"] == ["new"] and not result["sources"]
    assert datetime.fromisoformat(result["effective_at"]).utcoffset() is not None
    reopened = build_evidence_context(settings_for_tmp)
    try:
        current = await reopened.gate.inspect_context(
            user_id=settings_for_tmp.owner_user_id,
            project="p",
            evidence_ids=["new"],
            effective_at=NOW,
        )
        assert current["sources"][0]["content"] == "Отзываю разрешение. Done ✓"
    finally:
        reopened.conn.close()


async def test_legacy_naive_times_preserved_but_normalized_for_eligibility(settings_for_tmp):
    context = build_memory_context(settings_for_tmp)
    owner = settings_for_tmp.owner_user_id
    legacy = datetime(2026, 3, 1, tzinfo=UTC)
    try:
        await context.gate.store(
            Memory(
                id="legacy",
                user_id=owner,
                project="p",
                content="Legacy dated report",
                created_at=legacy,
                source=MemorySource.USER_STATED,
            )
        )
        await context.gate.upsert_fact(
            TemporalFact(
                id="legacy-fact",
                user_id=owner,
                project="p",
                subject="owner",
                predicate="reported",
                object="legacy",
                created_at=datetime(2026, 3, 2, tzinfo=UTC),
                valid_from=legacy,
                valid_to=datetime(2026, 3, 3, tzinfo=UTC),
                support_event_ids=["legacy"],
            )
        )
        # Synthetic legacy fixture: current writers intentionally reject naive event times.
        # Only timestamp metadata is adapted; every inspection still uses the public gate.
        context.conn.execute(
            "UPDATE memories SET created_at='2026-03-01T00:00:00' WHERE id='legacy'"
        )
        context.conn.execute(
            "UPDATE facts SET valid_from='2026-03-01T00:00:00', "
            "valid_to='2026-03-03T00:00:00' WHERE id='legacy-fact'"
        )
        context.conn.commit()
        result = await context.gate.inspect_context(
            user_id=owner,
            project="p",
            evidence_ids=["legacy", "legacy-fact"],
            selections=[
                selection("legacy", "Legacy"),
                selection("legacy-fact", "owner reported legacy"),
            ],
            effective_at=NOW,
        )
        assert len(result["sections"]["current_facts"]["items"]) == 2
        fact = result["sources"][1]
        assert fact["effective_at"] == fact["valid_from"] == "2026-03-01T00:00:00"
        assert fact["created_at"] is None
        assert result["sources"][0]["created_at"] == "2026-03-01T00:00:00"
        expired = await context.gate.inspect_context(
            user_id=owner,
            project="p",
            evidence_ids=["legacy-fact"],
            selections=[selection("legacy-fact", "owner reported legacy")],
            effective_at=datetime(2026, 3, 3, tzinfo=UTC),
        )
        assert expired["withheld"][0]["reason"] == "inactive_interval"
    finally:
        context.conn.close()
