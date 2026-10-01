"""Correction input contracts are bounded before DB/model work; old calls remain valid."""

import argparse
from datetime import UTC, datetime

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from morgan_brain.composition import build_memory_context
from morgan_brain.models import Memory, OriginKind, TemporalFact
from morgan_brain.surfaces.cli.__main__ import build_parser
from morgan_brain.surfaces.cli.commands import cmd_evidence, cmd_recall, cmd_remember
from morgan_brain.surfaces.cli.render import RENDERERS
from morgan_brain.surfaces.mcp_server import build_server


def remember_args(**fields):
    return argparse.Namespace(
        text="Synthetic correction", source="user_stated", author_id="person:owner", **fields
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"event_id": ""},
        {"event_id": 123},
        {"effective_at": "2026-01-01T00:00:00"},
        {"effective_at": "bad"},
        {"revises_event_ids": "parent"},
        {"revises_event_ids": [None]},
        {"revises_event_ids": ["x", "x"]},
        {"revises_event_ids": [str(n) for n in range(9)]},
        {"revises_event_ids": ["parent"]},
    ],
)
async def test_invalid_revision_refuses_before_context(settings_for_tmp, monkeypatch, fields):
    def unexpected(settings):
        raise AssertionError("Invalid correction opened database")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_memory_context", unexpected)
    with pytest.raises(ValueError):
        await cmd_remember(remember_args(**fields), settings_for_tmp, None)


@pytest.mark.parametrize("source,author", [("unknown", "person:owner"), ("user_stated", " ")])
async def test_correction_requires_explicit_attribution(
    settings_for_tmp, monkeypatch, source, author
):
    def unexpected(settings):
        raise AssertionError("Unattributed correction opened database")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_memory_context", unexpected)
    args = argparse.Namespace(
        text="Correction",
        source=source,
        author_id=author,
        effective_at="2026-01-01T00:00:00Z",
        revises_event_ids=["a"],
    )
    with pytest.raises(ValueError, match="Corrections require"):
        await cmd_remember(args, settings_for_tmp, None)


@pytest.mark.parametrize("command", ["recall", "evidence"])
async def test_naive_cutoff_refuses_before_context(settings_for_tmp, monkeypatch, command):
    def unexpected(settings):
        raise AssertionError("Invalid cutoff opened database")

    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_memory_context", unexpected)
    monkeypatch.setattr("morgan_brain.surfaces.cli.commands.build_evidence_context", unexpected)
    args = argparse.Namespace(
        ids=["a"], query="tea", top_k=8, all_projects=False, effective_at="2026-01-01T00:00:00"
    )
    with pytest.raises(ValueError, match="timezone"):
        await (cmd_recall if command == "recall" else cmd_evidence)(
            args, settings_for_tmp, "personal"
        )


def test_repeat_parent_cli_flags_and_old_remember():
    parser = build_parser()
    args = parser.parse_args(
        [
            "remember",
            "Corrected date",
            "--event-id",
            "d",
            "--effective-at",
            "2026-01-01T00:00:00+03:00",
            "--revises-event-id",
            "b",
            "--revises-event-id",
            "c",
        ]
    )
    assert args.revises_event_ids == ["b", "c"] and args.event_id == "d"
    assert parser.parse_args(["remember", "Plain assertion"]).event_id is None


async def test_personal_mcp_correction_retry_and_raw_history(settings_for_tmp):
    server = build_server(settings_for_tmp)
    async with create_connected_server_and_client_session(server.mcp) as client:
        base = {"source": "user_stated", "author_id": "person:owner"}
        first = await client.call_tool(
            "remember",
            {
                **base,
                "text": "Read 20 minutes",
                "event_id": "a",
                "effective_at": "2026-01-01T00:00:00Z",
            },
        )
        assert not first.isError and first.structuredContent["project_defaulted"]
        correction = {
            **base,
            "text": "Read 10 minutes",
            "event_id": "b",
            "effective_at": "2026-06-01T00:00:00Z",
            "revises_event_ids": ["a"],
        }
        result = await client.call_tool("remember", correction)
        assert not result.isError
        retry = await client.call_tool("remember", correction)
        assert not retry.isError and retry.structuredContent["id"] == "b"
        assert retry.structuredContent["recorded_at"] == result.structuredContent["recorded_at"]
        assert retry.structuredContent["effective_at"] == result.structuredContent["effective_at"]
        assert retry.structuredContent["revision_root_id"] == "a"
        changed = await client.call_tool("remember", {**correction, "text": "Read 5 minutes"})
        assert changed.isError
        before = await client.call_tool(
            "evidence", {"ids": ["a", "b"], "effective_at": "2026-03-01T00:00:00Z"}
        )
        after = await client.call_tool(
            "evidence", {"ids": ["a", "b"], "effective_at": "2026-07-01T00:00:00Z"}
        )
        assert not before.isError and not after.isError
        assert before.structuredContent["version"] == "morgan.evidence.v1"
        old, new = before.structuredContent["results"]
        assert old["revision_state"] == "active" and new["revision_state"] == "inactive"
        old_after, new_after = after.structuredContent["results"]
        assert old_after["revision_state"] == "inactive" and new_after["revision_state"] == "active"
        assert new_after["revises_event_ids"] == ["a"]
        assert new_after["revision_root_id"] == "a"
        assert new_after["recorded_at"] == new["recorded_at"]
        assert old_after["content"] == "Read 20 minutes"


async def test_ranked_fork_keeps_all_branch_references_and_resolution(settings_for_tmp):
    server = build_server(settings_for_tmp)
    async with create_connected_server_and_client_session(server.mcp) as client:
        base = {"source": "user_stated", "author_id": "person:owner"}
        for identity, text, parents, time in [
            ("a", "Поездка начинается 12 ноября", [], "2026-01-01T00:00:00Z"),
            ("b", "Поездка начинается 14 ноября", ["a"], "2026-02-01T00:00:00Z"),
            ("c", "Поездка начинается 16 ноября", ["a"], "2026-02-01T00:00:00Z"),
            ("d", "Верна дата 16 ноября", ["c", "b"], "2026-06-01T00:00:00Z"),
        ]:
            stored = await client.call_tool(
                "remember",
                {
                    **base,
                    "text": text,
                    "event_id": identity,
                    "revises_event_ids": parents,
                    "effective_at": time,
                },
            )
            assert not stored.isError
        before = await client.call_tool(
            "recall",
            {
                "query": "Поездка начинается",
                "top_k": 1,
                "effective_at": "2026-03-01T00:00:00Z",
            },
        )
        assert not before.isError
        rows = before.structuredContent["results"]
        assert len(rows) == 1
        assert rows[0]["revision_state"] == "conflicted"
        assert rows[0]["eligible_leaf_ids"] == ["b", "c"]
        assert rows[0]["eligible_leaf_count"] == 2
        assert not rows[0]["revision_truncated"]
        assert "conflicted" in RENDERERS["recall"](before.structuredContent)
        assert "eligible=b,c" in RENDERERS["recall"](before.structuredContent)
        after = await client.call_tool(
            "evidence",
            {
                "ids": ["a", "b", "c", "d"],
                "effective_at": "2026-07-01T00:00:00Z",
            },
        )
        assert not after.isError
        final = after.structuredContent["results"][-1]
        assert final["revises_event_ids"] == ["b", "c"]
        assert final["eligible_leaf_ids"] == ["d"]
        assert final["revision_state"] == "active"


async def test_same_client_retry_reuses_original_runtime_capture(
    settings_for_tmp, tmp_path, monkeypatch
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    args = remember_args(event_id="retry-stable", effective_at="2026-01-01T00:00:00Z")
    monkeypatch.chdir(first_dir)
    original = await cmd_remember(
        args, settings_for_tmp, None, client="test-client", session_id="first-session"
    )
    monkeypatch.chdir(second_dir)
    retry = await cmd_remember(
        args, settings_for_tmp, None, client="test-client", session_id="new-session"
    )
    assert retry["recorded_at"] == original["recorded_at"]
    assert retry["effective_at"] == original["effective_at"]
    ctx = build_memory_context(settings_for_tmp)
    try:
        stored = await ctx.gate.get("retry-stable", user_id=settings_for_tmp.owner_user_id)
        assert stored.session_id == "first-session"
        assert stored.cwd == str(first_dir)
    finally:
        ctx.conn.close()


async def test_mcp_retry_after_server_restart_preserves_recording(settings_for_tmp, monkeypatch):
    request = {
        "text": "User goal",
        "source": "user_stated",
        "author_id": "person:owner",
        "event_id": "restart-root",
        "effective_at": "2026-01-01T00:00:00Z",
    }
    async with create_connected_server_and_client_session(
        build_server(settings_for_tmp).mcp
    ) as client:
        original = await client.call_tool("remember", request)
        assert not original.isError

    async def unexpected_embedding(*args, **kwargs):
        raise AssertionError("Exact reconnect retry embedded again")

    monkeypatch.setattr("morgan_brain.memory.embedder.FakeEmbedder.embed", unexpected_embedding)
    async with create_connected_server_and_client_session(
        build_server(settings_for_tmp).mcp
    ) as client:
        retry = await client.call_tool("remember", request)
        assert not retry.isError
        assert retry.structuredContent["recorded_at"] == original.structuredContent["recorded_at"]


async def test_same_session_retry_and_no_id_assertions_stay_distinct(settings_for_tmp):
    stable = remember_args(event_id="same-session")
    original = await cmd_remember(stable, settings_for_tmp, None, session_id="session")
    retry = await cmd_remember(stable, settings_for_tmp, None, session_id="session")
    assert original["id"] == retry["id"] and original["recorded_at"] == retry["recorded_at"]
    assert original["effective_at"] == retry["effective_at"]
    fresh = remember_args()
    first = await cmd_remember(fresh, settings_for_tmp, None, session_id="session")
    second = await cmd_remember(fresh, settings_for_tmp, None, session_id="session")
    assert first["id"] != second["id"]
    assert first["effective_at"] is not None and second["recorded_at"] is not None


async def test_facts_surface_drops_stale_support_but_retains_labeled_legacy(settings_for_tmp):
    ctx = build_memory_context(settings_for_tmp)
    at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        await ctx.gate.store(
            Memory(
                id="basis",
                user_id="owner",
                content="20 minutes",
                source="user_stated",
                author_id="person",
                created_at=at,
            )
        )
        await ctx.gate.upsert_fact(
            TemporalFact(
                id="grounded",
                user_id="owner",
                subject="user",
                predicate="minutes",
                object="20",
                source="agent_inferred",
                support_event_ids=["basis"],
                valid_from=at,
            )
        )
        await ctx.gate.upsert_fact(
            TemporalFact(
                id="legacy",
                user_id="owner",
                subject="user",
                predicate="legacy",
                object="unverified",
                source="unknown",
                valid_from=at,
            )
        )
        await ctx.gate.store(
            Memory(
                id="correction",
                user_id="owner",
                content="10 minutes",
                source="user_stated",
                author_id="person",
                created_at=at,
                revises_event_ids=["basis"],
            )
        )
    finally:
        ctx.conn.close()
    async with create_connected_server_and_client_session(
        build_server(settings_for_tmp).mcp
    ) as client:
        facts = await client.call_tool("facts", {})
        assert not facts.isError
        assert [record["id"] for record in facts.structuredContent["facts"]] == ["legacy"]
        assert facts.structuredContent["facts"][0]["support_state"] == "unsupported"
        evidence = await client.call_tool("evidence", {"ids": ["grounded"]})
        assert not evidence.isError
        assert evidence.structuredContent["results"][0]["support_state"] == "inactive_support"


@pytest.mark.parametrize("changed", ["client", "origin", "content", "author", "time", "project"])
async def test_runtime_retry_does_not_relax_other_identity(settings_for_tmp, changed):
    args = remember_args(event_id="boundary-retry", effective_at="2026-01-01T00:00:00Z")
    if changed == "origin":
        ctx = build_memory_context(settings_for_tmp)
        try:
            await ctx.gate.store(
                Memory(
                    id="boundary-retry",
                    user_id=settings_for_tmp.owner_user_id,
                    content=args.text,
                    source="user_stated",
                    author_id="person:owner",
                    origin_kind=OriginKind.IMPORT,
                    client="first-client",
                )
            )
        finally:
            ctx.conn.close()
        with pytest.raises(ValueError, match="identity conflict"):
            await cmd_remember(
                args, settings_for_tmp, None, client="first-client", session_id="new"
            )
        return
    await cmd_remember(args, settings_for_tmp, None, client="first-client", session_id="first")
    if changed == "content":
        args.text = "Changed goal"
    if changed == "author":
        args.author_id = "person:other"
    if changed == "time":
        args.effective_at = "2026-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="identity conflict"):
        await cmd_remember(
            args,
            settings_for_tmp,
            "other" if changed == "project" else None,
            client="different-client" if changed == "client" else "first-client",
            session_id="new-session",
        )
