"""Real local working-context adapters: scoped reads/writes, no model or endpoints."""

import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings
from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.working_context import WorkingContextDraft, WorkingContextPreview
from morgan_brain.models import Memory, MemorySource
from morgan_brain.surfaces.cli.__main__ import build_parser
from morgan_brain.surfaces.cli.working_context import (
    apply_context,
    cmd_context,
    continue_context,
    propose_context,
    read_context,
    render_context,
)


@pytest.fixture
def local_settings(tmp_path):
    return Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        owner_user_id="synthetic-owner",
        embedding_backend="hash",
    )


async def prepare(settings, *, project="personal", identity="gift"):
    ctx = build_memory_context(settings)
    try:
        await ctx.gate.store(
            Memory(
                id="choice",
                user_id=settings.owner_user_id,
                project=project,
                content="Use four panels because the paper folds neatly.",
                source=MemorySource.USER_STATED,
                author_id="person:synthetic",
            )
        )
        basis = (
            await ctx.gate.evidence(
                user_id=settings.owner_user_id, project=project, evidence_ids=["choice"]
            )
        ).records
        state = WorkingContextDraft.model_validate(
            {
                "title": "Gift booklet",
                "decisions": [
                    {
                        "choice": {"event_id": "choice", "quote": "Use four panels"},
                        "reason": {"event_id": "choice", "quote": "because the paper folds neatly"},
                    }
                ],
            }
        ).normalize(basis)
        return WorkingContextPreview(
            context_id=identity,
            context=CheckpointContext(
                user_id=settings.owner_user_id, project=project, author_id="agent:synthetic"
            ),
            expected_fact_id=None,
            generation=ctx.gate.capture_erasure_generation(),
            state=state,
            evidence_basis=basis,
        )
    finally:
        ctx.conn.close()


async def test_personal_default_ignores_repository_cwd_and_dispatch_scope(
    local_settings, tmp_path, monkeypatch
):
    repo = tmp_path / "unrelated-repository"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)
    preview = await prepare(local_settings)
    applied = await apply_context(local_settings, preview.model_dump(mode="json"))
    assert applied["project"] == "personal" and applied["project_defaulted"]
    args = build_parser().parse_args(["context", "show", "gift"])
    shown = await cmd_context(args, local_settings, "unrelated-repository")
    assert shown["project"] == "personal" and shown["project_defaulted"]
    assert shown["view"]["state"]["title"] == "Gift booklet"
    assert (await read_context(local_settings, None, "unrelated-repository"))["contexts"] == []
    rendered = render_context(shown)
    assert "Reason: because the paper folds neatly" in rendered
    assert "unverified" in rendered
    assert "morgan evidence choice --project personal" in rendered


async def test_corrected_source_remains_discoverable_and_renders_rebuild(local_settings):
    preview = await prepare(local_settings)
    await apply_context(local_settings, preview.model_dump(mode="json"))
    ctx = build_memory_context(local_settings)
    try:
        await ctx.gate.store(
            Memory(
                id="updated",
                user_id=local_settings.owner_user_id,
                content="Use two panels instead.",
                source=MemorySource.USER_STATED,
                author_id="person:synthetic",
                revises_event_ids=["choice"],
                created_at=datetime.now(UTC),
            )
        )
    finally:
        ctx.conn.close()
    listing = await read_context(local_settings, None)
    assert listing["contexts"] == [
        {
            "context_id": "gift",
            "fact_id": listing["contexts"][0]["fact_id"],
            "eligibility": "needs_rebuild",
            "title": None,
        }
    ]
    assert "gift: rebuild required [needs_rebuild]" in render_context(listing)
    shown = await read_context(local_settings, "gift")
    assert shown["view"]["state"] is None and shown["view"]["sources"] == []
    rendered = render_context(shown)
    assert "context propose gift --rebuild --event-id CURRENT_SOURCE_ID" in rendered
    assert "Review the proposal, then apply it explicitly" in rendered
    assert "four panels" not in rendered


@pytest.mark.parametrize("field,value", [("user_id", "other-owner"), ("project", "other-project")])
async def test_apply_refuses_foreign_proposal_before_opening_database(
    local_settings, monkeypatch, field, value
):
    preview = (await prepare(local_settings)).model_dump(mode="json")
    preview["context"][field] = value

    def forbidden_open(settings):
        raise AssertionError("Rejected proposal opened storage")

    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.working_context.build_memory_context", forbidden_open
    )
    with pytest.raises(PermissionError, match="ownership/project"):
        await apply_context(local_settings, preview)


def test_resume_parser_requires_independent_session_and_rebuild_is_explicit():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["context", "resume", "gift", "Draft the next page"])
    args = parser.parse_args(
        ["context", "resume", "gift", "Draft the next page", "--session-id", "fresh-session"]
    )
    assert args.session_id == "fresh-session" and args.project is None
    propose = parser.parse_args(
        ["context", "propose", "gift", "--event-id", "updated", "--rebuild", "--json"]
    )
    assert propose.rebuild and propose.event_ids == ["updated"]


def test_continuation_renderer_keeps_evidence_access_and_project_visible():
    rendered = render_context(
        {
            "response": "Draft: fold the sheet.",
            "context_id": "gift",
            "project": "orchid",
            "source_event_ids": ["source-one", "source-two"],
        }
    )
    assert rendered.startswith("Draft: fold the sheet.")
    assert "Working context: gift (orchid)" in rendered
    assert "morgan evidence source-one source-two --project orchid" in rendered


def test_truncated_listing_explains_bound_and_direct_lookup():
    rendered = render_context(
        {
            "contexts": [{"context_id": "gift", "title": "Gift", "eligibility": "current"}],
            "truncated": True,
        }
    )
    assert "Only the first 32 names are shown" in rendered
    assert "read a known context by ID" in rendered


def test_rendered_commands_quote_names_and_source_ids_as_literal_arguments():
    rendered = render_context({"context_id": "gift zine", "project": "my repo", "view": None})
    assert "'gift zine'" in rendered and "'my repo'" in rendered
    draft = render_context(
        {
            "response": "Draft",
            "context_id": "gift",
            "project": "my repo",
            "source_event_ids": ["source $(execute)"],
        }
    )
    assert "'source $(execute)'" in draft and "'my repo'" in draft


@pytest.mark.parametrize("operation", ["propose", "resume"])
async def test_both_resource_closures_and_sqlite_close_run_after_chat_cleanup_error(
    local_settings, monkeypatch, operation
):
    ctx = build_memory_context(local_settings)
    closed = []

    async def chat_close():
        closed.append("chat")
        raise RuntimeError("synthetic chat cleanup failure")

    async def client_close():
        closed.append("client")

    async def body_failure(*args, **kwargs):
        raise ValueError("synthetic body failure")

    app = SimpleNamespace(
        conn=ctx.conn,
        gate=ctx.gate,
        history=ctx.history,
        chat=SimpleNamespace(aclose=chat_close),
        client=SimpleNamespace(aclose=client_close),
    )
    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.working_context.build_app_context", lambda _: app
    )
    monkeypatch.setattr(
        "morgan_brain.surfaces.cli.working_context.WorkingContextService.preview", body_failure
    )
    monkeypatch.setattr("morgan_brain.surfaces.cli.working_context.resume_work", body_failure)
    with pytest.raises(RuntimeError, match="chat cleanup failure"):
        if operation == "propose":
            await propose_context(local_settings, "gift", ["choice"])
        else:
            await continue_context(local_settings, "gift", "Draft next page", "fresh")
    assert closed == ["chat", "client"]
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        ctx.conn.execute("SELECT 1")


async def test_incremental_unicode_proposals_remain_applicable_or_refuse_before_return(
    local_settings, tmp_path, monkeypatch
):
    import json

    from morgan_brain.app.working_context import WorkingContextService
    from morgan_brain.surfaces.cli import working_context as adapter

    selected = []

    async def proposer(messages):
        return WorkingContextDraft.model_validate(
            {"title": "Bounded selection", "intentions": selected[:4], "progress": selected[4:]}
        )

    async def close():
        pass

    def app_context(settings):
        ctx = build_memory_context(settings)
        return SimpleNamespace(
            conn=ctx.conn,
            gate=ctx.gate,
            chat=SimpleNamespace(aclose=close),
            client=SimpleNamespace(aclose=close),
        )

    monkeypatch.setattr(adapter, "build_app_context", app_context)
    monkeypatch.setattr(
        adapter,
        "WorkingContextService",
        lambda **kwargs: WorkingContextService(**kwargs, proposer=proposer),
    )
    previous_fact = None
    for index in range(8):
        identity = f"unicode-{index}"
        quote = f"Event {index} "
        content = quote + chr(0x1F600) * (4096 - len(quote))
        ctx = build_memory_context(local_settings)
        try:
            await ctx.gate.store(
                Memory(
                    id=identity,
                    user_id=local_settings.owner_user_id,
                    content=content,
                    source=MemorySource.USER_STATED,
                    author_id="person:synthetic",
                )
            )
        finally:
            ctx.conn.close()
        selected.append({"event_id": identity, "quote": quote})
        if index == 7:
            with pytest.raises(ValueError, match="select fewer sources"):
                await propose_context(local_settings, "unicode", [identity])
            view = await read_context(local_settings, "unicode")
            assert view["view"]["fact_id"] == previous_fact
        else:
            result = await propose_context(local_settings, "unicode", [identity])
            raw = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            path = tmp_path / "proposal.json"
            path.write_bytes(raw.replace("\n", "\r\n").encode("utf-8"))
            loaded = adapter.load_proposal(str(path))
            applied = await apply_context(local_settings, loaded)
            previous_fact = applied["fact_id"]


async def test_proposal_lf_below_cap_refuses_when_windows_output_exceeds_cap(
    local_settings, monkeypatch
):
    import json

    from morgan_brain.surfaces.cli import working_context as adapter

    preview = await prepare(local_settings)
    payload = preview.model_dump(mode="json")
    result = {
        "project": "personal",
        "project_defaulted": True,
        "proposal": payload,
        "applied": False,
    }
    raw = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    # A long evidence body approaches the physical-file bound without adding
    # formatting lines. LF fits exactly; Windows CRLF must be refused.
    payload["evidence_basis"][0]["content"] += "x" * (
        adapter.MAX_PROPOSAL_BYTES - len(raw.encode("utf-8"))
    )
    raw = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    assert len(raw.encode("utf-8")) == adapter.MAX_PROPOSAL_BYTES
    assert len(raw.replace("\n", "\r\n").encode("utf-8")) > adapter.MAX_PROPOSAL_BYTES

    closed = []

    async def close():
        closed.append("resource")

    class PreviewService:
        def __init__(self, **kwargs):
            pass

        async def preview(self, *args, **kwargs):
            return SimpleNamespace(model_dump=lambda **kwargs: payload)

    def app_context(settings):
        ctx = build_memory_context(settings)
        return SimpleNamespace(
            conn=ctx.conn,
            gate=ctx.gate,
            chat=SimpleNamespace(aclose=close),
            client=SimpleNamespace(aclose=close),
        )

    monkeypatch.setattr(adapter, "build_app_context", app_context)
    monkeypatch.setattr(adapter, "WorkingContextService", PreviewService)
    with pytest.raises(ValueError, match="select fewer sources"):
        await propose_context(local_settings, "gift", ["choice"])
    assert closed == ["resource", "resource"]
    view = await read_context(local_settings, "gift")
    assert view["view"] is None
