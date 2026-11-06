"""Real preview/accept processes bind reviewed data and preserve normal scoped memory semantics."""

import asyncio
import hashlib
import json
from datetime import date

import pytest

from morgan_brain.composition import open_evidence_context
from morgan_brain.daily_brief import BriefRequest, prepare_brief
from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.tasks import TaskKey
from tests.integration.test_cli import _hash_env, _run


@pytest.fixture
def proposal(tmp_path):
    """Create an unapproved synthetic model proposal in a private directory."""
    path = tmp_path / "drafts"
    workspace = NoteWorkspace(path)
    note = "Нужно полить растения.\nЯ уже купил хлеб.\nПозвонить в библиотеку."
    try:
        workspace.publish("note.txt", note)
        workspace.publish(
            "proposal.json",
            json.dumps(
                {
                    "version": "morgan.note-proposal.v1",
                    "source_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest(),
                    "source": "agent_inferred",
                    "approved": False,
                    "review_required": True,
                    "selected_lines": ["Нужно полить растения.", "Позвонить в библиотеку."],
                },
                ensure_ascii=False,
            ),
        )
    finally:
        workspace.close()
    return path / "proposal.json"


def arguments(proposal):
    """Supply every planning and provenance field before review, without accepting it."""
    return [
        "review-proposal",
        str(proposal),
        "--note",
        str(proposal.parent / "note.txt"),
        "--selection",
        "1",
        "--due",
        "2026-11-06",
        "--minutes",
        "10",
        "--priority",
        "normal",
        "--author-id",
        "owner",
        "--event-id",
        "reviewed-plants",
        "--effective-at",
        "2020-01-01T00:00:00Z",
        "--project",
        "personal",
        "--json",
    ]


def preview(args, env, root):
    """Run preview as a separate process and require a valid protocol response."""
    result = _run(args, env, root)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_preview_accept_replay_and_plan(proposal, tmp_path):
    """Only acceptance stores a current user assertion, with review provenance and no tool grant."""
    memory = tmp_path / "memory"
    env = _hash_env(memory)
    args = arguments(proposal)
    draft = preview(args, env, tmp_path)
    assert draft["stored"] is False
    assert draft["review"]["original_line"] == "Нужно полить растения."
    assert draft["review"]["database"] == str(memory / "morgan.db")
    assert not memory.exists()
    accepted = [*args, "--accept", draft["review_sha256"]]
    result = preview(accepted, env, tmp_path)
    assert result["stored"] is True
    assert result["source"] == "user_stated"
    assert result["execution_authorized"] is False
    assert preview(accepted, env, tmp_path)["id"] == result["id"]
    context = open_evidence_context(memory / "morgan.db")
    try:
        evidence = asyncio.run(
            context.gate.evidence(user_id="owner", project="personal", evidence_ids=[result["id"]])
        )
        assert evidence.records[0].client == "reviewed-proposal-v1"
        assert evidence.records[0].session_id == draft["review_sha256"]
        plan = asyncio.run(
            prepare_brief(
                context.gate,
                BriefRequest(
                    TaskKey("owner", "personal", "focus"), (result["id"],), date(2026, 11, 6), 30
                ),
            )
        )
        assert plan.planned_minutes == 10
    finally:
        context.conn.close()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--selection", "2"),
        ("--due", "2026-11-07"),
        ("--minutes", "20"),
        ("--priority", "high"),
        ("--author-id", "different"),
        ("--event-id", "different"),
        ("--effective-at", "2021-01-01T00:00:00Z"),
        ("--project", "different"),
    ],
)
def test_changed_fields_reject_before_write(proposal, tmp_path, flag, value):
    """Every mutable claim or scope field invalidates a previously displayed confirmation."""
    memory = tmp_path / "memory"
    env = _hash_env(memory)
    args = arguments(proposal)
    digest = preview(args, env, tmp_path)["review_sha256"]
    args[args.index(flag) + 1] = value
    result = _run([*args, "--accept", digest], env, tmp_path)
    assert result.returncode != 0
    assert "review changed" in result.stdout
    assert not memory.exists()


def test_changed_file_rejects(proposal, tmp_path):
    """Replacement of the artifact after preview cannot silently change the accepted line."""
    env = _hash_env(tmp_path / "memory")
    args = arguments(proposal)
    digest = preview(args, env, tmp_path)["review_sha256"]
    proposal.write_text(proposal.read_text().replace("растения", "другие растения"))
    result = _run([*args, "--accept", digest], env, tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize("change", ["owner", "database"])
def test_changed_destination_rejects(proposal, tmp_path, change):
    """A token cannot be reused with a different configured owner or database."""
    env = _hash_env(tmp_path / "memory")
    args = arguments(proposal)
    digest = preview(args, env, tmp_path)["review_sha256"]
    env["MORGAN_OWNER_USER_ID" if change == "owner" else "MORGAN_DATA_DIR"] = (
        "other" if change == "owner" else str(tmp_path / "other-memory")
    )
    assert _run([*args, "--accept", digest], env, tmp_path).returncode != 0
    assert not (tmp_path / "memory").exists()
    assert not (tmp_path / "other-memory").exists()


def test_title_correction_requires_new_preview(proposal, tmp_path):
    """Reviewed edits are supported, but cannot inherit confirmation of the original wording."""
    env = _hash_env(tmp_path / "memory")
    args = arguments(proposal)
    old = preview(args, env, tmp_path)["review_sha256"]
    edited = [*args, "--title", "Полить цветы на балконе"]
    assert _run([*edited, "--accept", old], env, tmp_path).returncode != 0
    new = preview(edited, env, tmp_path)
    assert new["review"]["commitment"]["title"] == "Полить цветы на балконе"
    assert preview([*edited, "--accept", new["review_sha256"]], env, tmp_path)["stored"]


@pytest.mark.parametrize("payload", ["null", "[]", '{"approved":true}', '{"approved":0}'])
def test_invalid_artifact_no_memory(proposal, tmp_path, payload):
    """Malformed and forged metadata fail before a memory context is opened."""
    proposal.write_text(payload)
    assert _run(arguments(proposal), _hash_env(tmp_path / "memory"), tmp_path).returncode != 0
    assert not (tmp_path / "memory").exists()


def test_missing_directory_preview_is_read_only(tmp_path):
    """A preview of a missing artifact must not create even its input directory."""
    missing = tmp_path / "absent" / "proposal.json"
    assert _run(arguments(missing), _hash_env(tmp_path / "memory"), tmp_path).returncode != 0
    assert not missing.parent.exists()
    assert not (tmp_path / "memory").exists()


def test_symlink_rejected(proposal, tmp_path):
    """Private file handling refuses a symlink instead of following its target."""
    link = proposal.parent / "link.json"
    link.symlink_to(proposal)
    assert _run(arguments(link), _hash_env(tmp_path / "memory"), tmp_path).returncode != 0


def test_unknown_selection_rejected(proposal, tmp_path):
    """Out-of-range indices cannot select fabricated source text."""
    args = arguments(proposal)
    args[args.index("--selection") + 1] = "3"
    assert _run(args, _hash_env(tmp_path / "memory"), tmp_path).returncode != 0


def test_changed_source_note_rejects(proposal, tmp_path):
    """A later completion in the source invalidates approval of an earlier proposal."""
    env = _hash_env(tmp_path / "memory")
    args = arguments(proposal)
    digest = preview(args, env, tmp_path)["review_sha256"]
    note = proposal.parent / "note.txt"
    note.write_text(note.read_text() + "\nПозже: растения уже политы.")
    assert _run([*args, "--accept", digest], env, tmp_path).returncode != 0
    assert not (tmp_path / "memory").exists()


def test_unrelated_source_line_rejects(proposal, tmp_path):
    """Even a correctly hashed note cannot justify a selected line absent from that note."""
    data = json.loads(proposal.read_text())
    data["selected_lines"][0] = "Заказать другую вещь."
    proposal.write_text(json.dumps(data))
    assert _run(arguments(proposal), _hash_env(tmp_path / "memory"), tmp_path).returncode != 0
    assert not (tmp_path / "memory").exists()


def test_review_cannot_write_across_projects(proposal, tmp_path):
    """The normal single-project write guard also applies to review intake."""
    args = [*arguments(proposal), "--all-projects"]
    assert _run(args, _hash_env(tmp_path / "memory"), tmp_path).returncode != 0
    assert not (tmp_path / "memory").exists()


def test_human_preview_preserves_unicode_and_escapes_controls(proposal, tmp_path):
    """Unselected source context cannot inject terminal escapes or bidi formatting into preview."""
    note = proposal.parent / "note.txt"
    text = note.read_text() + "\nUntrusted: \x1b]0;hidden\x07\u202e"
    note.write_text(text)
    data = json.loads(proposal.read_text())
    data["source_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    proposal.write_text(json.dumps(data))
    args = arguments(proposal)
    args.remove("--json")
    result = _run(args, _hash_env(tmp_path / "memory"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Нужно полить растения" in result.stdout
    assert "Я уже купил хлеб" in result.stdout
    assert "--accept " in result.stdout
    assert all(control not in result.stdout for control in ("\x1b", "\x07", "\u202e"))
    assert not (tmp_path / "memory").exists()


def test_new_review_cannot_reuse_old_event_provenance(proposal, tmp_path):
    """Even unchanged commitment text cannot replace the review digest on an existing event."""
    env = _hash_env(tmp_path / "memory")
    args = arguments(proposal)
    first = preview(args, env, tmp_path)
    assert preview([*args, "--accept", first["review_sha256"]], env, tmp_path)["stored"]
    proposal.write_text(proposal.read_text() + "\n")
    changed = preview(args, env, tmp_path)
    assert changed["review_sha256"] != first["review_sha256"]
    result = _run([*args, "--accept", changed["review_sha256"]], env, tmp_path)
    assert result.returncode != 0
    assert "session_id" in result.stdout
