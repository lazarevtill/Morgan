"""Scoped source lookup follows current evidence rather than stale task receipts."""

import asyncio
import hashlib
import json
from contextlib import closing
from datetime import UTC, datetime

import pytest

from morgan_brain import guided_plan, owner_input
from morgan_brain.composition import build_memory_context
from morgan_brain.local_notes import NoteWorkspace
from tests.integration.test_owner_ui import drive, initialize


def capture(root, identity="book", text="Мне нужно вернуть библиотечную книгу."):
    owner_input.request(root, ["submit", identity], text=text)
    review = owner_input.request(
        root,
        [
            "review",
            identity,
            "--new",
            "--title",
            "Вернуть книгу",
            "--minutes",
            "10",
            "--priority",
            "normal",
            "--due",
            "2026-11-06",
        ],
    )["message"]
    owner_input.request(root, ["confirm", identity, "--digest", review["digest"]])
    return review["capture"]["event_id"]


def revise(root, parent, event, *, done=True):
    with closing(NoteWorkspace(root)) as workspace:
        session, _ = owner_input.load_pilot(workspace)
        task = session.captures[0].commitment.model_copy(
            update={"status": "done" if done else "open"}
        )
        update = guided_plan.Capture(
            event_id=event,
            effective_at=datetime.now(UTC).isoformat(),
            commitment=task,
            parents=[parent],
        )
        asyncio.run(guided_plan._capture(workspace, session, update))


def find(root, query="КНИГ"):
    return owner_input.request(root, ["find", query])


def test_search_quotes_memory_and_follows_completion_without_writes(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    parent = capture(root)
    first = find(root)["results"][0]
    assert first["source_kind"] == "verified_memory"
    assert first["source_id"] == parent + "-source"
    assert first["current_task"]["status"] == "open"
    revise(root, parent, "completed-book")
    paths = [root / "owner-input.json", root / "session.json", root / "memory/morgan.db"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    result = find(root)
    item = result["results"][0]
    assert item["current_evidence_id"] == "completed-book"
    assert item["current_task"]["status"] == "done"
    assert item["quote"] == "Мне нужно вернуть библиотечную книгу."
    assert not result["scope_complete"] and not result["execution_authorized"]
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_renewal_source_reports_latest_date_not_original_receipt(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    parent = capture(root)
    owner_input.request(root, ["submit", "renew"], text="Книгу не вернул, перенеси на завтра.")
    item = owner_input.request(
        root, ["review", "renew", "--target", parent, "--due", "2026-11-07"]
    )["message"]
    owner_input.request(root, ["confirm", "renew", "--digest", item["digest"]])
    result = find(root)["results"]
    assert len(result) == 2
    assert {r["current_task"]["due"] for r in result} == {"2026-11-07"}
    assert len({r["source_id"] for r in result}) == 2
    assert len({r["current_evidence_id"] for r in result}) == 1


def test_missing_memory_never_falls_back_to_applied_journal(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    capture(root)
    with closing(NoteWorkspace(root)) as workspace:
        session, _ = owner_input.load_pilot(workspace)
        context = build_memory_context(guided_plan._settings(workspace, session))
        try:
            asyncio.run(context.gate.forget(user_id=session.user, project=session.project))
        finally:
            context.conn.close()
    result = find(root)
    assert result["results"] == []
    assert result["unavailable_confirmed_sources"] == 1
    assert "библиотечную" not in json.dumps(result, ensure_ascii=False)


def test_conflicted_family_is_not_presented_as_current(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    parent = capture(root)
    revise(root, parent, "branch-one")
    revise(root, parent, "branch-two", done=False)
    result = find(root)
    assert result["results"] == [] and result["unavailable_confirmed_sources"] == 1


@pytest.mark.parametrize("query", ["", "   ", "x" * 201, "book\n"])
def test_query_bounds(tmp_path, query):
    root = tmp_path / "pilot"
    initialize(root)
    with pytest.raises(ValueError, match="search requires"):
        find(root, query)


def test_unconfirmed_and_dismissed_are_receipts_not_current_tasks(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    owner_input.request(root, ["submit", "note"], text="I already returned the BOOK.")
    owner_input.request(root, ["dismiss", "note"])
    item = find(root, "book")["results"][0]
    assert item["review_state"] == "dismissed"
    assert item["source_kind"] == "inbox_receipt" and item["current_task"] is None
    assert find(root, "not in this inbox")["results"] == []
    owner_input.request(root, ["stop"])
    assert find(root, "book")["results"] == [item]


def test_scripted_ui_shows_quote_and_current_completed_state(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    parent = capture(root)
    revise(root, parent, "done")
    dialogs = drive(
        root,
        [
            ("choose", "Найти источник"),
            ("text", "книг"),
            ("choose", 0),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    text = next(d["prompt"] for d in dialogs.seen if d["kind"] == "info")
    assert "Мне нужно вернуть библиотечную книгу." in text
    assert "выполнено" in text and parent + "-source" in text
    assert "Это сведения, не разрешение на действие." in text


def test_changed_owner_cannot_read_other_scope_through_receipt_ids(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    capture(root)
    for name, key in (("owner-input.json", "user"), ("session.json", "user")):
        path = root / name
        value = json.loads(path.read_text())
        value[key] = "different-owner"
        path.write_text(json.dumps(value))
    result = find(root)
    assert result["results"] == []
    assert result["unavailable_confirmed_sources"] == 1


def test_read_only_search_remains_available_after_expiry(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    owner_input.request(root, ["submit", "a"], text="Synthetic note.")
    path = root / "owner-input.json"
    value = json.loads(path.read_text())
    value["expires_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    assert find(root, "Synthetic")["results"][0]["current_task"] is None
    assert path.read_bytes() == before
