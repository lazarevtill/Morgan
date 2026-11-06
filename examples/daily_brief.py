"""Synthetic end-to-end personal planning demo; creates a new private scratch directory only."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from morgan_brain.composition import build_memory_module, open_evidence_context
from morgan_brain.daily_brief import BriefRequest, Commitment, prepare_brief
from morgan_brain.local_notes import NoteWorkspace, approve, run_local
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource
from morgan_brain.tasks import TaskKey

_DEMO_ITEMS = [
    ("meeting", "Подготовить вопросы к встрече", 30, "high", -1, "open"),
    ("papers", "Разобрать домашние документы", 20, "normal", 0, "open"),
    ("reading", "Почитать книгу", 25, "low", 0, "open"),
    ("shopping", "Проверить список покупок", 10, "high", 0, "done"),
    ("weekend", "Спланировать выходные", 20, "normal", 1, "open"),
]


async def _seed(path: Path, now: datetime) -> tuple[str, ...]:
    """Store only this synthetic fixture through the real gate in a new demo database."""
    conn = open_db(str(path))
    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4))
    try:
        for identity, title, minutes, priority, offset, status in _DEMO_ITEMS:
            commitment = Commitment.model_validate(
                {
                    "version": "morgan.commitment.v1",
                    "title": title,
                    "minutes": minutes,
                    "priority": priority,
                    "due": now.date() + timedelta(days=offset),
                    "status": status,
                }
            )
            await gate.store(
                Memory(
                    id=identity,
                    user_id="demo",
                    project="personal",
                    source=MemorySource.USER_STATED,
                    author_id="demo",
                    client="synthetic-daily-brief-demo",
                    created_at=now,
                    content=commitment.model_dump_json(),
                )
            )
    finally:
        conn.close()
    return tuple(item[0] for item in _DEMO_ITEMS)


async def demonstrate(root: Path) -> dict[str, object]:
    """Prepare a fixture plan and explicitly approve only its checked local delivery."""
    await asyncio.to_thread(root.mkdir, mode=0o700)
    now = datetime.now(UTC)
    evidence_ids = await _seed(root / "memory.db", now)
    source = open_evidence_context(root / "memory.db")
    try:
        draft = await prepare_brief(
            source.gate,
            BriefRequest(
                TaskKey("demo", "personal", "daily-brief"),
                evidence_ids,
                now.date(),
                50,
            ),
        )
    finally:
        source.conn.close()
    if draft.selected_ids != ("meeting", "papers") or draft.planned_minutes != 50:
        raise RuntimeError("synthetic fixture plan did not match the expected selection")
    folder = NoteWorkspace(root)
    try:
        folder.publish("draft.md", draft.task.payload)
    finally:
        folder.close()
    delivery = NoteWorkspace(root / "delivery")
    try:
        # This demonstration's trusted owner approves only the checked synthetic fixture.
        # Production preparation never calls approve; review the draft with the existing CLI.
        approve(delivery, [draft.task], ttl_seconds=60)
        first = await run_local(delivery)
        repeat = await run_local(delivery)
        files = sorted(delivery.path.glob("note-*.md"))
        return {
            "synthetic": True,
            "planner": draft.planner,
            "planned_minutes": draft.planned_minutes,
            "selected": list(draft.selected_ids),
            "deferred": list(draft.deferred_ids),
            "excluded": list(draft.excluded_ids),
            "first_actions": first.actions,
            "repeat_actions": repeat.actions,
            "delivered_files": [str(p) for p in files],
        }
    finally:
        delivery.close()


def main() -> None:
    """Run only when explicitly invoked with a new scratch directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(demonstrate(args.new_directory)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
