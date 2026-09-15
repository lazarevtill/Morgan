"""Two processes superseding the same facts at once leave one current fact and one history.

Two `morgan consolidate` runs over one database -- the owner's cron and a manual run, say --
each open their own connection and can assert a value for the same (user, project, subject,
predicate) at the same moment. Asserting closes whatever fact is currently valid and opens the
new one, which is a lookup followed by writes. Unless the two are one atomic step, both
processes find the same current fact, both insert, and the key ends with two currently-valid
facts. Nothing raises: the schema does not forbid it, so the only symptom is recall answering
with both values.

Each subject starts with one fact, and each writer asserts its own value, so a serialised run
leaves a single chain: seed, superseded by one writer's fact, superseded by the other's.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import TemporalFact

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

# Without an atomic upsert the writers collide on the first subject, where the barrier releases
# them together, and on one to four more per thousand as their commits interleave. Fewer
# subjects leave those later collisions too rare to back up the first.
_SUBJECTS = 1000
_WRITERS = 2
_TIMEOUT_S = 120


def _fact(subject: int, origin: str) -> TemporalFact:
    return TemporalFact(
        id=f"{origin}-s{subject}",
        user_id="u",
        subject=f"s{subject}",
        predicate="lives_in",
        object=f"city-from-{origin}",
    )


def _write_all(path: str, writer: int, start: Barrier, outcomes: Queue[str]) -> None:
    """Runs in a spawned process: assert a new value for every subject, then report."""
    try:
        conn = open_db(path)
        store = SqliteTemporalStore(conn=conn)
        start.wait(timeout=_TIMEOUT_S)

        async def supersede_every_subject() -> None:
            for s in range(_SUBJECTS):
                await store.upsert_fact(_fact(s, f"writer{writer}"), now=datetime.now(UTC))

        asyncio.run(supersede_every_subject())
        conn.close()
    except BaseException as exc:
        outcomes.put(f"writer {writer}: {type(exc).__name__}: {exc}")
        raise
    outcomes.put(f"writer {writer}: ok")


def test_two_processes_superseding_the_same_facts_leave_one_current_fact_each(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "morgan.db")
    setup = open_db(path)
    seeded = SqliteTemporalStore(conn=setup)

    async def seed() -> None:
        for s in range(_SUBJECTS):
            await seeded.upsert_fact(_fact(s, "seed"), now=datetime.now(UTC))

    asyncio.run(seed())
    setup.close()

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(_WRITERS)
    outcomes: Queue[str] = ctx.Queue()
    writers = [
        ctx.Process(target=_write_all, args=(path, w, start, outcomes)) for w in range(_WRITERS)
    ]
    for p in writers:
        p.start()
    try:
        reported = sorted(outcomes.get(timeout=_TIMEOUT_S) for _ in writers)
        for p in writers:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in writers:
            if p.is_alive():
                p.terminate()

    assert reported == [f"writer {w}: ok" for w in range(_WRITERS)]
    assert [p.exitcode for p in writers] == [0] * _WRITERS

    conn = open_db(path)
    try:
        rows = conn.execute("SELECT id, subject, valid_to, superseded_by FROM facts").fetchall()
    finally:
        conn.close()

    by_subject: dict[str, dict[str, dict[str, str | None]]] = defaultdict(dict)
    for r in rows:
        by_subject[r["subject"]][r["id"]] = {
            "valid_to": r["valid_to"],
            "superseded_by": r["superseded_by"],
        }
    assert len(by_subject) == _SUBJECTS

    broken: list[str] = []
    for subject, facts in by_subject.items():
        current = [fid for fid, f in facts.items() if f["valid_to"] is None]
        # Walk the supersession chain from the seed; it must visit every fact once and end at
        # the one current fact.
        chain = [f"seed-{subject}"]
        while (nxt := facts[chain[-1]]["superseded_by"]) in facts and nxt not in chain:
            chain.append(nxt)
        if len(facts) != 1 + _WRITERS or current != chain[-1:] or len(chain) != len(facts):
            broken.append(subject)
    assert broken == [], f"{len(broken)} of {_SUBJECTS} subjects forked: {broken[:5]}"
