"""Abrupt process exit before/after commit, without connection cleanup or an ACK."""

import asyncio
import multiprocessing
import os

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import Checkpoint, CheckpointContext, StaleCheckpoint
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.models import Memory, MemorySource


def _gate(conn):
    return MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4))


def _state(status="active"):
    return Checkpoint(kind="task", title="Synthetic task", objective="Test recovery", status=status)


async def _update(gate, basis):
    return await gate.put_checkpoint(
        _state("paused"),
        checkpoint_id="task",
        context=CheckpointContext(user_id="owner"),
        support_event_ids=["source"],
        expected_fact_id=basis,
    )


def _exit_during_update(path, basis, committed):
    conn = open_db(path)
    gate = _gate(conn)
    if committed:
        asyncio.run(_update(gate, basis))
        os._exit(42)
    with write_transaction(conn):
        asyncio.run(_update(gate, basis))
        # Neither __exit__, Python finalizers nor SQLite connection cleanup runs.
        os._exit(42)


@pytest.mark.parametrize("committed", [False, True])
async def test_checkpoint_recovery_after_abrupt_exit(tmp_path, committed):
    path = str(tmp_path / "recovery.db")
    conn = open_db(path)
    try:
        gate = _gate(conn)
        await gate.store(
            Memory(
                id="source",
                user_id="owner",
                content="Synthetic task requested",
                source=MemorySource.USER_STATED,
            )
        )
        basis = await gate.put_checkpoint(
            _state(),
            checkpoint_id="task",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["source"],
        )
    finally:
        conn.close()

    process = multiprocessing.get_context("spawn").Process(
        target=_exit_during_update, args=(path, basis, committed)
    )
    process.start()
    try:
        await asyncio.to_thread(process.join, 20)
        assert not process.is_alive(), "checkpoint child timed out"
        assert process.exitcode == 42
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()

    conn = open_db(path)
    try:
        gate = _gate(conn)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        recovered = await gate.get_checkpoint("task", user_id="owner")
        assert recovered.eligibility == "current"
        assert recovered.state.status == ("paused" if committed else "active")
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == (2 if committed else 1)
        assert await gate.get_checkpoint("task", user_id="another-owner") is None
        if committed:
            assert recovered.fact.id != basis
            # The caller lost the ACK: re-reading must precede a fresh decision, not blind replay.
            with pytest.raises(StaleCheckpoint):
                await _update(gate, basis)
        else:
            assert recovered.fact.id == basis
            await _update(gate, basis)
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2
        evidence = await gate.evidence(user_id="owner", project="personal", evidence_ids=[basis])
        assert len(evidence.records) == 1
    finally:
        conn.close()
