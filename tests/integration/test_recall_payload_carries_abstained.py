"""Both surfaces say whether recall abstained, and why.

``--json`` and the MCP tool result are contracts other programs parse, so the two new keys ride
beside the ones already there. A decline in a project that holds facts returns neither memories
nor facts: ``abstained`` beside returned facts would be a result contradicting itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings
from morgan_brain.models import Memory, TemporalFact
from morgan_brain.surfaces.mcp_server import build_server

#: A margin no hash embedding reaches: every one of its components is positive, so no
#: similarity falls far enough below the best one. Every judged query is declined.
_A_FLOOR_NOTHING_CLEARS = "0.9"
#: Shares no word with any memory below, so no exact entity match can overrule the floor.
_UNANSWERABLE = "zyzzyva quokka"


def _run(args: list[str], tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.surfaces.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "MORGAN_DATA_DIR": str(tmp_path),
            "MORGAN_EMBEDDING_BACKEND": "hash",
            **env,
        },
        cwd=tmp_path,
        check=False,
    )


def _settings(tmp_path: Path, floor_margin: float | None = None) -> Settings:
    return Settings(
        data_dir=str(tmp_path), embedding_backend="hash", recall_floor_margin=floor_margin
    )


async def _a_project_with_facts(tmp_path: Path) -> None:
    """Six memories -- enough for the floor to judge -- and one current fact, in ``acme``."""
    ctx = build_memory_context(_settings(tmp_path))
    try:
        for n in range(6):
            await ctx.gate.store(
                Memory(user_id="owner", project="acme", content=f"grocery list number {n}")
            )
        await ctx.gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                project="acme",
                subject="deploy",
                predicate="blocked_by",
                object="mirror",
            )
        )
    finally:
        ctx.conn.close()


def test_the_cli_says_an_empty_project_is_empty(tmp_path):
    out = _run(["recall", "anything", "--project", "acme", "--json"], tmp_path)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload == {
        "project": "acme",
        "all_projects": False,
        "abstained": True,
        "reason": "empty",
        "results": [],
    }

    words = _run(["recall", "anything", "--project", "acme"], tmp_path)
    assert "No memories found in project 'acme' (empty)" in words.stdout


def test_the_cli_says_a_decline_and_returns_no_facts(tmp_path):
    asyncio.run(_a_project_with_facts(tmp_path))
    floor = {"MORGAN_RECALL_FLOOR_MARGIN": _A_FLOOR_NOTHING_CLEARS}

    out = _run(["recall", _UNANSWERABLE, "--project", "acme", "--json"], tmp_path, **floor)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert (payload["abstained"], payload["reason"], payload["results"]) == (True, "declined", [])

    words = _run(["recall", _UNANSWERABLE, "--project", "acme"], tmp_path, **floor)
    assert "declined: nothing stood out above the background" in words.stdout


def test_the_cli_says_an_answer_was_not_judged_without_a_floor(tmp_path):
    asyncio.run(_a_project_with_facts(tmp_path))

    out = _run(["recall", "grocery", "--project", "acme", "--json"], tmp_path)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert (payload["abstained"], payload["reason"]) == (False, "no_floor")
    assert payload["results"]


async def test_the_mcp_result_carries_abstained_and_reason(tmp_path):
    empty = await build_server(_settings(tmp_path)).call_tool(
        "recall", {"query": "anything", "project": "acme"}
    )
    assert (empty["abstained"], empty["reason"], empty["results"]) == (True, "empty", [])

    await _a_project_with_facts(tmp_path)
    declined = await build_server(_settings(tmp_path, floor_margin=0.9)).call_tool(
        "recall", {"query": _UNANSWERABLE, "project": "acme"}
    )
    assert (declined["abstained"], declined["reason"], declined["results"]) == (
        True,
        "declined",
        [],
    )

    answered = await build_server(_settings(tmp_path)).call_tool(
        "recall", {"query": "grocery", "project": "acme"}
    )
    assert (answered["abstained"], answered["reason"]) == (False, "no_floor")
    assert answered["results"]
