"""A stopped import's suspects reach the owner through the real CLI entry point, not just
through ``ImportStopped``'s own attributes.

Re-review round-1 finding: nothing drove ``surfaces/cli/__main__.py``'s new
``except ImportStopped`` clause through the actual ``main``/argparse/``_dispatch`` path --
every other test either built ``ImportStopped`` directly or called ``cmd_import`` in-process,
bypassing the exception handler this proves. A real subprocess, against a real (fake) model
server, is the only way to be sure ``--json``'s ``suspect_ids`` key and the text-mode second
stderr line are not lost to a future refactor of that handler.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from morgan_brain.app.chatgpt_import import _memory_id
from tests.fakes import drifting_model_server

#: Turns in the export; well past where the drift stops the import, so the stop is never a
#: coincidence of the export simply running out.
_TURNS = 10
#: The model answers correctly through memory 3, then drifts -- fingerprint strings included.
_DRIFT_AFTER = 3
#: The interval is real configuration (MORGAN_IMPORT_CANARY_EVERY), set through the
#: environment like any other setting -- not a test-only parameter to production code.
_CANARY_EVERY = 3


def _run(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.surfaces.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
        check=False,
    )


def _env(data_dir: Path, embedding_url: str) -> dict[str, str]:
    return {
        **os.environ,
        "MORGAN_DATA_DIR": str(data_dir),
        "MORGAN_EMBEDDING_BACKEND": "provider",
        "MORGAN_EMBEDDING_ENDPOINT": embedding_url,
        "MORGAN_EMBEDDING_DIM": "4",
        "MORGAN_EMBEDDING_MODEL": "cli-canary-test-model",
        "MORGAN_IMPORT_CANARY_EVERY": str(_CANARY_EVERY),
    }


def _export(path: Path) -> Path:
    """One conversation of ``_TURNS`` distinct single-turn "user" messages, in a fixed order,
    so which memory ordinal is which turn is known without reading anything back."""
    conversation = {
        "conversation_id": "cli-canary",
        "title": "t",
        "mapping": {
            f"m{i}": {
                "id": f"m{i}",
                "message": {
                    "id": f"m{i}",
                    "author": {"role": "user"},
                    "create_time": 1_700_000_000.0 + i,
                    "content": {"content_type": "text", "parts": [f"turn {i} content"]},
                },
            }
            for i in range(_TURNS)
        },
    }
    path.write_text(json.dumps([conversation]), encoding="utf-8")
    return path


def _expected_suspect_ids() -> list[str]:
    """Memories 4-6 (1-based, this run's own store order): the model is still good through
    memory 3, the canary at 3 passes, memories 4-6 drift silently, and the canary at 6 --
    fingerprint strings drifted too -- is the first to notice. Turn index i is memory i+1."""
    return [_memory_id(f"m{i}", 0) for i in (3, 4, 5)]


def test_a_stopped_import_surfaces_suspect_ids_in_json(tmp_path):
    export = _export(tmp_path / "export.json")
    with drifting_model_server(embedding_dim=4, drift_after=_DRIFT_AFTER) as url:
        env = _env(tmp_path / "data", url)
        out = _run(["import", str(export), "--json"], env, tmp_path)

    assert out.returncode != 0, out.stderr
    assert "Traceback" not in out.stdout
    assert "Traceback" not in out.stderr
    payload = json.loads(out.stdout)
    assert "error" in payload
    assert "morgan doctor --vectors" in payload["error"] or "doctor --vectors" in payload["error"]
    assert payload["suspect_ids"] == _expected_suspect_ids()


def test_a_stopped_import_surfaces_suspect_ids_on_stderr_in_text_mode(tmp_path):
    export = _export(tmp_path / "export.json")
    with drifting_model_server(embedding_dim=4, drift_after=_DRIFT_AFTER) as url:
        env = _env(tmp_path / "data", url)
        out = _run(["import", str(export)], env, tmp_path)

    assert out.returncode != 0, out.stderr
    assert "Traceback" not in out.stdout
    assert "Traceback" not in out.stderr
    assert out.stdout == "", "the CLI's --json contract is stdout; text-mode errors go to stderr"
    assert "error:" in out.stderr
    assert "suspect memory ids:" in out.stderr
    for suspect_id in _expected_suspect_ids():
        assert suspect_id in out.stderr
