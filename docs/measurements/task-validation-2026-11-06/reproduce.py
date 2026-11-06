"""Frozen synthetic selection plus a removable-line validator; no owner-data inference."""

import argparse
import asyncio
import hashlib
import json
import resource
import time
from contextlib import closing
from pathlib import Path

import httpx

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle, managed_model
from morgan_brain.providers import note_probe

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser(description="Frozen local task validation experiment")
parser.add_argument("--name", choices=("granite", "ornith"), required=True)
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
NAME = args.name
DIGEST = args.model.name.removeprefix("sha256-")
SHARED = args.workspace.resolve()
CONFIG = ModelCycle(args.server, args.model, seconds=120)
OUTPUT = args.output.resolve()
OUTPUT.mkdir(mode=0o700)


async def validate(session, note, selected):
    proposed = [i for i, line in enumerate(note.splitlines(), 1) if line in selected]
    if not proposed:
        return (), 0.0
    started = time.monotonic()
    remaining = session.deadline - asyncio.get_running_loop().time()
    seconds = min(20, remaining)
    session.workspace.check_stop()
    async with (
        asyncio.timeout(seconds),
        httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(session.socket)),
            base_url="http://127.0.0.1",
            trust_env=False,
            timeout=seconds,
        ) as client,
    ):
        result = await note_probe._json_request(
            client,
            "/v1/chat/completions",
            json={
                "model": "morgan-cycle",
                "stream": False,
                "temperature": 0,
                "max_tokens": 128,
                "messages": [
                    {"role": "system", "content": (ROOT / "validator.txt").read_text()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "note": dict(enumerate(note.splitlines(), 1)),
                                "proposed_lines": proposed,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "selection",
                        "strict": True,
                        "schema": note_probe.Selection.model_json_schema(),
                    },
                },
            },
        )
    result = note_probe._chat_selection(note, result, "morgan-cycle")
    if not set(result) <= set(selected):
        raise ValueError("validator added a source line")
    session.workspace.check_stop()
    return result, round(time.monotonic() - started, 3)


def row(case, arm, selected, seconds, error=None):
    expected = {case["note"].splitlines()[i - 1] for i in case["expected"]}
    actual = set(selected)
    return {
        "case": case["name"],
        "language": case["language"],
        "kind": case["kind"],
        "arm": arm,
        "valid": error is None,
        "error": error,
        "selected": sorted(actual),
        "exact": error is None and expected == actual,
        "tp": len(actual & expected),
        "fp": len(actual - expected),
        "fn": len(expected - actual),
        "seconds": round(seconds, 3),
    }


async def evaluate_case(session, case, rows):
    """Record baseline and optional validator, returning only baseline failure count."""
    failed = 0
    start = time.monotonic()
    try:
        selected = await session.select(case["note"])
        elapsed = time.monotonic() - start
        rows.append(row(case, NAME, selected, elapsed))
        if NAME == "granite":
            try:
                checked, extra = await validate(session, case["note"], selected)
                rows.append(row(case, "granite_validated", checked, elapsed + extra))
            except (
                note_probe.ProbeFailure,
                ValueError,
                TimeoutError,
                httpx.HTTPError,
            ) as error:
                rows.append(
                    row(
                        case,
                        "granite_validated",
                        (),
                        time.monotonic() - start,
                        type(error).__name__,
                    )
                )
    except (note_probe.ProbeFailure, ValueError, TimeoutError) as error:
        failed = 1
        rows.append(row(case, NAME, (), time.monotonic() - start, type(error).__name__))
        if NAME == "granite":
            rows.append(
                row(
                    case,
                    "granite_validated",
                    (),
                    time.monotonic() - start,
                    "baseline_failed",
                )
            )
    return failed


async def main():
    raw = (ROOT / "cases.json").read_bytes()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if hashlib.sha256(raw).hexdigest() != protocol["corpus_sha256"]:
        raise ValueError("corpus changed")
    if (
        hashlib.sha256((ROOT / "validator.txt").read_bytes()).hexdigest()
        != protocol["validator_sha256"]
    ):
        raise ValueError("validator changed")
    cases = json.loads(raw)
    rows = []
    cycles = []
    failures = 0
    started = time.monotonic()
    status = "complete"
    try:
        for batch in range(0, len(cases), 4):
            if json.loads((SHARED / "owner-input.json").read_text())["messages"]:
                status = "deferred_for_owner"
                break
            cycle_start = time.monotonic()
            with closing(NoteWorkspace(SHARED, create=False)) as workspace:
                async with managed_model(workspace, CONFIG) as session:
                    load = time.monotonic() - cycle_start
                    for case in cases[batch : batch + 4]:
                        failures += await evaluate_case(session, case, rows)
                        (OUTPUT / (NAME + "-results.json")).write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                        )
                        print(
                            json.dumps({k: v for k, v in rows[-1].items() if k != "selected"}),
                            flush=True,
                        )
                        if failures >= 2:
                            status = "operational_failures"
                            break
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            cycles.append(
                {
                    "batch": batch,
                    "load_seconds": round(load, 3),
                    "wall_seconds": round(time.monotonic() - cycle_start, 3),
                    "child_cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
                    "child_maxrss_bytes": usage.ru_maxrss,
                }
            )
            if failures >= 2:
                break
    except (StopRequested, note_probe.ProbeFailure, OSError) as error:
        status = type(error).__name__
    summary = {
        "model": NAME,
        "digest": DIGEST,
        "status": status,
        "wall_seconds": round(time.monotonic() - started, 3),
        "cycles": cycles,
        "rows": len(rows),
    }
    (OUTPUT / (NAME + "-run.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


asyncio.run(main())
