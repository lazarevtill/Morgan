"""One preregistered prompt comparison on synthetic notes, no task or memory writes."""

import argparse
import ast
import asyncio
import hashlib
import inspect
import json
import time
from contextlib import closing
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.model_cycle import ModelCycle, managed_model
from morgan_brain.providers import note_probe

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
CONFIG = ModelCycle(args.server, args.model, seconds=120)
OUTPUT = args.output.resolve()
OUTPUT.mkdir(mode=0o700)


async def main():
    """Compare exactly the frozen baseline and candidate in finite isolated cycles."""
    baseline = (ROOT / "baseline.txt").read_text()
    constants = [
        node.value
        for node in ast.walk(ast.parse(inspect.getsource(note_probe.propose_chat)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    if baseline not in constants:
        raise ValueError("installed baseline differs from the recorded experiment")
    raw = (ROOT / "cases.json").read_bytes()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if hashlib.sha256(raw).hexdigest() != protocol["corpus_sha256"]:
        raise ValueError("corpus differs from the frozen protocol")
    candidate = (ROOT / "candidate.txt").read_text()
    if hashlib.sha256(candidate.encode()).hexdigest() != protocol["candidate_sha256"]:
        raise ValueError("candidate differs from the frozen protocol")
    cases = json.loads(raw)
    rows = []
    original = note_probe._json_request

    async def altered(client, path, **kwargs):
        if path == "/v1/chat/completions":
            kwargs["json"]["messages"][0]["content"] = candidate
        return await original(client, path, **kwargs)

    for batch in range(0, len(cases), 4):
        with closing(NoteWorkspace(OUTPUT / f"cycle-{batch}")) as workspace:
            async with managed_model(workspace, CONFIG) as session:
                for offset, case in enumerate(cases[batch : batch + 4]):
                    expected = {case["note"].splitlines()[i - 1] for i in case["expected"]}
                    for arm in (
                        ("baseline", "candidate") if offset % 2 == 0 else ("candidate", "baseline")
                    ):
                        note_probe._json_request = altered if arm == "candidate" else original
                        start = time.monotonic()
                        try:
                            actual = set(await session.select(case["note"]))
                            row = {
                                "case": case["name"],
                                "arm": arm,
                                "language": case["language"],
                                "kind": case["kind"],
                                "split": case["split"],
                                "valid": True,
                                "selected": sorted(actual),
                                "exact": actual == expected,
                                "tp": len(actual & expected),
                                "fp": len(actual - expected),
                                "fn": len(expected - actual),
                            }
                        except (note_probe.ProbeFailure, ValueError, TimeoutError) as error:
                            row = {
                                "case": case["name"],
                                "arm": arm,
                                "language": case["language"],
                                "kind": case["kind"],
                                "split": case["split"],
                                "valid": False,
                                "error": type(error).__name__,
                                "exact": False,
                                "tp": 0,
                                "fp": 0,
                                "fn": len(expected),
                            }
                        finally:
                            note_probe._json_request = original
                        row["seconds"] = round(time.monotonic() - start, 3)
                        rows.append(row)
                        (OUTPUT / "results.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                        )
                        print(
                            json.dumps({k: v for k, v in row.items() if k != "selected"}),
                            flush=True,
                        )
    summaries = []
    for split in ("development", "holdout"):
        for group in ("all", "en", "ru", "owner", "renew", "control"):
            for arm in ("baseline", "candidate"):
                subset = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["arm"] == arm
                    and (group == "all" or r["language"] == group or r["kind"] == group)
                ]
                summaries.append(
                    dict(
                        split=split,
                        group=group,
                        arm=arm,
                        n=len(subset),
                        **{
                            k: sum(r[k] for r in subset)
                            for k in ("valid", "exact", "tp", "fp", "fn")
                        },
                    )
                )
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries), flush=True)


asyncio.run(main())
