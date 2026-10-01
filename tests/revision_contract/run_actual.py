"""Materialize synthetic production observations; refuses to overwrite prior evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from tests.revision_contract.production_adapter import observe, operation_only
from tests.revision_contract.score_gold import load_fixture


async def collect(fixture: dict) -> dict:
    output = {"version": fixture["version"], "observations": [], "writes": []}
    for scenario in fixture["scenarios"]:
        result = await observe(operation_only(scenario))
        output["observations"].extend(result["observations"])
        output["writes"].extend(result["writes"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture_path = Path(__file__).with_name("revision_acceptance_v1.json")
    fixture = load_fixture(fixture_path)
    output = asyncio.run(collect(fixture))
    output["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    output["limitations"] = [
        "Authored contract only; no retrieval-quality or model-performance claim",
        "executable_instruction_ids means instruction-like recall promoted into native system "
        "prompt; no execution or model-obedience claim",
        "Legacy events and recorded historical fact snapshots are seeded; "
        "candidate writes exercise public admission",
    ]
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(output, target, ensure_ascii=False, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
