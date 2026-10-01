"""Score declared fixture outputs; this deliberately contains no revision algorithm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def load_fixture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    digest_path = path.with_suffix(".sha256")
    expected = digest_path.read_text(encoding="ascii").split()[0]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(
            "Frozen fixture hash mismatch; create a new version instead of rewriting it"
        )
    return json.loads(raw)


def declared_gold(fixture: dict[str, Any]) -> dict[str, Any]:
    observations = []
    writes = []
    for scenario in fixture["scenarios"]:
        for observation in scenario["observations"]:
            observations.append(
                {
                    "scenario_id": scenario["id"],
                    "observation_id": observation["id"],
                    **observation["expected"],
                }
            )
        for write in scenario["write_observations"]:
            writes.append(
                {
                    "scenario_id": scenario["id"],
                    "write_id": write["id"],
                    **write["expected"],
                }
            )
    return {
        "version": fixture["version"],
        "observations": observations,
        "writes": writes,
    }


def equivalent(expected: Any, actual: Any, field: str) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            equivalent(value, actual[key], key) for key, value in expected.items()
        )
    if isinstance(expected, list) and field.endswith("_ids"):
        return all(isinstance(item, str) for item in actual) and sorted(expected) == sorted(actual)
    return expected == actual


def score(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    failures = []
    total = 0
    passed = 0
    if actual.get("version") != expected["version"]:
        failures.append(
            {
                "field": "version",
                "expected": expected["version"],
                "actual": actual.get("version"),
            }
        )
    for section, identity_field in (
        ("observations", "observation_id"),
        ("writes", "write_id"),
    ):
        rows = actual.get(section, [])
        if not isinstance(rows, list):
            failures.append({"section": section, "error": "expected a list"})
            rows = []
        index = {}
        for row in rows:
            if not isinstance(row, dict):
                failures.append({"section": section, "error": "expected object rows"})
                continue
            key = (row.get("scenario_id"), row.get(identity_field))
            if not all(isinstance(value, str) for value in key):
                failures.append({"section": section, "error": "row identifiers must be strings"})
                continue
            if key in index:
                failures.append({"section": section, "key": key, "error": "duplicate output"})
            index[key] = row
        expected_keys = set()
        for row in expected[section]:
            total += 1
            key = (row["scenario_id"], row[identity_field])
            expected_keys.add(key)
            observed = index.get(key, {})
            errors = []
            for field, value in row.items():
                if field in ("scenario_id", identity_field):
                    continue
                if not equivalent(value, observed.get(field), field):
                    errors.append(
                        {
                            "field": field,
                            "expected": value,
                            "actual": observed.get(field),
                        }
                    )
            if errors:
                failures.append({"section": section, "key": key, "errors": errors})
            else:
                passed += 1
        for extra in index.keys() - expected_keys:
            failures.append({"section": section, "key": extra, "error": "undeclared output"})
    return {
        "version": expected["version"],
        "passed": passed,
        "total": total,
        "all_passed": not failures and passed == total,
        "failures": failures,
        "scope": "Authored contract acceptance only; no retrieval/model performance claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "revision_acceptance_v1.json")
    parser.add_argument("--actual", type=Path)
    parser.add_argument(
        "--emit-gold",
        action="store_true",
        help="Print the declared output format, not product results",
    )
    args = parser.parse_args()
    gold = declared_gold(load_fixture(args.fixture))
    if args.emit_gold:
        print(json.dumps(gold, ensure_ascii=False, indent=2))
        return 0
    if args.actual is None:
        parser.error("Provide --actual, or use --emit-gold to inspect the adapter format")
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    result = score(gold, actual)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
