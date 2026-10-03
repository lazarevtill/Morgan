"""Run against a new synthetic directory; no chat model or network is required.

PYTHONPATH=. python examples/context_inspect_zero_model.py /tmp/morgan-synthetic-demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from morgan_brain.composition import build_evidence_context, build_memory_context
from morgan_brain.config import Settings
from morgan_brain.models import Memory


async def seed_sources(
    settings: Settings, owner: str, project: str
) -> tuple[dict[str, str], dict[str, str]]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    inputs = [
        ("reported", "Reported preference: morning walks.", []),
        ("plan", "Reported plan: a walk on Saturday.", []),
        ("correction", "Reported correction: a walk on Sunday.", ["plan"]),
        ("fork", "Reported destination: the park.", []),
        ("fork-a", "Reported alternative: the lake.", ["fork"]),
        ("fork-b", "Reported alternative: the forest.", ["fork"]),
    ]
    ids = {}
    texts = {}
    writer = build_memory_context(settings)
    try:
        for index, (label, content, parents) in enumerate(inputs):
            ids[label] = await writer.gate.store(
                Memory(
                    user_id=owner,
                    project=project,
                    content=content,
                    source="user_stated",
                    author_id=owner,
                    scope="private",
                    effective_at=base + timedelta(days=index),
                    revises_event_ids=[ids[parent] for parent in parents],
                )
            )
            texts[label] = content
    finally:
        writer.conn.close()
    return ids, texts


async def demonstrate(directory: Path) -> dict:
    settings = Settings(
        _env_file=None,
        data_dir=str(directory),
        temporal_db_url=f"sqlite:///{directory}/morgan.db",
        snapshot_dir=str(directory / "snapshots"),
        owner_user_id="synthetic-demo-owner",
        embedding_backend="hash",
        embedding_dim=1024,
    )
    owner, project = settings.owner_user_id, "synthetic-demo"
    ids, texts = await seed_sources(settings, owner, project)
    # The caller owns this index. Morgan does not discover which IDs a task needs.
    (directory / "source-ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    selections = []
    for label, section in [
        ("reported", "current_facts"),
        ("correction", "completed_progress"),
        ("fork-a", "relevant_constraints"),
    ]:
        quote = texts[label]
        selections.append(
            {
                "section": section,
                "event_id": ids[label],
                "start": 0,
                "end": len(quote),
                "quote": quote,
            }
        )
    selections.append(
        {
            "section": "unresolved_questions",
            "event_id": "synthetic-missing-source",
            "start": 0,
            "end": 7,
            "quote": "missing",
        }
    )
    request = {
        "user_id": owner,
        "project": project,
        "evidence_ids": [*ids.values(), "synthetic-missing-source"],
        "selections": selections,
        "effective_at": datetime(2026, 10, 1, tzinfo=UTC),
    }
    # Reopen through the inspection-only composition: no writable migration or provider.
    reader = build_evidence_context(settings)
    try:
        envelope = await reader.gate.inspect_context(**request)
    finally:
        reader.conn.close()
    (directory / "selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    (directory / "context.json").write_text(json.dumps(envelope, indent=2) + "\n")
    return envelope


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_directory", type=Path)
    args = parser.parse_args()
    directory = args.new_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    print(json.dumps(asyncio.run(demonstrate(directory)), indent=2))
