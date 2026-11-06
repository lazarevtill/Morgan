"""Read one explicitly piped note, call a selected local model and save an unapproved proposal."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.providers.note_probe import ProbeFailure, propose_chat


def _check_size(raw: bytes) -> None:
    """Reject oversized stdin before decoding or making a request."""
    if len(raw) > 4096:
        raise ValueError("note exceeds input limit")


def main() -> None:
    """Keep source text off command-line arguments and stdout; never open memory or execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(4097)
        _check_size(raw)
        note = raw.decode("utf-8")
        selected = asyncio.run(propose_chat(args.endpoint, args.model, note, seconds=args.seconds))
        payload = json.dumps(
            {
                "version": "morgan.note-proposal.v1",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source": "agent_inferred",
                "approved": False,
                "review_required": True,
                "selected_lines": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        workspace = NoteWorkspace(args.output.parent)
        try:
            workspace.publish(args.output.name, payload)
        finally:
            workspace.close()
    except (ValueError, OSError, ProbeFailure) as error:
        reason = str(error) if isinstance(error, ProbeFailure) else "invalid_input_or_output"
        print(json.dumps({"written": False, "reason": reason, "approved": False}))
        raise SystemExit(1) from None
    print(json.dumps({"written": True, "proposals": len(selected), "approved": False}))


if __name__ == "__main__":
    main()
