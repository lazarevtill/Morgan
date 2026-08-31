"""Decision receipts — every promotion decision stays reconstructible.

Ouroboros, Principle 1: *"No optimization, compression, or caching strategy may destroy
the ability to recover the exact prompt/context, tool schema, model route, and model
output that shaped a decision. A mind that remembers conclusions but cannot replay how
they were formed remembers only a shadow of itself."*

Morgan's champion promotions were a log line. The champion preprompt is state that
shapes every subsequent turn, it is chosen automatically, and once promoted it becomes
the baseline everything after is measured against -- so "why is the champion this?" is a
question the owner will eventually ask about a decision made weeks earlier by a model
that is no longer running. A log line that scrolled away cannot answer it, and a rollback
chosen without that answer is a guess.

A receipt records the comparison, not the prose: which champion, at what score, against
what candidate, on which gate fingerprint, judged by which model, and -- when the answer
was "no" -- why. Rejections are recorded too, and they are the more useful half: a
candidate refused for gate integrity looks identical to one refused for scoring badly if
only promotions are kept.

The candidate body is stored by hash rather than in full. The bodies themselves already
live in the ``PromptRegistry`` with their versions; duplicating them here would make the
receipt table the second copy of a thing that must have exactly one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_receipts (
    -- A monotonic sequence, not just a timestamp. Two decisions in the same nightly run
    -- share a timestamp to the second, and ordering by a random uuid to break that tie
    -- means the history reorders itself between reads -- which is not a record.
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    id                TEXT NOT NULL UNIQUE,
    created_at        TEXT NOT NULL,
    prompt_name       TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT '',
    champion_version  INTEGER,
    champion_score    REAL,
    candidate_hash    TEXT NOT NULL,
    candidate_score   REAL,
    gate_fingerprint  TEXT NOT NULL DEFAULT '',
    -- The full gate description, not just its fingerprint. A fingerprint can only answer
    -- "did it change"; deciding whether a change *weakened* the gate needs the item count
    -- back, and the receipt is the only place that survives to be asked.
    gate_spec         TEXT NOT NULL DEFAULT '{}',
    judge_model       TEXT NOT NULL DEFAULT '',
    metrics           TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_receipts_prompt
    ON decision_receipts (prompt_name, seq DESC);
"""


def body_hash(body: str) -> str:
    """Identify a candidate body without storing a second copy of it."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Receipt:
    id: str
    created_at: datetime
    prompt_name: str
    verdict: str
    reason: str
    champion_version: int | None
    champion_score: float | None
    candidate_hash: str
    candidate_score: float | None
    gate_fingerprint: str
    judge_model: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_spec: dict[str, object] = field(default_factory=dict)


class ReceiptStore:
    """Decision receipts over the one shared SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)
        conn.commit()

    def record(
        self,
        *,
        prompt_name: str,
        verdict: str,
        candidate_body: str,
        now: datetime,
        reason: str = "",
        champion_version: int | None = None,
        champion_score: float | None = None,
        candidate_score: float | None = None,
        gate_fingerprint: str = "",
        gate_spec: dict[str, object] | None = None,
        judge_model: str = "",
        metrics: dict[str, float] | None = None,
    ) -> Receipt:
        """Record one promotion decision. *verdict* is ``promoted`` or ``rejected``."""
        receipt_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO decision_receipts (id, created_at, prompt_name, verdict, reason, "
            "champion_version, champion_score, candidate_hash, candidate_score, "
            "gate_fingerprint, gate_spec, judge_model, metrics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                receipt_id,
                now.isoformat(),
                prompt_name,
                verdict,
                reason,
                champion_version,
                champion_score,
                body_hash(candidate_body),
                candidate_score,
                gate_fingerprint,
                json.dumps(gate_spec or {}),
                judge_model,
                json.dumps(metrics or {}),
            ),
        )
        self._conn.commit()
        return self.get(receipt_id)

    def get(self, receipt_id: str) -> Receipt:
        row = self._conn.execute(
            "SELECT * FROM decision_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        return _to_receipt(row)

    def recent(self, *, prompt_name: str | None = None, limit: int = 20) -> list[Receipt]:
        """Most recent decisions first. Rejections included -- a history of only the
        promotions cannot explain the promotions that did not happen."""
        if prompt_name is None:
            rows = self._conn.execute(
                "SELECT * FROM decision_receipts ORDER BY seq DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM decision_receipts WHERE prompt_name = ? ORDER BY seq DESC LIMIT ?",
                (prompt_name, limit),
            ).fetchall()
        return [_to_receipt(r) for r in rows]

    def last_promotion(self, prompt_name: str) -> Receipt | None:
        """The receipt of the standing champion's promotion, or ``None``.

        This is what makes gate integrity checkable at all: it is the only record of the
        gate the current champion was certified on, so it is the thing the next
        candidate's gate has to match.
        """
        row = self._conn.execute(
            "SELECT * FROM decision_receipts WHERE prompt_name = ? AND verdict = 'promoted' "
            "ORDER BY seq DESC LIMIT 1",
            (prompt_name,),
        ).fetchone()
        return _to_receipt(row) if row is not None else None


def _to_receipt(row: sqlite3.Row) -> Receipt:
    return Receipt(
        id=str(row["id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        prompt_name=str(row["prompt_name"]),
        verdict=str(row["verdict"]),
        reason=str(row["reason"]),
        champion_version=(
            int(row["champion_version"]) if row["champion_version"] is not None else None
        ),
        champion_score=(
            float(row["champion_score"]) if row["champion_score"] is not None else None
        ),
        candidate_hash=str(row["candidate_hash"]),
        candidate_score=(
            float(row["candidate_score"]) if row["candidate_score"] is not None else None
        ),
        gate_fingerprint=str(row["gate_fingerprint"]),
        judge_model=str(row["judge_model"]),
        metrics=dict(json.loads(str(row["metrics"] or "{}"))),
        gate_spec=dict(json.loads(str(row["gate_spec"] or "{}"))),
    )
