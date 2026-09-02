"""The pattern register — learning at the level of classes, not instances.

Ouroboros, Principle 2 (Meta-over-Patch): *"When an error occurs -- any error -- the
response is not to fix the specific instance. The response is to ask: what must change so
this entire class of failure becomes structurally impossible?"* Its Pattern Register is
the durable projection of those classes, their counts, and the structural fix applied to
each, and its rule is: *"Before closing any bug, check the register: is this a known
pattern? If yes, escalate to architectural level immediately."*

Morgan's optimizer mines high-value signals -- edits beat retries beat thumbs -- and
hands them to the reflection model as *instances*: eleven separate corrections, each
looking like a one-off. The model proposes a patch for the most recent one, next week
sees the same eleven again, and proposes the same patch. Nothing accumulates. Ouroboros's
own test for this is sharp: *"If three behavioural rules exist for the same class and the
class still recurs, the problem is tooling or structure, not memory."*

So this register groups corrections into classes, counts them, and hands the *class* back
to the optimizer: not "the user edited this reply" but "this class of correction has
happened eleven times across four projects, and here is the fix that was supposed to
close it". A class seen once is noise; a class seen repeatedly after a fix was recorded
is evidence the fix was at the wrong depth, and the register can say so because it keeps
the count from before the fix and after it separately.

Cold path only. Nothing here is reachable from a request.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

#: Below this, a class is one thing that happened, not a pattern. The optimizer is told
#: about classes at or above it, so a single stray edit cannot reshape the champion.
DEFAULT_MIN_OCCURRENCES = 3


class PatternStatus(str, Enum):
    OPEN = "open"
    #: A structural fix was recorded against this class.
    ADDRESSED = "addressed"
    #: The class recurred *after* its fix was recorded -- the fix was at the wrong depth.
    REGRESSED = "regressed"


@dataclass(frozen=True)
class Pattern:
    class_id: str
    title: str
    description: str
    occurrences: int
    #: Occurrences recorded after a structural fix was set. Non-zero here is the signal
    #: that the fix did not close the class.
    occurrences_since_fix: int
    structural_fix: str
    status: PatternStatus
    first_seen: datetime
    last_seen: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS learned_patterns (
    user_id               TEXT    NOT NULL,
    project               TEXT    NOT NULL,
    class_id              TEXT    NOT NULL,
    title                 TEXT    NOT NULL,
    description           TEXT    NOT NULL DEFAULT '',
    occurrences           INTEGER NOT NULL DEFAULT 0,
    occurrences_since_fix INTEGER NOT NULL DEFAULT 0,
    structural_fix        TEXT    NOT NULL DEFAULT '',
    status                TEXT    NOT NULL DEFAULT 'open',
    first_seen            TEXT,
    last_seen             TEXT,
    PRIMARY KEY (user_id, project, class_id)
);

CREATE INDEX IF NOT EXISTS idx_patterns_recurring
    ON learned_patterns (user_id, project, occurrences DESC);
"""


def class_id_for(title: str) -> str:
    """Identity of a class, derived from its title.

    Content-derived rather than assigned, for the same reason as persona nodes: the cold
    path has no handle on a class it has not read, and reading first to decide whether to
    insert is the race that produces two registers for one pattern.
    """
    return hashlib.sha256(title.strip().casefold().encode("utf-8")).hexdigest()[:32]


class PatternRegister:
    """Error classes and their structural fixes, over the shared SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)
        conn.commit()

    def record(
        self,
        *,
        user_id: str,
        project: str,
        title: str,
        description: str = "",
        now: datetime,
    ) -> Pattern:
        """Record one occurrence of the class named by *title*."""
        cid = class_id_for(title)
        stamp = now.isoformat()
        row = self._conn.execute(
            "SELECT occurrences, occurrences_since_fix, structural_fix, status "
            "FROM learned_patterns WHERE user_id = ? AND project = ? AND class_id = ?",
            (user_id, project, cid),
        ).fetchone()

        if row is None:
            self._conn.execute(
                "INSERT INTO learned_patterns (user_id, project, class_id, title, description, "
                "occurrences, occurrences_since_fix, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, 1, 0, ?, ?)",
                (user_id, project, cid, title.strip(), description.strip(), stamp, stamp),
            )
        else:
            has_fix = bool(str(row["structural_fix"]).strip())
            # An occurrence after a fix was recorded is the interesting one: it says the
            # fix was at the wrong depth. Counting it in the same bucket as the ones that
            # motivated the fix would hide exactly that.
            since_fix = int(row["occurrences_since_fix"]) + (1 if has_fix else 0)
            status = PatternStatus.REGRESSED.value if has_fix else str(row["status"])
            self._conn.execute(
                "UPDATE learned_patterns SET occurrences = occurrences + 1, "
                "occurrences_since_fix = ?, status = ?, last_seen = ?, "
                "description = CASE WHEN ? != '' THEN ? ELSE description END "
                "WHERE user_id = ? AND project = ? AND class_id = ?",
                (
                    since_fix,
                    status,
                    stamp,
                    description.strip(),
                    description.strip(),
                    user_id,
                    project,
                    cid,
                ),
            )
        self._conn.commit()
        return self.get(user_id=user_id, project=project, class_id=cid)

    def set_structural_fix(self, *, user_id: str, project: str, class_id: str, fix: str) -> Pattern:
        """Record the structural change meant to close this class.

        Resets the post-fix counter, so the next occurrence starts a clean count of "did
        the fix hold" rather than inheriting the evidence that motivated it.
        """
        self._conn.execute(
            "UPDATE learned_patterns SET structural_fix = ?, status = ?, "
            "occurrences_since_fix = 0 WHERE user_id = ? AND project = ? AND class_id = ?",
            (fix.strip(), PatternStatus.ADDRESSED.value, user_id, project, class_id),
        )
        self._conn.commit()
        return self.get(user_id=user_id, project=project, class_id=class_id)

    def recurring(
        self,
        *,
        user_id: str,
        project: str | None,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        limit: int = 10,
    ) -> list[Pattern]:
        """Classes worth telling the optimizer about, most frequent first.

        ``project=None`` aggregates each class across every project the user has, which
        is the scope the optimizer actually needs: the champion preprompt is one document
        per user, not one per project, and a correction class that recurs in three
        different projects is *stronger* evidence about how the assistant should behave,
        not three weaker ones. Reading a single project here would have left the register
        invisible to the optimizer for every real project name, since corrections are
        recorded under the project of the turn that produced them.
        """
        if project is not None:
            rows = self._conn.execute(
                "SELECT * FROM learned_patterns WHERE user_id = ? AND project = ? "
                "AND occurrences >= ? ORDER BY occurrences DESC, class_id ASC LIMIT ?",
                (user_id, project, min_occurrences, limit),
            ).fetchall()
            return [_to_pattern(r) for r in rows]

        # Aggregated across projects. `status` is the worst case across the rows: a class
        # that regressed anywhere has regressed, and reporting it as merely "addressed"
        # because two other projects are quiet is the reassurance this register exists to
        # withhold.
        rows = self._conn.execute(
            """
            SELECT
                class_id,
                MIN(title)                        AS title,
                MAX(description)                  AS description,
                SUM(occurrences)                  AS occurrences,
                SUM(occurrences_since_fix)        AS occurrences_since_fix,
                MAX(structural_fix)               AS structural_fix,
                MAX(status = 'regressed')         AS any_regressed,
                MAX(structural_fix != '')         AS any_fixed,
                MIN(first_seen)                   AS first_seen,
                MAX(last_seen)                    AS last_seen
            FROM learned_patterns
            WHERE user_id = ?
            GROUP BY class_id
            HAVING SUM(occurrences) >= ?
            ORDER BY occurrences DESC, class_id ASC
            LIMIT ?
            """,
            (user_id, min_occurrences, limit),
        ).fetchall()
        out: list[Pattern] = []
        for r in rows:
            if r["any_regressed"]:
                status = PatternStatus.REGRESSED
            elif r["any_fixed"]:
                status = PatternStatus.ADDRESSED
            else:
                status = PatternStatus.OPEN
            out.append(
                Pattern(
                    class_id=str(r["class_id"]),
                    title=str(r["title"]),
                    description=str(r["description"]),
                    occurrences=int(r["occurrences"]),
                    occurrences_since_fix=int(r["occurrences_since_fix"]),
                    structural_fix=str(r["structural_fix"]),
                    status=status,
                    first_seen=datetime.fromisoformat(str(r["first_seen"])),
                    last_seen=datetime.fromisoformat(str(r["last_seen"])),
                )
            )
        return out

    def get(self, *, user_id: str, project: str, class_id: str) -> Pattern:
        row = self._conn.execute(
            "SELECT * FROM learned_patterns WHERE user_id = ? AND project = ? AND class_id = ?",
            (user_id, project, class_id),
        ).fetchone()
        return _to_pattern(row)

    def all_patterns(self, *, user_id: str, project: str) -> list[Pattern]:
        rows = self._conn.execute(
            "SELECT * FROM learned_patterns WHERE user_id = ? AND project = ? "
            "ORDER BY occurrences DESC, class_id ASC",
            (user_id, project),
        ).fetchall()
        return [_to_pattern(r) for r in rows]


def render_for_optimizer(patterns: list[Pattern]) -> str:
    """Render recurring classes as optimizer context.

    Deliberately says how *many* times and, where one exists, what the previous fix was.
    A model given eleven separate corrections proposes a twelfth patch; a model told the
    class has recurred eleven times and that the last fix did not hold has the
    information needed to propose something structural instead.
    """
    if not patterns:
        return ""
    lines = ["Recurring correction classes (fix the class, not the instance):"]
    for p in patterns:
        line = f"- {p.title} — seen {p.occurrences}×"
        if p.description:
            line += f": {p.description}"
        if p.structural_fix:
            line += f" | previous fix: {p.structural_fix}"
            if p.occurrences_since_fix:
                line += (
                    f" | RECURRED {p.occurrences_since_fix}× since that fix — "
                    "it was at the wrong depth"
                )
        lines.append(line)
    return "\n".join(lines)


def _to_pattern(row: sqlite3.Row) -> Pattern:
    return Pattern(
        class_id=str(row["class_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        occurrences=int(row["occurrences"]),
        occurrences_since_fix=int(row["occurrences_since_fix"]),
        structural_fix=str(row["structural_fix"]),
        status=PatternStatus(str(row["status"])),
        first_seen=datetime.fromisoformat(str(row["first_seen"])),
        last_seen=datetime.fromisoformat(str(row["last_seen"])),
    )
