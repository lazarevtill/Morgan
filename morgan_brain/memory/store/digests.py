"""``digests``, ``digest_refs``, ``digest_ratings`` and ``link_ratings``.

A digest row is the exact text a session start printed, with the line refs that number its
content lines and one ``digest_refs`` row per turn, memory or fact it quoted -- the index by
which an erasure of any of them can find every digest that rendered it. ``entrypoint_source``
says where the row's ``entrypoint`` came from (``env``, ``transcript``, ``backfill`` or
``none``); every writer stores it explicitly, and the column's ``DEFAULT ''`` marks only a row
whose writer did not set it, which no writer in this codebase leaves unset.

Every statement runs with ``execute``, never ``executescript``, so a schema function can run
inside a caller's own write transaction without an implicit commit ending it partway through.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast, get_args

RefKind = Literal["turn", "memory", "fact"]
EntrypointSource = Literal["env", "transcript", "backfill", "none"]

_ENTRYPOINT_SOURCES: frozenset[str] = frozenset(get_args(EntrypointSource))


def _checked_entrypoint_source(value: str) -> EntrypointSource:
    """Validate *value* against ``EntrypointSource``'s four members, raising ``ValueError``
    naming it otherwise."""
    if value not in _ENTRYPOINT_SOURCES:
        raise ValueError(f"unknown entrypoint_source: {value!r}")
    return cast(EntrypointSource, value)


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS digests (
        id                TEXT PRIMARY KEY,
        ts                TEXT NOT NULL,
        user_id           TEXT NOT NULL,
        project           TEXT NOT NULL,
        harness           TEXT NOT NULL,
        native_session_id TEXT NOT NULL,
        source            TEXT NOT NULL DEFAULT '',
        entrypoint        TEXT NOT NULL DEFAULT '',
        entrypoint_source TEXT NOT NULL DEFAULT '',
        first_for_session INTEGER NOT NULL DEFAULT 0,
        text              TEXT NOT NULL,
        lines             TEXT NOT NULL,
        chars             INTEGER NOT NULL,
        ms                INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_digests_session ON digests (harness, native_session_id)",
    """
    CREATE TABLE IF NOT EXISTS digest_refs (
        digest_id TEXT NOT NULL REFERENCES digests(id) ON DELETE CASCADE,
        kind      TEXT NOT NULL CHECK (kind IN ('turn', 'memory', 'fact')),
        ref_id    TEXT NOT NULL,
        PRIMARY KEY (digest_id, kind, ref_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_digest_refs_ref ON digest_refs (kind, ref_id)",
    """
    CREATE TABLE IF NOT EXISTS digest_ratings (
        digest_id TEXT NOT NULL REFERENCES digests(id) ON DELETE CASCADE,
        line_no   INTEGER NOT NULL,
        rating    TEXT NOT NULL CHECK (rating IN ('right', 'wrong', 'useless')),
        rated_at  TEXT NOT NULL,
        PRIMARY KEY (digest_id, line_no)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS link_ratings (
        turn_id         INTEGER NOT NULL,
        earlier_turn_id INTEGER NOT NULL,
        user_id         TEXT NOT NULL,
        project         TEXT NOT NULL,
        rating          TEXT NOT NULL CHECK (rating IN ('right', 'wrong')),
        rated_at        TEXT NOT NULL,
        PRIMARY KEY (turn_id, earlier_turn_id)
    )
    """,
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the four tables and their indexes, joining *conn*'s current transaction."""
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


class DigestStore:
    """Creates the digest tables; every query over them is a module-level function, like
    ``ProjectStore``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()


@dataclass(frozen=True)
class DigestRef:
    """One id a digest quoted."""

    kind: RefKind
    ref_id: str


@dataclass(frozen=True)
class DigestLineRef:
    """One printed line: ``no`` is its 1-based number among the content lines (``fact``,
    ``memory``, ``corrected``) and ``None`` for a ``project``, ``session``, ``pending`` or
    ``cut`` line; ``ref`` is the id the line quotes, the same id its ``DigestRef`` carries."""

    no: int | None
    kind: str
    ref: str | None


@dataclass(frozen=True)
class DigestRow:
    id: str
    ts: str
    user_id: str
    project: str
    harness: str
    native_session_id: str
    source: str
    entrypoint: str
    entrypoint_source: EntrypointSource
    first_for_session: bool
    text: str
    lines: tuple[DigestLineRef, ...]
    refs: tuple[DigestRef, ...]
    chars: int
    ms: int


@dataclass(frozen=True)
class DigestRating:
    digest_id: str
    line_no: int
    rating: str
    rated_at: str


@dataclass(frozen=True)
class LinkRating:
    turn_id: int
    earlier_turn_id: int
    user_id: str
    project: str
    rating: str
    rated_at: str


def _lines_json(lines: tuple[DigestLineRef, ...]) -> str:
    return json.dumps([{"no": line.no, "kind": line.kind, "ref": line.ref} for line in lines])


def _row_to_digest(conn: sqlite3.Connection, row: sqlite3.Row) -> DigestRow:
    refs = tuple(
        DigestRef(kind=r["kind"], ref_id=r["ref_id"])
        for r in conn.execute(
            "SELECT kind, ref_id FROM digest_refs WHERE digest_id = ? ORDER BY kind, ref_id",
            (row["id"],),
        )
    )
    lines = tuple(
        DigestLineRef(no=line["no"], kind=line["kind"], ref=line["ref"])
        for line in json.loads(row["lines"])
    )
    return DigestRow(
        id=row["id"],
        ts=row["ts"],
        user_id=row["user_id"],
        project=row["project"],
        harness=row["harness"],
        native_session_id=row["native_session_id"],
        source=row["source"],
        entrypoint=row["entrypoint"],
        entrypoint_source=row["entrypoint_source"],
        first_for_session=bool(row["first_for_session"]),
        text=row["text"],
        lines=lines,
        refs=refs,
        chars=row["chars"],
        ms=row["ms"],
    )


def insert_digest(conn: sqlite3.Connection, row: DigestRow) -> None:
    """The row, and one ``digest_refs`` row per ref. Joins the caller's transaction.

    Raises ``ValueError`` before writing anything if ``row.entrypoint_source`` is not one of
    ``EntrypointSource``'s four values, and ``sqlite3.IntegrityError`` if a ref's kind or id
    fails ``digest_refs``' constraints -- the caller's ``write_transaction`` then rolls back
    the row already written, so a digest is never committed short of a ref an erasure needs to
    find it.
    """
    entrypoint_source = _checked_entrypoint_source(row.entrypoint_source)
    conn.execute(
        "INSERT INTO digests (id, ts, user_id, project, harness, native_session_id, source, "
        "entrypoint, entrypoint_source, first_for_session, text, lines, chars, ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row.id,
            row.ts,
            row.user_id,
            row.project,
            row.harness,
            row.native_session_id,
            row.source,
            row.entrypoint,
            entrypoint_source,
            int(row.first_for_session),
            row.text,
            _lines_json(row.lines),
            row.chars,
            row.ms,
        ),
    )
    conn.executemany(
        "INSERT INTO digest_refs (digest_id, kind, ref_id) VALUES (?, ?, ?) "
        "ON CONFLICT (digest_id, kind, ref_id) DO NOTHING",
        [(row.id, ref.kind, ref.ref_id) for ref in row.refs],
    )


def has_first_for_session(
    conn: sqlite3.Connection, *, harness: str, native_session_id: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM digests WHERE harness = ? AND native_session_id = ? "
        "AND first_for_session = 1",
        (harness, native_session_id),
    ).fetchone()
    return row is not None


def get_digest(conn: sqlite3.Connection, *, user_id: str, digest_id: str) -> DigestRow | None:
    row = conn.execute(
        "SELECT * FROM digests WHERE id = ? AND user_id = ?", (digest_id, user_id)
    ).fetchone()
    return None if row is None else _row_to_digest(conn, row)


def last_digest(conn: sqlite3.Connection, *, user_id: str, project: str) -> DigestRow | None:
    row = conn.execute(
        "SELECT * FROM digests WHERE user_id = ? AND project = ? ORDER BY ts DESC, id DESC LIMIT 1",
        (user_id, project),
    ).fetchone()
    return None if row is None else _row_to_digest(conn, row)


def _rated_lines(conn: sqlite3.Connection, digest_id: str) -> set[int]:
    return {
        int(r["line_no"])
        for r in conn.execute(
            "SELECT line_no FROM digest_ratings WHERE digest_id = ?", (digest_id,)
        )
    }


def newest_unrated_first(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    project: str | None,
    excluded_entrypoints: Callable[[str], bool],
) -> DigestRow | None:
    """The newest ``first_for_session`` digest with an unrated content line whose entrypoint
    the predicate does not exclude; ``None`` when every offered line is rated. Only a
    session's first render is offered; a later render is stored and never offered."""
    sql = "SELECT * FROM digests WHERE user_id = ? AND first_for_session = 1"
    params: list[object] = [user_id]
    if project is not None:
        sql += " AND project = ?"
        params.append(project)
    sql += " ORDER BY ts DESC, id DESC"
    for row in conn.execute(sql, params):
        if excluded_entrypoints(str(row["entrypoint"])):
            continue
        digest = _row_to_digest(conn, row)
        rated = _rated_lines(conn, digest.id)
        if any(line.no is not None and line.no not in rated for line in digest.lines):
            return digest
    return None


def digests_between(
    conn: sqlite3.Connection, *, user_id: str, since: str, until: str
) -> list[DigestRow]:
    """*user_id*'s digests with ``since <= ts < until``, oldest first."""
    rows = conn.execute(
        "SELECT * FROM digests WHERE user_id = ? AND ts >= ? AND ts < ? ORDER BY ts, id",
        (user_id, since, until),
    ).fetchall()
    return [_row_to_digest(conn, r) for r in rows]


def ratings_of(conn: sqlite3.Connection, digest_id: str) -> list[DigestRating]:
    return [
        DigestRating(
            digest_id=r["digest_id"],
            line_no=r["line_no"],
            rating=r["rating"],
            rated_at=r["rated_at"],
        )
        for r in conn.execute(
            "SELECT * FROM digest_ratings WHERE digest_id = ? ORDER BY line_no", (digest_id,)
        )
    ]


def rate_line(
    conn: sqlite3.Connection, *, digest_id: str, line_no: int, rating: str, now: str
) -> None:
    """Upsert one line's rating; the table's CHECK refuses a rating outside ``right``,
    ``wrong``, ``useless``."""
    conn.execute(
        "INSERT INTO digest_ratings (digest_id, line_no, rating, rated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (digest_id, line_no) DO UPDATE SET rating = excluded.rating, "
        "rated_at = excluded.rated_at",
        (digest_id, line_no, rating, now),
    )


def rate_link(
    conn: sqlite3.Connection,
    *,
    turn_id: int,
    earlier_turn_id: int,
    user_id: str,
    project: str,
    rating: str,
    now: str,
) -> None:
    """Upsert one link's rating (``right`` or ``wrong``, by the table's CHECK)."""
    conn.execute(
        "INSERT INTO link_ratings (turn_id, earlier_turn_id, user_id, project, rating, rated_at) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (turn_id, earlier_turn_id) DO UPDATE SET "
        "rating = excluded.rating, rated_at = excluded.rated_at",
        (turn_id, earlier_turn_id, user_id, project, rating, now),
    )


def link_ratings_between(
    conn: sqlite3.Connection, *, user_id: str, since: str, until: str
) -> list[LinkRating]:
    """*user_id*'s link ratings with ``since <= rated_at < until``."""
    return [
        LinkRating(
            turn_id=r["turn_id"],
            earlier_turn_id=r["earlier_turn_id"],
            user_id=r["user_id"],
            project=r["project"],
            rating=r["rating"],
            rated_at=r["rated_at"],
        )
        for r in conn.execute(
            "SELECT * FROM link_ratings WHERE user_id = ? AND rated_at >= ? AND rated_at < ? "
            "ORDER BY rated_at, turn_id, earlier_turn_id",
            (user_id, since, until),
        )
    ]


def digests_quoting(conn: sqlite3.Connection, *, kind: RefKind, ref_ids: str) -> list[str]:
    """The ids of every digest whose refs name any id in the JSON array *ref_ids* under
    *kind*, through ``idx_digest_refs_ref``. The ids are cast to text: a turn id is an
    integer in the array and text in the column."""
    rows = conn.execute(
        "SELECT DISTINCT digest_id FROM digest_refs WHERE kind = ? "
        "AND ref_id IN (SELECT CAST(value AS TEXT) FROM json_each(?)) ORDER BY digest_id",
        (kind, ref_ids),
    )
    return [str(r["digest_id"]) for r in rows]


def backfill_entrypoint(
    conn: sqlite3.Connection, *, harness: str, native_session_id: str, entrypoint: str
) -> int:
    """Set *entrypoint*, with ``entrypoint_source = 'backfill'``, on this native session's
    digests whose entrypoint is still unknown (``''``), and return how many. A known
    entrypoint is never overwritten. Joins the caller's transaction."""
    if not entrypoint:
        return 0
    updated = conn.execute(
        "UPDATE digests SET entrypoint = ?, entrypoint_source = 'backfill' "
        "WHERE harness = ? AND native_session_id = ? AND entrypoint = ''",
        (entrypoint, harness, native_session_id),
    )
    return updated.rowcount
