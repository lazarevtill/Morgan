"""doctor's row counts say what they counted.

Run from the home folder, doctor counted the project 'personal' and printed the number under a
label that read like a total. The owner concluded the import had not landed.
"""

from __future__ import annotations

from pathlib import Path

from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor

#: ``memories`` exactly as every Morgan before phase 0 created it (frozen in
#: ``tests/unit/memory/test_provenance_columns.py::_PRE_PHASE_ZERO_DDL`` too) -- no
#: ``author_id``, the column ``rows_missing_provenance`` checks first.
_PRE_PHASE_ZERO_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    project    TEXT NOT NULL DEFAULT 'default',
    kind       TEXT NOT NULL,
    source     TEXT NOT NULL,
    content    TEXT NOT NULL,
    importance REAL NOT NULL,
    entities   TEXT NOT NULL,
    created_at TEXT
);
"""


def _settings(tmp_path: Path) -> Settings:
    # Port 1 refuses at once: doctor's chat probe contacts no server of the developer's own.
    return Settings(
        data_dir=str(tmp_path), embedding_backend="hash", llm_endpoint="http://127.0.0.1:1/v1"
    )


async def _store(tmp_path: Path, *, project: str, count: int) -> None:
    """*count* memories in *project*, each carrying provenance the way a real writer does
    (``surfaces/cli/commands.py::cmd_remember`` sets ``author_id=settings.owner_user_id``) --
    so a test that wants a row *missing* provenance blanks it explicitly, rather than getting
    one for free from a fixture that never set it.
    """
    settings = _settings(tmp_path)
    ctx = build_memory_context(settings)
    try:
        for i in range(count):
            await ctx.gate.store(
                Memory(
                    user_id=settings.owner_user_id,
                    project=project,
                    content=f"{project} memory {i}",
                    author_id=settings.owner_user_id,
                )
            )
    finally:
        ctx.conn.close()


async def _report(tmp_path: Path, *, project: str, all_projects: bool = False) -> dict:
    return await build_doctor_report(
        _settings(tmp_path), project=project, all_projects=all_projects
    )


def _blank_the_author(tmp_path: Path) -> None:
    """What a pre-phase-0 process's write left behind: every column phase-0 code fills in
    with something, this one left empty."""
    settings = _settings(tmp_path)
    conn = open_db(sqlite_path(settings.temporal_db_url))
    try:
        conn.execute("UPDATE memories SET author_id = ''")
        conn.commit()
    finally:
        conn.close()


async def test_the_scoped_count_and_the_total_are_both_there(tmp_path):
    await _store(tmp_path, project="Morgan", count=2)
    await _store(tmp_path, project="personal", count=3)

    report = await _report(tmp_path, project="personal")

    assert report["rows"] == {"scope": "project 'personal'", "memories": 3, "fts": 3, "vectors": 3}
    assert report["rows_all_projects"]["memories"] == 5
    assert report["rows_by_project"] == {"Morgan": 2, "personal": 3}


async def test_the_same_totals_from_either_project(tmp_path):
    await _store(tmp_path, project="Morgan", count=2)
    from_morgan = await _report(tmp_path, project="Morgan")
    from_personal = await _report(tmp_path, project="personal")
    assert from_morgan["rows_all_projects"] == from_personal["rows_all_projects"]


async def test_a_row_written_without_provenance_is_counted(tmp_path):
    await _store(tmp_path, project="Morgan", count=1)
    _blank_the_author(tmp_path)
    assert (await _report(tmp_path, project="Morgan"))["rows_missing_provenance"] == 1


async def test_an_unmigrated_database_reports_missing_provenance_as_null_with_a_reason(tmp_path):
    """Before migration step 4 has run, ``memories`` has no ``author_id`` column at all -- a
    count against it would be meaningless, not an honest zero, so the report says why instead
    of guessing or raising."""
    settings = _settings(tmp_path)
    conn = open_db(sqlite_path(settings.temporal_db_url))
    conn.execute(_PRE_PHASE_ZERO_MEMORIES)
    conn.commit()
    conn.close()

    report = await _report(tmp_path, project="p")

    assert report["rows_missing_provenance"] is None
    assert report["rows_missing_provenance_reason"] is not None
    assert "migrate" in report["rows_missing_provenance_reason"]


async def test_a_vec_items_row_with_a_null_status_is_counted(tmp_path):
    """sqlite-vec 0.1.9 itself refuses NULL for a declared TEXT metadata column -- verified
    against this pin: both ``INSERT`` and a plain ``UPDATE`` raise ``Expected text for TEXT
    metadata column status, received NULL``, matching the upstream tracking issue
    (asg017/sqlite-vec#141, "NULL values are not supported yet"). A real vec0 table can
    therefore never hold the row this test counts: the pre-phase-0 writer the controller
    ruling describes would crash before landing one, not write one quietly. The counting SQL
    doesn't know or care whether ``vec_items`` is the real vec0 table or not, so a plain table
    of the same name and shape exercises exactly the query `doctor` runs, without fighting the
    extension's own validation to prove it -- `SqliteVectorIndex`'s `CREATE VIRTUAL TABLE IF
    NOT EXISTS` leaves a same-named table already there alone, whatever kind it is.
    """
    await _store(tmp_path, project="Morgan", count=1)
    settings = _settings(tmp_path)
    conn = open_db(sqlite_path(settings.temporal_db_url))
    conn.execute("DROP TABLE vec_items")
    conn.execute(
        """
        CREATE TABLE vec_items (
            rowid     INTEGER PRIMARY KEY,
            embedding BLOB,
            user_id   TEXT,
            project   TEXT,
            status    TEXT,
            scope     TEXT,
            author_id TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO vec_items (rowid, user_id, project, status, scope, author_id) "
        "VALUES (1, ?, 'Morgan', NULL, 'private', ?)",
        (settings.owner_user_id, settings.owner_user_id),
    )
    conn.commit()
    conn.close()

    report = await _report(tmp_path, project="Morgan")

    assert report["rows_missing_provenance"] == 1


async def test_a_checked_table_entirely_absent_is_named_not_silently_zeroed(tmp_path):
    """A checked table the database does not have is named in the reason, never counted as
    zero rows. doctor builds no store, so a table missing from the file stays missing -- here
    ``vec_items``, beside a ``memories`` that holds a row: a clean ``0`` would describe a
    database whose vector index is not there at all."""
    await _store(tmp_path, project="p", count=1)
    conn = open_db(sqlite_path(_settings(tmp_path).temporal_db_url))
    conn.execute("DROP TABLE vec_items")
    conn.commit()
    conn.close()

    report = await _report(tmp_path, project="p")

    assert report["rows_missing_provenance"] is None
    assert "vec_items" in report["rows_missing_provenance_reason"]


async def test_the_scoped_line_renders_with_its_total(tmp_path):
    """`render.py` is on this task's file list and otherwise has no test touching doctor's
    plain-text form: the spec's own example, `memories: 3 in project 'personal' (5 across all
    projects)`, rendered from a real report rather than a hand-built dict."""
    await _store(tmp_path, project="Morgan", count=2)
    await _store(tmp_path, project="personal", count=3)

    report = await _report(tmp_path, project="personal")
    rendered = _render_doctor(report)

    assert "memories: 3 in project 'personal' (5 across all projects)" in rendered.splitlines()
