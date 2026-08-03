"""Step 7 writes the turn's project, not `default`.

Two shipped bug fixes had no regression test, and mutation proved it: hardcoding
`record_turn(project="default")` or `history_store.append(..., project="default")` in
`Orchestrator._persist_turn` left the whole suite green. Both are one assertion, and both
matter because `forget()` finds rows by project -- a signal or transcript written under
`default` is unreachable when the owner erases the project it actually belongs to.
"""

from __future__ import annotations

from typing import Any

from morgan_brain.core.orchestrator import Orchestrator


class _RecordingHistoryStore:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str]] = []

    def append(self, key: str, message: Any, *, project: str) -> None:
        self.appended.append((key, project))

    def recent(self, key: str, *, limit: int = 10, project: str = "default") -> list[Any]:
        return []


class _RecordingRecorder:
    def __init__(self) -> None:
        self.projects: list[str] = []

    async def record_turn(self, *, project: str, **_: Any) -> str:
        self.projects.append(project)
        return "sig-1"


async def test_persist_turn_threads_the_project_to_both_writers() -> None:
    history = _RecordingHistoryStore()
    recorder = _RecordingRecorder()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._history_store = history  # type: ignore[attr-defined]
    orchestrator._recorder = recorder  # type: ignore[attr-defined]

    await orchestrator._persist_turn(  # type: ignore[attr-defined]
        user_id="u",
        project="plata",
        session_id="s",
        turn_id="t",
        text="what blocked the deploy?",
        reply="the Harbor mirror",
    )

    # Both user and assistant messages, both under the turn's project.
    assert [p for _, p in history.appended] == ["plata", "plata"], history.appended
    assert recorder.projects == ["plata"], recorder.projects
