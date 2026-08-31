"""The pattern register — Ouroboros Principle 2, made checkable.

The point is not that corrections are stored; the optimizer already sees those. The point
is that a class which recurs *after* its fix was recorded is distinguishable from one
that has never been addressed. Without that distinction the register is just a second
copy of the signal store, and the optimizer keeps proposing the same patch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.learning.patterns import (
    PatternRegister,
    PatternStatus,
    class_id_for,
    render_for_optimizer,
)
from morgan_brain.modules.memory.stores.db import open_db

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def register():
    conn = open_db(":memory:")
    yield PatternRegister(conn)
    conn.close()


def _record(reg, title, *, day=0, description=""):
    return reg.record(
        user_id=U, project=P, title=title, description=description, now=T0 + timedelta(days=day)
    )


# ---------------------------------------------------------------------------
# Classes, not instances
# ---------------------------------------------------------------------------


def test_the_same_class_recorded_twice_is_one_row_with_a_count(register):
    _record(register, "replies are too long")
    p = _record(register, "replies are too long", day=1)
    assert p.occurrences == 2
    assert len(register.all_patterns(user_id=U, project=P)) == 1


def test_the_class_is_matched_on_the_title_regardless_of_case_and_padding(register):
    _record(register, "Replies are too long")
    _record(register, "  replies are too long  ")
    assert len(register.all_patterns(user_id=U, project=P)) == 1


def test_different_classes_stay_separate(register):
    _record(register, "replies are too long")
    _record(register, "code blocks are missing the language tag")
    assert len(register.all_patterns(user_id=U, project=P)) == 2


def test_classes_are_user_and_project_scoped(register):
    _record(register, "replies are too long")
    register.record(user_id="u2", project=P, title="replies are too long", now=T0)
    assert len(register.all_patterns(user_id=U, project=P)) == 1


# ---------------------------------------------------------------------------
# Did the fix hold?
# ---------------------------------------------------------------------------


def test_a_new_class_is_open(register):
    assert _record(register, "replies are too long").status is PatternStatus.OPEN


def test_recording_a_fix_marks_the_class_addressed_and_resets_the_counter(register):
    for i in range(4):
        _record(register, "replies are too long", day=i)
    p = register.set_structural_fix(
        user_id=U,
        project=P,
        class_id=class_id_for("replies are too long"),
        fix="champion preprompt now states a length ceiling",
    )
    assert p.status is PatternStatus.ADDRESSED
    assert p.occurrences == 4
    assert p.occurrences_since_fix == 0


def test_a_recurrence_after_the_fix_marks_the_class_regressed(register):
    """The signal the whole register exists for: the fix was at the wrong depth."""
    for i in range(4):
        _record(register, "replies are too long", day=i)
    register.set_structural_fix(
        user_id=U, project=P, class_id=class_id_for("replies are too long"), fix="a length ceiling"
    )
    p = _record(register, "replies are too long", day=9)
    assert p.status is PatternStatus.REGRESSED
    assert p.occurrences == 5
    assert p.occurrences_since_fix == 1


def test_the_pre_fix_evidence_is_not_mixed_into_the_post_fix_count(register):
    for i in range(4):
        _record(register, "replies are too long", day=i)
    register.set_structural_fix(
        user_id=U, project=P, class_id=class_id_for("replies are too long"), fix="a ceiling"
    )
    _record(register, "replies are too long", day=9)
    p = register.get(user_id=U, project=P, class_id=class_id_for("replies are too long"))
    assert (p.occurrences, p.occurrences_since_fix) == (5, 1)


# ---------------------------------------------------------------------------
# What the optimizer is told
# ---------------------------------------------------------------------------


def test_a_class_seen_once_is_not_offered_to_the_optimizer(register):
    """One correction is a thing that happened. Reshaping the champion around it is how
    a single stray edit becomes policy."""
    _record(register, "replies are too long")
    assert register.recurring(user_id=U, project=P) == []


def test_a_recurring_class_is_offered(register):
    for i in range(3):
        _record(register, "replies are too long", day=i)
    assert [p.title for p in register.recurring(user_id=U, project=P)] == ["replies are too long"]


def test_recurring_classes_come_back_most_frequent_first(register):
    for i in range(3):
        _record(register, "replies are too long", day=i)
    for i in range(6):
        _record(register, "code blocks lack a language tag", day=i)
    assert [p.title for p in register.recurring(user_id=U, project=P)] == [
        "code blocks lack a language tag",
        "replies are too long",
    ]


def test_the_rendered_context_says_how_many_times(register):
    for i in range(3):
        _record(register, "replies are too long", day=i)
    text = render_for_optimizer(register.recurring(user_id=U, project=P))
    assert "seen 3×" in text
    assert "fix the class, not the instance" in text


def test_the_rendered_context_names_a_fix_that_did_not_hold(register):
    """This is the sentence that changes what the model proposes: not "here is a
    correction" but "your previous fix did not close this"."""
    for i in range(3):
        _record(register, "replies are too long", day=i)
    register.set_structural_fix(
        user_id=U,
        project=P,
        class_id=class_id_for("replies are too long"),
        fix="a length ceiling in the preprompt",
    )
    for i in range(2):
        _record(register, "replies are too long", day=10 + i)

    text = render_for_optimizer(register.recurring(user_id=U, project=P))
    assert "previous fix: a length ceiling in the preprompt" in text
    assert "RECURRED 2×" in text
    assert "wrong depth" in text


def test_rendering_nothing_produces_nothing(register):
    assert render_for_optimizer([]) == ""


# ---------------------------------------------------------------------------
# The register reaching the optimizer prompt
# ---------------------------------------------------------------------------


async def test_recurring_classes_reach_the_reflection_prompt(register):
    """The register only matters if the model sees it. This asserts on the actual
    message sent, not on a helper's return value."""
    from morgan_brain.learning.optimizer import ReflectiveOptimizer
    from morgan_brain.providers.capability import CapabilityRegistry
    from morgan_brain.providers.router import Binding, RoleRouter

    class _Recorder:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def agenerate(self, messages, *, model, **kwargs):
            self.messages.append(messages[-1].content)

            class _R:
                text = "- be terse"

            return _R()

    for i in range(4):
        _record(register, "replies are too long", day=i, description="the user trims them")

    client = _Recorder()
    reg = CapabilityRegistry.from_seed({})
    optimizer = ReflectiveOptimizer(
        router=RoleRouter(reg=reg, bindings={"reflection": [Binding("fake", "m", client)]}),
        patterns=register,
        pattern_user_id=U,
    )
    await optimizer.optimize("p", train=[], scorer=lambda body: 0.0, max_calls=1, current_body="")

    assert "RECURRING CLASSES" in client.messages[0]
    assert "seen 4×" in client.messages[0]


async def test_no_register_leaves_the_prompt_as_it_was(register):
    from morgan_brain.learning.optimizer import ReflectiveOptimizer
    from morgan_brain.providers.capability import CapabilityRegistry
    from morgan_brain.providers.router import Binding, RoleRouter

    class _Recorder:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def agenerate(self, messages, *, model, **kwargs):
            self.messages.append(messages[-1].content)

            class _R:
                text = "- be terse"

            return _R()

    client = _Recorder()
    reg = CapabilityRegistry.from_seed({})
    optimizer = ReflectiveOptimizer(
        router=RoleRouter(reg=reg, bindings={"reflection": [Binding("fake", "m", client)]})
    )
    await optimizer.optimize("p", train=[], scorer=lambda body: 0.0, max_calls=1, current_body="")

    assert "RECURRING CLASSES" not in client.messages[0]


def test_a_class_is_aggregated_across_projects_for_the_optimizer(register):
    """The champion preprompt is one document per user. A class recurring in three
    projects is one strong class, not three weak ones -- and reading a single project
    would have made the register invisible for every real project name, since
    corrections are recorded under the project of the turn that produced them."""
    for project in ("acme", "morgan", "homelab"):
        register.record(user_id=U, project=project, title="replies are too long", now=T0)

    assert register.recurring(user_id=U, project="acme") == []
    across = register.recurring(user_id=U, project=None)
    assert [(p.title, p.occurrences) for p in across] == [("replies are too long", 3)]


def test_a_class_that_regressed_in_one_project_is_reported_as_regressed(register):
    """Reporting it as merely "addressed" because two other projects are quiet is the
    reassurance this register exists to withhold."""
    for project in ("acme", "morgan", "homelab"):
        register.record(user_id=U, project=project, title="replies are too long", now=T0)
    register.set_structural_fix(
        user_id=U,
        project="acme",
        class_id=class_id_for("replies are too long"),
        fix="a length ceiling",
    )
    register.record(user_id=U, project="acme", title="replies are too long", now=T0)

    across = register.recurring(user_id=U, project=None)
    assert across[0].status is PatternStatus.REGRESSED
    assert across[0].occurrences_since_fix == 1
