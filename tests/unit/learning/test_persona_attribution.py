"""Cold-path persona attribution.

The rule the tests exist for: an inference with no target never becomes a disposition.
Only the user's own statement about themselves may enter the graph unanchored.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.learning.persona_attribution import (
    LLMAttributor,
    NullAttributor,
    Observation,
    ObservationBatch,
    PersonaAttributor,
)
from morgan_brain.models.message import Message, Role
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.personalization.persona_graph import PersonaGraph, PersonaKind
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def graph():
    conn = open_db(":memory:")
    yield PersonaGraph(conn)
    conn.close()


class _Fixed:
    def __init__(self, batch: ObservationBatch) -> None:
        self._batch = batch

    async def observe(self, messages):
        return self._batch


def _msgs(text: str) -> list[Message]:
    return [Message(user_id=U, role=Role.USER, content=text)]


async def _run(graph, batch) -> int:
    return await PersonaAttributor(graph=graph, attributor=_Fixed(batch)).attribute(
        user_id=U, project=P, session_id="s1", messages=_msgs("whatever"), now=T0
    )


async def test_a_targeted_observation_becomes_a_cross_entity_node(graph):
    written = await _run(
        graph,
        ObservationBatch(
            observations=[Observation(description="impatient", entity="harbor sync", valence=-0.6)]
        ),
    )
    assert written == 1
    node = graph.all_nodes(user_id=U, project=P)[0]
    assert node.kind is PersonaKind.CROSS_ENTITY
    assert node.entity == "harbor sync"


async def test_an_untargeted_inference_is_dropped(graph):
    """The load-bearing refusal: "the user is impatient", inferred, with nothing it is
    about. Recording it is the over-personalization failure."""
    written = await _run(
        graph,
        ObservationBatch(observations=[Observation(description="impatient", valence=-0.6)]),
    )
    assert written == 0
    assert graph.all_nodes(user_id=U, project=P) == []


async def test_an_untargeted_statement_by_the_user_is_kept(graph):
    """ "I hate long answers" is the user describing themselves. That is not an inference,
    and it is the one thing allowed in unanchored."""
    written = await _run(
        graph,
        ObservationBatch(
            observations=[
                Observation(description="prefers short answers", stated=True, valence=0.2)
            ]
        ),
    )
    assert written == 1
    assert graph.all_nodes(user_id=U, project=P)[0].kind is PersonaKind.INTRINSIC


async def test_an_empty_description_is_ignored(graph):
    assert await _run(graph, ObservationBatch(observations=[Observation(description="  ")])) == 0


async def test_the_null_attributor_writes_nothing(graph):
    written = await PersonaAttributor(graph=graph, attributor=NullAttributor()).attribute(
        user_id=U, project=P, session_id="s1", messages=_msgs("I am furious"), now=T0
    )
    assert written == 0
    assert graph.all_nodes(user_id=U, project=P) == []


async def test_an_unreachable_model_records_nothing_rather_than_guessing(graph):
    reg = CapabilityRegistry.from_seed({})
    attributor = LLMAttributor(router=RoleRouter(reg=reg, bindings={}), capability_registry=reg)
    written = await PersonaAttributor(graph=graph, attributor=attributor).attribute(
        user_id=U, project=P, session_id="s1", messages=_msgs("I am furious"), now=T0
    )
    assert written == 0


async def test_the_llm_reading_reaches_the_graph(graph):
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    client = FakeChatClient(
        replies=[
            '{"observations": [{"description": "impatient", "entity": "harbor sync", '
            '"valence": -0.6, "stated": false}]}'
        ]
    )
    router = RoleRouter(reg=reg, bindings={"reflection": [Binding("fake", "test-model", client)]})
    attributor = LLMAttributor(router=router, capability_registry=reg)

    written = await PersonaAttributor(graph=graph, attributor=attributor).attribute(
        user_id=U, project=P, session_id="s1", messages=_msgs("that sync again"), now=T0
    )
    assert written == 1
    assert graph.all_nodes(user_id=U, project=P)[0].entity == "harbor sync"


async def test_a_turn_with_no_user_text_is_not_sent_to_the_model(graph):
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    client = FakeChatClient(replies=[])
    router = RoleRouter(reg=reg, bindings={"reflection": [Binding("fake", "test-model", client)]})
    attributor = LLMAttributor(router=router, capability_registry=reg)

    batch = await attributor.observe([Message(user_id=U, role=Role.ASSISTANT, content="hi")])
    assert batch.observations == []
