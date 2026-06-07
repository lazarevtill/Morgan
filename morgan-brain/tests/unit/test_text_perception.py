from morgan_brain.modules.perception.text.analyzer import TextPerception
from morgan_brain.models.perception import Modality


async def test_returns_fused_perception_for_text():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="Remind me about Berlin on Monday")
    assert out.text == "Remind me about Berlin on Monday"
    assert out.modalities_used == [Modality.TEXT]
    assert out.intent.name in {"chat", "command", "question"}


async def test_extracts_capitalized_entities():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="I met Alice in Berlin")
    names = {e.name for e in out.entities}
    assert "Alice" in names and "Berlin" in names


async def test_question_intent_detected():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="What time is it?")
    assert out.intent.name == "question"
