import pytest

from morgan_brain.providers.wire import (
    ChatMessage,
    ChatResult,
    EmbeddingSpaceMismatch,
    StreamDelta,
    ToolCall,
    Usage,
)


def test_chatmessage_roundtrips_openai_dict():
    m = ChatMessage(role="user", content="hi")
    assert m.to_openai() == {"role": "user", "content": "hi"}


def test_chatresult_holds_text_and_usage():
    r = ChatResult(text="ok", model="m", usage=Usage(input_tokens=3, output_tokens=1))
    assert r.text == "ok" and r.usage.output_tokens == 1 and r.tool_calls == []


def test_stream_delta_kinds():
    d = StreamDelta(kind="text_delta", text="x")
    assert d.kind == "text_delta" and d.text == "x"


def test_to_openai_serializes_tool_calls():
    from morgan_brain.providers.wire import ChatMessage

    m = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
    )
    d = m.to_openai()
    assert d["tool_calls"][0]["id"] == "c1"
    assert d["tool_calls"][0]["type"] == "function"
    assert d["tool_calls"][0]["function"]["name"] == "search"
    import json

    assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}


_SPACE = {"space_id": 1, "model": "a-model", "dims": 4, "setting": "MORGAN_EMBEDDING_ENDPOINT"}


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingSpaceMismatch.cosines(
            **_SPACE,
            against="fingerprint",
            min_cosine=0.41,
            tolerance=0.995,
            failed=5,
            compared=5,
        ),
        EmbeddingSpaceMismatch.cosines(
            **_SPACE,
            against="stored-row",
            min_cosine=0.41,
            tolerance=0.995,
            failed=1,
            compared=5,
        ),
        EmbeddingSpaceMismatch.width(**_SPACE, got=3),
    ],
    ids=["fingerprint-cosines", "stored-row-cosines", "width"],
)
def test_a_different_model_answering_says_how_to_change_models_on_purpose(error):
    """Nothing re-embeds a database's vectors or retires its space: a model changed on purpose
    needs a database of its own, and only a mismatch a different model causes says so."""
    message = str(error)
    assert "check MORGAN_EMBEDDING_MODEL or run `morgan doctor --vectors`" in message
    assert "point MORGAN_DATA_DIR at a new database" in message


def test_a_model_of_another_width_is_told_the_width_its_new_database_needs():
    message = str(EmbeddingSpaceMismatch.width(**_SPACE, got=3))

    assert "set MORGAN_EMBEDDING_DIM=3 and point MORGAN_DATA_DIR at a new database" in message


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingSpaceMismatch.non_finite(**_SPACE, value=float("nan")),
        EmbeddingSpaceMismatch.wrong_count(**_SPACE, expected=5, got=3),
        EmbeddingSpaceMismatch.unverified(**_SPACE),
        EmbeddingSpaceMismatch.no_fingerprint(**_SPACE),
        EmbeddingSpaceMismatch.no_active_space(setting="MORGAN_EMBEDDING_ENDPOINT"),
    ],
    ids=["non-finite", "wrong-count", "unverified", "no-fingerprint", "no-active-space"],
)
def test_a_failure_that_is_not_a_model_change_does_not_send_the_owner_to_a_new_database(error):
    """A server answering NaN or too few vectors, or a space that cannot be checked yet, is not
    a different model: leaving the database would lose every memory in it for nothing."""
    assert "MORGAN_DATA_DIR" not in str(error)
