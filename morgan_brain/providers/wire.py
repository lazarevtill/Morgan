"""Provider-neutral wire types (OpenAI Chat Completions shape). No provider SDK imported here."""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResult(BaseModel):
    text: str = ""
    model: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"


class StreamDelta(BaseModel):
    kind: Literal["text_delta", "tool_call_delta", "usage", "finish"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: str | None = None


class ProviderUnreachable(ConnectionError):
    """The model endpoint could not be reached, or gave no answer in time.

    Raised by both adapters (chat and embeddings). Carries the endpoint so the message can
    name it: "Connection error." tells the owner nothing about *which* server is down. It also
    names ``setting``, the variable that addresses the endpoint, which the adapter is given by
    the factory: embeddings go to the chat endpoint unless MORGAN_EMBEDDING_ENDPOINT is set,
    so the embedding adapter alone cannot tell which of the two to check.
    """

    def __init__(self, endpoint: str, detail: str, *, setting: str) -> None:
        self.endpoint = endpoint
        self.detail = detail
        self.setting = setting
        super().__init__(
            f"model endpoint {endpoint} is unreachable ({detail}); check {setting} "
            "and run `morgan doctor`"
        )


class EmbeddingSpaceMismatch(Exception):
    """The model answering embedding requests is not the one that wrote the stored vectors.

    Two models of the same width are indistinguishable by width alone, so without this every
    stored vector would be searched by a model that never wrote it: wrong answers, no error.
    Raised by ``memory/checked_embedder.py`` on a process's first embedding call, when the
    model's answers fall outside the measured tolerance of the active space's fingerprint, or
    of the vectors already stored, or come back at another width. The message names the space,
    the setting that addresses the model, what went wrong, what it means and what to check.
    """

    def __init__(self, *, space_id: int, model: str, dims: int, setting: str, detail: str) -> None:
        self.space_id = space_id
        self.model = model
        self.dims = dims
        self.setting = setting
        self.detail = detail
        super().__init__(
            f"embedding space {space_id} ({model}, {dims} dims) does not match the model at "
            f"{setting}: {detail}; stored vectors would be searched with the wrong model; "
            "check MORGAN_EMBEDDING_MODEL or run `morgan doctor --vectors`"
        )

    @classmethod
    def cosines(
        cls,
        *,
        space_id: int,
        model: str,
        dims: int,
        setting: str,
        against: Literal["fingerprint", "stored-row"],
        min_cosine: float,
        tolerance: float,
        failed: int,
        compared: int,
    ) -> EmbeddingSpaceMismatch:
        """Fresh vectors fell below *tolerance* against *against*: ``"fingerprint"`` (the five
        fixed strings) or ``"stored-row"`` (memories whose vectors are already stored)."""
        what = "strings" if against == "fingerprint" else "stored rows"
        detail = f"{against} cosine {min_cosine:.4f} < {tolerance} on {failed} of {compared} {what}"
        return cls(space_id=space_id, model=model, dims=dims, setting=setting, detail=detail)

    @classmethod
    def width(
        cls, *, space_id: int, model: str, dims: int, setting: str, got: int
    ) -> EmbeddingSpaceMismatch:
        """The model answered at a width other than the space's."""
        detail = f"it returned a {got}-dimensional vector but the active space is {dims} wide"
        return cls(space_id=space_id, model=model, dims=dims, setting=setting, detail=detail)


@runtime_checkable
class ChatClient(Protocol):
    """What the core needs from a chat model: one call, messages in, a result out."""

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...
