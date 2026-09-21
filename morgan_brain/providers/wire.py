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


#: What became of a model request that did not succeed: no connection could be made, or one
#: was made and the answer never came, came as a server error, or was cut off.
Outcome = Literal["unreachable", "slow"]

#: How long a cold embedding host takes to load its model, said with every failure that looks
#: like one. Measured: 7-8 s over three runs, 43 s on a first load from disk
#: (docs/measurements/2026-09-phase0-baseline.md).
COLD_LOAD = "a cold host loads the model in seconds, 43 s on a first load from disk"


class ProviderUnreachable(ConnectionError):
    """The model endpoint could not be reached, or gave no answer in time.

    Raised by both adapters (chat and embeddings). Carries the endpoint so the message can
    name it: "Connection error." tells the owner nothing about *which* server is down. It also
    names ``setting``, the variable that addresses the endpoint, which the adapter is given by
    the factory: embeddings go to the chat endpoint unless MORGAN_EMBEDDING_ENDPOINT is set,
    so the embedding adapter alone cannot tell which of the two to check.

    ``outcome`` says which of two things happened. ``unreachable``: no connection was made --
    the host is off or the address is wrong. ``slow``: a connection was made and the answer
    did not come in time, came as a server error, or was cut off -- which is also how a cold
    host loading its model looks, so that message says how long a load takes, and never
    calls the host unreachable. ``retried`` builds the embedder's error once its budget is
    spent; the chat adapter, which does not retry, raises the plain form.
    """

    def __init__(
        self,
        endpoint: str,
        detail: str,
        *,
        setting: str,
        outcome: Outcome = "unreachable",
        verdict: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.detail = detail
        self.setting = setting
        self.outcome: Outcome = outcome
        verdict = verdict or f"is unreachable ({detail})"
        super().__init__(
            f"model endpoint {endpoint} {verdict}; check {setting} and run `morgan doctor`"
        )

    @classmethod
    def retried(
        cls,
        endpoint: str,
        *,
        setting: str,
        outcome: Outcome,
        error: str,
        attempts: int,
        seconds: float,
    ) -> ProviderUnreachable:
        """The embedder's budget for *outcome* is spent: *attempts* attempts over *seconds*,
        the last of which failed with *error* (an exception's name, or ``HTTP 503``)."""
        plural = "" if attempts == 1 else "s"
        detail = f"{error} after {attempts} attempt{plural} over {seconds:.1f} s"
        if outcome == "unreachable":
            verdict = f"is unreachable: {detail}"
        else:
            verdict = f"answered too slowly or dropped: {detail}; {COLD_LOAD}"
        return cls(endpoint, detail, setting=setting, outcome=outcome, verdict=verdict)


class ProviderRefused(Exception):
    """The model endpoint answered, and refused the request: a 4xx other than 429, or a
    redirect, which is not followed.

    Not retried -- the same request gets the same answer -- and not a ``ProviderUnreachable``:
    the host is up, so checking whether it runs sends the owner to the wrong place. On a 401
    or 403 ``setting`` is the key setting whose value was sent; on any other status it is the
    setting that addresses the endpoint. ``detail`` is the start of what the server said,
    which names the problem when the server does (a model it has never heard of, say).
    """

    def __init__(self, endpoint: str, status: int, setting: str, *, detail: str = "") -> None:
        self.endpoint = endpoint
        self.status = status
        self.setting = setting
        self.detail = detail
        said = f" ({detail})" if detail else ""
        super().__init__(
            f"model endpoint {endpoint} refused the request: HTTP {status}{said}; check {setting} "
            "and run `morgan doctor`"
        )


class EmbeddingSpaceMismatch(Exception):
    """The model answering embedding requests is not the one that wrote the stored vectors.

    Two models of the same width are indistinguishable by width alone, so without this every
    stored vector would be searched by a model that never wrote it: wrong answers, no error.
    Raised by ``memory/checked_embedder.py`` on a process's first embedding call, when the
    model's answers fall outside the measured tolerance of the active space's fingerprint, or
    of the vectors already stored, come back at another width or with a non-finite component,
    or cannot be checked against the stored vectors at all (``established`` is then false).
    The message names the space, the setting that addresses the model, what went wrong, what
    it means and what to check.
    """

    def __init__(
        self,
        *,
        space_id: int,
        model: str,
        dims: int,
        setting: str,
        detail: str,
        established: bool = True,
    ) -> None:
        self.space_id = space_id
        self.model = model
        self.dims = dims
        self.setting = setting
        self.detail = detail
        self.established = established
        verdict = "does not match" if established else "cannot be verified against"
        risk = "would" if established else "might"
        super().__init__(
            f"embedding space {space_id} ({model}, {dims} dims) {verdict} the model at "
            f"{setting}: {detail}; stored vectors {risk} be searched with the wrong model; "
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

    @classmethod
    def non_finite(
        cls, *, space_id: int, model: str, dims: int, setting: str, value: float
    ) -> EmbeddingSpaceMismatch:
        """The model answered with a NaN or infinite component: no vector the space holds."""
        detail = f"it returned a vector with a non-finite component ({value})"
        return cls(space_id=space_id, model=model, dims=dims, setting=setting, detail=detail)

    @classmethod
    def unverified(
        cls, *, space_id: int, model: str, dims: int, setting: str
    ) -> EmbeddingSpaceMismatch:
        """The space holds vectors, but none could be sampled to check the model against, so
        no fingerprint was recorded: the model is neither proven nor disproven."""
        detail = (
            "the space holds stored vectors but none could be sampled to re-embed, so no "
            "fingerprint was recorded"
        )
        return cls(
            space_id=space_id,
            model=model,
            dims=dims,
            setting=setting,
            detail=detail,
            established=False,
        )


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
