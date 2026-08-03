"""ReasoningModule — interfaces.Reasoner.

Builds the context window from a ``ReasoningRequest``, routes to a model via
``RoleRouter``, optionally runs a bounded tool-call loop (model → tool → model),
then returns a ``ReasoningResult``.

The module depends on the provider seam (``RoleRouter``) rather than a concrete
LLM client, so adapters can be swapped (Ollama, OpenAI, vLLM…) without touching
this code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from morgan_brain.interfaces.reasoning import ReasoningRequest, ReasoningResult
from morgan_brain.modules.reasoning.context.builder import build_messages
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.wire import ChatMessage

if TYPE_CHECKING:
    from morgan_brain.interfaces.tools import ToolExecutor

_DEFAULT_MAX_TOOL_ITERS = 4


class ReasoningModule:
    def __init__(
        self,
        *,
        router: RoleRouter,
        role: str = "strong",
        executor: "ToolExecutor | None" = None,
        max_tool_iters: int = _DEFAULT_MAX_TOOL_ITERS,
    ) -> None:
        self._router = router
        self._role = role
        self._executor = executor
        self._max_tool_iters = max_tool_iters

    async def generate(self, request: ReasoningRequest) -> ReasoningResult:
        has_tools = bool(request.tools) and self._executor is not None

        # Attempt to get a tool-capable binding when tools are present.
        if has_tools:
            try:
                client, model = self._router.chat_for(self._role, needs_tools=True)
            except LookupError:
                # Fall back to the plain path — no tool-capable model registered.
                has_tools = False
                client, model = self._router.chat_for(self._role)
        else:
            needs_tools = bool(request.tools)
            client, model = self._router.chat_for(self._role, needs_tools=needs_tools)

        messages = build_messages(request)
        invoked: list[str] = []

        if has_tools and self._executor is not None:
            # Bounded tool-call loop: model may call tools up to max_tool_iters times.
            result = await client.agenerate(messages, model=model, tools=request.tools)
            for _ in range(self._max_tool_iters):
                if not result.tool_calls:
                    break
                # Append the assistant message with its tool_calls.
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=result.text,
                        tool_calls=result.tool_calls,
                    )
                )
                # Execute each tool call and append the tool-result messages.
                for tc in result.tool_calls:
                    # The model does not get to choose the project: a `project` key in its
                    # tool-call arguments is dropped, and the turn's project is passed
                    # explicitly. Without the strip this call would also raise TypeError on
                    # the duplicate keyword.
                    model_args = {k: v for k, v in tc.arguments.items() if k != "project"}
                    tr = await self._executor.execute(
                        tc.name,
                        user_id=request.user_id,
                        project=request.project,
                        **model_args,
                    )
                    tool_content = str(tr.output) if tr.ok else f"ERROR: {tr.error}"
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=tool_content,
                            tool_call_id=tc.id,
                        )
                    )
                    invoked.append(tc.name)
                # Ask the model again with the tool results in context.
                result = await client.agenerate(messages, model=model, tools=request.tools)
        else:
            result = await client.agenerate(messages, model=model)

        return ReasoningResult(text=result.text, model_used=model, tools_invoked=invoked)

    async def stream(self, request: ReasoningRequest) -> AsyncIterator[str]:
        """Streaming variant.

        When tools are present: runs the (non-streaming) tool-call loop to completion
        then yields the final answer as a single chunk.  Full streaming of tool-call
        deltas is a later refinement.

        When no tools: streams token deltas as they arrive (original behaviour).
        """
        has_tools = bool(request.tools) and self._executor is not None

        if has_tools:
            # Run the full loop synchronously, then yield the result.
            result = await self.generate(request)
            yield result.text
            return

        needs_tools = bool(request.tools)
        try:
            client, model = self._router.chat_for(self._role, needs_tools=needs_tools)
        except LookupError:
            client, model = self._router.chat_for(self._role)

        messages = build_messages(request)
        async for delta in client.astream(messages, model=model):
            if delta.kind == "text_delta" and delta.text:
                yield delta.text
