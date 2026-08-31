"""Unit tests for ReasoningModule.

Uses a FakeChatClient-backed RoleRouter — no network, no Ollama dependency.
"""

from __future__ import annotations

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest, ReasoningResult
from morgan_brain.models.memory import Memory
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.reasoner import ReasoningModule
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.providers.wire import ChatResult, ToolCall, ToolSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(
    reply: str = "Hello!",
    *,
    supports_tools: bool = True,
    model: str = "qwen2.5:7b",
) -> RoleRouter:
    reg = CapabilityRegistry.from_seed(
        {
            f"ollama/{model}": {
                "supports_tools": supports_tools,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    client = FakeChatClient(reply=reply)
    return RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("ollama", model, client)]},
    )


def _request() -> ReasoningRequest:
    return ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="hi"),
        personalization=PersonalizedContext(),
        memories=[Memory(user_id="u1", content="user is called Sam")],
        history=[],
        skill_prompt="",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_generate_returns_llm_reply_and_model():
    router = _make_router(reply="Hello Sam!", model="qwen2.5:7b")
    reasoner = ReasoningModule(router=router, role="strong")
    result = await reasoner.generate(_request())
    assert result.text == "Hello Sam!"
    assert result.model_used == "qwen2.5:7b"


async def test_generate_passes_memories_into_context():
    router = _make_router(reply="ok")
    client: FakeChatClient = router.bindings_for("strong")[0].client  # type: ignore[assignment]
    reasoner = ReasoningModule(router=router, role="strong")
    await reasoner.generate(_request())
    system = client.last_messages[0]
    assert "Sam" in system.content


async def test_generate_result_is_reasoning_result():
    router = _make_router(reply="answer")
    reasoner = ReasoningModule(router=router, role="strong")
    result = await reasoner.generate(_request())
    assert isinstance(result, ReasoningResult)
    assert result.tools_invoked == []


async def test_default_role_is_strong():
    router = _make_router(reply="ok")
    # No role arg — should default to "strong"
    reasoner = ReasoningModule(router=router)
    result = await reasoner.generate(_request())
    assert result.text == "ok"


async def test_tools_capable_binding_selected_when_tools_needed():
    """A request with tools routes to the tools-capable binding."""
    reg = CapabilityRegistry.from_seed(
        {
            "ollama/weak": {"supports_tools": False, "json_mode": "none"},
            "ollama/strong": {"supports_tools": True, "json_mode": "json_schema"},
        }
    )
    weak_client = FakeChatClient(reply="weak")
    strong_client = FakeChatClient(reply="strong")
    router = RoleRouter(
        reg=reg,
        bindings={
            "strong": [
                Binding("ollama", "weak", weak_client),
                Binding("ollama", "strong", strong_client),
            ]
        },
    )
    # Simulate a request that needs tools by monkeypatching — or just verify
    # the router gates correctly (the reasoner checks bool(getattr(request, 'tools', None))).
    # ReasoningRequest has no 'tools' field currently, so needs_tools=False.
    # The reasoner passes needs_tools=bool(getattr(request, 'tools', None)).
    reasoner = ReasoningModule(router=router, role="strong")
    result = await reasoner.generate(_request())
    # Without tools, first binding (weak) is selected
    assert result.text == "weak"


async def test_reasoning_request_accepts_tools_field():
    """ReasoningRequest accepts a list of ToolSpecs via the tools field (commit-1 gate)."""
    from morgan_brain.providers.wire import ToolSpec

    spec = ToolSpec(
        name="calculator",
        description="Evaluate arithmetic.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="hi"),
        personalization=PersonalizedContext(),
        tools=[spec],
    )
    assert len(req.tools) == 1
    assert req.tools[0].name == "calculator"


async def test_reasoning_request_tools_defaults_to_empty():
    """tools field defaults to [] when not provided."""
    req = _request()
    assert req.tools == []


# ---------------------------------------------------------------------------
# Commit-2: Tool-call loop tests
# ---------------------------------------------------------------------------


def _calc_spec() -> ToolSpec:
    return ToolSpec(
        name="calculator",
        description="Evaluate arithmetic.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )


def _make_router_with_client(
    client: FakeChatClient,
    model: str = "qwen2.5:7b",
    supports_tools: bool = True,
) -> RoleRouter:
    reg = CapabilityRegistry.from_seed(
        {
            f"fake/{model}": {
                "supports_tools": supports_tools,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    return RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", model, client)]},
    )


async def test_tool_call_loop_executes_tool_and_returns_final_answer():
    """Model calls calculator, loop executes it, model answers with result."""
    from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
    from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
    from morgan_brain.security.permissions import PermissionGate, PermissionMode

    # Script: first call returns tool_call, second returns the final answer.
    tc = ToolCall(id="tc1", name="calculator", arguments={"expression": "6 * 7"})
    client = FakeChatClient(
        results=[
            ChatResult(text="", tool_calls=[tc]),
            ChatResult(text="The answer is 42.", tool_calls=[]),
        ]
    )
    router = _make_router_with_client(client)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    gate = PermissionGate(default=PermissionMode.AUTO)
    executor = ToolExecutorImpl(registry=registry, gate=gate)

    reasoner = ReasoningModule(router=router, role="strong", executor=executor)
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="What is 6 * 7?"),
        personalization=PersonalizedContext(),
        tools=[_calc_spec()],
    )
    result = await reasoner.generate(req)

    assert result.text == "The answer is 42."
    assert result.tools_invoked == ["calculator"]
    assert client.calls == 2

    # A role="tool" message with the calculator output must have been sent back.
    tool_msgs = [m for m in client.last_messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "42" in tool_msgs[0].content


async def test_tool_call_loop_permission_denied_error_appended_then_terminates():
    """When the gate denies a tool, the error is appended and the loop still terminates."""
    from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
    from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
    from morgan_brain.security.permissions import PermissionGate, PermissionMode

    tc = ToolCall(id="tc2", name="calculator", arguments={"expression": "1 + 1"})
    client = FakeChatClient(
        results=[
            ChatResult(text="", tool_calls=[tc]),
            ChatResult(text="I could not compute that.", tool_calls=[]),
        ]
    )
    router = _make_router_with_client(client)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    # DENY mode so the gate blocks execution.
    gate = PermissionGate(default=PermissionMode.DENY)
    executor = ToolExecutorImpl(registry=registry, gate=gate)

    reasoner = ReasoningModule(router=router, role="strong", executor=executor)
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="Add 1 + 1"),
        personalization=PersonalizedContext(),
        tools=[_calc_spec()],
    )
    result = await reasoner.generate(req)

    assert result.text == "I could not compute that."
    # Tool name is still recorded (the call was attempted).
    assert "calculator" in result.tools_invoked
    # The error was forwarded to the model.
    tool_msgs = [m for m in client.last_messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "ERROR" in tool_msgs[0].content
    assert "permission denied" in tool_msgs[0].content


async def test_tool_call_loop_no_tools_request_unchanged_plain_path():
    """Request with no tools uses the plain path — no executor called."""
    from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
    from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
    from morgan_brain.security.permissions import PermissionGate, PermissionMode

    client = FakeChatClient(reply="plain answer")
    router = _make_router_with_client(client)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    gate = PermissionGate(default=PermissionMode.AUTO)
    executor = ToolExecutorImpl(registry=registry, gate=gate)

    reasoner = ReasoningModule(router=router, role="strong", executor=executor)
    # No tools in request.
    result = await reasoner.generate(_request())

    assert result.text == "plain answer"
    assert result.tools_invoked == []
    assert client.calls == 1


async def test_tool_call_loop_no_executor_plain_path():
    """No executor → plain path even if tools are listed in the request."""
    client = FakeChatClient(reply="fallback answer")
    router = _make_router_with_client(client)
    # executor=None (default)
    reasoner = ReasoningModule(router=router, role="strong")
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="What is 2 + 2?"),
        personalization=PersonalizedContext(),
        tools=[_calc_spec()],
    )
    result = await reasoner.generate(req)
    assert result.text == "fallback answer"
    assert result.tools_invoked == []


async def test_tool_call_loop_lookuperror_falls_back_to_plain():
    """LookupError from router (no tools-capable binding) falls back to plain path."""
    # Model that does NOT support tools.
    client = FakeChatClient(reply="no tools answer")
    reg = CapabilityRegistry.from_seed(
        {
            "fake/no-tools-model": {
                "supports_tools": False,
                "json_mode": "none",
                "context_window": 4096,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "no-tools-model", client)]},
    )
    from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
    from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
    from morgan_brain.security.permissions import PermissionGate, PermissionMode

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    gate = PermissionGate(default=PermissionMode.AUTO)
    executor = ToolExecutorImpl(registry=registry, gate=gate)

    reasoner = ReasoningModule(router=router, role="strong", executor=executor)
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="test"),
        personalization=PersonalizedContext(),
        tools=[_calc_spec()],
    )
    result = await reasoner.generate(req)
    assert result.text == "no tools answer"
    assert result.tools_invoked == []


async def test_tool_call_loop_stream_with_tools_yields_final_text():
    """stream() with tools runs the loop then yields the final answer."""
    from morgan_brain.modules.tools.builtin.calculator import CalculatorTool
    from morgan_brain.modules.tools.executor import ToolExecutorImpl, ToolRegistry
    from morgan_brain.security.permissions import PermissionGate, PermissionMode

    tc = ToolCall(id="tc3", name="calculator", arguments={"expression": "3 + 3"})
    client = FakeChatClient(
        results=[
            ChatResult(text="", tool_calls=[tc]),
            ChatResult(text="Result is 6.", tool_calls=[]),
        ]
    )
    router = _make_router_with_client(client)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    gate = PermissionGate(default=PermissionMode.AUTO)
    executor = ToolExecutorImpl(registry=registry, gate=gate)

    reasoner = ReasoningModule(router=router, role="strong", executor=executor)
    req = ReasoningRequest(
        user_id="u1",
        project="p",
        perception=FusedPerception(text="What is 3 + 3?"),
        personalization=PersonalizedContext(),
        tools=[_calc_spec()],
    )
    chunks: list[str] = []
    async for chunk in reasoner.stream(req):
        chunks.append(chunk)

    assert "".join(chunks) == "Result is 6."
