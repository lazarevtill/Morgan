"""Unit tests for built-in tools.

All tests are deterministic:
* No network — FetchUrlTool receives a fake HTTP client.
* No real embedding — MemorySearchTool receives a MemoryGate over a fake MemoryStore.
* Clock is injected for CurrentTimeTool.
* CalculatorTool uses a safe AST evaluator (no eval).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, TemporalFact
from morgan_brain.modules.tools.builtin.calculator import CalculatorTool, safe_eval
from morgan_brain.modules.tools.builtin.clock_tool import CurrentTimeTool
from morgan_brain.modules.tools.builtin.fetch_url import FetchUrlTool
from morgan_brain.modules.tools.builtin.memory_search import MemorySearchTool
from morgan_brain.security.memory_gate import MemoryGate


# ===========================================================================
# CalculatorTool
# ===========================================================================


class TestCalculatorTool:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    # -----------------------------------------------------------------------
    # Correct arithmetic
    # -----------------------------------------------------------------------

    async def test_addition(self) -> None:
        r = await self.tool.run(user_id="u1", expression="2 + 2")
        assert r.ok is True
        assert r.output == 4

    async def test_operator_precedence(self) -> None:
        """2 + 3 * 4 must equal 14, not 20."""
        r = await self.tool.run(user_id="u1", expression="2+3*4")
        assert r.ok is True
        assert r.output == 14

    async def test_parentheses(self) -> None:
        r = await self.tool.run(user_id="u1", expression="(2+3)*4")
        assert r.ok is True
        assert r.output == 20

    async def test_float_division(self) -> None:
        r = await self.tool.run(user_id="u1", expression="7 / 2")
        assert r.ok is True
        assert r.output == pytest.approx(3.5)

    async def test_floor_division(self) -> None:
        r = await self.tool.run(user_id="u1", expression="7 // 2")
        assert r.ok is True
        assert r.output == 3

    async def test_power(self) -> None:
        r = await self.tool.run(user_id="u1", expression="2 ** 10")
        assert r.ok is True
        assert r.output == 1024

    async def test_modulo(self) -> None:
        r = await self.tool.run(user_id="u1", expression="17 % 5")
        assert r.ok is True
        assert r.output == 2

    async def test_unary_negation(self) -> None:
        r = await self.tool.run(user_id="u1", expression="-3 + 5")
        assert r.ok is True
        assert r.output == 2

    # -----------------------------------------------------------------------
    # Safety: must REJECT dangerous inputs (returns ok=False, does NOT exec)
    # -----------------------------------------------------------------------

    async def test_rejects_import_statement(self) -> None:
        r = await self.tool.run(user_id="u1", expression="__import__('os')")
        assert r.ok is False
        assert r.error is not None

    async def test_rejects_dunder_name(self) -> None:
        r = await self.tool.run(user_id="u1", expression="__class__")
        assert r.ok is False

    async def test_rejects_function_call(self) -> None:
        r = await self.tool.run(user_id="u1", expression="print('hello')")
        assert r.ok is False

    async def test_rejects_name_access(self) -> None:
        r = await self.tool.run(user_id="u1", expression="os.getcwd()")
        assert r.ok is False

    async def test_rejects_string_literal(self) -> None:
        r = await self.tool.run(user_id="u1", expression="'hello'")
        assert r.ok is False

    async def test_rejects_list_literal(self) -> None:
        r = await self.tool.run(user_id="u1", expression="[1, 2, 3]")
        assert r.ok is False

    async def test_rejects_invalid_syntax(self) -> None:
        r = await self.tool.run(user_id="u1", expression="2 +* 3")
        assert r.ok is False

    async def test_rejects_division_by_zero(self) -> None:
        r = await self.tool.run(user_id="u1", expression="1 / 0")
        assert r.ok is False
        assert r.error is not None

    # -----------------------------------------------------------------------
    # safe_eval unit tests
    # -----------------------------------------------------------------------

    def test_safe_eval_basic(self) -> None:
        assert safe_eval("2+3*4") == 14

    def test_safe_eval_raises_on_name(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("x + 1")

    def test_safe_eval_raises_on_dunder_injection(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("__import__('os').system('echo pwned')")

    # -----------------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------------

    def test_schema_has_expression_property(self) -> None:
        s = self.tool.schema()
        assert "expression" in s["properties"]

    def test_protocol_attributes(self) -> None:
        assert self.tool.name == "calculator"
        assert self.tool.description


# ===========================================================================
# CurrentTimeTool
# ===========================================================================


class TestCurrentTimeTool:
    def _fixed_clock(self) -> datetime:
        return datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)

    def setup_method(self) -> None:
        self.tool = CurrentTimeTool(clock=self._fixed_clock)

    async def test_returns_injected_time(self) -> None:
        r = await self.tool.run(user_id="u1")
        assert r.ok is True
        assert r.output == "2026-06-08T12:00:00+00:00"

    async def test_returns_iso_string(self) -> None:
        r = await self.tool.run(user_id="u1")
        # ISO-8601 parse must not raise
        parsed = datetime.fromisoformat(str(r.output))
        assert parsed.year == 2026

    def test_default_clock_is_used_when_none_injected(self) -> None:
        tool = CurrentTimeTool()
        # Just verify it has a callable clock (won't fail)
        assert callable(tool._clock)

    def test_schema_is_dict(self) -> None:
        s = self.tool.schema()
        assert isinstance(s, dict)

    def test_protocol_attributes(self) -> None:
        assert self.tool.name == "current_time"
        assert self.tool.description


# ===========================================================================
# MemorySearchTool
# ===========================================================================


class _FakeMemoryStore:
    """In-memory store backed by a simple list. No embedding, no network."""

    def __init__(self) -> None:
        self._memories: list[Memory] = []

    async def store(self, memory: Memory) -> str:
        self._memories.append(memory)
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        # Return all memories for the user, up to top_k.
        return [m for m in self._memories if m.user_id == query.user_id][: query.top_k]

    async def upsert_fact(self, fact: TemporalFact) -> str:
        return fact.id

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        return []


class TestMemorySearchTool:
    def setup_method(self) -> None:
        self.store = _FakeMemoryStore()
        self.gate = MemoryGate(self.store)  # type: ignore[arg-type]
        self.tool = MemorySearchTool(gate=self.gate)

    async def _seed(self, user_id: str, contents: list[str]) -> None:
        for text in contents:
            m = Memory(user_id=user_id, content=text, kind=MemoryKind.EPISODIC)
            await self.store.store(m)

    async def test_returns_matching_memory_contents(self) -> None:
        await self._seed("u1", ["I love hiking", "I prefer tea over coffee"])
        r = await self.tool.run(user_id="u1", query="hobbies")
        assert r.ok is True
        assert isinstance(r.output, list)
        assert "I love hiking" in r.output

    async def test_top_k_limits_results(self) -> None:
        await self._seed("u1", [f"memory {i}" for i in range(10)])
        r = await self.tool.run(user_id="u1", query="memory", top_k=3)
        assert r.ok is True
        assert len(r.output) <= 3  # type: ignore[arg-type]

    async def test_user_isolation(self) -> None:
        """Memories from another user must not appear."""
        await self._seed("u1", ["u1 secret"])
        await self._seed("u2", ["u2 data"])
        r = await self.tool.run(user_id="u1", query="data")
        assert r.ok is True
        contents: list[str] = r.output  # type: ignore[assignment]
        assert all("u2" not in c for c in contents)

    async def test_empty_memory_returns_empty_list(self) -> None:
        r = await self.tool.run(user_id="u1", query="anything")
        assert r.ok is True
        assert r.output == []

    def test_schema_has_query_property(self) -> None:
        s = self.tool.schema()
        assert "query" in s["properties"]

    def test_protocol_attributes(self) -> None:
        assert self.tool.name == "memory_search"
        assert self.tool.description


# ===========================================================================
# FetchUrlTool — fake HTTP client (no network)
# ===========================================================================


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeHttpClient:
    """In-process fake; returns canned text for any URL."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._default = "<html>default</html>"

    async def get(self, url: str, **_: Any) -> _FakeResponse:
        return _FakeResponse(self._responses.get(url, self._default))


class _ErrorHttpClient:
    async def get(self, url: str, **_: Any) -> _FakeResponse:
        raise ConnectionError("network unavailable")


class TestFetchUrlTool:
    def setup_method(self) -> None:
        self.fake_client = _FakeHttpClient({"https://example.com": "<p>Hello</p>"})
        self.tool = FetchUrlTool(http_client=self.fake_client)  # type: ignore[arg-type]

    async def test_returns_canned_text(self) -> None:
        r = await self.tool.run(user_id="u1", url="https://example.com")
        assert r.ok is True
        assert r.output == "<p>Hello</p>"

    async def test_network_error_returns_ok_false(self) -> None:
        tool = FetchUrlTool(http_client=_ErrorHttpClient())  # type: ignore[arg-type]
        r = await tool.run(user_id="u1", url="https://example.com")
        assert r.ok is False
        assert r.error is not None

    async def test_truncation(self) -> None:
        long_text = "x" * 20_000
        client = _FakeHttpClient({"https://big.com": long_text})
        tool = FetchUrlTool(http_client=client, max_chars=100)  # type: ignore[arg-type]
        r = await tool.run(user_id="u1", url="https://big.com")
        assert r.ok is True
        assert "[truncated]" in str(r.output)
        assert len(str(r.output)) <= 120  # 100 + len("[truncated]\n")

    async def test_unknown_url_returns_default(self) -> None:
        r = await self.tool.run(user_id="u1", url="https://other.com")
        assert r.ok is True
        assert "default" in str(r.output)

    def test_default_permission_is_ask(self) -> None:
        assert FetchUrlTool.default_permission == "ask"

    def test_schema_has_url_property(self) -> None:
        s = self.tool.schema()
        assert "url" in s["properties"]

    def test_protocol_attributes(self) -> None:
        assert self.tool.name == "fetch_url"
        assert self.tool.description
