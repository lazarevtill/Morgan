"""
Tests for the Agent/Subagent system.

Covers:
- AgentDefinition and AgentResult dataclasses
- AgentSpawner with mock run_fn
- AgentSpawner default fallback to get_llm_service
- Frontmatter parsing from markdown content
- Agent loading from .md files on disk
- Built-in agent definitions (researcher, coder, planner)
- Error handling in spawner
"""

import asyncio
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Coroutine, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morgan.agents.base import AgentDefinition, AgentResult
from morgan.agents.loader import load_agents_from_dir, parse_frontmatter
from morgan.agents.spawner import AgentSpawner
from morgan.agents.builtin import BUILTIN_AGENTS
from morgan.agents.builtin.researcher import ResearcherAgent
from morgan.agents.builtin.coder import CoderAgent
from morgan.agents.builtin.planner import PlannerAgent


# =============================================================================
# AgentDefinition Tests
# =============================================================================


class TestAgentDefinition:
    """Tests for the AgentDefinition dataclass."""

    def test_basic_creation(self):
        defn = AgentDefinition(
            agent_type="test",
            when_to_use="When testing things",
            get_system_prompt=lambda: "You are a test agent.",
            tools=["echo"],
        )
        assert defn.agent_type == "test"
        assert defn.when_to_use == "When testing things"
        assert defn.get_system_prompt() == "You are a test agent."
        assert defn.tools == ["echo"]

    def test_defaults(self):
        defn = AgentDefinition(
            agent_type="minimal",
            when_to_use="Minimal agent",
            get_system_prompt=lambda: "Minimal",
            tools=[],
        )
        assert defn.disallowed_tools == []
        assert defn.model is None
        assert defn.effort == "balanced"
        assert defn.max_turns == 10
        assert defn.source == "builtin"

    def test_custom_fields(self):
        defn = AgentDefinition(
            agent_type="custom",
            when_to_use="Custom agent",
            get_system_prompt=lambda: "Custom prompt",
            tools=["bash", "file_read"],
            disallowed_tools=["file_write"],
            model="gpt-4",
            effort="thorough",
            max_turns=20,
            source="user_defined",
        )
        assert defn.disallowed_tools == ["file_write"]
        assert defn.model == "gpt-4"
        assert defn.effort == "thorough"
        assert defn.max_turns == 20
        assert defn.source == "user_defined"

    def test_get_system_prompt_closure(self):
        """Verify get_system_prompt works as a closure capturing state."""
        context = {"role": "analyst"}

        def make_prompt():
            return f"You are an {context['role']}."

        defn = AgentDefinition(
            agent_type="closure_test",
            when_to_use="Testing closures",
            get_system_prompt=make_prompt,
            tools=[],
        )
        assert defn.get_system_prompt() == "You are an analyst."
        # Mutate the captured variable
        context["role"] = "engineer"
        assert defn.get_system_prompt() == "You are an engineer."


# =============================================================================
# AgentResult Tests
# =============================================================================


class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_successful_result(self):
        result = AgentResult(
            output="Analysis complete.",
            agent_type="researcher",
            success=True,
        )
        assert result.output == "Analysis complete."
        assert result.agent_type == "researcher"
        assert result.success is True
        assert result.error is None
        assert result.metadata == {}

    def test_failed_result(self):
        result = AgentResult(
            output="",
            agent_type="coder",
            success=False,
            error="LLM service unavailable",
        )
        assert result.success is False
        assert result.error == "LLM service unavailable"

    def test_result_with_metadata(self):
        result = AgentResult(
            output="Done",
            agent_type="planner",
            success=True,
            metadata={"turns_used": 3, "tokens": 1500},
        )
        assert result.metadata["turns_used"] == 3
        assert result.metadata["tokens"] == 1500


# =============================================================================
# AgentSpawner Tests
# =============================================================================


class TestAgentSpawner:
    """Tests for AgentSpawner with mock run_fn."""

    @pytest.mark.asyncio
    async def test_spawn_with_custom_run_fn(self):
        """Spawner uses the provided run_fn instead of LLM service."""

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return f"Mock response to: {prompt}"

        defn = AgentDefinition(
            agent_type="mock_agent",
            when_to_use="Testing",
            get_system_prompt=lambda: "You are a test agent.",
            tools=["echo"],
        )
        spawner = AgentSpawner(run_fn=mock_run)
        result = await spawner.spawn(defn, prompt="Hello agent")

        assert result.success is True
        assert result.output == "Mock response to: Hello agent"
        assert result.agent_type == "mock_agent"

    @pytest.mark.asyncio
    async def test_spawn_passes_system_prompt(self):
        """Verify system prompt from definition is passed to run_fn."""
        captured = {}

        async def capture_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            captured["system_prompt"] = system_prompt
            captured["tools"] = tools
            captured["model"] = model
            return "ok"

        defn = AgentDefinition(
            agent_type="capture_agent",
            when_to_use="Capture test",
            get_system_prompt=lambda: "Be precise and thorough.",
            tools=["web_search", "memory_search"],
            model="custom-model",
        )
        spawner = AgentSpawner(run_fn=capture_run)
        await spawner.spawn(defn, prompt="Search for X")

        assert captured["system_prompt"] == "Be precise and thorough."
        assert captured["tools"] == ["web_search", "memory_search"]
        assert captured["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_spawn_with_context(self):
        """Verify context is appended to the prompt."""

        async def echo_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return prompt

        defn = AgentDefinition(
            agent_type="ctx_agent",
            when_to_use="Context test",
            get_system_prompt=lambda: "System",
            tools=[],
        )
        spawner = AgentSpawner(run_fn=echo_run)
        result = await spawner.spawn(
            defn, prompt="Do X", context="Background info here"
        )

        assert "Do X" in result.output
        assert "Background info here" in result.output

    @pytest.mark.asyncio
    async def test_spawn_handles_run_fn_error(self):
        """Spawner returns a failed AgentResult on run_fn exception."""

        async def failing_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            raise RuntimeError("LLM exploded")

        defn = AgentDefinition(
            agent_type="failing_agent",
            when_to_use="Failure test",
            get_system_prompt=lambda: "System",
            tools=[],
        )
        spawner = AgentSpawner(run_fn=failing_run)
        result = await spawner.spawn(defn, prompt="Boom")

        assert result.success is False
        assert "LLM exploded" in result.error
        assert result.agent_type == "failing_agent"

    @pytest.mark.asyncio
    async def test_spawn_default_uses_llm_service(self):
        """When no run_fn provided, spawner falls back to get_llm_service."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM service response"
        mock_llm.agenerate = AsyncMock(return_value=mock_response)

        with patch(
            "morgan.services.get_llm_service", return_value=mock_llm
        ):
            defn = AgentDefinition(
                agent_type="default_agent",
                when_to_use="Default fallback test",
                get_system_prompt=lambda: "Default system prompt",
                tools=["bash"],
            )
            spawner = AgentSpawner()
            result = await spawner.spawn(defn, prompt="Use default LLM")

        assert result.success is True
        assert result.output == "LLM service response"
        mock_llm.agenerate.assert_awaited_once()
        call_kwargs = mock_llm.agenerate.call_args
        assert call_kwargs.kwargs["system_prompt"] == "Default system prompt"


# =============================================================================
# Frontmatter Parsing Tests
# =============================================================================


class TestParseFrontmatter:
    """Tests for parse_frontmatter utility."""

    def test_basic_frontmatter(self):
        content = textwrap.dedent("""\
            ---
            agent_type: researcher
            when_to_use: Research tasks
            tools:
              - web_search
              - memory_search
            effort: thorough
            ---
            You are a researcher agent. Be thorough.
        """)
        meta, body = parse_frontmatter(content)
        assert meta["agent_type"] == "researcher"
        assert meta["when_to_use"] == "Research tasks"
        assert meta["tools"] == ["web_search", "memory_search"]
        assert meta["effort"] == "thorough"
        assert "You are a researcher agent." in body

    def test_no_frontmatter(self):
        content = "Just a plain prompt body."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == "Just a plain prompt body."

    def test_empty_frontmatter(self):
        content = textwrap.dedent("""\
            ---
            ---
            Body only.
        """)
        meta, body = parse_frontmatter(content)
        assert meta == {} or meta is None  # empty YAML
        assert "Body only." in body

    def test_frontmatter_with_all_fields(self):
        content = textwrap.dedent("""\
            ---
            agent_type: custom
            when_to_use: Custom tasks
            tools:
              - bash
              - file_read
            disallowed_tools:
              - file_write
            model: gpt-4
            effort: balanced
            max_turns: 15
            ---
            Custom agent prompt.
        """)
        meta, body = parse_frontmatter(content)
        assert meta["agent_type"] == "custom"
        assert meta["disallowed_tools"] == ["file_write"]
        assert meta["model"] == "gpt-4"
        assert meta["max_turns"] == 15
        assert "Custom agent prompt." in body

    def test_frontmatter_multiline_body(self):
        content = textwrap.dedent("""\
            ---
            agent_type: writer
            when_to_use: Writing tasks
            tools: []
            ---
            Line one.

            Line two.

            Line three.
        """)
        meta, body = parse_frontmatter(content)
        assert "Line one." in body
        assert "Line two." in body
        assert "Line three." in body


# =============================================================================
# Agent Loader Tests
# =============================================================================


class TestLoadAgentsFromDir:
    """Tests for load_agents_from_dir."""

    def test_load_single_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_file = Path(tmpdir) / "test_agent.md"
            agent_file.write_text(textwrap.dedent("""\
                ---
                agent_type: test_loader
                when_to_use: Testing the loader
                tools:
                  - echo
                effort: balanced
                max_turns: 5
                ---
                You are a test agent loaded from disk.
            """))

            agents = load_agents_from_dir(tmpdir)
            assert len(agents) == 1
            agent = agents[0]
            assert agent.agent_type == "test_loader"
            assert agent.when_to_use == "Testing the loader"
            assert agent.tools == ["echo"]
            assert agent.effort == "balanced"
            assert agent.max_turns == 5
            assert "test agent loaded from disk" in agent.get_system_prompt()
            assert agent.source == str(agent_file)

    def test_load_multiple_agents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                f = Path(tmpdir) / f"agent_{i}.md"
                f.write_text(textwrap.dedent(f"""\
                    ---
                    agent_type: agent_{i}
                    when_to_use: Task {i}
                    tools: []
                    ---
                    Prompt for agent {i}.
                """))

            agents = load_agents_from_dir(tmpdir)
            assert len(agents) == 3
            types = {a.agent_type for a in agents}
            assert types == {"agent_0", "agent_1", "agent_2"}

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents = load_agents_from_dir(tmpdir)
            assert agents == []

    def test_load_nonexistent_dir(self):
        agents = load_agents_from_dir("/nonexistent/path/to/agents")
        assert agents == []

    def test_load_skips_non_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # .md file
            md = Path(tmpdir) / "good.md"
            md.write_text(textwrap.dedent("""\
                ---
                agent_type: good
                when_to_use: Good
                tools: []
                ---
                Good agent.
            """))
            # .txt file (should be skipped)
            txt = Path(tmpdir) / "bad.txt"
            txt.write_text("Not an agent.")

            agents = load_agents_from_dir(tmpdir)
            assert len(agents) == 1
            assert agents[0].agent_type == "good"

    def test_load_handles_malformed_frontmatter(self):
        """Loader should skip files with missing required frontmatter fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.md"
            bad.write_text("No frontmatter at all, just text.")

            good = Path(tmpdir) / "good.md"
            good.write_text(textwrap.dedent("""\
                ---
                agent_type: good_agent
                when_to_use: Valid agent
                tools:
                  - bash
                ---
                Valid agent prompt.
            """))

            agents = load_agents_from_dir(tmpdir)
            # Only the good agent should load
            assert len(agents) == 1
            assert agents[0].agent_type == "good_agent"


# =============================================================================
# Built-in Agent Tests
# =============================================================================


class TestBuiltinAgents:
    """Tests for built-in agent definitions."""

    def test_builtin_agents_list(self):
        assert len(BUILTIN_AGENTS) == 3
        types = {a.agent_type for a in BUILTIN_AGENTS}
        assert types == {"researcher", "coder", "planner"}

    def test_researcher_agent(self):
        agent = ResearcherAgent
        assert agent.agent_type == "researcher"
        assert "web_search" in agent.tools
        assert "memory_search" in agent.tools
        assert "file_read" in agent.tools
        assert agent.effort == "thorough"
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_coder_agent(self):
        agent = CoderAgent
        assert agent.agent_type == "coder"
        assert "bash" in agent.tools
        assert "file_read" in agent.tools
        assert "file_write" in agent.tools
        assert "calculator" in agent.tools
        assert agent.effort == "thorough"
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_planner_agent(self):
        agent = PlannerAgent
        assert agent.agent_type == "planner"
        assert "file_read" in agent.tools
        assert "web_search" in agent.tools
        assert "memory_search" in agent.tools
        assert agent.effort == "balanced"
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_all_builtins_are_valid_definitions(self):
        for agent in BUILTIN_AGENTS:
            assert isinstance(agent, AgentDefinition)
            assert agent.agent_type
            assert agent.when_to_use
            assert callable(agent.get_system_prompt)
            assert isinstance(agent.tools, list)
            assert agent.source == "builtin"


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestAgentSpawnerIntegration:
    """Integration tests combining spawner with built-in definitions."""

    @pytest.mark.asyncio
    async def test_spawn_builtin_researcher(self):
        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            assert "web_search" in tools
            return "Research findings: ..."

        spawner = AgentSpawner(run_fn=mock_run)
        result = await spawner.spawn(ResearcherAgent, prompt="Find info on X")
        assert result.success is True
        assert result.agent_type == "researcher"

    @pytest.mark.asyncio
    async def test_spawn_builtin_coder(self):
        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            assert "bash" in tools
            assert "file_write" in tools
            return "```python\nprint('hello')\n```"

        spawner = AgentSpawner(run_fn=mock_run)
        result = await spawner.spawn(CoderAgent, prompt="Write a script")
        assert result.success is True
        assert result.agent_type == "coder"

    @pytest.mark.asyncio
    async def test_spawn_loaded_agent(self):
        """Load an agent from .md and spawn it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "helper.md"
            f.write_text(textwrap.dedent("""\
                ---
                agent_type: helper
                when_to_use: Helping with things
                tools:
                  - echo
                effort: balanced
                max_turns: 3
                ---
                You are a helpful assistant.
            """))

            agents = load_agents_from_dir(tmpdir)
            assert len(agents) == 1

            async def mock_run(
                prompt: str,
                system_prompt: str,
                tools: List[str],
                model: Optional[str],
            ) -> str:
                return f"Helped with: {prompt}"

            spawner = AgentSpawner(run_fn=mock_run)
            result = await spawner.spawn(agents[0], prompt="Help me")
            assert result.success is True
            assert result.output == "Helped with: Help me"
            assert result.agent_type == "helper"
